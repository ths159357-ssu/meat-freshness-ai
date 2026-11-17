# ============================================================
# 4조 육류 신선도 판단 모델 - 최종 모델 학습 + 앱용 내보내기
#   모델 구조·전처리는 최종 코드(chicken_spoilage_cnn_v2_colab.py)와 같음
#   학습 데이터:
#     - Wijaya 외(2018) TS1~TS5 (기존 학습 데이터)
#     - Wijaya 외(2022) 소고기 12부위 중 1~8번 부위 (새 장비 데이터, 추가 학습)
#       → 9~12번 부위는 학습에 쓰지 않고 평가·앱 시연에만 사용
#   출력: model/model_web.json  (앱이 읽는 모델 파일)
# ============================================================
import os, glob, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
CLASS_NAMES = ["신선", "비교적 신선", "부패"]
SENSORS = ["MQ135", "MQ3", "MQ4", "MQ8"]     # 최종 모델 입력 센서 4개
BASE_MIN = 10                                # 기준선: 보관 시작 10분
EVAL_FROM = 30                               # 학습·평가는 30분 이후부터
WIN, STEP = 720, 20                          # 최근 12시간, 20분 간격 → 36칸
GROWTH_B, GROWTH_TMIN = 0.0259, -5.0         # Ratkowsky 식 + Cauchie(2020) Pseudomonas
LABEL_MAP = {"excellent": 0, "good": 0, "acceptable": 1, "spoiled": 2}

DATA_DIR = os.environ.get("DATA_DIR", "data/wijaya2018")
NEW_XLSX = os.environ.get("NEW_XLSX", "data/wijaya2022/e-nose_dataset_12_beef_cuts.xlsx")
OUT_DIR = os.environ.get("OUT_DIR", "model")
NEW_TEMP_C = 29.0                            # 2022 실험은 약 29 °C로 유지 (온도 열 없음)
NEW_LABEL = {1: "excellent", 2: "good", 3: "acceptable", 4: "spoiled"}
TRAIN_CUTS = ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8."]       # 학습에 추가하는 부위
HOLDOUT_CUTS = ["9.", "10.", "11.", "12."]                          # 평가·시연 전용 부위


# ---------- 1. 데이터 불러오기 ----------
def load_ts():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "TS*.csv")))
    if not files:   # 콜랩: TS1.csv ~ TS5.csv 업로드
        from google.colab import files as cf
        for name, content in cf.upload().items():
            open(name, "wb").write(content)
        files = sorted(glob.glob("TS*.csv"))
    out = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d = d.rename(columns={c: c.capitalize() for c in d.columns if not c.upper().startswith("MQ") and c != "TVC"})
        out.append((os.path.basename(f)[:-4], d))
    return out


def load_new(cuts):
    """Wijaya 2022 엑셀(12개 시트)에서 지정한 부위만 불러오기"""
    book = pd.read_excel(NEW_XLSX, sheet_name=None)
    out = []
    for name, d in book.items():
        if not any(name.startswith(c) for c in cuts): continue
        d = d.copy()
        d["Temperature"] = NEW_TEMP_C
        d["Class"] = d["Label"].map(NEW_LABEL)
        out.append((name, d))
    return out


# ---------- 2. 전처리 (앱도 똑같이 계산) ----------
def growth_tt(T, minutes=None):
    """온도 기록(1분 간격) → 온도 누적 지표 tt
       중간에 빠진 분이 있으면(일부 구간만 넣은 엑셀) 온도를 직선으로 이어 그 분만큼 더함 → 앱과 같은 계산"""
    rate = lambda t: (GROWTH_B * np.clip(np.asarray(t, float) - GROWTH_TMIN, 0, None)) ** 2 / 60.0
    T = np.asarray(T, float)
    if minutes is None or (np.all(np.diff(minutes) == 1) and minutes[0] == 1):
        return np.cumsum(rate(T)) / np.log(10)
    out, acc = np.empty(len(T)), rate(T[0]) * max(minutes[0], 1); out[0] = acc
    for i in range(1, len(T)):
        g = float(minutes[i] - minutes[i - 1]); n = int(np.floor(g))   # 측정 간격(분, 소수 가능)
        acc += rate(T[i - 1] + (T[i] - T[i - 1]) * np.arange(1, n + 1) / g).sum() + (g - n) * rate(T[i])
        out[i] = acc
    return out / np.log(10)

def prepare(name, d, label_col="Class"):
    d = d.sort_values("Minute").reset_index(drop=True)
    R = np.log(np.clip(d[SENSORS].astype(float).values, 1e-6, None))
    x = R - R[:BASE_MIN].mean(0)                                               # 보관 시작 10분 대비 변화
    x = pd.DataFrame(x).rolling(5, min_periods=1).median().values.astype("float32")  # 5분 이동 중앙값
    y = d[label_col].astype(str).str.strip().str.lower().map(LABEL_MAP).values if label_col in d else None
    return dict(name=name, x=x, y=y, tt=growth_tt(d["Temperature"].values, d["Minute"].values))

def windows(x, L=WIN, step=STEP):
    idx = np.arange(len(x))[:, None] - np.arange(0, L, step)[::-1][None, :]
    return x[np.clip(idx, 0, None)]

def level_tt(s):
    lv = pd.DataFrame(s["x"]).rolling(30, min_periods=1).mean().values
    return np.c_[lv, s["tt"]]


# ---------- 3. 모델 ----------
def build_cnn(T, ch):
    m = keras.Sequential([layers.Input((T, ch)),
                          layers.Conv1D(16, 5, activation="relu", padding="same"),
                          layers.MaxPooling1D(2),
                          layers.Conv1D(16, 3, activation="relu", padding="same"),
                          layers.GlobalAveragePooling1D(),
                          layers.Dropout(0.2),
                          layers.Dense(3, activation="softmax")])
    m.compile(optimizer=keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy")
    return m

def train_all(tr):
    Xtr = np.concatenate([windows(s["x"])[EVAL_FROM:] for s in tr])
    ytr = np.concatenate([s["y"][EVAL_FROM:] for s in tr])
    mu, sd = Xtr.reshape(-1, Xtr.shape[2]).mean(0), Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    perm = np.random.default_rng(SEED).permutation(len(ytr))
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    tf.keras.utils.set_random_seed(SEED)
    cnn = build_cnn(Xtr.shape[1], Xtr.shape[2])
    cnn.fit(((Xtr - mu) / sd)[perm], ytr[perm], epochs=30, batch_size=256, verbose=0, validation_split=0.15,
            class_weight=dict(zip(np.unique(ytr), cw)),
            callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(
        np.concatenate([s["tt"][EVAL_FROM:] for s in tr])[:, None], ytr)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                         random_state=SEED).fit(np.concatenate([level_tt(s)[EVAL_FROM:] for s in tr]), ytr)
    return dict(cnn=cnn, mu=mu, sd=sd, lr=lr, hgb=hgb)

def monotone(pred, k=10):
    """등급은 내려가지 않음: 더 높은 등급이 10분 연속일 때만 올림"""
    state, run, cand, out = 0, 0, None, np.empty_like(pred)
    for i, p in enumerate(pred):
        if p > state:
            run = run + 1 if cand == p else 1; cand = p
            if run >= k: state, run = p, 0
        else:
            run, cand = 0, None
        out[i] = state
    return out

def predict_proba(M, s):
    pc = M["cnn"].predict((windows(s["x"]) - M["mu"]) / M["sd"], verbose=0, batch_size=2048)
    return (pc + M["lr"].predict_proba(s["tt"][:, None]) + M["hgb"].predict_proba(level_tt(s))) / 3

def predict_grade(M, s):
    return monotone(predict_proba(M, s).argmax(1))


# ---------- 4. 앱용 내보내기 (JSON) ----------
def export_web(M, path):
    w = M["cnn"].get_weights()   # conv1 W,b / conv2 W,b / dense W,b
    trees = []
    for it in M["hgb"]._predictors:
        per = []
        for p in it:
            n = p.nodes
            per.append(dict(f=n["feature_idx"].tolist(), t=[float(v) for v in n["num_threshold"]],
                            l=n["left"].tolist(), r=n["right"].tolist(), leaf=n["is_leaf"].astype(int).tolist(),
                            v=[float(v) for v in n["value"]], ml=n["missing_go_to_left"].astype(int).tolist()))
        trees.append(per)
    out = dict(
        version="v2-web-1", classes=CLASS_NAMES, sensors=SENSORS, base_minutes=BASE_MIN,
        window=WIN, step=STEP, growth_b=GROWTH_B, growth_tmin=GROWTH_TMIN, monotone_minutes=10,
        mu=M["mu"].tolist(), sd=M["sd"].tolist(),
        conv1=dict(W=w[0].tolist(), b=w[1].tolist()), conv2=dict(W=w[2].tolist(), b=w[3].tolist()),
        dense=dict(W=w[4].tolist(), b=w[5].tolist()),
        lr=dict(coef=M["lr"].coef_[:, 0].tolist(), intercept=M["lr"].intercept_.tolist()),
        hgb=dict(baseline=np.ravel(M["hgb"]._baseline_prediction).tolist(), trees=trees))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, separators=(",", ":"))
    return out


if __name__ == "__main__":
    old = [prepare(n, d) for n, d in load_ts()]
    new = [prepare(n, d) for n, d in load_new(TRAIN_CUTS)]
    print("학습 데이터:", [s["name"] for s in old + new])

    M = train_all(old + new)
    os.makedirs(OUT_DIR, exist_ok=True)
    M["cnn"].save(os.path.join(OUT_DIR, "spoilage_cnn_v2.keras"))
    export_web(M, os.path.join(OUT_DIR, "model_web.json"))
    import joblib; joblib.dump(dict(lr=M["lr"], hgb=M["hgb"], mu=M["mu"], sd=M["sd"]), os.path.join(OUT_DIR, "spoilage_aux_v2.joblib"))
    print("저장 완료:", os.listdir(OUT_DIR))
