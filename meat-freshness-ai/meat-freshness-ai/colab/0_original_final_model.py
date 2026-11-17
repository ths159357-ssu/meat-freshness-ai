# ============================================================
# 1. 설정
# ============================================================
import io, glob, os, json, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from IPython.display import display
import joblib

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
CLASS_NAMES = ["Fresh", "Fairly fresh", "Spoiled"]

# 사용 센서 (전체 9개 / 실험 간 반응 방향이 일관된 4개)
SENSORS_ALL = ["MQ135", "MQ136", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ8", "MQ9"]
SENSORS_SEL = ["MQ135", "MQ3", "MQ4", "MQ8"]

# 기준선 구간(분), 평가 시작 시점(분)
BASE_MIN = 10
EVAL_FROM = 30

# 개선 모델 입력 창: 최근 12시간을 20분 간격으로 (36스텝)
WIN_LONG, STEP_LONG = 720, 20

# 성장 모델 파라미터 (온도 누적 지표 계산용)
GROWTH_B, GROWTH_TMIN = 0.0259, -5.0

DATASET_PAGE = "https://data.mendeley.com/datasets/mwmhh766fc/3"
DATASET_API = "https://data.mendeley.com/public-api/datasets/mwmhh766fc/files?folder_id=root&version=3"
DATA_DIR = "wijaya"
print("TensorFlow", tf.__version__)


# ============================================================
# 2. Wijaya 데이터 불러오기 (폴더 확인 → 자동 다운로드 → 수동 업로드)
# ============================================================
# 열 이름 정리
def tidy(d):
    d.columns = [str(c).strip() for c in d.columns]
    ren = {c: c.upper() if c.upper().startswith("MQ") else c.capitalize() for c in d.columns}
    ren.update({c: "TVC" for c in d.columns if c.upper() == "TVC"})
    return d.rename(columns=ren)

# 현재 폴더·하위 폴더의 Wijaya CSV 찾기 (MQ 센서 열이 있는 CSV만)
def find_local_csv():
    out = []
    for f in sorted(set(glob.glob("TS*.csv") + glob.glob(f"{DATA_DIR}/**/*.csv", recursive=True))):
        try:
            head = pd.read_csv(f, sep=None, engine="python", nrows=3)
            if any(str(c).strip().upper().startswith("MQ") for c in head.columns):
                out.append(f)
        except Exception:
            pass
    return out

# Mendeley 공개 API로 데이터 자동 다운로드
def try_download():
    import requests
    r = requests.get(DATASET_API, timeout=30); r.raise_for_status()
    os.makedirs(DATA_DIR, exist_ok=True)
    for it in r.json():
        name = it.get("filename") or it.get("name") or ""
        url = (it.get("content_details") or {}).get("download_url")
        if not url or not name.lower().endswith((".csv", ".zip")):
            continue
        content = requests.get(url, timeout=120).content
        if name.lower().endswith(".zip"):
            zipfile.ZipFile(io.BytesIO(content)).extractall(DATA_DIR)
        else:
            open(os.path.join(DATA_DIR, os.path.basename(name)), "wb").write(content)

files_found = find_local_csv()
if not files_found:
    try:
        print("Mendeley에서 데이터 자동 다운로드 시도 중...")
        try_download()
        files_found = find_local_csv()
    except Exception as e:
        print("자동 다운로드 실패:", type(e).__name__, e)
if not files_found:
    print("\n아래 링크에서 'Download All'로 받은 뒤 압축을 풀고 TS1.csv ~ TS5.csv 5개를 한 번에 선택하세요.")
    print(DATASET_PAGE)
    from google.colab import files
    for name, content in sorted(files.upload().items()):
        open(name, "wb").write(content)
    files_found = find_local_csv()
assert files_found, "TS1.csv ~ TS5.csv 파일이 필요합니다."

raw = [(os.path.basename(f).replace(".csv", ""), tidy(pd.read_csv(f, sep=None, engine="python"))) for f in files_found]
print("실험:", [n for n, _ in raw])


# ============================================================
# 3. 전처리 + 라벨 2종 + 온도 누적 지표
# ============================================================
LABEL_MAP = {"excellent": 0, "good": 0, "acceptable": 1, "spoiled": 2}

# 실험 1개 → 센서 신호, 라벨, 온도 누적 지표
def prepare(name, d):
    d = d.sort_values("Minute").reset_index(drop=True)
    R = np.log(np.clip(d[SENSORS_ALL].astype(float).values, 1e-6, None))
    x = R - R[:BASE_MIN].mean(0)
    x = pd.DataFrame(x, columns=SENSORS_ALL).rolling(5, min_periods=1).median().values.astype("float32")
    tvc = pd.to_numeric(d["TVC"], errors="coerce").values
    y_tvc = np.where(tvc < 4, 0, np.where(tvc < 5, 1, 2))
    y_auth = d["Class"].astype(str).str.strip().str.lower().map(LABEL_MAP).values.astype(int)
    T = d["Temperature"].astype(float).values
    tt = np.cumsum((GROWTH_B * np.clip(T - GROWTH_TMIN, 0, None)) ** 2 / 60.0) / np.log(10)
    return dict(name=name, x=x, y_tvc=y_tvc, y_auth=y_auth, tt=tt, tvc=tvc)

series = [prepare(n, d) for n, d in raw]
sel_idx = [SENSORS_ALL.index(s) for s in SENSORS_SEL]

# TVC 기준 라벨과 연구팀(원 데이터) 라벨 비교
fig, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 3), sharey=True)
for ax, s in zip(np.atleast_1d(axes), series):
    h = np.arange(len(s["x"])) / 60
    ax.step(h, s["y_tvc"], label="TVC 4/5 threshold", alpha=.7)
    ax.step(h, s["y_auth"], label="author label", lw=2)
    ax.set_title(s["name"]); ax.set_xlabel("hour"); ax.set_yticks([0, 1, 2], CLASS_NAMES)
np.atleast_1d(axes)[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
for s in series:
    back = int(((np.diff(s["y_tvc"]) < 0)).sum())
    print(f"{s['name']}: TVC 기준 라벨이 등급이 내려가는 횟수 {back}회, 연구팀 라벨과 일치율 {np.mean(s['y_tvc'] == s['y_auth']):.2f}")


# ============================================================
# 4. 모델 구성 요소
# ============================================================
# 과거 L분을 step 간격으로 자른 입력 창 (앞부분은 첫 값으로 채움, 미래 값 사용 안 함)
def windows(x, L, step):
    idx = np.arange(len(x))[:, None] - np.arange(0, L, step)[::-1][None, :]
    return x[np.clip(idx, 0, None)]

# 1D-CNN
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

# CNN 학습 → 테스트 확률 반환
def cnn_proba(tr, ts, lab, ch_idx, L, step, return_model=False):
    Xtr = np.concatenate([windows(s["x"][:, ch_idx], L, step)[EVAL_FROM:] for s in tr])
    ytr = np.concatenate([s[lab][EVAL_FROM:] for s in tr])
    mu, sd = Xtr.reshape(-1, len(ch_idx)).mean(0), Xtr.reshape(-1, len(ch_idx)).std(0) + 1e-6
    perm = np.random.default_rng(SEED).permutation(len(ytr))
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    tf.keras.utils.set_random_seed(SEED)
    m = build_cnn(Xtr.shape[1], len(ch_idx))
    m.fit(((Xtr - mu) / sd)[perm], ytr[perm], epochs=30, batch_size=256, verbose=0, validation_split=0.15,
          class_weight=dict(zip(np.unique(ytr), cw)),
          callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    p = None if ts is None else m.predict((windows(ts["x"][:, ch_idx], L, step) - mu) / sd, verbose=0, batch_size=2048)
    return (p, m, (mu, sd)) if return_model else p

# 센서 수준(30분 평균) + 온도 누적 지표 입력
def level_tt(s):
    lv = pd.DataFrame(s["x"][:, sel_idx]).rolling(30, min_periods=1).mean().values
    return np.c_[lv, s["tt"]]

# 온도 누적 지표 모델, 센서 수준+온도 모델 학습 → 테스트 확률 반환
def aux_models(tr, lab):
    ytr = np.concatenate([s[lab][EVAL_FROM:] for s in tr])
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(
        np.concatenate([s["tt"][EVAL_FROM:] for s in tr])[:, None], ytr)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED).fit(
        np.concatenate([level_tt(s)[EVAL_FROM:] for s in tr]), ytr)
    return lr, hgb

# 등급은 내려가지 않음: 더 높은 등급이 10분 연속일 때만 올림
def monotone(pred, k=10):
    state, run, cand, out = 0, 0, None, np.empty_like(pred)
    for i, p in enumerate(pred):
        if p > state:
            run = run + 1 if cand == p else 1; cand = p
            if run >= k: state, run = p, 0
        else:
            run, cand = 0, None
        out[i] = state
    return out


# ============================================================
# 5. 실험 단위 교차검증: 기존 모델 vs 개선 단계별 비교
# ============================================================
ALL_IDX = list(range(len(SENSORS_ALL)))
stages = {
    "S0 기존 (TVC 라벨, 30분, 센서 9개)": [],
    "S1 + 연구팀 라벨": [],
    "S2 + 12시간 창, 센서 4개, 단조 판단": [],
    "S3 + 온도 누적 지표 앙상블 (최종)": [],
    "참고: 온도 누적 지표만": [],
}
truth = {k: [] for k in stages}
for te, ts in enumerate(series):
    tr = [s for j, s in enumerate(series) if j != te]
    p0 = cnn_proba(tr, ts, "y_tvc", ALL_IDX, 30, 1).argmax(1)
    p1 = cnn_proba(tr, ts, "y_auth", ALL_IDX, 30, 1).argmax(1)
    pc = cnn_proba(tr, ts, "y_auth", sel_idx, WIN_LONG, STEP_LONG)
    lr, hgb = aux_models(tr, "y_auth")
    p_tt = lr.predict_proba(ts["tt"][:, None]); p_h = hgb.predict_proba(level_tt(ts))
    outs = [(p0, "y_tvc"), (p1, "y_auth"), (monotone(pc.argmax(1)), "y_auth"),
            (monotone(((pc + p_tt + p_h) / 3).argmax(1)), "y_auth"), (monotone(p_tt.argmax(1)), "y_auth")]
    for k, (p, lab) in zip(stages, outs):
        stages[k].append(p[EVAL_FROM:]); truth[k].append(ts[lab][EVAL_FROM:])
    print(f"✔ {ts['name']} 테스트 완료")

rows = []
for k in stages:
    P, Y = np.concatenate(stages[k]), np.concatenate(truth[k])
    rows.append(dict(단계=k, 정확도=accuracy_score(Y, P), macro_F1=f1_score(Y, P, average="macro"),
                     부패_재현율=recall_score(Y, P, labels=[2], average="macro"),
                     부패를_신선으로=((Y == 2) & (P == 0)).sum() / max((Y == 2).sum(), 1),
                     실험별_정확도=[round(accuracy_score(y, p), 3) for p, y in zip(stages[k], truth[k])]))
result = pd.DataFrame(rows).set_index("단계").round(4)
display(result)

# 단계별 정확도 그래프
fig, ax = plt.subplots(figsize=(9, 3.8))
names = ["S0", "S1", "S2", "S3 (final)", "ref: temp only"]
ax.bar(names, result["정확도"].values, color=["#9AA5B1", "#9AA5B1", "#3A7CA5", "#1F8A70", "#D6E6E1"])
ax.axhline(0.9, color="#C8394F", ls="--", lw=1)
for i, v in enumerate(result["정확도"].values): ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy (leave-one-experiment-out)")
plt.tight_layout(); plt.show()

# 최종 모델 혼동행렬
k_final = "S3 + 온도 누적 지표 앙상블 (최종)"
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ConfusionMatrixDisplay(confusion_matrix(np.concatenate(truth[k_final]), np.concatenate(stages[k_final]), normalize="true"),
                       display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, values_format=".2f")
ax.set_title("Final model"); plt.tight_layout(); plt.show()


# ============================================================
# 6. 참고 검사: 경과 시간만으로 맞힐 수 있는지
# ============================================================
from sklearn.tree import DecisionTreeClassifier
P, Y = [], []
for te, ts in enumerate(series):
    tr = [s for j, s in enumerate(series) if j != te]
    Xt = np.concatenate([np.arange(len(s["x"]))[EVAL_FROM:, None] for s in tr]); yt = np.concatenate([s["y_auth"][EVAL_FROM:] for s in tr])
    P.append(DecisionTreeClassifier(max_depth=3).fit(Xt, yt).predict(np.arange(len(ts["x"]))[:, None])[EVAL_FROM:]); Y.append(ts["y_auth"][EVAL_FROM:])
print(f"경과 시간(분)만 사용한 정확도: {accuracy_score(np.concatenate(Y), np.concatenate(P)):.1%}")
print("→ 이 데이터는 모든 실험이 비슷한 일정으로 상해서, 시간만으로도 등급이 대부분 맞음. 센서 기반 성능은 S2 행을 함께 볼 것.")


# ============================================================
# 7. 최종 모델 학습 (5개 실험 전체) + 저장
# ============================================================
_, final_cnn, (MU, SD) = cnn_proba(series, None, "y_auth", sel_idx, WIN_LONG, STEP_LONG, return_model=True)
final_lr, final_hgb = aux_models(series, "y_auth")
final_cnn.save("spoilage_cnn_v2.keras")
joblib.dump(dict(lr=final_lr, hgb=final_hgb), "spoilage_aux_v2.joblib")
json.dump(dict(sensors=SENSORS_SEL, base_minutes=BASE_MIN, window=WIN_LONG, step=STEP_LONG, mean=MU.tolist(), std=SD.tolist(),
               growth_b=GROWTH_B, growth_tmin=GROWTH_TMIN, classes=CLASS_NAMES, monotone_minutes=10),
          open("spoilage_v2_config.json", "w"), ensure_ascii=False, indent=1)

# 새 센서 기록(1분 간격, 9개 센서 원시값 + 온도)으로 등급 판단
def predict_series(df):
    s = prepare("new", tidy(df.copy()).assign(Class="excellent", TVC=0))
    pc = final_cnn.predict((windows(s["x"][:, sel_idx], WIN_LONG, STEP_LONG) - MU) / SD, verbose=0, batch_size=2048)
    p = (pc + final_lr.predict_proba(s["tt"][:, None]) + final_hgb.predict_proba(level_tt(s))) / 3
    return monotone(p.argmax(1))

demo = predict_series(raw[-1][1])
print(f"{raw[-1][0]} 판단 예시: 비교적 신선 진입 {np.argmax(demo >= 1)}분, 부패 진입 {np.argmax(demo >= 2)}분")
for f in ["spoilage_cnn_v2.keras", "spoilage_aux_v2.joblib", "spoilage_v2_config.json"]:
    print(f, f"{os.path.getsize(f)/1024:.1f} KB")
