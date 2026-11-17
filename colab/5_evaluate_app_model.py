# ============================================================
# 4조 육류 신선도 판단 모델 - 앱에 들어간 모델 그대로 평가하기
#   model/model_web.json (앱이 쓰는 가중치)을 numpy로 계산 → 라이브러리 버전과 무관하게 앱과 같은 결과
#   평가 데이터: Wijaya 외(2022) 9~12번 부위 (학습에 사용 안 함)
#   출력: 부위별 정확도, 혼동행렬, CNN만 썼을 때 정확도, 시연용 엑셀 판정
# ============================================================
import os, json, glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import importlib.util
_spec = importlib.util.spec_from_file_location("m", os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_train_export.py"))
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)   # 전처리·데이터 불러오기 함수만 사용

W = json.load(open(os.environ.get("MODEL_JSON", "model/model_web.json"), encoding="utf-8"))

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def conv_same_relu(x, Wt, b):            # x (N,T,C), Wt (k,C,F)
    k = Wt.shape[0]; p = (k - 1) // 2
    xp = np.pad(x, ((0, 0), (p, k - 1 - p), (0, 0)))
    out = sum(np.einsum("ntc,cf->ntf", xp[:, j:j + x.shape[1]], Wt[j]) for j in range(k)) + b
    return np.maximum(out, 0)

def cnn_proba(s):
    X = (m.windows(s["x"]) - np.asarray(W["mu"])) / np.asarray(W["sd"])
    h = conv_same_relu(X, np.asarray(W["conv1"]["W"]), np.asarray(W["conv1"]["b"]))
    h = h[:, : h.shape[1] // 2 * 2].reshape(h.shape[0], -1, 2, h.shape[2]).max(2)          # MaxPool 2
    h = conv_same_relu(h, np.asarray(W["conv2"]["W"]), np.asarray(W["conv2"]["b"]))
    return softmax(h.mean(1) @ np.asarray(W["dense"]["W"]) + np.asarray(W["dense"]["b"]))   # GAP → Dense

def lr_proba(s):
    return softmax(s["tt"][:, None] * np.asarray(W["lr"]["coef"]) + np.asarray(W["lr"]["intercept"]))

def hgb_proba(s):
    F = m.level_tt(s); raw = np.tile(np.asarray(W["hgb"]["baseline"], float), (len(F), 1))
    for it in W["hgb"]["trees"]:
        for k, t in enumerate(it):
            f, th, l, r, leaf, v, ml = (np.asarray(t[c]) for c in ("f", "t", "l", "r", "leaf", "v", "ml"))
            node = np.zeros(len(F), int)
            while True:
                act = leaf[node] == 0
                if not act.any(): break
                i = np.where(act)[0]; nd = node[i]; xv = F[i, f[nd]]
                go_left = np.where(np.isnan(xv), ml[nd] == 1, xv <= th[nd])
                node[i] = np.where(go_left, l[nd], r[nd])
            raw[:, k] += v[node]
    return softmax(raw)

def judge(s):
    pc, pl, ph = cnn_proba(s), lr_proba(s), hgb_proba(s)
    return dict(ens=m.monotone(((pc + pl + ph) / 3).argmax(1)), cnn=m.monotone(pc.argmax(1)), p=(pc + pl + ph) / 3)

if __name__ == "__main__":
    rows, Y, P, Pc = [], [], [], []
    for name, d in m.load_new(m.HOLDOUT_CUTS):
        s = m.prepare(name, d.sort_values("Minute").reset_index(drop=True)); r = judge(s)
        y = s["y"][m.EVAL_FROM:]; Y.append(y); P.append(r["ens"][m.EVAL_FROM:]); Pc.append(r["cnn"][m.EVAL_FROM:])
        rows.append({"평가 부위": name, "최종 모델 정확도": round(accuracy_score(y, P[-1]), 3), "CNN만 정확도": round(accuracy_score(y, Pc[-1]), 3)})
    Y, P, Pc = map(np.concatenate, (Y, P, Pc))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n최종 모델(앱) 정확도: {accuracy_score(Y, P):.4f}   |   CNN만: {accuracy_score(Y, Pc):.4f}")
    print("\n혼동행렬 (행=실제, 열=판단; 신선 / 비교적 신선 / 부패)\n", confusion_matrix(Y, P, labels=[0, 1, 2]))
    print("\n", classification_report(Y, P, target_names=m.CLASS_NAMES, digits=3))

    # 시연용 엑셀(상온 원본 / 냉장 환산)도 같은 모델로 판정
    files = sorted(glob.glob("data/samples*/*.xlsx")) + sorted(glob.glob("*.xlsx"))
    out = []
    for fpath in files:
        e = pd.read_excel(fpath)
        if not set(["Minute", "Temperature"] + m.SENSORS) <= set(e.columns): continue
        r = judge(m.prepare(os.path.basename(fpath), e.sort_values("Minute").reset_index(drop=True)))
        out.append({"파일": fpath, "판단": m.CLASS_NAMES[r["ens"][-1]],
                    "확률(신선/비교적/부패)": " / ".join(f"{v:.1%}" for v in r["p"][-1]),
                    "마지막 시점": f"{e['Minute'].max():,.0f}분 ({e['Minute'].max()/1440:.1f}일)"})
    if out: print(pd.DataFrame(out).to_string(index=False))
