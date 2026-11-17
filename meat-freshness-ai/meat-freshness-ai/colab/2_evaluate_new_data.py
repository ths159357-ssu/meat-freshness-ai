# ============================================================
# 4조 육류 신선도 판단 모델 - 새 데이터로 평가 + 앱 시연 구간 만들기
#   평가 데이터: Wijaya 외(2022) 소고기 12부위 중 9~12번 부위
#               (1_train_export.py 에서 학습에 쓰지 않은 부위)
#   1) 부위별 정확도, 혼동행렬
#   2) 앱 시연 구간 4개(비교적 신선 2, 부패 2)를 엑셀로 잘라 저장 + 앱용 samples.json
# ============================================================
import os, json
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix

import importlib.util
_spec = importlib.util.spec_from_file_location("m", os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_train_export.py"))
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
OUT_DIR = os.environ.get("SAMPLE_DIR", "data/samples")

aux = joblib.load(os.path.join(MODEL_DIR, "spoilage_aux_v2.joblib"))
M = dict(cnn=keras.models.load_model(os.path.join(MODEL_DIR, "spoilage_cnn_v2.keras")), **aux)

# ---------- 1. 평가 ----------
res, rows, allP, allY = {}, [], [], []
for name, d in m.load_new(m.HOLDOUT_CUTS):
    d = d.sort_values("Minute").reset_index(drop=True)
    s = m.prepare(name, d)
    p = m.predict_proba(M, s); g = m.monotone(p.argmax(1)); y = s["y"]
    res[name] = dict(d=d, g=g, y=y)
    P, Y = g[m.EVAL_FROM:], y[m.EVAL_FROM:]; allP.append(P); allY.append(Y)
    first = lambda a, k: int(d["Minute"].iloc[np.argmax(a >= k)]) if (a >= k).any() else None
    rows.append({"부위": name, "정확도": round(accuracy_score(Y, P), 3),
                 "비교적신선 진입(실제)": first(y, 1), "비교적신선 진입(모델)": first(g, 1),
                 "부패 진입(실제)": first(y, 2), "부패 진입(모델)": first(g, 2)})
table = pd.DataFrame(rows); P, Y = np.concatenate(allP), np.concatenate(allY)
acc = accuracy_score(Y, P)
print(table.to_string(index=False))
print(f"\n평가 부위 전체 정확도: {acc:.4f}")
print("혼동행렬 (행=실제, 열=판단; 신선/비교적 신선/부패)\n", confusion_matrix(Y, P, labels=[0, 1, 2]))
os.makedirs(OUT_DIR, exist_ok=True)
table.to_csv(os.path.join(OUT_DIR, "eval_holdout.csv"), index=False, encoding="utf-8-sig")

# ---------- 2. 앱 시연 구간 ----------
#  규칙(미리 정함): 각 부위에서 실제 등급 구간의 한가운데 시점을 판단 시점으로 하고
#                   [보관 시작 10분] + [판단 시점까지 720분]을 잘라 저장
COLS = ["Minute"] + m.SENSORS + ["Temperature"]
def excerpt(d, end_i):
    return pd.concat([d.iloc[:m.BASE_MIN], d.iloc[max(m.BASE_MIN, end_i - m.WIN + 1): end_i + 1]])[COLS]

cands = {1: [], 2: []}
for name, r in res.items():
    for k in (1, 2):
        idx = np.where(r["y"] == k)[0]
        mid = int(idx[len(idx) // 2])
        ex = excerpt(r["d"], mid)
        ge = m.monotone(m.predict_proba(M, m.prepare("ex", ex)).argmax(1))[-1]
        cands[k].append(dict(cut=name, end=int(r["d"]["Minute"].iloc[mid]), judged=int(ge), ok=bool(ge == k), ex=ex))

print("\n후보 구간 (부위별 실제 등급 구간의 한가운데):")
for k, v in cands.items():
    for z in v: print(f"  실제 {m.CLASS_NAMES[k]:<6} {z['cut']:<14} {z['end']:>5}분 → 모델 {m.CLASS_NAMES[z['judged']]}")

items = []
for k, tag in ((1, "fairly_fresh"), (2, "spoiled")):
    pick = [z for z in cands[k] if z["ok"]][:2]
    for j, z in enumerate(pick, 1):
        fn = f"sample_{tag}_{j}.xlsx"
        z["ex"].to_excel(os.path.join(OUT_DIR, fn), index=False)
        items.append(dict(label=f"{m.CLASS_NAMES[k]} {'①②'[j - 1]}", source=f"Wijaya 2022 · {z['cut']} · {z['end']}분",
                          file=fn, truth=m.CLASS_NAMES[k], cols=COLS, data=z["ex"].round(4).values.tolist()))
json.dump(dict(source="Wijaya 외(2022) 소고기 공개 데이터 9~12번 부위(학습에 사용 안 함)",
               note=f"학습에 쓰지 않은 부위의 실제 등급 구간 한가운데를 잘라 넣었습니다. 평가 부위 전체 정확도 {acc:.1%}.",
               accuracy=float(acc), items=items),
          open(os.path.join(OUT_DIR, "samples.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("\n시연 구간:", [(i["label"], i["source"]) for i in items])
