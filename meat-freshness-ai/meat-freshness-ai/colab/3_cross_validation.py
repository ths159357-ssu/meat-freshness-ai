# ============================================================
# 4조 육류 신선도 판단 모델 - 새 데이터 교차검증 (부위 1개씩 빼고 시험)
#   12개 부위마다: 2018 데이터 + 나머지 11개 부위로 학습 → 뺀 부위로 시험
#   비교: 최종 앙상블 / CNN만 / 경과 시간(tt)만
# ============================================================
import os
import numpy as np
import pandas as pd
import importlib.util
_spec = importlib.util.spec_from_file_location("m", os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_train_export.py"))
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)

old = [m.prepare(n, d) for n, d in m.load_ts()]
new = [m.prepare(n, d) for n, d in m.load_new(m.TRAIN_CUTS + m.HOLDOUT_CUTS)]
rows = []
for te in new:
    M = m.train_all(old + [s for s in new if s is not te])
    pc = M["cnn"].predict((m.windows(te["x"]) - M["mu"]) / M["sd"], verbose=0, batch_size=4096)
    pl = M["lr"].predict_proba(te["tt"][:, None]); ph = M["hgb"].predict_proba(m.level_tt(te))
    y = te["y"][m.EVAL_FROM:]
    acc = lambda p: float((m.monotone(p.argmax(1))[m.EVAL_FROM:] == y).mean())
    rows.append({"시험 부위": te["name"], "최종 앙상블": acc((pc + pl + ph) / 3), "CNN만": acc(pc), "경과 시간(tt)만": acc(pl)})
    print(rows[-1], flush=True)
t = pd.DataFrame(rows)
t.loc[len(t)] = ["평균"] + t.iloc[:, 1:].mean().tolist()
print(t.round(3).to_string(index=False))
os.makedirs("data/samples", exist_ok=True)
t.round(4).to_csv("data/samples/cross_validation_new_data.csv", index=False, encoding="utf-8-sig")
