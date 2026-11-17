# ============================================================
# 4조 육류 신선도 판단 모델 - 상온 데이터를 냉장(4 °C) 기준으로 환산
#   성장 제곱근 모델(Ratkowsky): μ = [b (T − Tmin)]²,  Tmin = −5 °C (Cauchie 2020, Pseudomonas)
#   같은 부패 정도(같은 온도 누적 지표 tt)에 도달하는 시간은 성장 속도에 반비례
#     냉장 시간 = 상온 시간 × [(T상온 − Tmin) / (T냉장 − Tmin)]²
#   → 센서 값은 그대로 두고, 시간(Minute)만 배율만큼 늘리고 온도를 4 °C로 바꾼다.
#     (같은 tt에서는 같은 가스 상태라고 가정한 '환산 데이터'이며 실제 냉장 측정값이 아님)
# ============================================================
import os, json, glob
import pandas as pd

TMIN, FRIDGE_C = -5.0, 4.0
SAMPLE_DIR = os.environ.get("SAMPLE_DIR", "data/samples")
OUT_DIR = os.environ.get("FRIDGE_DIR", "data/samples_fridge")

def to_fridge(d, fridge_c=FRIDGE_C):
    T = float(d["Temperature"].mean())
    ratio = ((T - TMIN) / (fridge_c - TMIN)) ** 2
    f = d.copy()
    f["Minute"] = (d["Minute"] * ratio).round(2)
    f["Temperature"] = fridge_c
    return f, T, ratio

os.makedirs(OUT_DIR, exist_ok=True)
S = json.load(open(os.path.join(SAMPLE_DIR, "samples.json"), encoding="utf-8"))
items, rows = [], []
for it in S["items"]:
    d = pd.DataFrame(it["data"], columns=it["cols"])
    f, T, ratio = to_fridge(d)
    fn = it["file"].replace(".xlsx", "_fridge4C.xlsx")
    f.to_excel(os.path.join(OUT_DIR, fn), index=False)
    m0, m1 = d["Minute"].iloc[-1], f["Minute"].iloc[-1]
    rows.append({"구간": it["label"], "상온": f"{T:.0f} °C · {m0:.0f}분 ({m0/60:.1f}시간)",
                 "배율": round(ratio, 2), "냉장 4 °C 환산": f"{m1:,.0f}분 ({m1/1440:.1f}일)"})
    items.append(dict(it, file=fn, source=it["source"].replace("분", f"분(상온 {T:.0f} °C) → 냉장 4 °C 환산 {m1/1440:.1f}일"),
                      data=f.values.tolist()))
print(pd.DataFrame(rows).to_string(index=False))
json.dump(dict(S, source=S["source"] + " → 성장 모델로 냉장(4 °C) 환산", items=items,
               note=S["note"] + " 시간은 성장 제곱근 모델로 냉장(4 °C) 기준으로 환산했습니다(추정치)."),
          open(os.path.join(OUT_DIR, "samples.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("저장:", sorted(os.listdir(OUT_DIR)))
