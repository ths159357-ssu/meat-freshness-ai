# ============================================================
# 8. 평가 데이터(Wijaya 2022 소고기 12부위)로 평가
#    ※ 기존 코랩 최종 코드(1~7번 셀)를 모두 실행한 뒤, 맨 아래 새 셀에 붙여 넣어 실행
#    평가 부위: 9~12번 (학습에 사용 안 함) / 추가 학습 부위: 1~8번
#    온도 열이 없어 실험 조건(약 29 °C)으로 채움
# ============================================================
import glob
XLSX = "e-nose_dataset_12_beef_cuts.xlsx"
if not os.path.exists(XLSX):
    found = glob.glob("**/*12_beef_cuts*.xlsx", recursive=True)
    if found: XLSX = found[0]
    else:
        from google.colab import files
        XLSX = list(files.upload().keys())[0]          # e-nose_dataset_12_beef_cuts.xlsx 업로드

NUM2CLASS = {1: "excellent", 2: "good", 3: "acceptable", 4: "spoiled"}
book = pd.read_excel(XLSX, sheet_name=None)
new = {}
for name, d in book.items():
    d = tidy(d.copy())
    d["Class"] = d["Label"].map(NUM2CLASS)
    d["Temperature"] = 29.0
    new[name] = prepare(name, d)
train_cuts = [new[k] for k in new if int(k.split(".")[0]) <= 8]
test_cuts = [new[k] for k in new if int(k.split(".")[0]) >= 9]
print("평가 부위:", [s["name"] for s in test_cuts])

def evaluate(cnn, mu, sd, lr, hgb, title):
    Y, P, Pc, rows = [], [], [], []
    for s in test_cuts:
        pc = cnn.predict((windows(s["x"][:, sel_idx], WIN_LONG, STEP_LONG) - mu) / sd, verbose=0, batch_size=2048)
        p = (pc + lr.predict_proba(s["tt"][:, None]) + hgb.predict_proba(level_tt(s))) / 3
        g, gc, y = monotone(p.argmax(1))[EVAL_FROM:], monotone(pc.argmax(1))[EVAL_FROM:], s["y_auth"][EVAL_FROM:]
        Y.append(y); P.append(g); Pc.append(gc)
        rows.append({"평가 부위": s["name"], "최종 모델": round(accuracy_score(y, g), 3), "CNN만": round(accuracy_score(y, gc), 3)})
    Y, P, Pc = map(np.concatenate, (Y, P, Pc))
    print(f"\n[{title}]"); print(pd.DataFrame(rows).to_string(index=False))
    print(f"전체 정확도: 최종 모델 {accuracy_score(Y, P):.1%} | CNN만 {accuracy_score(Y, Pc):.1%}")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(Y, P, labels=[0, 1, 2], normalize="true"),
                           display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(title); plt.tight_layout(); plt.show()

# (A) 기존 최종 모델 그대로 (Wijaya 2018만 학습)
evaluate(final_cnn, MU, SD, final_lr, final_hgb, "A. Original model (2018 only) on new data")

# (B) 같은 구조로 새 장비 데이터 1~8번 부위를 추가 학습 → 앱에 들어간 방식
_, cnn_b, (mu_b, sd_b) = cnn_proba(series + train_cuts, None, "y_auth", sel_idx, WIN_LONG, STEP_LONG, return_model=True)
lr_b, hgb_b = aux_models(series + train_cuts, "y_auth")
evaluate(cnn_b, mu_b, sd_b, lr_b, hgb_b, "B. + new-device data (cuts 1-8) on cuts 9-12")
