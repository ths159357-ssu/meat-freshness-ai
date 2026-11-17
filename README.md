# 육류 신선도 판단 AI 앱 (4조 · 기초공학설계)

전자코(MQ 가스 센서) 기록을 엑셀로 넣으면, 1D-CNN 기반 판단 모델이 **신선 / 비교적 신선 / 부패**를 판단해 보여 주는 웹 앱입니다.
하드웨어가 없으므로 실제 측정값 대신 **공개 전자코 데이터의 일부 구간**을 넣어 판단 기능을 시연합니다.

- 앱 주소(GitHub Pages): `https://ths159357-ssu.github.io/meat-freshness-ai/`
- QR 코드: `qr_app.png` (앱 주소로 연결)

## 폴더 구성

| 경로 | 내용 |
| --- | --- |
| `index.html` | 앱 (파일 1개, 모델 포함). GitHub Pages가 이 파일을 보여 줌 |
| `engine.js` | 앱 안에서 모델 계산을 하는 코드 (전처리 · CNN · 성장 모델 · 단조 규칙) |
| `app_template.html`, `build_app.py` | 앱 화면 틀 + 모델을 합쳐 `index.html`을 만드는 코드 |
| `model/` | 학습된 모델 (`model_web.json` = 앱용, `.keras`·`.joblib` = 파이썬용) |
| `colab/0_original_final_model.py` | 기존 최종 코드 (참고용, Wijaya 2018만 사용) |
| `colab/1_train_export.py` | 모델 학습 + 앱용 모델 파일 만들기 |
| `colab/2_evaluate_new_data.py` | 학습에 안 쓴 부위로 평가 + 시연 구간 엑셀 4개 만들기 |
| `colab/3_cross_validation.py` | 부위 1개씩 빼고 시험하는 교차검증 |
| `colab/4_make_fridge_data.py` | 상온 시연 데이터를 성장 제곱근 모델로 냉장(4 °C) 기준으로 환산 |
| `colab/5_evaluate_app_model.py` | 앱에 들어간 모델(`model_web.json`) 그대로 평가 데이터 판정 (학습 없이, 버전 무관) |
| `colab/run_in_colab.ipynb` | 위 과정을 콜랩에서 차례로 실행하는 노트북 |
| `data/wijaya2018/` | 기존 학습 데이터 TS1~TS5 |
| `data/wijaya2022/` | 새 데이터 (소고기 12부위) |
| `data/samples/` | 시연용 엑셀 4개(상온 29 °C 원본), 평가 결과 표 |
| `data/samples_fridge/` | 시연용 엑셀 4개(냉장 4 °C 환산) — 앱 예시 버튼에 쓰임 |
| `make_qr.py` | 앱 주소로 QR 코드 만들기 |

## 모델

- 입력: 센서 4개(MQ135, MQ3, MQ4, MQ8)의 최근 12시간 기록 + 온도
- 전처리: 로그 → 보관 시작 10분 평균과의 차이 → 5분 이동 중앙값
- 판단: 1D-CNN 확률 + 성장 모델(온도 누적 지표 tt) 로지스틱 회귀 확률 + 센서 수준·tt 부스팅 확률의 평균
  → 더 높은 등급이 10분 연속일 때만 등급을 올림
- 성장 모델: Ratkowsky 제곱근 식, Pseudomonas 값(Cauchie 2020: Tmin −5 °C, 20 °C에서 μ = 0.42 → b = 0.0259)

### 기존 모델에서 바꾼 점 (새 장비 데이터 추가 학습)

기존 모델(Wijaya 2018만 학습)을 새 데이터(Wijaya 2022)에 그대로 넣으면 정확도가 30.8%였습니다.
새 실험 장비는 센서 변화 폭이 기존의 1/4~1/15 수준이고, 부패로 넘어가는 시점도 더 빨랐기 때문입니다.
그래서 **모델 구조와 전처리는 그대로 두고**, 새 데이터 12부위 중 1~8번 부위를 학습에 추가했습니다.
9~12번 부위는 학습에 쓰지 않고 평가와 앱 시연에만 사용합니다.

## 결과

| 평가 | 정확도 |
| --- | --- |
| 기존 데이터, 실험 1개씩 빼고 시험 (기존 모델) | 92.3% |
| 새 데이터, 기존 모델 그대로 | 30.8% |
| 새 데이터, 부위 1개씩 빼고 시험 (추가 학습) | 88.8% (실행마다 몇 %p 차이) |
| 새 데이터 9~12번 부위 (앱에 들어간 모델) | 85.8% (부위별 56~99%) |
| 같은 평가, CNN만 사용 | 47.4% |

참고: 새 데이터는 12부위 모두 거의 같은 시각에 등급이 바뀝니다. 그래서 경과 시간만으로도 74%가 맞습니다.
CNN 하나만으로는 72%였고, 세 모델을 합친 결과가 가장 높았습니다. 9번 부위(Brisket)는 부패 판단이 크게 늦었습니다.

### 앱 시연 구간 (학습에 안 쓴 부위)

미리 정한 규칙(실제 등급 구간의 한가운데 시점)으로 후보 8개를 만들었고, 그중 6개를 모델이 맞혔습니다(9번 Brisket 2개는 틀림).
맞힌 후보 중 앞의 2개씩을 시연 구간으로 썼습니다.

| 시연 구간 | 부위 · 판단 시점 | 앱 판단 |
| --- | --- | --- |
| 비교적 신선 ① | 10.Clod_Chuck · 781분 | 비교적 신선 |
| 비교적 신선 ② | 11.Shin · 781분 | 비교적 신선 |
| 부패 ① | 10.Clod_Chuck · 1561분 | 부패 |
| 부패 ② | 11.Shin · 1561분 | 부패 |

## 냉장(4 °C) 기준 환산

공개 데이터는 상온(약 29 °C) 실험이라 부패까지 하루 정도로 짧습니다. 그래서 성장 제곱근 모델로 시간을 냉장 기준으로 바꿨습니다.

- 같은 부패 정도(같은 온도 누적 지표 tt)에 도달하는 시간은 성장 속도에 반비례합니다.
- 냉장 시간 = 상온 시간 × [(29 − (−5)) ÷ (4 − (−5))]² = 상온 시간 × 14.27
- 센서 값은 그대로 두고 `Minute`만 14.27배로 늘리고 `Temperature`를 4로 바꿨습니다. 측정 간격은 1분에서 약 14.3분이 됩니다.
- 모델은 온도 누적 지표로 시간과 온도를 함께 보기 때문에, 환산 데이터의 판정과 확률은 상온 원본과 같습니다(확인함).

| 시연 구간 | 상온 29 °C | 냉장 4 °C 환산 |
| --- | --- | --- |
| 비교적 신선 | 781분 (13시간) | 약 7.7일 |
| 부패 | 1,561분 (26시간) | 약 15.5일 |

한계: 실제 냉장고에서 측정한 값이 아니라 "같은 tt면 같은 가스 상태"라고 가정한 추정입니다. 상온 구간에서 제곱근 식이 성장 속도를 크게 잡을 수 있어, 냉장 기간이 실제보다 길게 나올 수 있습니다.

## 엑셀 입력 형식

첫 시트에 `Minute, MQ135, MQ3, MQ4, MQ8, Temperature` 열.
첫 10행은 보관을 막 시작했을 때의 기록(센서 기준값), 그 뒤는 판단할 구간(일정한 간격, 최대 720행). `Minute`는 보관 시작부터 지난 분이며 1분 간격이 아니어도 됩니다.
마지막 행 시점의 상태를 판단합니다. Wijaya 2022에는 온도 열이 없어 실험 조건(약 29 °C)으로 채웠습니다.

## GitHub에 올리고 앱 열기 (프로그램 설치 없이 웹에서)

1. github.com 로그인 → 오른쪽 위 `+` → **New repository** → 이름 `meat-freshness-ai`, **Public** → Create
2. 저장소 화면의 **uploading an existing file** → 이 폴더 안의 파일·폴더를 모두 끌어다 놓기 → **Commit changes**
3. 저장소 **Settings → Pages** → Source: `Deploy from a branch`, Branch: `main` / `(root)` → Save
4. 1~2분 뒤 `https://ths159357-ssu.github.io/meat-freshness-ai/` 접속
5. QR 코드: 콜랩 노트북 마지막 셀, 또는 `python make_qr.py https://ths159357-ssu.github.io/meat-freshness-ai/`

## 콜랩에서 실행

GitHub에 올린 뒤 `https://colab.research.google.com/github/ths159357-ssu/meat-freshness-ai/blob/main/colab/run_in_colab.ipynb` 를 열고, 첫 셀의 아이디를 바꿔 차례로 실행합니다.
(학습할 때마다 결과가 조금씩 달라질 수 있습니다. 앱에는 이미 학습된 모델이 들어 있습니다.)

## 데이터 출처

- Wijaya, D. R. et al. (2018). Electronic nose dataset for beef quality monitoring in uncontrolled ambient conditions. *Data in Brief*. (TS1~TS5)
- Wijaya, D. R. et al. (2022). Electronic nose homogeneous data sets for beef quality classification and microbial population prediction. *BMC Research Notes*. 데이터: https://doi.org/10.7910/DVN/XNFVTS
- Ratkowsky, D. A. et al. (1982), *Journal of Bacteriology* · Cauchie, E. et al. (2020), *Frontiers in Microbiology* · Parvaneh, S. et al. (2023), *Innovative Food Technologies*

데이터는 각 원 출처의 이용 조건을 따르며, 교육 목적으로 출처를 밝혀 사용했습니다.
