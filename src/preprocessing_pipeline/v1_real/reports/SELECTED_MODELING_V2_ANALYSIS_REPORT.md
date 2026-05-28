# 새 selected_modeling 데이터셋 v2 재분석 리포트

## 데이터 검증

- 전체 dataset: `selected_modeling_dataset (1).csv`
- X 파일: `selected_modeling_X (1).csv`
- y 파일: `selected_modeling_y (1).csv`
- 행 수: 161,703명
- X 컬럼 수: 58개
- y 컬럼: `current_ecig_use`
- `dataset = X + y` 검증: 정상
- 타깃 양성 수: 4,766명
- 타깃 양성률: 2.947%
- 결측률 40% 이상 컬럼: 0개
- 상수 컬럼: `body_image_missing`

## 이전 파일 대비 핵심 변화

이전 `selected_modeling_X.csv`에서 전부 0이던 가족구성 더미 8개가 수정되었다. 수정 후 분포는 원본 가족구성 코드와 일치한다.

- `live_with_stepfather`: 1 = 4,069명
- `live_with_mother`: 1 = 140,126명
- `live_with_stepmother`: 1 = 3,412명
- `live_with_grandfather`: 1 = 36,389명
- `live_with_grandmother`: 1 = 48,488명
- `live_with_older_sibling`: 1 = 69,667명
- `live_with_younger_sibling`: 1 = 66,270명
- `live_with_no_family`: 1 = 521명

또한 `PR_HT`, `PR_BI` 원 컬럼은 빠지고, `subjective_unhealthy_level` 및 체형 인식 더미(`body_image_*`)로 가공되었다. `body_image_missing`은 전부 0인 상수 컬럼이라 모델 feature에서 제외했다.

## 모델링 방법

- 2023, 2024년을 학습 데이터로 사용하고 2025년을 테스트 데이터로 사용했다.
- `YEAR`는 시간 분할에만 사용하고 feature에서는 제외했다.
- 범주형 변수는 One-Hot Encoding, 수치형 변수는 중앙값 대체와 표준화를 적용했다.
- 불균형 데이터이므로 class weight 또는 scale_pos_weight를 적용했다.

## 모델 성능

F1-score 기준 최고 모델은 `random_forest`이다.

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| random_forest | 0.9495 | 0.3460 | 0.8819 | 0.4970 | 0.9647 | 0.5161 |
| lightgbm | 0.9484 | 0.3411 | 0.8851 | 0.4925 | 0.9620 | 0.5185 |
| xgboost | 0.9430 | 0.3202 | 0.9047 | 0.4730 | 0.9654 | 0.5305 |
| logistic_regression | 0.9402 | 0.3070 | 0.8864 | 0.4561 | 0.9602 | 0.4881 |

전자담배 사용군은 전체의 약 2.95%로 불균형이 크기 때문에 Accuracy보다 Recall, F1, ROC-AUC를 함께 해석해야 한다.

## 중요 피처

Random Forest 중요도 기준 상위 피처는 다음과 같다.

| 순위 | 피처 | 모델 가중치(%) | 순열 AUC 감소 | 사용률 격차(%p) |
|---:|---|---:|---:|---:|
| 1 | `ever_cigarette_use` | 35.77 | 0.1053 | 32.14 |
| 2 | `alcohol_days_30d_cat` | 12.94 | 0.0033 | 32.94 |
| 3 | `alcohol_start_age_cat` | 10.98 | 0.0005 | 13.04 |
| 4 | `ever_alcohol_use` | 10.41 | 0.0008 | 7.92 |
| 5 | `INT_SPWD_TM` | 3.36 | 0.0016 | 5.43 |
| 6 | `INT_SPWK_TM` | 2.47 | 0.0019 | 3.86 |
| 7 | `secondhand_smoke_public` | 2.27 | 0.0016 | 9.08 |
| 8 | `AGE` | 1.54 | 0.0002 | 4.71 |
| 9 | `M_SAD` | 1.52 | 0.0006 | 3.41 |
| 10 | `SCHOOL` | 1.43 | 0.0010 | 6.49 |

SHAP 원 변수 기준 중요도도 비슷한 흐름을 보였다. 상위 5개는 `ever_cigarette_use`, `alcohol_days_30d_cat`, `ever_alcohol_use`, `alcohol_start_age_cat`, `INT_SPWD_TM`이다.

## 해석 요약

- 새 데이터는 이전보다 가족구성 정보가 정상적으로 반영되어 사회환경/가정 관련 변수 해석이 가능해졌다.
- 타깃 분포는 기존과 동일하므로, 성능과 중요도 차이는 타깃 변경이 아니라 feature 구성 및 가족구성 더미 수정의 영향으로 보면 된다.
- 가장 강한 신호는 일반담배 평생 경험(`ever_cigarette_use`)이다. 이는 액상형 전자담배 현재 사용과 일반담배 경험이 강하게 동반된다는 의미이며, 인과효과로 단정해서는 안 된다.
- 음주 관련 변수도 상위권에 있다. 특히 최근 30일 음주일수와 음주 시작 시기 범주는 위험군 구분에 강한 신호를 제공했다.
- `live_with_no_family`는 사용률이 높지만 표본이 521명으로 작아 과해석을 피해야 한다.

## 산출물

- 모델 성능: `tables/model_metrics.csv`
- 혼동행렬: `tables/confusion_matrices.csv`
- 피처 중요도: `tables/feature_importance_ranking.csv`
- SHAP 원 변수 중요도: `tables/shap_original_feature_importance.csv`
- SHAP 인코딩 피처 중요도: `tables/shap_encoded_feature_importance.csv`
- 가족구성 사용률: `tables/family_dummy_target_rates.csv`
- ROC 그래프: `figures/roc_curves.png`
- 피처 중요도 그래프: `figures/feature_importance_top25.png`
- SHAP 원 변수 중요도 그래프: `figures/shap_original_feature_importance_top25.png`
- SHAP summary plot: `figures/shap_summary_encoded_top30.png`
- 가족구성 사용률 그래프: `figures/family_dummy_target_rate_value1.png`

## 주의사항

본 분석은 예측 분석이며 인과효과 분석이 아니다. 변수 중요도가 높다는 것은 해당 변수가 모델의 위험군 구분에 많이 쓰였다는 뜻이지, 그 변수가 실제로 액상형 전자담배 사용을 증가시킨다는 뜻은 아니다.
