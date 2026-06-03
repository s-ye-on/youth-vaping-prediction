# 팀원 공유용 결과 정리

생성 시각: 20260530_164006

## 1. 폴더 구성

```text
share_with_teammate_20260530_164006/
  01_tables/
    model_results/
    seungyeon_final_candidate_threshold_summary.csv
    teammate_vs_seungyeon_comparison_checklist.csv
    questions_for_teammate.csv

  02_threshold_experiment/
    final_candidate_threshold_.../
      reports/
      plots/

  03_plots/
    model_comparison_visualization_.../

  04_feature_reduction/
    reduced_feature_dataset_summary.csv
    removed_features.txt

  05_notes/
    README_for_teammate.md
    message_to_teammate.txt
```

## 2. 승연 실험 방식

최종 후보 실험은 아래 방식으로 진행했습니다.

| 구분 | 비율 | 용도 |
|---|---:|---|
| train | 60% | 모델 학습 |
| validation | 20% | F1이 최대가 되는 threshold 선택 |
| test | 20% | 최종 성능 평가 |

threshold는 validation set에서 F1이 최대가 되는 값으로 선택했고, test set은 최종 평가에만 사용했습니다.

## 3. 승연 최종 후보 결과 요약

| 순위 | dataset | model | threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 2021~2025 reduced with ever | XGBoost | 0.924001 | 0.4790 | 0.6315 | 0.5448 | **0.5081** | 0.9640 |
| 2 | 2021~2025 original with ever | XGBoost | 0.921907 | 0.4791 | 0.6460 | **0.5501** | 0.5080 | **0.9645** |
| 3 | 2021~2025 reduced with ever | LightGBM | 0.895586 | 0.4599 | 0.6196 | 0.5279 | 0.5019 | 0.9609 |
| 4 | 2021~2025 original with ever | Random Forest | 0.349737 | 0.4738 | 0.6472 | 0.5471 | 0.5006 | 0.9611 |
| 5 | 2021~2025 original with ever | LightGBM | 0.895434 | 0.4631 | 0.6102 | 0.5265 | 0.5001 | 0.9611 |
| 6 | 2021~2025 reduced with ever | Random Forest | 0.347812 | 0.4561 | 0.6623 | 0.5402 | 0.4987 | 0.9618 |

## 4. 승연 기준 해석

- F1 기준 최고: `2021_2025_original_with_ever + XGBoost`
- PR-AUC 기준 최고: `2021_2025_reduced_with_ever + XGBoost`
- 두 XGBoost 결과의 PR-AUC 차이는 매우 작음
- original XGBoost가 F1, Recall, ROC-AUC에서 더 좋아 최종 메인 후보로 더 자연스러움
- Random Forest도 F1 0.5471로 매우 근접한 안정적 대안
- LightGBM은 승연 코드 기준에서는 XGBoost/Random Forest보다 낮게 나옴

## 5. 팀원 결과와 비교할 때 확인할 점

팀원 결과에서는 LightGBM이 PR-AUC 0.5402로 가장 높게 나왔고,
승연 결과에서는 XGBoost가 F1 기준으로 가장 높게 나왔습니다.

따라서 아래 항목이 같은지 확인해야 합니다.

1. 정확한 입력 데이터셋 파일
2. rows / positive target count / feature count
3. train/validation/test split 비율과 random_state
4. threshold 선택 기준
5. LightGBM 하이퍼파라미터
6. categorical 변수 처리 방식
7. scale_pos_weight / is_unbalance / class_weight / SMOTE 사용 여부
8. PR-AUC 계산 대상 test set
9. 같은 split에서 XGBoost와 LightGBM을 비교했는지 여부

## 6. 결론

승연 개인 재현 실험에서는 validation set 기준 threshold 조정 후
`2021~2025 original with ever_cigarette_use + XGBoost`가 F1 기준 최종 후보로 가장 적절하게 나타났습니다.

다만 팀원 결과에서는 LightGBM의 PR-AUC가 더 높게 보고되었으므로,
최종 결론 전에 같은 데이터, 같은 split, 같은 전처리, 같은 threshold 선택 방식으로
XGBoost와 LightGBM을 다시 비교하는 것이 가장 안전합니다.
