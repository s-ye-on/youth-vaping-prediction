# 랜덤 포레스트, xgboost, 

Random Forest는 현재 threshold 기준에서 precision과 recall의 균형이 가장 좋아 실제 분류 모델로는 가장 안정적이다.
XGBoost는 PR-AUC와 recall이 높아 사용자를 위험 점수 상위권에 올리는 능력은 좋지만, 현재 threshold에서는 오탐이 많아 F1이 낮다.