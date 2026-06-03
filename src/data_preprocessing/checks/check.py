# 여러가지 체크할 때 쓰는 파일

from pathlib import Path
import pandas as pd

DATA_PATH = Path("/Users/choi-seung-yeon/PyCharmMiscProject/data/processed/modeling/selected_modeling_dataset_2021_2025.csv")

TARGET_COL = "current_ecig_use"

df = pd.read_csv(DATA_PATH)

total_rows, total_columns = df.shape

if TARGET_COL not in df.columns:
    raise ValueError(f"{TARGET_COL} 컬럼이 없습니다.")

feature_columns = [col for col in df.columns if col != TARGET_COL]

print("=" * 80)
print("5개년 최종 데이터셋 feature 수 확인")
print("=" * 80)

print(f"데이터셋 경로: {DATA_PATH}")
print(f"전체 행 수: {total_rows}")
print(f"전체 컬럼 수: {total_columns}")
print(f"target 컬럼: {TARGET_COL}")
print(f"모델 입력 feature 수: {len(feature_columns)}")

print("\n[feature 목록]")
for i, col in enumerate(feature_columns, start=1):
    print(f"{i:03d}. {col}")