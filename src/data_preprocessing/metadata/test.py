import pandas as pd

from src.data_preprocessing.project_paths import TABLES_DIR

df = pd.read_excel(TABLES_DIR / "variables_by_year.xlsx")

common_df = pd.read_excel(TABLES_DIR / "common_variables.xlsx")
common_vars = common_df["variable"].tolist()

rows = []

for var in common_vars:
    temp = df[df["variable"] == var]

    label_2023 = temp[temp["year"] == 2023]["label"].iloc[0] if not temp[temp["year"] == 2023].empty else None
    label_2024 = temp[temp["year"] == 2024]["label"].iloc[0] if not temp[temp["year"] == 2024].empty else None
    label_2025 = temp[temp["year"] == 2025]["label"].iloc[0] if not temp[temp["year"] == 2025].empty else None

    labels = [
        "" if pd.isna(label_2023) else str(label_2023).strip(),
        "" if pd.isna(label_2024) else str(label_2024).strip(),
        "" if pd.isna(label_2025) else str(label_2025).strip(),
    ]

    rows.append({
        "variable": var,
        "label_2023": label_2023,
        "label_2024": label_2024,
        "label_2025": label_2025,
        "is_same_label": len(set(labels)) == 1
    })

label_compare_df = pd.DataFrame(rows)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
label_compare_df.to_excel(TABLES_DIR / "label_compare_2023_2024_2025.xlsx", index=False)

diff_df = label_compare_df[label_compare_df["is_same_label"] == False]
diff_df.to_excel(TABLES_DIR / "different_label_variables.xlsx", index=False)

print("공통 변수 개수:", len(label_compare_df))
print("라벨 동일 변수 개수:", label_compare_df["is_same_label"].sum())
print("라벨 다른 변수 개수:", len(diff_df))
print(diff_df.head(30))
