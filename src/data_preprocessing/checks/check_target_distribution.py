import pandas as pd

from src.data_preprocessing.project_paths import TABLES_DIR

input_file = TABLES_DIR / "key_variable_value_distribution.xlsx"
output_file = TABLES_DIR / "target_TC_EC_MN_distribution.xlsx"

freq_df = pd.read_excel(input_file, sheet_name="frequency")

target_df = freq_df[freq_df["variable"] == "TC_EC_MN"].copy()

target_df = target_df.sort_values(["year", "value"])

target_df.to_excel(output_file, index=False)

print("저장 완료:", output_file)
print(target_df.to_string(index=False))
