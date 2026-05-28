import pyreadstat
import pandas as pd

from src.data_preprocessing.project_paths import TABLES_DIR, sav_files

# value label 비교
# 응답값 라벨 확인

files = sav_files()

check_vars = [
    "E_BORN_F",
    "E_BORN_M",
    "F_BR",
    "F_FASTFOOD",
    "F_FRUIT",
    "F_SWD_A",
    "F_WAT",
    "M_SAD",
    "TC_EC_MN",
]

rows = []

for year, path in files.items():
    df, meta = pyreadstat.read_sav(str(path), metadataonly=True)

    for var in check_vars:
        value_labels = meta.variable_value_labels.get(var)

        rows.append({
            "year": year,
            "variable": var,
            "column_label": meta.column_names_to_labels.get(var),
            "value_labels": str(value_labels)
        })

value_label_compare_df = pd.DataFrame(rows)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
output_file = TABLES_DIR / "value_label_compare_key_variables.xlsx"
value_label_compare_df.to_excel(output_file, index=False)

print("저장 완료:", output_file)
print(value_label_compare_df.to_string(index=False))
