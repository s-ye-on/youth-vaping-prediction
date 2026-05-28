import pyreadstat
import pandas as pd

from src.data_preprocessing.project_paths import TABLES_DIR, sav_files

files = sav_files()

year_to_vars = {}
year_to_labels = {}

for year, path in files.items():
    df, meta = pyreadstat.read_sav(str(path), metadataonly=True)

    year_to_vars[year] = set(meta.column_names)
    year_to_labels[year] = dict(zip(meta.column_names, meta.column_labels))

common_vars = set.intersection(*year_to_vars.values())
common_vars = sorted(common_vars)

rows = []

for var in common_vars:
    row = {"variable": var}

    for year in files.keys():
        row[f"label_{year}"] = year_to_labels[year].get(var)

    rows.append(row)

common_df = pd.DataFrame(rows)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
output_file = TABLES_DIR / "common_variables.xlsx"
common_df.to_excel(output_file, index=False)

print("3개년 공통 변수 개수:", len(common_df))
print("저장 완료:", output_file)
print(common_df.head(30))


target_candidates = ["TC_EC_MN", "TC_EC_LT", "TC_EC_FAGE"]

for target in target_candidates:
    print(target, ":", target in common_vars)
