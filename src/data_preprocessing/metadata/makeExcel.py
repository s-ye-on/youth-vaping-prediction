import pyreadstat
import pandas as pd

from src.data_preprocessing.project_paths import TABLES_DIR, sav_files

files = sav_files()

rows = []

for year, path in files.items():
    df, meta = pyreadstat.read_sav(str(path), metadataonly=True)

    for name, label in zip(meta.column_names, meta.column_labels):
        rows.append({
            "year": year,
            "variable": name,
            "label": label
        })

variables_df = pd.DataFrame(rows)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
output_file = TABLES_DIR / "variables_by_year.csv"
variables_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("저장 완료:", output_file)
print(variables_df.head())
