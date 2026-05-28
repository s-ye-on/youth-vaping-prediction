import pyreadstat

from src.data_preprocessing.project_paths import sav_file, sav_files

df, meta = pyreadstat.read_sav(str(sav_file(2023)))

for name, label in zip(meta.column_names, meta.column_labels):
    print(name, ":", label)

files = sav_files()

for year, path in files.items():
    df, meta = pyreadstat.read_sav(str(path), metadataonly=True)

    print(f"\n===== {year} =====")
    print("행 개수:", meta.number_rows)
    print("열 개수:", meta.number_columns)

    for name, label in zip(meta.column_names, meta.column_labels):
        print(name, ":", label)
