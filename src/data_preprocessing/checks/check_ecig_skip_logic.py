# TC_EC_LT와 TC_EC_MN 교차표

import pandas as pd
import pyreadstat

from src.data_preprocessing.project_paths import TABLES_DIR, sav_files

FILES = sav_files()

OUTPUT_DIR = TABLES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ecig_skip_logic_check.xlsx"

rows = []
cross_tables = {}

for year, path in FILES.items():
    df, meta = pyreadstat.read_sav(str(path))

    required_vars = ["TC_EC_LT", "TC_EC_MN"]

    for var in required_vars:
        if var not in df.columns:
            raise ValueError(f"{year}년 데이터에 {var} 변수가 없습니다.")

    temp = df[["TC_EC_LT", "TC_EC_MN"]].copy()

    cross = pd.crosstab(
        temp["TC_EC_LT"],
        temp["TC_EC_MN"],
        dropna=False
    )

    cross_tables[year] = cross

    for tc_ec_lt_value in sorted(temp["TC_EC_LT"].dropna().unique()):
        sub = temp[temp["TC_EC_LT"] == tc_ec_lt_value]

        for tc_ec_mn_value, count in sub["TC_EC_MN"].value_counts(dropna=False).sort_index().items():
            rows.append({
                "year": year,
                "TC_EC_LT_value": tc_ec_lt_value,
                "TC_EC_MN_value": tc_ec_mn_value,
                "count": int(count),
                "ratio_within_TC_EC_LT": float(count / len(sub)),
            })

result_df = pd.DataFrame(rows)

with pd.ExcelWriter(OUTPUT_FILE) as writer:
    result_df.to_excel(writer, sheet_name="long_table", index=False)

    for year, cross in cross_tables.items():
        cross.to_excel(writer, sheet_name=f"cross_{year}")

print("저장 완료:", OUTPUT_FILE)

for year, cross in cross_tables.items():
    print(f"\n===== {year}년 TC_EC_LT x TC_EC_MN 교차표 =====")
    print(cross)
