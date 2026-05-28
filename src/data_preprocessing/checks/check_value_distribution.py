# check_value_distribution.py
# value label이 없는 변수들의 실제 값 범위와 빈도표를 연도별로 비교한다.

import pandas as pd
import pyreadstat

from src.data_preprocessing.project_paths import PROJECT_ROOT, TABLES_DIR, sav_files

# ============================================================
# 1. 경로 설정
# ============================================================
FILES = sav_files()

OUTPUT_DIR = TABLES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "key_variable_value_distribution.xlsx"


# ============================================================
# 2. 확인할 주요 변수 목록
# ============================================================
CHECK_VARS = [
    # 라벨 표현이 달랐던 변수
    "E_BORN_F",
    "E_BORN_M",
    "F_BR",
    "F_FASTFOOD",
    "F_FRUIT",
    "F_SWD_A",
    "F_WAT",
    "M_SAD",

    # 타깃 변수
    "TC_EC_MN",

    # 살리기로 한 지역/도시규모 변수
    "CITY",
    "CTYPE",
]


# ============================================================
# 3. 파일 존재 여부 확인
# ============================================================
print("프로젝트 루트:", PROJECT_ROOT)
print("\n확인할 .sav 파일 경로")

for year, path in FILES.items():
    print(f"{year}: {path}")

missing_files = [str(path) for path in FILES.values() if not path.exists()]

if missing_files:
    raise FileNotFoundError(
        "다음 .sav 파일을 찾을 수 없습니다.\n"
        + "\n".join(missing_files)
        + "\n\n해결 방법:\n"
        + "1. kyrbs2023.sav, kyrbs2024.sav, kyrbs2025.sav 파일이 프로젝트 루트 또는 data/raw 아래에 있는지 확인하세요.\n"
        + "2. 파일명이 정확히 kyrbs2023.sav / kyrbs2024.sav / kyrbs2025.sav 인지 확인하세요."
    )


# ============================================================
# 4. 값 분포 확인
# ============================================================
summary_rows = []
freq_rows = []

for year, path in FILES.items():
    print(f"\n{year}년 데이터 읽는 중: {path.name}")

    df, meta = pyreadstat.read_sav(str(path))

    print(f"{year}년 데이터 크기: {df.shape}")

    for var in CHECK_VARS:
        if var not in df.columns:
            summary_rows.append({
                "year": year,
                "variable": var,
                "exists": False,
                "column_label": None,
                "row_count": len(df),
                "non_null_count": None,
                "missing_count": None,
                "missing_ratio": None,
                "unique_count": None,
                "min": None,
                "max": None,
            })
            continue

        s = df[var]

        summary_rows.append({
            "year": year,
            "variable": var,
            "exists": True,
            "column_label": meta.column_names_to_labels.get(var),
            "row_count": len(df),
            "non_null_count": int(s.notna().sum()),
            "missing_count": int(s.isna().sum()),
            "missing_ratio": float(s.isna().mean()),
            "unique_count": int(s.nunique(dropna=True)),
            "min": s.min(skipna=True),
            "max": s.max(skipna=True),
        })

        counts = s.value_counts(dropna=False).sort_index()

        for value, count in counts.items():
            freq_rows.append({
                "year": year,
                "variable": var,
                "value": value,
                "count": int(count),
                "ratio": float(count / len(s)),
            })


# ============================================================
# 5. 결과 저장
# ============================================================
summary_df = pd.DataFrame(summary_rows)
freq_df = pd.DataFrame(freq_rows)

with pd.ExcelWriter(OUTPUT_FILE) as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    freq_df.to_excel(writer, sheet_name="frequency", index=False)

print("\n저장 완료:", OUTPUT_FILE)

print("\n요약 결과:")
print(summary_df.to_string(index=False))
