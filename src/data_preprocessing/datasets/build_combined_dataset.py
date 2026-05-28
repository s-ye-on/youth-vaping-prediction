# TC_EC_MN 기준으로 현재 전자담배 사용자 생성
# 2023, 2024, 2025 데이터 통합

# build_combined_dataset.py

import pandas as pd
import pyreadstat

from src.data_preprocessing.project_paths import PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, sav_files


# ============================================================
# 1. 경로 설정
# ============================================================
FILES = sav_files()

OUTPUT_DATA_DIR = PROCESSED_DATA_DIR
OUTPUT_TABLE_DIR = TABLES_DIR

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_FILE = OUTPUT_DATA_DIR / "combined_kyrbs_2023_2025.csv"
OUTPUT_XLSX_FILE = OUTPUT_TABLE_DIR / "combined_dataset_summary.xlsx"
OUTPUT_MISSING_FEATURE_FILE = OUTPUT_TABLE_DIR / "missing_candidate_features.xlsx"


# ============================================================
# 2. 후보 feature 목록
# ============================================================
# YEAR는 아래에서 직접 생성해서 항상 포함한다.
# 따라서 FEATURES에는 YEAR를 넣지 않는다.
#
# 주의:
# TC_EC_MN은 타깃 생성용 원본 변수이므로 최종 feature에서는 제거한다.
# TC_EC_LT, TC_EC_FAGE 등 액상형 전자담배 직접 관련 변수는 누수 가능성이 있어 제외한다.
FEATURES = [
    # 기본 정보
    "CITY",
    "CTYPE",
    "SEX",
    "AGE",
    "GRADE",
    "SCHOOL",

    # 가정/사회경제
    "E_SES",
    "E_RES",
    "E_BORN_F",
    "E_BORN_M",
    "E_EDU_F",
    "E_EDU_M",

    # 정신건강
    "M_STR",
    "M_SAD",
    "M_SLP_EN",

    # 음주
    "AC_LT",
    "AC_DAYS",
    "AC_AMNT",
    "AC_DRUNK",

    # 식생활
    "F_BR",
    "F_FRUIT",
    "F_FASTFOOD",
    "F_SWD_A",
    "F_WAT",

    # 신체활동
    "PA_TOT",
    "PA_VIG_D",
    "PA_MSC",

    # 스마트폰/생활
    "INT_SPWD_TM",
    "INT_SPWK_TM",
]

TARGET_SOURCE = "TC_EC_MN"
TARGET_NAME = "current_ecig_use"


# ============================================================
# 3. 타깃 생성 함수
# ============================================================
def make_current_ecig_use(value):
    """
    현재 액상형 전자담배 사용 여부 생성 규칙

    TC_EC_MN = 9999 -> 비해당, 현재 비사용자, 0
    TC_EC_MN = 1    -> 최근 30일 동안 사용 없음, 0
    TC_EC_MN = 2~7  -> 최근 30일 동안 1일 이상 사용, 1
    """
    if pd.isna(value):
        return pd.NA

    value = int(value)

    if value == 9999:
        return 0

    if value == 1:
        return 0

    if value in [2, 3, 4, 5, 6, 7]:
        return 1

    return pd.NA


# ============================================================
# 4. 파일 존재 확인
# ============================================================
print("프로젝트 루트:", PROJECT_ROOT)
print("\n확인할 원본 .sav 파일")

for year, path in FILES.items():
    print(f"{year}: {path}")

missing_files = [str(path) for path in FILES.values() if not path.exists()]

if missing_files:
    raise FileNotFoundError(
        "다음 .sav 파일을 찾을 수 없습니다.\n"
        + "\n".join(missing_files)
        + "\n\n확인 사항:\n"
        + "1. kyrbs2023.sav, kyrbs2024.sav, kyrbs2025.sav가 프로젝트 루트에 있는지 확인하세요.\n"
        + "2. 파일명이 정확히 일치하는지 확인하세요."
    )


# ============================================================
# 5. 3개년 데이터 읽기 + 타깃 생성 + 통합 준비
# ============================================================
combined_dfs = []
missing_feature_rows = []
year_summary_rows = []

for year, path in FILES.items():
    print(f"\n===== {year}년 데이터 처리 시작 =====")
    print("읽는 파일:", path.name)

    # pyreadstat.read_sav는 SPSS .sav 파일을 pandas DataFrame과 metadata로 읽어온다.
    df, meta = pyreadstat.read_sav(str(path))

    print("원본 데이터 크기:", df.shape)

    if TARGET_SOURCE not in df.columns:
        raise ValueError(f"{year}년 데이터에 타깃 원본 변수 {TARGET_SOURCE}가 없습니다.")

    available_features = [feature for feature in FEATURES if feature in df.columns]
    missing_features = [feature for feature in FEATURES if feature not in df.columns]

    for feature in missing_features:
        missing_feature_rows.append({
            "year": year,
            "missing_feature": feature,
        })

    selected_columns = available_features + [TARGET_SOURCE]
    temp = df[selected_columns].copy()

    # ------------------------------------------------------------
    # YEAR는 통합 후에도 반드시 필요하므로 명시적으로 생성한다.
    # 원자료에 YEAR가 있더라도, 파일 기준 연도를 신뢰해서 덮어쓴다.
    # ------------------------------------------------------------
    temp.insert(0, "YEAR", year)

    # ------------------------------------------------------------
    # 타깃 생성
    # ------------------------------------------------------------
    temp[TARGET_NAME] = temp[TARGET_SOURCE].apply(make_current_ecig_use)

    invalid_target_count = int(temp[TARGET_NAME].isna().sum())

    # ------------------------------------------------------------
    # 모델링용 데이터에서는 원본 타깃 변수 제거
    # ------------------------------------------------------------
    temp = temp.drop(columns=[TARGET_SOURCE])

    # ------------------------------------------------------------
    # 타깃 생성 실패 행 제거
    # 현재 규칙상 정상 데이터라면 제거 행은 없어야 한다.
    # ------------------------------------------------------------
    before_drop = len(temp)
    temp = temp.dropna(subset=[TARGET_NAME])
    after_drop = len(temp)

    # 타깃을 정수형으로 변환
    temp[TARGET_NAME] = temp[TARGET_NAME].astype(int)

    # ------------------------------------------------------------
    # 최종 컬럼 순서 정리
    # YEAR가 첫 번째, current_ecig_use가 마지막에 오도록 한다.
    # ------------------------------------------------------------
    feature_columns = [col for col in temp.columns if col not in ["YEAR", TARGET_NAME]]
    ordered_columns = ["YEAR"] + feature_columns + [TARGET_NAME]
    temp = temp[ordered_columns]

    target_counts = temp[TARGET_NAME].value_counts().sort_index()
    target_ratios = temp[TARGET_NAME].value_counts(normalize=True).sort_index()

    count_0 = int(target_counts.get(0, 0))
    count_1 = int(target_counts.get(1, 0))

    ratio_0 = float(target_ratios.get(0, 0))
    ratio_1 = float(target_ratios.get(1, 0))

    year_summary_rows.append({
        "year": year,
        "original_rows": len(df),
        "processed_rows": len(temp),
        "dropped_rows": before_drop - after_drop,
        "invalid_target_count": invalid_target_count,
        "current_non_user_count_y0": count_0,
        "current_user_count_y1": count_1,
        "current_non_user_ratio_y0": ratio_0,
        "current_user_ratio_y1": ratio_1,
        "available_feature_count": len(available_features),
        "missing_feature_count": len(missing_features),
    })

    print("처리 후 데이터 크기:", temp.shape)
    print("컬럼 목록:")
    print(list(temp.columns))

    print("\n타깃 분포:")
    print(target_counts)

    print("\n타깃 비율:")
    print(target_ratios)

    if missing_features:
        print("\n누락 feature:", missing_features)
    else:
        print("\n모든 후보 feature가 존재합니다.")

    combined_dfs.append(temp)


# ============================================================
# 6. 세로 통합
# ============================================================
combined_df = pd.concat(
    combined_dfs,
    axis=0,
    ignore_index=True,
)

print("\n===== 3개년 통합 완료 =====")
print("통합 데이터 크기:", combined_df.shape)

combined_target_counts = combined_df[TARGET_NAME].value_counts().sort_index()
combined_target_ratios = combined_df[TARGET_NAME].value_counts(normalize=True).sort_index()

print("\n통합 타깃 분포:")
print(combined_target_counts)

print("\n통합 타깃 비율:")
print(combined_target_ratios)

print("\n연도별 행 수:")
print(combined_df["YEAR"].value_counts().sort_index())

print("\n최종 컬럼 목록:")
print(list(combined_df.columns))


# ============================================================
# 7. 통합 데이터 저장
# ============================================================
# utf-8-sig는 엑셀에서 한글 CSV를 열 때 깨짐을 줄이기 위해 사용한다.
combined_df.to_csv(
    OUTPUT_CSV_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\n통합 데이터 CSV 저장 완료:", OUTPUT_CSV_FILE)


# ============================================================
# 8. 요약 엑셀 저장
# ============================================================
year_summary_df = pd.DataFrame(year_summary_rows)

target_distribution_df = pd.DataFrame({
    "target_value": combined_target_counts.index,
    "count": combined_target_counts.values,
    "ratio": combined_target_ratios.values,
})

year_distribution_df = (
    combined_df["YEAR"]
    .value_counts()
    .sort_index()
    .rename_axis("year")
    .reset_index(name="row_count")
)

column_info_df = pd.DataFrame({
    "column_order": range(1, len(combined_df.columns) + 1),
    "column_name": combined_df.columns,
})

dataset_info_df = pd.DataFrame([
    {
        "item": "파일명",
        "content": "combined_kyrbs_2023_2025.csv",
    },
    {
        "item": "파일 목적",
        "content": "2023~2025년 청소년건강행태조사 원자료를 통합하고 현재 액상형 전자담배 사용 여부 타깃을 생성한 모델링용 데이터",
    },
    {
        "item": "원본 파일",
        "content": "kyrbs2023.sav, kyrbs2024.sav, kyrbs2025.sav",
    },
    {
        "item": "연도 변수 포함 여부",
        "content": "YEAR 컬럼을 첫 번째 컬럼으로 포함",
    },
    {
        "item": "생성 타깃 변수",
        "content": TARGET_NAME,
    },
    {
        "item": "타깃 정의",
        "content": "최근 30일 동안 액상형 전자담배를 1일 이상 사용했으면 1, 그렇지 않으면 0",
    },
    {
        "item": "y=0",
        "content": "TC_EC_MN=9999 또는 TC_EC_MN=1",
    },
    {
        "item": "y=1",
        "content": "TC_EC_MN=2~7",
    },
    {
        "item": "주의사항",
        "content": "TC_EC_MN, TC_EC_LT, TC_EC_FAGE 등 액상형 전자담배 직접 관련 변수는 feature에서 제외",
    },
])

with pd.ExcelWriter(OUTPUT_XLSX_FILE, engine="openpyxl") as writer:
    dataset_info_df.to_excel(writer, sheet_name="README", index=False)
    year_summary_df.to_excel(writer, sheet_name="year_summary", index=False)
    target_distribution_df.to_excel(writer, sheet_name="target_distribution", index=False)
    year_distribution_df.to_excel(writer, sheet_name="year_distribution", index=False)
    column_info_df.to_excel(writer, sheet_name="column_info", index=False)

print("통합 데이터 요약 엑셀 저장 완료:", OUTPUT_XLSX_FILE)


# ============================================================
# 9. 누락 feature 저장
# ============================================================
if missing_feature_rows:
    missing_feature_df = pd.DataFrame(missing_feature_rows)
    missing_feature_df.to_excel(
        OUTPUT_MISSING_FEATURE_FILE,
        index=False,
    )

    print("누락 feature 목록 저장 완료:", OUTPUT_MISSING_FEATURE_FILE)
else:
    print("누락 feature가 없어 missing_candidate_features.xlsx는 생성하지 않았습니다.")


# ============================================================
# 10. 최종 안내 출력
# ============================================================
print("\n===== 생성된 파일 =====")
print("1. 모델링용 통합 데이터:", OUTPUT_CSV_FILE)
print("2. 통합 데이터 요약:", OUTPUT_XLSX_FILE)

if missing_feature_rows:
    print("3. 누락 feature 목록:", OUTPUT_MISSING_FEATURE_FILE)

print("\n작업 완료")
