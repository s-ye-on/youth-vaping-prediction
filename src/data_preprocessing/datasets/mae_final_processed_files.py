# make_final_processed_files.py

import pandas as pd

from src.data_preprocessing.project_paths import PROCESSED_DATA_DIR, TABLES_DIR

# ============================================================
# 1. 경로 설정
# ============================================================
DATA_DIR = PROCESSED_DATA_DIR
OUTPUT_TABLE_DIR = TABLES_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "combined_kyrbs_2023_2025.csv"

HUMAN_READABLE_XLSX = DATA_DIR / "combined_kyrbs_2023_2025.xlsx"
MODELING_CSV = DATA_DIR / "modeling_dataset_encoded.csv"
MODELING_XLSX = DATA_DIR / "modeling_dataset_encoded.xlsx"
ENCODING_MAP_XLSX = OUTPUT_TABLE_DIR / "encoding_map.xlsx"


# ============================================================
# 2. 파일 존재 확인
# ============================================================
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"통합 CSV 파일을 찾을 수 없습니다: {INPUT_CSV}\n"
        "먼저 build_combined_dataset.py를 실행해서 combined_kyrbs_2023_2025.csv를 생성하세요."
    )


# ============================================================
# 3. 통합 데이터 읽기
# ============================================================
df = pd.read_csv(INPUT_CSV)

print("통합 CSV 읽기 완료:", INPUT_CSV)
print("데이터 크기:", df.shape)
print("컬럼 목록:")
print(list(df.columns))


# ============================================================
# 4. feature 한글 설명표
# ============================================================
feature_description_rows = [
    {
        "variable": "YEAR",
        "korean_description": "조사연도",
        "role": "feature",
        "type": "numeric/categorical",
        "note": "2023, 2024, 2025 구분용 연도 변수",
    },
    {
        "variable": "CITY",
        "korean_description": "시도",
        "role": "feature",
        "type": "categorical",
        "note": "외부 지역 단위 변수와 결합할 때 사용할 수 있음",
    },
    {
        "variable": "CTYPE",
        "korean_description": "도시규모",
        "role": "feature",
        "type": "categorical",
        "note": "대도시, 중소도시, 군지역",
    },
    {
        "variable": "SEX",
        "korean_description": "성별",
        "role": "feature",
        "type": "categorical",
        "note": "성별 변수",
    },
    {
        "variable": "AGE",
        "korean_description": "나이",
        "role": "feature",
        "type": "numeric/ordinal",
        "note": "응답자 나이",
    },
    {
        "variable": "GRADE",
        "korean_description": "학년",
        "role": "feature",
        "type": "ordinal",
        "note": "학년 변수",
    },
    {
        "variable": "SCHOOL",
        "korean_description": "학교급",
        "role": "feature",
        "type": "categorical",
        "note": "중학교, 일반계고, 특성화계고 등",
    },
    {
        "variable": "E_SES",
        "korean_description": "주관적 경제상태",
        "role": "feature",
        "type": "ordinal",
        "note": "건강형평성 계열 변수",
    },
    {
        "variable": "E_RES",
        "korean_description": "현재 거주형태",
        "role": "feature",
        "type": "categorical",
        "note": "가정/거주 관련 변수",
    },
    {
        "variable": "E_BORN_F",
        "korean_description": "아버지 출신 국적 또는 태어난 나라",
        "role": "feature",
        "type": "categorical",
        "note": "연도별 라벨 표현 차이는 있으나 값 체계가 동일하여 통합",
    },
    {
        "variable": "E_BORN_M",
        "korean_description": "어머니 출신 국적 또는 태어난 나라",
        "role": "feature",
        "type": "categorical",
        "note": "연도별 라벨 표현 차이는 있으나 값 체계가 동일하여 통합",
    },
    {
        "variable": "E_EDU_F",
        "korean_description": "아버지 학력",
        "role": "feature",
        "type": "ordinal/categorical",
        "note": "무응답/비해당 코드 확인 필요",
    },
    {
        "variable": "E_EDU_M",
        "korean_description": "어머니 학력",
        "role": "feature",
        "type": "ordinal/categorical",
        "note": "무응답/비해당 코드 확인 필요",
    },
    {
        "variable": "M_STR",
        "korean_description": "스트레스 인지",
        "role": "feature",
        "type": "ordinal",
        "note": "정신건강 계열 변수",
    },
    {
        "variable": "M_SAD",
        "korean_description": "최근 12개월 동안 우울감 또는 슬픔·절망감 경험",
        "role": "feature",
        "type": "binary/categorical",
        "note": "라벨 표현 차이는 있으나 값 체계가 동일하여 통합",
    },
    {
        "variable": "M_SLP_EN",
        "korean_description": "주관적 수면 충족",
        "role": "feature",
        "type": "ordinal",
        "note": "수면건강 계열 변수",
    },
    {
        "variable": "AC_LT",
        "korean_description": "평생 음주 경험",
        "role": "feature",
        "type": "binary/categorical",
        "note": "음주 경험 변수",
    },
    {
        "variable": "AC_DAYS",
        "korean_description": "최근 30일 동안 음주일수",
        "role": "feature",
        "type": "ordinal/categorical",
        "note": "음주 빈도 변수",
    },
    {
        "variable": "AC_AMNT",
        "korean_description": "최근 30일 동안 1회 평균 음주량",
        "role": "feature",
        "type": "ordinal/categorical",
        "note": "음주량 변수",
    },
    {
        "variable": "AC_DRUNK",
        "korean_description": "최근 30일 동안 만취 경험",
        "role": "feature",
        "type": "ordinal/categorical",
        "note": "음주 관련 변수",
    },
    {
        "variable": "F_BR",
        "korean_description": "최근 7일 동안 아침식사 빈도 또는 횟수",
        "role": "feature",
        "type": "ordinal",
        "note": "연도별 라벨 표현 차이는 있으나 값 체계가 동일하여 통합",
    },
    {
        "variable": "F_FRUIT",
        "korean_description": "최근 7일 동안 과일 섭취 빈도 또는 횟수",
        "role": "feature",
        "type": "ordinal",
        "note": "식생활 계열 변수",
    },
    {
        "variable": "F_FASTFOOD",
        "korean_description": "최근 7일 동안 패스트푸드 섭취 빈도 또는 횟수",
        "role": "feature",
        "type": "ordinal",
        "note": "식생활 계열 변수",
    },
    {
        "variable": "F_SWD_A",
        "korean_description": "최근 7일 동안 단맛 나는 음료 섭취 빈도 또는 횟수",
        "role": "feature",
        "type": "ordinal",
        "note": "식생활 계열 변수",
    },
    {
        "variable": "F_WAT",
        "korean_description": "최근 7일 동안 물 섭취 빈도 또는 섭취량",
        "role": "feature",
        "type": "ordinal",
        "note": "라벨 표현 차이가 있어 보고서에 정합화 기준 명시 필요",
    },
    {
        "variable": "PA_TOT",
        "korean_description": "하루 60분 이상 신체활동 일수",
        "role": "feature",
        "type": "ordinal",
        "note": "신체활동 계열 변수",
    },
    {
        "variable": "PA_VIG_D",
        "korean_description": "고강도 신체활동 일수",
        "role": "feature",
        "type": "ordinal",
        "note": "신체활동 계열 변수",
    },
    {
        "variable": "PA_MSC",
        "korean_description": "근력강화운동 일수",
        "role": "feature",
        "type": "ordinal",
        "note": "신체활동 계열 변수",
    },
    {
        "variable": "INT_SPWD_TM",
        "korean_description": "주중 스마트폰 사용 시간",
        "role": "feature",
        "type": "numeric/ordinal",
        "note": "생활습관 또는 인터넷중독 계열 변수",
    },
    {
        "variable": "INT_SPWK_TM",
        "korean_description": "주말 스마트폰 사용 시간",
        "role": "feature",
        "type": "numeric/ordinal",
        "note": "생활습관 또는 인터넷중독 계열 변수",
    },
    {
        "variable": "current_ecig_use",
        "korean_description": "현재 액상형 전자담배 사용 여부",
        "role": "target",
        "type": "binary",
        "note": "TC_EC_MN 기준으로 생성. 0=현재 비사용자, 1=현재 사용자",
    },
]

feature_description_df = pd.DataFrame(feature_description_rows)

# 실제 데이터에 존재하는 컬럼만 남기고, 혹시 누락된 설명도 확인
feature_description_df = feature_description_df[
    feature_description_df["variable"].isin(df.columns)
].copy()

described_columns = set(feature_description_df["variable"])
undocumented_columns = [col for col in df.columns if col not in described_columns]

if undocumented_columns:
    extra_rows = [
        {
            "variable": col,
            "korean_description": "설명 미작성",
            "role": "feature",
            "type": "unknown",
            "note": "feature_description_rows에 설명 추가 필요",
        }
        for col in undocumented_columns
    ]
    feature_description_df = pd.concat(
        [feature_description_df, pd.DataFrame(extra_rows)],
        ignore_index=True,
    )


# ============================================================
# 5. 타깃 규칙표
# ============================================================
target_rule_df = pd.DataFrame([
    {
        "source_variable": "TC_EC_MN",
        "source_value": "9999",
        "target_variable": "current_ecig_use",
        "target_value": 0,
        "meaning": "분기형 문항의 비해당 코드",
        "decision": "현재 비사용자",
    },
    {
        "source_variable": "TC_EC_MN",
        "source_value": "1",
        "target_variable": "current_ecig_use",
        "target_value": 0,
        "meaning": "최근 30일 동안 액상형 전자담배 사용 없음",
        "decision": "현재 비사용자",
    },
    {
        "source_variable": "TC_EC_MN",
        "source_value": "2~7",
        "target_variable": "current_ecig_use",
        "target_value": 1,
        "meaning": "최근 30일 동안 1일 이상 액상형 전자담배 사용",
        "decision": "현재 사용자",
    },
])


# ============================================================
# 6. 타깃 분포표
# ============================================================
target_counts = df["current_ecig_use"].value_counts().sort_index()
target_ratios = df["current_ecig_use"].value_counts(normalize=True).sort_index()

target_distribution_df = pd.DataFrame({
    "current_ecig_use": target_counts.index,
    "count": target_counts.values,
    "ratio": target_ratios.values,
})


year_target_distribution_df = (
    df.groupby(["YEAR", "current_ecig_use"])
    .size()
    .reset_index(name="count")
)

year_total = (
    df.groupby("YEAR")
    .size()
    .reset_index(name="year_total")
)

year_target_distribution_df = year_target_distribution_df.merge(
    year_total,
    on="YEAR",
    how="left",
)

year_target_distribution_df["ratio"] = (
    year_target_distribution_df["count"]
    / year_target_distribution_df["year_total"]
)


# ============================================================
# 7. 9999 값 점검표
# ============================================================
missing_code_rows = []

for col in df.columns:
    count_9999 = int((df[col] == 9999).sum()) if pd.api.types.is_numeric_dtype(df[col]) else 0

    if count_9999 > 0:
        missing_code_rows.append({
            "variable": col,
            "count_9999": count_9999,
            "ratio_9999": count_9999 / len(df),
            "recommended_handling": (
                "타깃이 아닌 feature의 9999는 무조건 0/1로 바꾸지 말고 "
                "unknown/missing 범주로 처리하거나 별도 결측 표시 변수로 관리"
            ),
        })

missing_code_df = pd.DataFrame(missing_code_rows)


# ============================================================
# 8. 모델링용 데이터 생성
# ============================================================
modeling_df = df.copy()

# ------------------------------------------------------------
# 8-1. 9999 처리
# ------------------------------------------------------------
# 타깃은 이미 0/1로 생성되어 있어야 함.
# feature의 9999는 의미상 비해당/무응답 계열이므로,
# 일단 -1로 치환해서 "unknown/missing category"로 보존한다.
#
# 주의:
# 모든 9999를 0으로 바꾸면 실제 0이라는 값과 섞일 수 있어 위험하다.
# 따라서 모델링용에서는 -1을 unknown 코드로 사용한다.
# ------------------------------------------------------------
feature_cols = [col for col in modeling_df.columns if col != "current_ecig_use"]

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(modeling_df[col]):
        modeling_df[col] = modeling_df[col].replace(9999, -1)

# ------------------------------------------------------------
# 8-2. 문자열 범주형 변수 숫자 인코딩
# ------------------------------------------------------------
# 예: CITY, CTYPE 등이 문자열이면 category code로 변환한다.
# 인코딩 매핑표는 따로 저장한다.
# ------------------------------------------------------------
encoding_map_rows = []

for col in feature_cols:
    if modeling_df[col].dtype == "object":
        original_series = modeling_df[col].astype("category")
        categories = list(original_series.cat.categories)

        mapping = {
            category: code
            for code, category in enumerate(categories)
        }

        for category, code in mapping.items():
            encoding_map_rows.append({
                "variable": col,
                "original_value": category,
                "encoded_value": code,
            })

        modeling_df[col] = original_series.cat.codes

# ------------------------------------------------------------
# 8-3. 혹시 남아 있는 결측 처리
# ------------------------------------------------------------
# 초기 모델링용으로 결측은 -1로 채운다.
# 나중에 더 정교하게 하려면 SimpleImputer를 사용할 수 있다.
# ------------------------------------------------------------
modeling_df = modeling_df.fillna(-1)

# 타깃은 int 보장
modeling_df["current_ecig_use"] = modeling_df["current_ecig_use"].astype(int)

encoding_map_df = pd.DataFrame(encoding_map_rows)


# ============================================================
# 9. 사람용 전체 데이터 Excel 저장
# ============================================================
with pd.ExcelWriter(HUMAN_READABLE_XLSX, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="combined_data", index=False)
    feature_description_df.to_excel(writer, sheet_name="feature_description", index=False)
    target_rule_df.to_excel(writer, sheet_name="target_rule", index=False)
    target_distribution_df.to_excel(writer, sheet_name="target_distribution", index=False)
    year_target_distribution_df.to_excel(writer, sheet_name="year_target_distribution", index=False)

    if not missing_code_df.empty:
        missing_code_df.to_excel(writer, sheet_name="missing_code_9999", index=False)

print("사람이 보기 좋은 통합 Excel 저장 완료:", HUMAN_READABLE_XLSX)


# ============================================================
# 10. 모델링용 데이터 저장
# ============================================================
modeling_df.to_csv(MODELING_CSV, index=False, encoding="utf-8-sig")

with pd.ExcelWriter(MODELING_XLSX, engine="openpyxl") as writer:
    modeling_df.to_excel(writer, sheet_name="modeling_data", index=False)

print("모델링용 CSV 저장 완료:", MODELING_CSV)
print("모델링용 Excel 저장 완료:", MODELING_XLSX)


# ============================================================
# 11. 인코딩 매핑표 저장
# ============================================================
with pd.ExcelWriter(ENCODING_MAP_XLSX, engine="openpyxl") as writer:
    if not encoding_map_df.empty:
        encoding_map_df.to_excel(writer, sheet_name="encoding_map", index=False)

    feature_description_df.to_excel(writer, sheet_name="feature_description", index=False)

    if not missing_code_df.empty:
        missing_code_df.to_excel(writer, sheet_name="missing_code_9999", index=False)

print("인코딩 매핑표 저장 완료:", ENCODING_MAP_XLSX)


# ============================================================
# 12. 결과 출력
# ============================================================
print("\n===== 생성 완료 =====")
print("1. 사람이 보기 좋은 전체 통합 Excel:")
print("   ", HUMAN_READABLE_XLSX)

print("2. 모델링용 CSV:")
print("   ", MODELING_CSV)

print("3. 모델링용 Excel:")
print("   ", MODELING_XLSX)

print("4. 인코딩 매핑표:")
print("   ", ENCODING_MAP_XLSX)

print("\n원본 통합 데이터 크기:", df.shape)
print("모델링용 데이터 크기:", modeling_df.shape)

print("\n타깃 분포:")
print(target_distribution_df.to_string(index=False))

if not missing_code_df.empty:
    print("\n9999 값이 남아 있던 변수:")
    print(missing_code_df.to_string(index=False))
else:
    print("\n9999 값이 남아 있는 변수는 없습니다.")
