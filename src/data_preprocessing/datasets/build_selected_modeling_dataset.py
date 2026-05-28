# 모델링에 사용될 변수 추출

import numpy as np
import pandas as pd

from src.data_preprocessing.project_paths import PROCESSED_DATA_DIR, TABLES_DIR

INPUT_FILE = PROCESSED_DATA_DIR / "full_combined_kyrbs_2023_2025.csv"
CATALOG_FILE = TABLES_DIR / "full_feature_catalog_2023_2025.xlsx"

OUTPUT_DIR = PROCESSED_DATA_DIR
OUTPUT_FILE = OUTPUT_DIR / "selected_modeling_dataset.csv"
OUTPUT_X_FILE = OUTPUT_DIR / "selected_modeling_X.csv"
OUTPUT_Y_FILE = OUTPUT_DIR / "selected_modeling_y.csv"

TARGET_COL = "current_ecig_use"


# ============================================================
# 1. 원자료 코드값
# ============================================================

NO_CODE = 1
YES_CODE = 2

HOUSEHOLD_CONSENT_CODE = 1
HOUSEHOLD_PRIVATE_CODE = 2

HOUSEHOLD_PRIVATE_VALUE = 8888
NOT_APPLICABLE_CODE = 9999


# ============================================================
# 2. 같은 변수명으로 3개년 공통인 변수
#    PR_HT, PR_BI는 아래에서 재구성하므로 여기에는 넣지 않는다.
# ============================================================

SAME_NAME_COLS = [
    # 기본 정보
    "YEAR",
    "CITY",
    "CTYPE",
    "SEX",
    "AGE",
    "GRADE",
    "SCHOOL",
    "STYPE",

    # 정신건강
    "M_STR",
    "M_SAD",

    # 신체활동
    "PA_TOT",

    # 스마트폰 사용 시간
    # 이번 정책: 합산하지 않고 주중/주말 그대로 유지
    "INT_SPWD_TM",
    "INT_SPWK_TM",

    # 경제 / 거주
    "E_SES",
    "E_RES",
]


# ============================================================
# 3. 3개년에 존재하지만 변수명이 다를 수 있는 변수
#    카탈로그의 한글 라벨을 기준으로 후보 컬럼을 찾아 표준 컬럼으로 통합
# ============================================================

CONCEPT_SPECS = {
    "breakfast_freq": {
        "description": "최근 7일 아침식사 빈도",
        "must_include": ["아침식사"],
        "must_exclude": ["편의점", "라면", "음료", "탄산"],
        "fallback_candidates": [
            "F_BR",
            "F_BR_FQ",
            "F_BREAKFAST",
            "F_BREAKFAST_FREQ",
        ],
    },
    "fruit_freq": {
        "description": "최근 7일 과일 섭취 빈도",
        "must_include": ["과일"],
        "must_exclude": ["주스", "음료", "탄산"],
        "fallback_candidates": [
            "F_FRUIT",
            "F_FRUIT_FQ",
            "F_FRUIT_FREQ",
        ],
    },
    "fastfood_freq": {
        "description": "최근 7일 패스트푸드 섭취 빈도",
        "must_include": ["패스트푸드"],
        "must_exclude": [],
        "fallback_candidates": [
            "F_FASTFOOD",
            "F_FASTFOOD_FQ",
            "F_FASTFOOD_FREQ",
        ],
    },
    "secondhand_smoke_home": {
        "description": "최근 7일 가정 내 간접흡연",
        "must_include": ["가정", "간접흡연"],
        "must_exclude": ["학교", "공공", "궐련형", "전자담배"],
        "fallback_candidates": [
            "TC_SHS_H",
            "TC_SHS_HOME",
            "TC_HOME_SHS",
            "TC_HSHS",
            "TC_SND_H",
        ],
    },
    "secondhand_smoke_public": {
        "description": "최근 7일 공공장소 간접흡연",
        "must_include": ["공공", "간접흡연"],
        "must_exclude": ["학교", "가정", "궐련형", "전자담배"],
        "fallback_candidates": [
            "TC_SHS_P",
            "TC_SHS_PUBLIC",
            "TC_PUBLIC_SHS",
            "TC_PSHS",
            "TC_SND_P",
        ],
    },
    "academic_performance": {
        "description": "주관적 학업성적",
        "must_include": ["학업", "성적"],
        "must_exclude": ["부모", "아버지", "어머니"],
        "fallback_candidates": [
            "E_ACDM",
            "E_S_RCRD",
            "E_RCRD",
            "E_SCHOOL_RECORD",
            "E_GRADE_RECORD",
        ],
    },
}


# ============================================================
# 4. 재구성할 변수
# ============================================================

HEALTH_COL = "PR_HT"
BODY_IMAGE_COL = "PR_BI"

SUBJECTIVE_UNHEALTHY_COL = "subjective_unhealthy_level"

BODY_IMAGE_DUMMY_MAP = {
    1: "body_image_very_thin",
    2: "body_image_thin",
    3: "body_image_normal",
    4: "body_image_fat",
    5: "body_image_very_fat",
}

ALCOHOL_COLS = [
    "AC_LT",
    "AC_FAGE",
    "AC_DAYS",
]

TOBACCO_COLS = [
    "TC_LT",
]

FAMILY_CONSENT_COL = "A_FM"

# 중요:
# 가족구성원 다중응답 문항은 각 컬럼마다 선택되었을 때의 코드가 다르다.
# 기존 오류는 모든 컬럼을 == 1로 처리한 것.
FAMILY_MEMBER_COLS = {
    "E_FM_F_1": ("live_with_father", 1),
    "E_FM_SF_2": ("live_with_stepfather", 2),
    "E_FM_M_3": ("live_with_mother", 3),
    "E_FM_SM_4": ("live_with_stepmother", 4),
    "E_FM_GF_5": ("live_with_grandfather", 5),
    "E_FM_GM_6": ("live_with_grandmother", 6),
    "E_FM_OBS_7": ("live_with_older_sibling", 7),
    "E_FM_YBS_8": ("live_with_younger_sibling", 8),
    "E_FM_NO_9": ("live_with_no_family", 9),
}

PARENT_EDU_COLS = {
    "E_EDU_F": "father",
    "E_EDU_M": "mother",
}

PARENT_KOREAN_BIRTH_COLS = {
    "E_KRN_F": "father",
    "E_KRN_M": "mother",
}


# ============================================================
# 5. 누수 방지
# ============================================================

LEAKAGE_PREFIXES = [
    "TC_EC",   # 액상형 전자담배 직접 문항
    "TC_HTP",  # 궐련형 전자담배 직접 문항
]

LEAKAGE_EXACT_COLS = {
    "TC_EC_MN",
    "TC_EC_LT",
    "TC_EC_FAGE",
}


def read_feature_catalog() -> pd.DataFrame:
    if not CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"카탈로그 파일을 찾지 못했습니다: {CATALOG_FILE}"
        )

    catalog = pd.read_excel(CATALOG_FILE, sheet_name="feature_catalog")

    lower_map = {col.lower(): col for col in catalog.columns}

    variable_col = (
        lower_map.get("variable")
        or lower_map.get("var")
        or lower_map.get("column")
        or lower_map.get("column_name")
    )

    if variable_col is None:
        raise ValueError(
            "feature_catalog 시트에서 변수명 컬럼을 찾지 못했습니다. "
            "variable 컬럼이 있는지 확인해주세요."
        )

    label_candidates = [
        "final_label",
        "label",
        "korean_label",
        "description",
        "question",
        "variable_label",
    ]

    label_cols = [
        lower_map[name]
        for name in label_candidates
        if name in lower_map
    ]

    if not label_cols:
        label_cols = [
            col for col in catalog.columns
            if col != variable_col and catalog[col].dtype == "object"
        ]

    result = catalog.copy()
    result["_variable"] = result[variable_col].astype(str)
    result["_search_text"] = ""

    for col in label_cols:
        result["_search_text"] += " " + result[col].fillna("").astype(str)

    return result[["_variable", "_search_text"]].drop_duplicates()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            "\n필수 컬럼이 데이터에 없습니다.\n"
            f"없는 컬럼: {missing}\n\n"
            "해결 방법:\n"
            "1. full_feature_catalog_2023_2025.xlsx에서 실제 변수명을 확인하세요.\n"
            "2. 코드 상단의 SAME_NAME_COLS, FAMILY_MEMBER_COLS, PARENT_EDU_COLS 등을 확인하세요.\n"
            "3. 변수명이 연도별로 다르면 SAME_NAME_COLS가 아니라 CONCEPT_SPECS로 옮기세요.\n"
        )


def is_leakage_column(col: str) -> bool:
    if col in LEAKAGE_EXACT_COLS:
        return True

    return any(col.startswith(prefix) for prefix in LEAKAGE_PREFIXES)


def find_candidate_columns(
    catalog: pd.DataFrame,
    df_columns: set[str],
    must_include: list[str],
    must_exclude: list[str],
    fallback_candidates: list[str],
) -> list[str]:
    candidates: list[str] = []

    for candidate in fallback_candidates:
        if candidate in df_columns:
            candidates.append(candidate)

    matched = catalog.copy()

    for keyword in must_include:
        matched = matched[matched["_search_text"].str.contains(keyword, na=False)]

    for keyword in must_exclude:
        matched = matched[~matched["_search_text"].str.contains(keyword, na=False)]

    matched = matched[matched["_variable"].isin(df_columns)]

    for col in matched["_variable"].tolist():
        if col not in candidates:
            candidates.append(col)

    candidates = [col for col in candidates if not is_leakage_column(col)]

    return candidates


def coalesce_columns(
    df: pd.DataFrame,
    candidates: list[str],
    new_col: str,
) -> pd.Series:
    existing = [col for col in candidates if col in df.columns]

    if not existing:
        raise ValueError(
            f"{new_col}에 해당하는 후보 컬럼을 찾지 못했습니다.\n"
            f"후보 목록: {candidates}"
        )

    if len(existing) == 1:
        return df[existing[0]]

    return df[existing].bfill(axis=1).iloc[:, 0]


def add_concept_features(
    result: pd.DataFrame,
    df: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    df_columns = set(df.columns)

    print("\n[변수명 통합 결과]")

    for new_col, spec in CONCEPT_SPECS.items():
        candidates = find_candidate_columns(
            catalog=catalog,
            df_columns=df_columns,
            must_include=spec["must_include"],
            must_exclude=spec["must_exclude"],
            fallback_candidates=spec["fallback_candidates"],
        )

        if not candidates:
            raise ValueError(
                f"\n'{new_col}' 후보 변수를 찾지 못했습니다.\n"
                f"설명: {spec['description']}\n"
                f"포함 키워드: {spec['must_include']}\n"
                f"제외 키워드: {spec['must_exclude']}\n"
                "카탈로그에서 실제 변수명을 확인한 뒤 fallback_candidates에 추가하세요."
            )

        result[new_col] = coalesce_columns(df, candidates, new_col)

        print(f"- {new_col}: {spec['description']}")
        print(f"  사용 후보: {candidates}")

    return result


def to_binary_yes_no(series: pd.Series) -> pd.Series:
    return np.select(
        [
            series == YES_CODE,
            series == NO_CODE,
        ],
        [
            1,
            0,
        ],
        default=np.nan,
    )


def add_subjective_health_feature(result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    PR_HT 주관적 건강인지 처리.

    원래 의미:
    1 = 매우 건강한 편이다
    2 = 건강한 편이다
    3 = 보통이다
    4 = 건강하지 못한 편이다
    5 = 매우 건강하지 못한 편이다

    처리:
    값은 그대로 두고 컬럼명만 subjective_unhealthy_level로 변경.
    숫자가 클수록 주관적으로 건강하지 않다고 느끼는 방향이다.
    """
    result[SUBJECTIVE_UNHEALTHY_COL] = df[HEALTH_COL]
    return result


def add_body_image_features(result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    PR_BI 주관적 체형인지 처리.

    원래 의미:
    1 = 매우 마른 편이다
    2 = 약간 마른 편이다
    3 = 보통이다
    4 = 약간 살찐 편이다
    5 = 매우 살찐 편이다

    처리:
    1~5를 각각 별도 0/1 컬럼으로 변환.
    결측은 body_image_missing으로 보존한다.
    """
    body_image = df[BODY_IMAGE_COL]

    result["body_image_missing"] = body_image.isna().astype(int)

    for code, new_col in BODY_IMAGE_DUMMY_MAP.items():
        result[new_col] = (body_image == code).astype(int)

    return result


def make_family_info_private(df: pd.DataFrame) -> pd.Series:
    private = pd.Series(False, index=df.index)

    if FAMILY_CONSENT_COL in df.columns:
        private = private | (df[FAMILY_CONSENT_COL] == HOUSEHOLD_PRIVATE_CODE)

    family_related_cols = (
        list(FAMILY_MEMBER_COLS.keys())
        + list(PARENT_EDU_COLS.keys())
        + list(PARENT_KOREAN_BIRTH_COLS.keys())
    )

    for col in family_related_cols:
        if col in df.columns:
            private = private | (df[col] == HOUSEHOLD_PRIVATE_VALUE)

    return private.astype(int)


def add_family_member_features(result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    가족구성원 다중응답 변수 처리.

    원자료:
    E_FM_F_1   선택 시 1
    E_FM_SF_2  선택 시 2
    E_FM_M_3   선택 시 3
    ...
    E_FM_NO_9  선택 시 9
    8888       가구조사 미동의

    변환:
    각 원본 컬럼별 선택 코드를 정확히 비교해 0/1로 변환한다.
    가구조사 미동의 여부는 family_info_private에서 따로 표현한다.
    """
    for raw_col, (new_col, selected_code) in FAMILY_MEMBER_COLS.items():
        result[new_col] = (df[raw_col] == selected_code).astype(int)

    return result


def add_parent_education_features(result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    for raw_col, parent in PARENT_EDU_COLS.items():
        result[f"{parent}_edu_middle_or_less"] = (df[raw_col] == 1).astype(int)
        result[f"{parent}_edu_high_school"] = (df[raw_col] == 2).astype(int)
        result[f"{parent}_edu_college_or_more"] = (df[raw_col] == 3).astype(int)
        result[f"{parent}_edu_unknown"] = (df[raw_col] == 4).astype(int)
        result[f"{parent}_absent_by_edu"] = (
            df[raw_col] == NOT_APPLICABLE_CODE
        ).astype(int)

    return result


def add_parent_korean_birth_features(result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    for raw_col, parent in PARENT_KOREAN_BIRTH_COLS.items():
        result[f"{parent}_korean_birth"] = (df[raw_col] == 1).astype(int)
        result[f"{parent}_foreign_birth"] = (df[raw_col] == 2).astype(int)
        result[f"{parent}_absent_by_birth"] = (
            df[raw_col] == NOT_APPLICABLE_CODE
        ).astype(int)

    return result


def add_alcohol_features(result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    result["ever_alcohol_use"] = to_binary_yes_no(df["AC_LT"])

    result["alcohol_start_age_cat"] = np.where(
        df["AC_FAGE"] == NOT_APPLICABLE_CODE,
        0,
        df["AC_FAGE"],
    )

    result["alcohol_days_30d_cat"] = np.where(
        df["AC_DAYS"] == NOT_APPLICABLE_CODE,
        0,
        df["AC_DAYS"],
    )

    return result


def add_tobacco_features(result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    result["ever_cigarette_use"] = to_binary_yes_no(df["TC_LT"])
    return result


def assert_no_forbidden_features(result: pd.DataFrame) -> None:
    forbidden = []

    for col in result.columns:
        if col == TARGET_COL:
            continue

        if is_leakage_column(col):
            forbidden.append(col)

    if forbidden:
        raise ValueError(
            "\n누수 또는 목표 외 담배제품 직접 문항이 포함되었습니다.\n"
            f"문제 컬럼: {forbidden}\n"
            "TC_EC_* 액상형 전자담배 직접 문항과 "
            "TC_HTP* 궐련형 전자담배 직접 문항은 X 변수에서 제외해야 합니다."
        )


def assert_target_valid(result: pd.DataFrame) -> None:
    values = sorted(result[TARGET_COL].dropna().unique().tolist())

    if values != [0, 1]:
        raise ValueError(
            f"{TARGET_COL} 값이 0/1이 아닙니다. 현재 값: {values}"
        )


def assert_family_no_family_consistency(result: pd.DataFrame) -> None:
    member_cols = [
        new_col
        for new_col, _ in FAMILY_MEMBER_COLS.values()
        if new_col != "live_with_no_family"
    ]

    inconsistent = (
        (result["live_with_no_family"] == 1)
        & (result[member_cols].sum(axis=1) > 0)
    )

    count = int(inconsistent.sum())

    print("\n[가족구성 더미 검증]")
    if count == 0:
        print("논리 충돌 없음: live_with_no_family=1이면서 다른 가족구성원도 1인 행이 없습니다.")
    else:
        print(
            f"주의: live_with_no_family=1인데 다른 가족구성원도 1인 행이 {count}개 있습니다."
        )
        print("원자료 확인 또는 후처리 기준이 필요할 수 있습니다.")


def print_family_dummy_summary(result: pd.DataFrame) -> None:
    print("\n[가족구성 더미 합계 확인]")
    for _, (new_col, _) in FAMILY_MEMBER_COLS.items():
        print(f"{new_col}: {int(result[new_col].sum())}")


def print_missing_summary(result: pd.DataFrame) -> None:
    missing_ratio = result.isna().mean().sort_values(ascending=False)
    missing_ratio = missing_ratio[missing_ratio > 0]

    if missing_ratio.empty:
        print("\n결측치가 있는 최종 컬럼이 없습니다.")
        return

    print("\n[최종 데이터셋 결측 비율 상위]")
    print(missing_ratio.head(30).round(4))


def move_target_to_last(result: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in result.columns if col != TARGET_COL]
    columns.append(TARGET_COL)
    return result[columns]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("전체 병합 데이터 읽는 중...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("원본 데이터 shape:", df.shape)

    catalog = read_feature_catalog()

    required_cols = (
        SAME_NAME_COLS
        + [HEALTH_COL, BODY_IMAGE_COL]
        + ALCOHOL_COLS
        + TOBACCO_COLS
        + [FAMILY_CONSENT_COL]
        + list(FAMILY_MEMBER_COLS.keys())
        + list(PARENT_EDU_COLS.keys())
        + list(PARENT_KOREAN_BIRTH_COLS.keys())
        + [TARGET_COL]
    )

    require_columns(df, required_cols)

    result = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------
    # 같은 이름으로 존재하는 변수 복사
    # ------------------------------------------------------------
    for col in SAME_NAME_COLS:
        result[col] = df[col]

    # ------------------------------------------------------------
    # PR_HT, PR_BI 재구성
    # ------------------------------------------------------------
    result = add_subjective_health_feature(result, df)
    result = add_body_image_features(result, df)

    # ------------------------------------------------------------
    # 변수명이 다른 3개년 공통 변수 통합
    # ------------------------------------------------------------
    result = add_concept_features(result, df, catalog)

    # ------------------------------------------------------------
    # 재구성 변수 추가
    # ------------------------------------------------------------
    result["family_info_private"] = make_family_info_private(df)

    result = add_alcohol_features(result, df)
    result = add_tobacco_features(result, df)
    result = add_family_member_features(result, df)
    result = add_parent_education_features(result, df)
    result = add_parent_korean_birth_features(result, df)

    # ------------------------------------------------------------
    # target 추가
    # selected_modeling_dataset에는 포함하지만,
    # 학습용 X에서는 반드시 제외한다.
    # ------------------------------------------------------------
    result[TARGET_COL] = df[TARGET_COL].astype(int)

    result = move_target_to_last(result)

    # ------------------------------------------------------------
    # 검증
    # ------------------------------------------------------------
    assert_no_forbidden_features(result)
    assert_target_valid(result)
    assert_family_no_family_consistency(result)

    # ------------------------------------------------------------
    # 저장
    # ------------------------------------------------------------
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    X = result.drop(columns=[TARGET_COL])
    y = result[[TARGET_COL]]

    X.to_csv(OUTPUT_X_FILE, index=False, encoding="utf-8-sig")
    y.to_csv(OUTPUT_Y_FILE, index=False, encoding="utf-8-sig")

    print("\n저장 완료")
    print("selected dataset:", OUTPUT_FILE)
    print("X only:", OUTPUT_X_FILE)
    print("y only:", OUTPUT_Y_FILE)

    print("\nselected dataset shape:", result.shape)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\ntarget 분포:")
    print(result[TARGET_COL].value_counts(dropna=False))

    print("\ntarget 비율:")
    print(result[TARGET_COL].value_counts(normalize=True, dropna=False).round(4))

    print("\nfamily_info_private 분포:")
    print(result["family_info_private"].value_counts(dropna=False))

    print("\n[주관적 건강인지 변환 확인]")
    print(result[SUBJECTIVE_UNHEALTHY_COL].value_counts(dropna=False).sort_index())

    print("\n[주관적 체형인지 one-hot 합계 확인]")
    body_cols = list(BODY_IMAGE_DUMMY_MAP.values()) + ["body_image_missing"]
    print(result[body_cols].sum().sort_index())

    print_family_dummy_summary(result)
    print_missing_summary(result)

    print("\n최종 컬럼 목록:")
    for col in result.columns:
        print("-", col)


if __name__ == "__main__":
    main()
