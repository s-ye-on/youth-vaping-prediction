# 21~25년 데이터셋 빌드

from pathlib import Path

import pandas as pd

from src.data_preprocessing.project_paths import PROCESSED_MODELING_DIR, PROJECT_ROOT, TABLES_DIR

# ============================================================
# 0. 기본 설정
# ============================================================

PROCESSED_DIR = PROCESSED_MODELING_DIR
OUTPUT_DIR = TABLES_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025]

TARGET_COL = "current_ecig_use"

OUTPUT_DATASET = PROCESSED_DIR / "selected_modeling_dataset_2021_2025.csv"
OUTPUT_X = PROCESSED_DIR / "selected_modeling_X_2021_2025.csv"
OUTPUT_Y = PROCESSED_DIR / "selected_modeling_y_2021_2025.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "selected_modeling_dataset_2021_2025_summary.csv"

OUTPUT_DATASET_WITHOUT_EVER_CIGARETTE = PROCESSED_DIR / "selected_modeling_dataset_2021_2025_without_ever_cigarette_use.csv"
OUTPUT_X_WITHOUT_EVER_CIGARETTE = PROCESSED_DIR / "selected_modeling_X_2021_2025_without_ever_cigarette_use.csv"
OUTPUT_Y_WITHOUT_EVER_CIGARETTE = PROCESSED_DIR / "selected_modeling_y_2021_2025_without_ever_cigarette_use.csv"


# ============================================================
# 1. 원본 변수 목록
# ============================================================

RAW_COLUMNS = [
    # target
    "TC_EC_MN",

    # basic
    "CITY",
    "CTYPE",
    "SEX",
    "AGE",
    "GRADE",
    "SCHOOL",
    "STYPE",

    # mental health / activity
    "M_STR",
    "M_SAD",
    "PA_TOT",

    # smartphone
    "INT_SPWD_TM",
    "INT_SPWK_TM",

    # economic / residence
    "E_SES",
    "E_RES",

    # health / body image
    "PR_HT",
    "PR_BI",

    # diet
    "F_BR",
    "F_FRUIT",
    "F_FASTFOOD",

    # secondhand smoke
    "TC_SND_H",
    "TC_SND_P",

    # academic
    "E_S_RCRD",

    # alcohol
    "AC_LT",
    "AC_FAGE",
    "AC_DAYS",

    # general cigarette
    "TC_LT",

    # family
    "A_FM",
    "E_FM_F_1",
    "E_FM_SF_2",
    "E_FM_M_3",
    "E_FM_SM_4",
    "E_FM_GF_5",
    "E_FM_GM_6",
    "E_FM_OBS_7",
    "E_FM_YBS_8",
    "E_FM_NO_9",

    # parent education / birth
    "E_EDU_F",
    "E_EDU_M",
    "E_KRN_F",
    "E_KRN_M",
]


# ============================================================
# 2. 최종 selected dataset 컬럼 순서
# ============================================================

SELECTED_COLUMNS = [
    "YEAR",
    "CITY",
    "CTYPE",
    "SEX",
    "AGE",
    "GRADE",
    "SCHOOL",
    "STYPE",

    "M_STR",
    "M_SAD",
    "PA_TOT",

    "INT_SPWD_TM",
    "INT_SPWK_TM",

    "E_SES",
    "E_RES",

    "subjective_unhealthy_level",

    "body_image_missing",
    "body_image_very_thin",
    "body_image_thin",
    "body_image_normal",
    "body_image_fat",
    "body_image_very_fat",

    "breakfast_freq",
    "fruit_freq",
    "fastfood_freq",

    "secondhand_smoke_home",
    "secondhand_smoke_public",

    "academic_performance",

    "family_info_private",

    "ever_alcohol_use",
    "alcohol_start_age_cat",
    "alcohol_days_30d_cat",

    "ever_cigarette_use",

    "live_with_father",
    "live_with_stepfather",
    "live_with_mother",
    "live_with_stepmother",
    "live_with_grandfather",
    "live_with_grandmother",
    "live_with_older_sibling",
    "live_with_younger_sibling",
    "live_with_no_family",

    "father_edu_middle_or_less",
    "father_edu_high_school",
    "father_edu_college_or_more",
    "father_edu_unknown",
    "father_absent_by_edu",

    "mother_edu_middle_or_less",
    "mother_edu_high_school",
    "mother_edu_college_or_more",
    "mother_edu_unknown",
    "mother_absent_by_edu",

    "father_korean_birth",
    "father_foreign_birth",
    "father_absent_by_birth",

    "mother_korean_birth",
    "mother_foreign_birth",
    "mother_absent_by_birth",

    TARGET_COL,
]


# ============================================================
# 3. 가족구성 코드
# ============================================================

FAMILY_MEMBER_SPECS = {
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

HOUSEHOLD_PRIVATE_VALUE = 8888


# ============================================================
# 4. 유틸
# ============================================================

def import_pyreadstat():
    try:
        import pyreadstat
        return pyreadstat
    except ImportError as e:
        raise ImportError(
            "\npyreadstat이 설치되어 있지 않습니다.\n\n"
            "아래 명령어로 설치하세요.\n\n"
            "/Users/choi-seung-yeon/.virtualenvs/.venv/bin/python -m pip install pyreadstat\n"
        ) from e


def find_sav_file(year: int) -> Path:
    patterns = [
        f"KYRBS{year}.sav",
        f"kyrbs{year}.sav",
        f"*{year}*.sav",
    ]

    ignore_dirs = {
        ".git",
        ".idea",
        ".venv",
        "venv",
        "__pycache__",
        "outputs",
        "data",
    }

    candidates = []

    for pattern in patterns:
        for path in PROJECT_ROOT.rglob(pattern):
            if any(part in ignore_dirs for part in path.parts):
                continue

            if path.suffix.lower() == ".sav":
                candidates.append(path)

    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            f"\n{year}년 SAV 파일을 찾지 못했습니다.\n"
            f"프로젝트 루트: {PROJECT_ROOT}\n"
        )

    if len(candidates) > 1:
        print(f"\n[주의] {year}년 SAV 후보가 여러 개 있습니다. 첫 번째를 사용합니다.")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"{idx}. {candidate}")

    return candidates[0]


def assert_required_columns_exist(df: pd.DataFrame, year: int) -> None:
    missing = [col for col in RAW_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"\n{year}년 데이터에 필요한 원본 변수가 없습니다.\n"
            f"누락 변수: {missing}\n"
        )


def to_binary_from_code(series: pd.Series, positive_code: int) -> pd.Series:
    return (series == positive_code).astype(int)


def create_current_ecig_use(tc_ec_mn: pd.Series) -> pd.Series:
    return tc_ec_mn.isin([2, 3, 4, 5, 6, 7]).astype(int)


def create_ever_alcohol_use(ac_lt: pd.Series) -> pd.Series:
    # 기존 결정: AC_LT 2 = 평생 음주 경험 있음
    return (ac_lt == 2).astype(int)


def create_ever_cigarette_use(tc_lt: pd.Series) -> pd.Series:
    # 기존 결정: TC_LT 2 = 평생 일반담배 흡연 경험 있음
    return (tc_lt == 2).astype(int)


def create_non_applicable_zero(series: pd.Series) -> pd.Series:
    # 기존 결정: 9999는 비해당/없음 범주 0으로 정리
    return series.where(series != 9999, 0)


def create_family_info_private(df: pd.DataFrame) -> pd.Series:
    family_cols = [
        "A_FM",
        "E_FM_F_1",
        "E_FM_SF_2",
        "E_FM_M_3",
        "E_FM_SM_4",
        "E_FM_GF_5",
        "E_FM_GM_6",
        "E_FM_OBS_7",
        "E_FM_YBS_8",
        "E_FM_NO_9",
        "E_EDU_F",
        "E_EDU_M",
        "E_KRN_F",
        "E_KRN_M",
    ]

    return df[family_cols].eq(HOUSEHOLD_PRIVATE_VALUE).any(axis=1).astype(int)


def add_body_image_features(selected: pd.DataFrame, pr_bi: pd.Series) -> None:
    selected["body_image_missing"] = pr_bi.isna().astype(int)
    selected["body_image_very_thin"] = (pr_bi == 1).astype(int)
    selected["body_image_thin"] = (pr_bi == 2).astype(int)
    selected["body_image_normal"] = (pr_bi == 3).astype(int)
    selected["body_image_fat"] = (pr_bi == 4).astype(int)
    selected["body_image_very_fat"] = (pr_bi == 5).astype(int)


def add_family_member_features(selected: pd.DataFrame, raw: pd.DataFrame) -> None:
    for raw_col, (dummy_col, selected_code) in FAMILY_MEMBER_SPECS.items():
        selected[dummy_col] = (raw[raw_col] == selected_code).astype(int)


def add_parent_education_features(selected: pd.DataFrame, raw: pd.DataFrame) -> None:
    selected["father_edu_middle_or_less"] = (raw["E_EDU_F"] == 1).astype(int)
    selected["father_edu_high_school"] = (raw["E_EDU_F"] == 2).astype(int)
    selected["father_edu_college_or_more"] = (raw["E_EDU_F"] == 3).astype(int)
    selected["father_edu_unknown"] = (raw["E_EDU_F"] == 4).astype(int)
    selected["father_absent_by_edu"] = (raw["E_EDU_F"] == 9999).astype(int)

    selected["mother_edu_middle_or_less"] = (raw["E_EDU_M"] == 1).astype(int)
    selected["mother_edu_high_school"] = (raw["E_EDU_M"] == 2).astype(int)
    selected["mother_edu_college_or_more"] = (raw["E_EDU_M"] == 3).astype(int)
    selected["mother_edu_unknown"] = (raw["E_EDU_M"] == 4).astype(int)
    selected["mother_absent_by_edu"] = (raw["E_EDU_M"] == 9999).astype(int)


def add_parent_birth_features(selected: pd.DataFrame, raw: pd.DataFrame) -> None:
    selected["father_korean_birth"] = (raw["E_KRN_F"] == 1).astype(int)
    selected["father_foreign_birth"] = (raw["E_KRN_F"] == 2).astype(int)
    selected["father_absent_by_birth"] = (raw["E_KRN_F"] == 9999).astype(int)

    selected["mother_korean_birth"] = (raw["E_KRN_M"] == 1).astype(int)
    selected["mother_foreign_birth"] = (raw["E_KRN_M"] == 2).astype(int)
    selected["mother_absent_by_birth"] = (raw["E_KRN_M"] == 9999).astype(int)


def validate_no_leakage_columns(selected: pd.DataFrame) -> None:
    forbidden_prefixes = ["TC_EC", "TC_HTP"]

    leaked = []
    for col in selected.columns:
        if col == TARGET_COL:
            continue

        if any(col.startswith(prefix) for prefix in forbidden_prefixes):
            leaked.append(col)

    if leaked:
        raise ValueError(
            "\n전자담배/궐련형 전자담배 관련 누수 컬럼이 selected dataset에 포함되어 있습니다.\n"
            f"누수 의심 컬럼: {leaked}\n"
        )


def validate_family_conflict(selected: pd.DataFrame, year: int) -> None:
    family_member_cols = [
        "live_with_father",
        "live_with_stepfather",
        "live_with_mother",
        "live_with_stepmother",
        "live_with_grandfather",
        "live_with_grandmother",
        "live_with_older_sibling",
        "live_with_younger_sibling",
    ]

    conflict = (
        (selected["live_with_no_family"] == 1)
        & (selected[family_member_cols].sum(axis=1) > 0)
    )

    conflict_count = int(conflict.sum())

    if conflict_count > 0:
        raise ValueError(
            f"\n{year}년 가족구성 논리 충돌 발생.\n"
            f"live_with_no_family=1인데 다른 가족구성원이 1인 행 수: {conflict_count}\n"
        )


def build_selected_for_year(year: int) -> pd.DataFrame:
    pyreadstat = import_pyreadstat()
    sav_path = find_sav_file(year)

    print("\n" + "=" * 80)
    print(f"{year}년 selected dataset 생성 시작")
    print("=" * 80)
    print("SAV:", sav_path)

    raw, _ = pyreadstat.read_sav(
        str(sav_path),
        usecols=RAW_COLUMNS,
        apply_value_formats=False,
        user_missing=True,
    )

    assert_required_columns_exist(raw, year)

    selected = pd.DataFrame(index=raw.index)

    # --------------------------------------------------------
    # basic
    # --------------------------------------------------------
    selected["YEAR"] = year
    selected["CITY"] = raw["CITY"]
    selected["CTYPE"] = raw["CTYPE"]
    selected["SEX"] = raw["SEX"]
    selected["AGE"] = raw["AGE"]
    selected["GRADE"] = raw["GRADE"]
    selected["SCHOOL"] = raw["SCHOOL"]
    selected["STYPE"] = raw["STYPE"]

    # --------------------------------------------------------
    # mental health / activity
    # --------------------------------------------------------
    selected["M_STR"] = raw["M_STR"]
    selected["M_SAD"] = raw["M_SAD"]
    selected["PA_TOT"] = raw["PA_TOT"]

    # --------------------------------------------------------
    # smartphone
    # --------------------------------------------------------
    selected["INT_SPWD_TM"] = raw["INT_SPWD_TM"]
    selected["INT_SPWK_TM"] = raw["INT_SPWK_TM"]

    # --------------------------------------------------------
    # economic / residence
    # --------------------------------------------------------
    selected["E_SES"] = raw["E_SES"]
    selected["E_RES"] = raw["E_RES"]

    # --------------------------------------------------------
    # health / body image
    # --------------------------------------------------------
    selected["subjective_unhealthy_level"] = raw["PR_HT"]
    add_body_image_features(selected, raw["PR_BI"])

    # --------------------------------------------------------
    # diet
    # --------------------------------------------------------
    selected["breakfast_freq"] = raw["F_BR"]
    selected["fruit_freq"] = raw["F_FRUIT"]
    selected["fastfood_freq"] = raw["F_FASTFOOD"]

    # --------------------------------------------------------
    # secondhand smoke
    # --------------------------------------------------------
    selected["secondhand_smoke_home"] = raw["TC_SND_H"]
    selected["secondhand_smoke_public"] = raw["TC_SND_P"]

    # --------------------------------------------------------
    # academic
    # --------------------------------------------------------
    selected["academic_performance"] = raw["E_S_RCRD"]

    # --------------------------------------------------------
    # family private
    # --------------------------------------------------------
    selected["family_info_private"] = create_family_info_private(raw)

    # --------------------------------------------------------
    # alcohol
    # --------------------------------------------------------
    selected["ever_alcohol_use"] = create_ever_alcohol_use(raw["AC_LT"])
    selected["alcohol_start_age_cat"] = create_non_applicable_zero(raw["AC_FAGE"])
    selected["alcohol_days_30d_cat"] = create_non_applicable_zero(raw["AC_DAYS"])

    # --------------------------------------------------------
    # general cigarette
    # --------------------------------------------------------
    selected["ever_cigarette_use"] = create_ever_cigarette_use(raw["TC_LT"])

    # --------------------------------------------------------
    # family composition
    # --------------------------------------------------------
    add_family_member_features(selected, raw)

    # --------------------------------------------------------
    # parent
    # --------------------------------------------------------
    add_parent_education_features(selected, raw)
    add_parent_birth_features(selected, raw)

    # --------------------------------------------------------
    # target
    # --------------------------------------------------------
    selected[TARGET_COL] = create_current_ecig_use(raw["TC_EC_MN"])

    selected = selected[SELECTED_COLUMNS].copy()

    validate_no_leakage_columns(selected)
    validate_family_conflict(selected, year)

    print(f"{year}년 shape:", selected.shape)
    print(f"{year}년 target 분포:")
    print(selected[TARGET_COL].value_counts(dropna=False).sort_index())
    print(f"{year}년 target 비율:")
    print(selected[TARGET_COL].value_counts(normalize=True, dropna=False).sort_index().round(6))

    return selected


def build_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for year, group in dataset.groupby("YEAR"):
        target_counts = group[TARGET_COL].value_counts(dropna=False).sort_index()
        total = len(group)
        negative = int(target_counts.get(0, 0))
        positive = int(target_counts.get(1, 0))

        rows.append(
            {
                "YEAR": int(year),
                "total_count": total,
                "target_0_count": negative,
                "target_1_count": positive,
                "target_1_ratio": positive / total if total else 0,
            }
        )

    target_counts = dataset[TARGET_COL].value_counts(dropna=False).sort_index()
    total = len(dataset)
    negative = int(target_counts.get(0, 0))
    positive = int(target_counts.get(1, 0))

    rows.append(
        {
            "YEAR": "TOTAL",
            "total_count": total,
            "target_0_count": negative,
            "target_1_count": positive,
            "target_1_ratio": positive / total if total else 0,
        }
    )

    return pd.DataFrame(rows)


def save_dataset_outputs(dataset: pd.DataFrame) -> None:
    X = dataset.drop(columns=[TARGET_COL])
    y = dataset[[TARGET_COL]]

    dataset.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8-sig")
    X.to_csv(OUTPUT_X, index=False, encoding="utf-8-sig")
    y.to_csv(OUTPUT_Y, index=False, encoding="utf-8-sig")

    dataset_without = dataset.drop(columns=["ever_cigarette_use"])
    X_without = dataset_without.drop(columns=[TARGET_COL])
    y_without = dataset_without[[TARGET_COL]]

    dataset_without.to_csv(OUTPUT_DATASET_WITHOUT_EVER_CIGARETTE, index=False, encoding="utf-8-sig")
    X_without.to_csv(OUTPUT_X_WITHOUT_EVER_CIGARETTE, index=False, encoding="utf-8-sig")
    y_without.to_csv(OUTPUT_Y_WITHOUT_EVER_CIGARETTE, index=False, encoding="utf-8-sig")

    summary = build_summary(dataset)
    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("저장 완료")
    print("=" * 80)
    print("5개년 selected dataset:", OUTPUT_DATASET)
    print("5개년 X:", OUTPUT_X)
    print("5개년 y:", OUTPUT_Y)
    print("5개년 selected dataset without ever_cigarette_use:", OUTPUT_DATASET_WITHOUT_EVER_CIGARETTE)
    print("5개년 X without ever_cigarette_use:", OUTPUT_X_WITHOUT_EVER_CIGARETTE)
    print("5개년 y without ever_cigarette_use:", OUTPUT_Y_WITHOUT_EVER_CIGARETTE)
    print("요약:", OUTPUT_SUMMARY)


def main() -> None:
    yearly_datasets = []

    for year in YEARS:
        selected = build_selected_for_year(year)
        yearly_datasets.append(selected)

    dataset = pd.concat(yearly_datasets, ignore_index=True)

    dataset = dataset[SELECTED_COLUMNS].copy()

    validate_no_leakage_columns(dataset)

    print("\n" + "=" * 80)
    print("2021~2025 통합 selected dataset 생성 완료")
    print("=" * 80)
    print("shape:", dataset.shape)

    print("\n[target 분포]")
    print(dataset[TARGET_COL].value_counts(dropna=False).sort_index())

    print("\n[target 비율]")
    print(dataset[TARGET_COL].value_counts(normalize=True, dropna=False).sort_index().round(6))

    print("\n[연도별 target 요약]")
    summary = build_summary(dataset)
    print(summary.to_string(index=False))

    print("\n[컬럼 수]")
    print(len(dataset.columns))

    print("\n[컬럼 목록]")
    print(dataset.columns.tolist())

    save_dataset_outputs(dataset)


if __name__ == "__main__":
    main()
