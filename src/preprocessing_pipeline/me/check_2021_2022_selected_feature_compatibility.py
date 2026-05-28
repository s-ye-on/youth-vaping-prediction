# 2021년 2022년 변수 선택

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 0. 기본 설정
# ============================================================

CHECK_YEARS = [2021, 2022]
REFERENCE_YEARS = [2023, 2024, 2025]
ALL_YEARS = CHECK_YEARS + REFERENCE_YEARS

TARGET_COL = "current_ecig_use"

SELECTED_DATASET_RELATIVE_PATH = Path("data") / "processed" / "selected_modeling_dataset.csv"

OUTPUT_DIR_RELATIVE_PATH = Path("outputs") / "compatibility_check"

OUTPUT_EXCEL_NAME = "selected_feature_compatibility_2021_2022.xlsx"
OUTPUT_FEATURE_SUMMARY_CSV = "selected_feature_compatibility_summary_2021_2022.csv"
OUTPUT_RAW_DETAIL_CSV = "selected_feature_raw_variable_detail_2021_2022.csv"
OUTPUT_FAMILY_CODE_CSV = "family_code_check_2021_2022.csv"
OUTPUT_ATTENTION_CSV = "compatibility_attention_needed_2021_2022.csv"


# ============================================================
# 1. selected feature -> 원본 변수 매핑
# ============================================================

# kind 의미:
# - generated: SAV 원본 변수 없이 파일 연도 등에서 생성 가능
# - direct: 원본 변수를 그대로 사용
# - derived: 원본 변수에서 파생
# - concept: 연도별 변수명이 다를 수 있어 후보 중 존재하는 변수 사용
# - family_dummy: 가족구성 다중응답 코드 확인 필요
# - target: target 생성 원본 변수
#
# required_all:
# - True: source_candidates의 모든 변수가 있어야 해당 feature 생성 가능
# - False: 후보 중 하나라도 있으면 생성 가능

FEATURE_SOURCE_SPECS = [
    # --------------------------------------------------------
    # target
    # --------------------------------------------------------
    {
        "selected_feature": "current_ecig_use",
        "kind": "target",
        "source_candidates": ["TC_EC_MN"],
        "required_all": False,
        "rule": "TC_EC_MN에서 current_ecig_use 생성. 9999/1은 0, 2~7은 1.",
        "note": "5개년 통합에서 가장 중요한 target source.",
    },

    # --------------------------------------------------------
    # 기본 정보
    # --------------------------------------------------------
    {
        "selected_feature": "YEAR",
        "kind": "generated",
        "source_candidates": [],
        "required_all": False,
        "rule": "파일 연도에서 생성 가능.",
        "note": "SAV 원본 변수 존재 여부와 무관.",
    },
    {"selected_feature": "CITY", "kind": "direct", "source_candidates": ["CITY"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {"selected_feature": "CTYPE", "kind": "direct", "source_candidates": ["CTYPE"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {"selected_feature": "SEX", "kind": "direct", "source_candidates": ["SEX"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {"selected_feature": "AGE", "kind": "direct", "source_candidates": ["AGE"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {"selected_feature": "GRADE", "kind": "direct", "source_candidates": ["GRADE"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {"selected_feature": "SCHOOL", "kind": "direct", "source_candidates": ["SCHOOL"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {"selected_feature": "STYPE", "kind": "direct", "source_candidates": ["STYPE"], "required_all": False, "rule": "그대로 사용", "note": ""},

    # --------------------------------------------------------
    # 건강 / 체형
    # --------------------------------------------------------
    {
        "selected_feature": "subjective_unhealthy_level",
        "kind": "derived",
        "source_candidates": ["PR_HT"],
        "required_all": False,
        "rule": "PR_HT를 그대로 사용하되 컬럼명 변경. 값이 클수록 주관적으로 건강하지 않음.",
        "note": "1=매우 건강한 편, 5=매우 건강하지 못한 편인지 확인.",
    },
    {
        "selected_feature": "body_image_missing",
        "kind": "derived",
        "source_candidates": ["PR_BI"],
        "required_all": False,
        "rule": "PR_BI 결측 여부.",
        "note": "",
    },
    {
        "selected_feature": "body_image_very_thin",
        "kind": "derived",
        "source_candidates": ["PR_BI"],
        "required_all": False,
        "rule": "PR_BI == 1",
        "note": "1=매우 마른 편인지 확인.",
    },
    {
        "selected_feature": "body_image_thin",
        "kind": "derived",
        "source_candidates": ["PR_BI"],
        "required_all": False,
        "rule": "PR_BI == 2",
        "note": "2=약간 마른 편인지 확인.",
    },
    {
        "selected_feature": "body_image_normal",
        "kind": "derived",
        "source_candidates": ["PR_BI"],
        "required_all": False,
        "rule": "PR_BI == 3",
        "note": "3=보통인지 확인.",
    },
    {
        "selected_feature": "body_image_fat",
        "kind": "derived",
        "source_candidates": ["PR_BI"],
        "required_all": False,
        "rule": "PR_BI == 4",
        "note": "4=약간 살찐 편인지 확인.",
    },
    {
        "selected_feature": "body_image_very_fat",
        "kind": "derived",
        "source_candidates": ["PR_BI"],
        "required_all": False,
        "rule": "PR_BI == 5",
        "note": "5=매우 살찐 편인지 확인.",
    },

    # --------------------------------------------------------
    # 정신건강 / 신체활동
    # --------------------------------------------------------
    {"selected_feature": "M_STR", "kind": "direct", "source_candidates": ["M_STR"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {"selected_feature": "M_SAD", "kind": "direct", "source_candidates": ["M_SAD"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {"selected_feature": "PA_TOT", "kind": "direct", "source_candidates": ["PA_TOT"], "required_all": False, "rule": "그대로 사용", "note": ""},

    # --------------------------------------------------------
    # 스마트폰
    # --------------------------------------------------------
    {
        "selected_feature": "INT_SPWD_TM",
        "kind": "direct",
        "source_candidates": ["INT_SPWD_TM"],
        "required_all": False,
        "rule": "주중 스마트폰 사용 시간. 합산하지 않고 그대로 사용.",
        "note": "분 단위인지 확인.",
    },
    {
        "selected_feature": "INT_SPWK_TM",
        "kind": "direct",
        "source_candidates": ["INT_SPWK_TM"],
        "required_all": False,
        "rule": "주말 스마트폰 사용 시간. 합산하지 않고 그대로 사용.",
        "note": "분 단위인지 확인.",
    },

    # --------------------------------------------------------
    # 경제 / 거주 / 학업
    # --------------------------------------------------------
    {"selected_feature": "E_SES", "kind": "direct", "source_candidates": ["E_SES"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {"selected_feature": "E_RES", "kind": "direct", "source_candidates": ["E_RES"], "required_all": False, "rule": "그대로 사용", "note": ""},
    {
        "selected_feature": "academic_performance",
        "kind": "concept",
        "source_candidates": ["E_S_RCRD", "E_RCRD", "E_ACDM", "E_SCHOOL_RECORD", "E_GRADE_RECORD"],
        "required_all": False,
        "rule": "주관적 학업성적 변수 후보 중 존재하는 변수 사용.",
        "note": "문항 의미가 주관적 학업성적인지 확인.",
    },

    # --------------------------------------------------------
    # 식생활
    # --------------------------------------------------------
    {
        "selected_feature": "breakfast_freq",
        "kind": "concept",
        "source_candidates": ["F_BR", "F_BR_FQ", "F_BREAKFAST", "F_BREAKFAST_FREQ"],
        "required_all": False,
        "rule": "최근 7일 아침식사 빈도 후보 중 존재하는 변수 사용.",
        "note": "",
    },
    {
        "selected_feature": "fruit_freq",
        "kind": "concept",
        "source_candidates": ["F_FRUIT", "F_FRUIT_FQ", "F_FRUIT_FREQ"],
        "required_all": False,
        "rule": "최근 7일 과일 섭취 빈도 후보 중 존재하는 변수 사용.",
        "note": "",
    },
    {
        "selected_feature": "fastfood_freq",
        "kind": "concept",
        "source_candidates": ["F_FASTFOOD", "F_FASTFOOD_FQ", "F_FASTFOOD_FREQ"],
        "required_all": False,
        "rule": "최근 7일 패스트푸드 섭취 빈도 후보 중 존재하는 변수 사용.",
        "note": "",
    },

    # --------------------------------------------------------
    # 간접흡연
    # --------------------------------------------------------
    {
        "selected_feature": "secondhand_smoke_home",
        "kind": "concept",
        "source_candidates": ["TC_SND_H", "TC_SHS_H", "TC_SHS_HOME", "TC_HOME_SHS", "TC_HSHS"],
        "required_all": False,
        "rule": "최근 7일 가정 내 간접흡연 후보 중 존재하는 변수 사용.",
        "note": "TC_HTP와 혼동 금지.",
    },
    {
        "selected_feature": "secondhand_smoke_public",
        "kind": "concept",
        "source_candidates": ["TC_SND_P", "TC_SHS_P", "TC_SHS_PUBLIC", "TC_PUBLIC_SHS", "TC_PSHS"],
        "required_all": False,
        "rule": "최근 7일 공공장소 간접흡연 후보 중 존재하는 변수 사용.",
        "note": "TC_HTP와 혼동 금지.",
    },

    # --------------------------------------------------------
    # 음주 / 흡연
    # --------------------------------------------------------
    {
        "selected_feature": "ever_alcohol_use",
        "kind": "derived",
        "source_candidates": ["AC_LT"],
        "required_all": False,
        "rule": "AC_LT에서 평생 음주 경험 0/1 생성.",
        "note": "1=없음, 2=있음 구조인지 확인.",
    },
    {
        "selected_feature": "alcohol_start_age_cat",
        "kind": "derived",
        "source_candidates": ["AC_FAGE"],
        "required_all": False,
        "rule": "AC_FAGE가 9999면 0, 아니면 원래 범주.",
        "note": "",
    },
    {
        "selected_feature": "alcohol_days_30d_cat",
        "kind": "derived",
        "source_candidates": ["AC_DAYS"],
        "required_all": False,
        "rule": "AC_DAYS가 9999면 0, 아니면 원래 범주.",
        "note": "",
    },
    {
        "selected_feature": "ever_cigarette_use",
        "kind": "derived",
        "source_candidates": ["TC_LT"],
        "required_all": False,
        "rule": "TC_LT에서 평생 일반담배 흡연 경험 0/1 생성.",
        "note": "강한 예측변수. 포함/제거 실험 둘 다 가능.",
    },

    # --------------------------------------------------------
    # 가족 정보 미동의
    # --------------------------------------------------------
    {
        "selected_feature": "family_info_private",
        "kind": "derived",
        "source_candidates": [
            "A_FM",
            "E_FM_F_1", "E_FM_SF_2", "E_FM_M_3", "E_FM_SM_4",
            "E_FM_GF_5", "E_FM_GM_6", "E_FM_OBS_7", "E_FM_YBS_8", "E_FM_NO_9",
            "E_EDU_F", "E_EDU_M", "E_KRN_F", "E_KRN_M",
        ],
        "required_all": True,
        "rule": "A_FM 또는 가족 관련 변수의 8888 값을 이용해 가구조사 미동의 여부 생성.",
        "note": "가족 관련 변수 전체 존재 여부 확인.",
    },

    # --------------------------------------------------------
    # 가족 구성 더미
    # --------------------------------------------------------
    {
        "selected_feature": "live_with_father",
        "kind": "family_dummy",
        "source_candidates": ["E_FM_F_1"],
        "required_all": False,
        "rule": "E_FM_F_1 == 1",
        "note": "선택 코드 1 확인.",
    },
    {
        "selected_feature": "live_with_stepfather",
        "kind": "family_dummy",
        "source_candidates": ["E_FM_SF_2"],
        "required_all": False,
        "rule": "E_FM_SF_2 == 2",
        "note": "선택 코드 2 확인.",
    },
    {
        "selected_feature": "live_with_mother",
        "kind": "family_dummy",
        "source_candidates": ["E_FM_M_3"],
        "required_all": False,
        "rule": "E_FM_M_3 == 3",
        "note": "선택 코드 3 확인.",
    },
    {
        "selected_feature": "live_with_stepmother",
        "kind": "family_dummy",
        "source_candidates": ["E_FM_SM_4"],
        "required_all": False,
        "rule": "E_FM_SM_4 == 4",
        "note": "선택 코드 4 확인.",
    },
    {
        "selected_feature": "live_with_grandfather",
        "kind": "family_dummy",
        "source_candidates": ["E_FM_GF_5"],
        "required_all": False,
        "rule": "E_FM_GF_5 == 5",
        "note": "선택 코드 5 확인.",
    },
    {
        "selected_feature": "live_with_grandmother",
        "kind": "family_dummy",
        "source_candidates": ["E_FM_GM_6"],
        "required_all": False,
        "rule": "E_FM_GM_6 == 6",
        "note": "선택 코드 6 확인.",
    },
    {
        "selected_feature": "live_with_older_sibling",
        "kind": "family_dummy",
        "source_candidates": ["E_FM_OBS_7"],
        "required_all": False,
        "rule": "E_FM_OBS_7 == 7",
        "note": "선택 코드 7 확인.",
    },
    {
        "selected_feature": "live_with_younger_sibling",
        "kind": "family_dummy",
        "source_candidates": ["E_FM_YBS_8"],
        "required_all": False,
        "rule": "E_FM_YBS_8 == 8",
        "note": "선택 코드 8 확인.",
    },
    {
        "selected_feature": "live_with_no_family",
        "kind": "family_dummy",
        "source_candidates": ["E_FM_NO_9"],
        "required_all": False,
        "rule": "E_FM_NO_9 == 9",
        "note": "선택 코드 9 확인.",
    },

    # --------------------------------------------------------
    # 부모 학력
    # --------------------------------------------------------
    {
        "selected_feature": "father_edu_middle_or_less",
        "kind": "derived",
        "source_candidates": ["E_EDU_F"],
        "required_all": False,
        "rule": "E_EDU_F == 1",
        "note": "",
    },
    {
        "selected_feature": "father_edu_high_school",
        "kind": "derived",
        "source_candidates": ["E_EDU_F"],
        "required_all": False,
        "rule": "E_EDU_F == 2",
        "note": "",
    },
    {
        "selected_feature": "father_edu_college_or_more",
        "kind": "derived",
        "source_candidates": ["E_EDU_F"],
        "required_all": False,
        "rule": "E_EDU_F == 3",
        "note": "",
    },
    {
        "selected_feature": "father_edu_unknown",
        "kind": "derived",
        "source_candidates": ["E_EDU_F"],
        "required_all": False,
        "rule": "E_EDU_F == 4",
        "note": "",
    },
    {
        "selected_feature": "father_absent_by_edu",
        "kind": "derived",
        "source_candidates": ["E_EDU_F"],
        "required_all": False,
        "rule": "E_EDU_F == 9999",
        "note": "",
    },
    {
        "selected_feature": "mother_edu_middle_or_less",
        "kind": "derived",
        "source_candidates": ["E_EDU_M"],
        "required_all": False,
        "rule": "E_EDU_M == 1",
        "note": "",
    },
    {
        "selected_feature": "mother_edu_high_school",
        "kind": "derived",
        "source_candidates": ["E_EDU_M"],
        "required_all": False,
        "rule": "E_EDU_M == 2",
        "note": "",
    },
    {
        "selected_feature": "mother_edu_college_or_more",
        "kind": "derived",
        "source_candidates": ["E_EDU_M"],
        "required_all": False,
        "rule": "E_EDU_M == 3",
        "note": "",
    },
    {
        "selected_feature": "mother_edu_unknown",
        "kind": "derived",
        "source_candidates": ["E_EDU_M"],
        "required_all": False,
        "rule": "E_EDU_M == 4",
        "note": "",
    },
    {
        "selected_feature": "mother_absent_by_edu",
        "kind": "derived",
        "source_candidates": ["E_EDU_M"],
        "required_all": False,
        "rule": "E_EDU_M == 9999",
        "note": "",
    },

    # --------------------------------------------------------
    # 부모 한국 출생 여부
    # --------------------------------------------------------
    {
        "selected_feature": "father_korean_birth",
        "kind": "derived",
        "source_candidates": ["E_KRN_F"],
        "required_all": False,
        "rule": "E_KRN_F == 1",
        "note": "",
    },
    {
        "selected_feature": "father_foreign_birth",
        "kind": "derived",
        "source_candidates": ["E_KRN_F"],
        "required_all": False,
        "rule": "E_KRN_F == 2",
        "note": "",
    },
    {
        "selected_feature": "father_absent_by_birth",
        "kind": "derived",
        "source_candidates": ["E_KRN_F"],
        "required_all": False,
        "rule": "E_KRN_F == 9999",
        "note": "",
    },
    {
        "selected_feature": "mother_korean_birth",
        "kind": "derived",
        "source_candidates": ["E_KRN_M"],
        "required_all": False,
        "rule": "E_KRN_M == 1",
        "note": "",
    },
    {
        "selected_feature": "mother_foreign_birth",
        "kind": "derived",
        "source_candidates": ["E_KRN_M"],
        "required_all": False,
        "rule": "E_KRN_M == 2",
        "note": "",
    },
    {
        "selected_feature": "mother_absent_by_birth",
        "kind": "derived",
        "source_candidates": ["E_KRN_M"],
        "required_all": False,
        "rule": "E_KRN_M == 9999",
        "note": "",
    },
]


FAMILY_SELECTED_CODE_SPECS = [
    {"raw_variable": "E_FM_F_1", "selected_feature": "live_with_father", "expected_selected_code": 1, "expected_keyword": "아버지"},
    {"raw_variable": "E_FM_SF_2", "selected_feature": "live_with_stepfather", "expected_selected_code": 2, "expected_keyword": "새아버지"},
    {"raw_variable": "E_FM_M_3", "selected_feature": "live_with_mother", "expected_selected_code": 3, "expected_keyword": "어머니"},
    {"raw_variable": "E_FM_SM_4", "selected_feature": "live_with_stepmother", "expected_selected_code": 4, "expected_keyword": "새어머니"},
    {"raw_variable": "E_FM_GF_5", "selected_feature": "live_with_grandfather", "expected_selected_code": 5, "expected_keyword": "할아버지"},
    {"raw_variable": "E_FM_GM_6", "selected_feature": "live_with_grandmother", "expected_selected_code": 6, "expected_keyword": "할머니"},
    {"raw_variable": "E_FM_OBS_7", "selected_feature": "live_with_older_sibling", "expected_selected_code": 7, "expected_keyword": "형"},
    {"raw_variable": "E_FM_YBS_8", "selected_feature": "live_with_younger_sibling", "expected_selected_code": 8, "expected_keyword": "동생"},
    {"raw_variable": "E_FM_NO_9", "selected_feature": "live_with_no_family", "expected_selected_code": 9, "expected_keyword": "없음"},
]


# ============================================================
# 2. 공통 유틸
# ============================================================

def import_pyreadstat():
    try:
        import pyreadstat
        return pyreadstat
    except ImportError as e:
        raise ImportError(
            "\npyreadstat이 설치되어 있지 않습니다.\n\n"
            "아래 명령어로 현재 가상환경에 설치하세요.\n\n"
            "/Users/choi-seung-yeon/.virtualenvs/.venv/bin/python -m pip install pyreadstat\n"
        ) from e


def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "data").exists() or (parent / "kyrbs2021.sav").exists():
            return parent

    return current.parent


def find_sav_file(project_root: Path, year: int) -> Path | None:
    ignore_dir_names = {
        ".git",
        ".idea",
        ".venv",
        "venv",
        "__pycache__",
        "outputs",
    }

    patterns = [
        f"kyrbs{year}.sav",
        f"KYRBS{year}.sav",
        f"*{year}*.sav",
    ]

    candidates: list[Path] = []

    for pattern in patterns:
        for path in project_root.rglob(pattern):
            if any(part in ignore_dir_names for part in path.parts):
                continue

            if path.suffix.lower() == ".sav":
                candidates.append(path)

    candidates = sorted(set(candidates))

    if not candidates:
        return None

    if len(candidates) > 1:
        print(f"\n[주의] {year}년 SAV 후보가 여러 개 있습니다. 첫 번째를 사용합니다.")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"{idx}. {candidate}")

    return candidates[0]


def read_sav_metadata(sav_path: Path):
    pyreadstat = import_pyreadstat()

    _, meta = pyreadstat.read_sav(
        str(sav_path),
        metadataonly=True,
        apply_value_formats=False,
        user_missing=True,
    )

    return meta


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().replace("\n", " ").replace("\r", " ")


def normalize_value_key(value: Any) -> str:
    if value is None:
        return ""

    try:
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return str(numeric_value)
    except (TypeError, ValueError):
        return str(value).strip()


def normalize_value_labels(labels: dict[Any, Any]) -> dict[str, str]:
    if not labels:
        return {}

    return {
        normalize_value_key(key): normalize_text(value)
        for key, value in labels.items()
    }


def get_column_label(meta, variable: str) -> str:
    labels = getattr(meta, "column_names_to_labels", {}) or {}
    return normalize_text(labels.get(variable, ""))


def get_value_labels(meta, variable: str) -> dict[str, str]:
    variable_value_labels = getattr(meta, "variable_value_labels", {}) or {}
    raw_labels = variable_value_labels.get(variable, {}) or {}
    return normalize_value_labels(raw_labels)


def dict_to_compact_string(value: dict[str, str]) -> str:
    if not value:
        return ""

    items = [f"{key}:{label}" for key, label in sorted(value.items(), key=lambda x: x[0])]
    return " | ".join(items)


def values_equal(a: dict[str, str], b: dict[str, str]) -> bool:
    return a == b


def labels_equal(a: str, b: str) -> bool:
    return normalize_text(a) == normalize_text(b)


def find_existing_candidate(meta, candidates: list[str]) -> str | None:
    column_names = set(meta.column_names)

    for candidate in candidates:
        if candidate in column_names:
            return candidate

    return None


def find_all_existing_candidates(meta, candidates: list[str]) -> list[str]:
    column_names = set(meta.column_names)
    return [candidate for candidate in candidates if candidate in column_names]


def choose_reference_variable(reference_metas: dict[int, Any], candidates: list[str]) -> tuple[int | None, str | None]:
    for year in sorted(reference_metas.keys(), reverse=True):
        actual = find_existing_candidate(reference_metas[year], candidates)
        if actual is not None:
            return year, actual

    return None, None


def evaluate_compatibility(
    check_label: str,
    check_value_labels: dict[str, str],
    reference_label: str,
    reference_value_labels: dict[str, str],
) -> str:
    same_label = labels_equal(check_label, reference_label)
    same_values = values_equal(check_value_labels, reference_value_labels)

    if same_label and same_values:
        return "SAME_LABEL_AND_VALUES"

    if same_label and not same_values:
        return "SAME_LABEL_BUT_VALUE_LABEL_DIFFERS"

    if not same_label and same_values:
        return "LABEL_DIFFERS_BUT_VALUES_SAME"

    return "LABEL_AND_VALUE_LABEL_DIFFERS"


# ============================================================
# 3. selected dataset 컬럼 확인
# ============================================================

def read_selected_dataset_columns(project_root: Path) -> list[str]:
    selected_dataset_path = project_root / SELECTED_DATASET_RELATIVE_PATH

    if not selected_dataset_path.exists():
        raise FileNotFoundError(
            "\nselected_modeling_dataset.csv 파일을 찾지 못했습니다.\n"
            f"기대 경로: {selected_dataset_path}\n\n"
            "먼저 build_selected_modeling_dataset.py를 실행했는지 확인하세요."
        )

    df_head = pd.read_csv(selected_dataset_path, nrows=5, low_memory=False)
    return df_head.columns.tolist()


def build_selected_dataset_column_report(selected_columns: list[str]) -> pd.DataFrame:
    mapped_features = {spec["selected_feature"] for spec in FEATURE_SOURCE_SPECS}

    rows = []

    for col in selected_columns:
        rows.append(
            {
                "selected_dataset_column": col,
                "has_source_mapping": col in mapped_features,
                "is_target": col == TARGET_COL,
            }
        )

    mapped_but_not_in_dataset = sorted(mapped_features - set(selected_columns))

    for col in mapped_but_not_in_dataset:
        rows.append(
            {
                "selected_dataset_column": col,
                "has_source_mapping": True,
                "is_target": col == TARGET_COL,
                "note": "매핑에는 있으나 현재 selected_modeling_dataset에는 없는 컬럼",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. 원본 변수 detail report
# ============================================================

def build_raw_variable_detail_report(
    selected_columns: list[str],
    check_metas: dict[int, Any],
    reference_metas: dict[int, Any],
) -> pd.DataFrame:
    selected_column_set = set(selected_columns)
    rows: list[dict[str, Any]] = []

    for spec in FEATURE_SOURCE_SPECS:
        selected_feature = spec["selected_feature"]
        source_candidates = spec["source_candidates"]
        required_all = spec["required_all"]
        kind = spec["kind"]

        if selected_feature not in selected_column_set:
            selected_feature_in_dataset = False
        else:
            selected_feature_in_dataset = True

        if kind == "generated":
            for check_year in CHECK_YEARS:
                rows.append(
                    {
                        "check_year": check_year,
                        "selected_feature": selected_feature,
                        "selected_feature_in_dataset": selected_feature_in_dataset,
                        "kind": kind,
                        "source_candidates": "",
                        "required_all": required_all,
                        "actual_variable": "",
                        "source_status": "GENERATED",
                        "reference_year": "",
                        "reference_variable": "",
                        "compatibility": "GENERATED_FROM_YEAR",
                        "check_label": "",
                        "reference_label": "",
                        "check_value_labels": "",
                        "reference_value_labels": "",
                        "rule": spec["rule"],
                        "note": spec["note"],
                    }
                )
            continue

        reference_year, reference_variable = choose_reference_variable(
            reference_metas=reference_metas,
            candidates=source_candidates,
        )

        reference_label = ""
        reference_value_labels: dict[str, str] = {}

        if reference_year is not None and reference_variable is not None:
            reference_label = get_column_label(reference_metas[reference_year], reference_variable)
            reference_value_labels = get_value_labels(reference_metas[reference_year], reference_variable)

        for check_year, check_meta in check_metas.items():
            if required_all:
                existing_variables = find_all_existing_candidates(check_meta, source_candidates)
                missing_variables = [var for var in source_candidates if var not in existing_variables]

                if missing_variables:
                    source_status = "MISSING_REQUIRED_SOME"
                    compatibility = "MISSING"
                else:
                    source_status = "ALL_REQUIRED_PRESENT"
                    compatibility = "ALL_REQUIRED_PRESENT"

                rows.append(
                    {
                        "check_year": check_year,
                        "selected_feature": selected_feature,
                        "selected_feature_in_dataset": selected_feature_in_dataset,
                        "kind": kind,
                        "source_candidates": ", ".join(source_candidates),
                        "required_all": required_all,
                        "actual_variable": ", ".join(existing_variables),
                        "missing_variables": ", ".join(missing_variables),
                        "source_status": source_status,
                        "reference_year": reference_year if reference_year is not None else "",
                        "reference_variable": reference_variable if reference_variable is not None else "",
                        "compatibility": compatibility,
                        "check_label": "",
                        "reference_label": reference_label,
                        "check_value_labels": "",
                        "reference_value_labels": dict_to_compact_string(reference_value_labels),
                        "rule": spec["rule"],
                        "note": spec["note"],
                    }
                )
                continue

            actual_variable = find_existing_candidate(check_meta, source_candidates)

            if actual_variable is None:
                rows.append(
                    {
                        "check_year": check_year,
                        "selected_feature": selected_feature,
                        "selected_feature_in_dataset": selected_feature_in_dataset,
                        "kind": kind,
                        "source_candidates": ", ".join(source_candidates),
                        "required_all": required_all,
                        "actual_variable": "",
                        "missing_variables": ", ".join(source_candidates),
                        "source_status": "MISSING",
                        "reference_year": reference_year if reference_year is not None else "",
                        "reference_variable": reference_variable if reference_variable is not None else "",
                        "compatibility": "MISSING",
                        "check_label": "",
                        "reference_label": reference_label,
                        "check_value_labels": "",
                        "reference_value_labels": dict_to_compact_string(reference_value_labels),
                        "rule": spec["rule"],
                        "note": spec["note"],
                    }
                )
                continue

            check_label = get_column_label(check_meta, actual_variable)
            check_value_labels = get_value_labels(check_meta, actual_variable)

            if reference_year is None or reference_variable is None:
                compatibility = "REFERENCE_NOT_FOUND"
            else:
                compatibility = evaluate_compatibility(
                    check_label=check_label,
                    check_value_labels=check_value_labels,
                    reference_label=reference_label,
                    reference_value_labels=reference_value_labels,
                )

            rows.append(
                {
                    "check_year": check_year,
                    "selected_feature": selected_feature,
                    "selected_feature_in_dataset": selected_feature_in_dataset,
                    "kind": kind,
                    "source_candidates": ", ".join(source_candidates),
                    "required_all": required_all,
                    "actual_variable": actual_variable,
                    "missing_variables": "",
                    "source_status": "FOUND",
                    "reference_year": reference_year if reference_year is not None else "",
                    "reference_variable": reference_variable if reference_variable is not None else "",
                    "compatibility": compatibility,
                    "check_label": check_label,
                    "reference_label": reference_label,
                    "check_value_labels": dict_to_compact_string(check_value_labels),
                    "reference_value_labels": dict_to_compact_string(reference_value_labels),
                    "rule": spec["rule"],
                    "note": spec["note"],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 5. feature summary
# ============================================================

def summarize_feature_compatibility(raw_detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    attention_compatibilities = {
        "MISSING",
        "REFERENCE_NOT_FOUND",
        "SAME_LABEL_BUT_VALUE_LABEL_DIFFERS",
        "LABEL_AND_VALUE_LABEL_DIFFERS",
    }

    for selected_feature, group in raw_detail.groupby("selected_feature", sort=False):
        row: dict[str, Any] = {
            "selected_feature": selected_feature,
            "kind": group["kind"].iloc[0],
            "selected_feature_in_dataset": bool(group["selected_feature_in_dataset"].iloc[0]),
            "rule": group["rule"].iloc[0],
            "note": group["note"].iloc[0],
        }

        for year in CHECK_YEARS:
            year_group = group[group["check_year"] == year]

            if year_group.empty:
                row[f"{year}_usable"] = False
                row[f"{year}_status"] = "NO_ROW"
                continue

            compatibilities = set(year_group["compatibility"].astype(str).tolist())
            source_statuses = set(year_group["source_status"].astype(str).tolist())

            has_attention = bool(compatibilities & attention_compatibilities)
            has_missing = "MISSING" in compatibilities or "MISSING" in source_statuses or "MISSING_REQUIRED_SOME" in source_statuses

            if has_missing:
                usable = False
            elif has_attention:
                usable = "REVIEW"
            else:
                usable = True

            row[f"{year}_usable"] = usable
            row[f"{year}_status"] = " | ".join(sorted(compatibilities))
            row[f"{year}_actual_variable"] = " | ".join(
                sorted(set(v for v in year_group["actual_variable"].astype(str).tolist() if v))
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. 가족구성 코드 체크
# ============================================================

def label_contains_keyword(label: str, keyword: str) -> bool:
    if not keyword:
        return True

    return keyword in normalize_text(label)


def build_family_code_report(
    check_metas: dict[int, Any],
    reference_metas: dict[int, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for check_year, check_meta in check_metas.items():
        for spec in FAMILY_SELECTED_CODE_SPECS:
            raw_variable = spec["raw_variable"]
            expected_code = spec["expected_selected_code"]
            expected_keyword = spec["expected_keyword"]

            actual_variable = find_existing_candidate(check_meta, [raw_variable])

            reference_year, reference_variable = choose_reference_variable(
                reference_metas=reference_metas,
                candidates=[raw_variable],
            )

            if actual_variable is None:
                rows.append(
                    {
                        "check_year": check_year,
                        "selected_feature": spec["selected_feature"],
                        "raw_variable": raw_variable,
                        "actual_variable": "",
                        "status": "MISSING",
                        "expected_selected_code": expected_code,
                        "expected_keyword": expected_keyword,
                        "selected_code_label": "",
                        "selected_code_label_matches_keyword": False,
                        "reference_year": reference_year if reference_year is not None else "",
                        "reference_selected_code_label": "",
                        "check_value_labels": "",
                        "reference_value_labels": "",
                    }
                )
                continue

            check_value_labels = get_value_labels(check_meta, actual_variable)
            selected_code_label = check_value_labels.get(str(expected_code), "")

            reference_value_labels: dict[str, str] = {}
            reference_selected_code_label = ""

            if reference_year is not None and reference_variable is not None:
                reference_value_labels = get_value_labels(reference_metas[reference_year], reference_variable)
                reference_selected_code_label = reference_value_labels.get(str(expected_code), "")

            rows.append(
                {
                    "check_year": check_year,
                    "selected_feature": spec["selected_feature"],
                    "raw_variable": raw_variable,
                    "actual_variable": actual_variable,
                    "status": "FOUND",
                    "expected_selected_code": expected_code,
                    "expected_keyword": expected_keyword,
                    "selected_code_label": selected_code_label,
                    "selected_code_label_matches_keyword": label_contains_keyword(selected_code_label, expected_keyword),
                    "reference_year": reference_year if reference_year is not None else "",
                    "reference_selected_code_label": reference_selected_code_label,
                    "check_value_labels": dict_to_compact_string(check_value_labels),
                    "reference_value_labels": dict_to_compact_string(reference_value_labels),
                    "same_as_reference_value_labels": values_equal(check_value_labels, reference_value_labels),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 7. 주의 필요 행 추출
# ============================================================

def build_attention_report(
    feature_summary: pd.DataFrame,
    raw_detail: pd.DataFrame,
    family_code: pd.DataFrame,
) -> pd.DataFrame:
    attention_rows: list[dict[str, Any]] = []

    for _, row in raw_detail.iterrows():
        compatibility = str(row.get("compatibility", ""))
        source_status = str(row.get("source_status", ""))

        if compatibility in {
            "MISSING",
            "REFERENCE_NOT_FOUND",
            "SAME_LABEL_BUT_VALUE_LABEL_DIFFERS",
            "LABEL_AND_VALUE_LABEL_DIFFERS",
        } or source_status in {"MISSING", "MISSING_REQUIRED_SOME"}:
            attention_rows.append(
                {
                    "type": "raw_variable",
                    "check_year": row.get("check_year", ""),
                    "selected_feature": row.get("selected_feature", ""),
                    "problem": f"{source_status} / {compatibility}",
                    "actual_variable": row.get("actual_variable", ""),
                    "source_candidates": row.get("source_candidates", ""),
                    "check_label": row.get("check_label", ""),
                    "reference_label": row.get("reference_label", ""),
                    "note": row.get("note", ""),
                }
            )

    for _, row in family_code.iterrows():
        status = str(row.get("status", ""))
        matches = bool(row.get("selected_code_label_matches_keyword", False))
        same_reference = bool(row.get("same_as_reference_value_labels", True))

        if status != "FOUND" or not matches or not same_reference:
            attention_rows.append(
                {
                    "type": "family_code",
                    "check_year": row.get("check_year", ""),
                    "selected_feature": row.get("selected_feature", ""),
                    "problem": (
                        f"status={status}, "
                        f"keyword_match={matches}, "
                        f"same_as_reference_value_labels={same_reference}"
                    ),
                    "actual_variable": row.get("actual_variable", ""),
                    "source_candidates": row.get("raw_variable", ""),
                    "check_label": row.get("selected_code_label", ""),
                    "reference_label": row.get("reference_selected_code_label", ""),
                    "note": f"expected_code={row.get('expected_selected_code', '')}, expected_keyword={row.get('expected_keyword', '')}",
                }
            )

    if not attention_rows:
        return pd.DataFrame(
            columns=[
                "type",
                "check_year",
                "selected_feature",
                "problem",
                "actual_variable",
                "source_candidates",
                "check_label",
                "reference_label",
                "note",
            ]
        )

    return pd.DataFrame(attention_rows)


# ============================================================
# 8. 저장
# ============================================================

def save_outputs(
    output_dir: Path,
    selected_column_report: pd.DataFrame,
    feature_summary: pd.DataFrame,
    raw_detail: pd.DataFrame,
    family_code: pd.DataFrame,
    attention: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_summary_csv = output_dir / OUTPUT_FEATURE_SUMMARY_CSV
    raw_detail_csv = output_dir / OUTPUT_RAW_DETAIL_CSV
    family_code_csv = output_dir / OUTPUT_FAMILY_CODE_CSV
    attention_csv = output_dir / OUTPUT_ATTENTION_CSV
    excel_path = output_dir / OUTPUT_EXCEL_NAME

    feature_summary.to_csv(feature_summary_csv, index=False, encoding="utf-8-sig")
    raw_detail.to_csv(raw_detail_csv, index=False, encoding="utf-8-sig")
    family_code.to_csv(family_code_csv, index=False, encoding="utf-8-sig")
    attention.to_csv(attention_csv, index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(excel_path) as writer:
            feature_summary.to_excel(writer, sheet_name="feature_summary", index=False)
            raw_detail.to_excel(writer, sheet_name="raw_variable_detail", index=False)
            family_code.to_excel(writer, sheet_name="family_code_check", index=False)
            attention.to_excel(writer, sheet_name="attention_needed", index=False)
            selected_column_report.to_excel(writer, sheet_name="selected_columns", index=False)

        print("\n엑셀 저장 완료:")
        print(excel_path)

    except ImportError:
        print("\n[주의] Excel 저장에 필요한 패키지가 없어 xlsx 저장은 건너뜁니다.")
        print("필요하면 설치:")
        print("/Users/choi-seung-yeon/.virtualenvs/.venv/bin/python -m pip install openpyxl")

    print("\nCSV 저장 완료:")
    print(feature_summary_csv)
    print(raw_detail_csv)
    print(family_code_csv)
    print(attention_csv)


# ============================================================
# 9. 출력 요약
# ============================================================

def print_summary(
    sav_paths: dict[int, Path],
    feature_summary: pd.DataFrame,
    raw_detail: pd.DataFrame,
    family_code: pd.DataFrame,
    attention: pd.DataFrame,
) -> None:
    print("\n" + "=" * 80)
    print("SAV 파일 확인")
    print("=" * 80)

    for year in ALL_YEARS:
        print(f"{year}: {sav_paths.get(year, '없음')}")

    print("\n" + "=" * 80)
    print("선택 feature 호환성 요약")
    print("=" * 80)

    for year in CHECK_YEARS:
        col = f"{year}_usable"
        if col in feature_summary.columns:
            print(f"\n[{year}_usable]")
            print(feature_summary[col].value_counts(dropna=False))

    print("\n" + "=" * 80)
    print("raw compatibility 요약")
    print("=" * 80)
    print(pd.crosstab(raw_detail["check_year"], raw_detail["compatibility"]))

    print("\n" + "=" * 80)
    print("가족구성 코드 체크 요약")
    print("=" * 80)

    if family_code.empty:
        print("가족구성 코드 체크 결과가 없습니다.")
    else:
        print(pd.crosstab(family_code["check_year"], family_code["selected_code_label_matches_keyword"]))

    print("\n" + "=" * 80)
    print("주의 필요 행 수")
    print("=" * 80)
    print(len(attention))

    if not attention.empty:
        print("\n[주의 필요 상위 50개]")
        print(attention.head(50).to_string(index=False, max_colwidth=100))


# ============================================================
# 10. main
# ============================================================

def main() -> None:
    project_root = find_project_root()
    output_dir = project_root / OUTPUT_DIR_RELATIVE_PATH

    print("프로젝트 루트:", project_root)
    print("출력 폴더:", output_dir)

    selected_columns = read_selected_dataset_columns(project_root)
    selected_column_report = build_selected_dataset_column_report(selected_columns)

    print("\nselected_modeling_dataset 컬럼 수:", len(selected_columns))

    sav_paths: dict[int, Path] = {}
    metas: dict[int, Any] = {}

    for year in ALL_YEARS:
        sav_path = find_sav_file(project_root, year)

        if sav_path is None:
            print(f"\n[주의] {year}년 SAV 파일을 찾지 못했습니다.")
            continue

        sav_paths[year] = sav_path
        print(f"\n{year}년 SAV metadata 읽는 중...")
        print(sav_path)

        metas[year] = read_sav_metadata(sav_path)
        print(f"{year}년 변수 수:", len(metas[year].column_names))

    check_metas = {
        year: metas[year]
        for year in CHECK_YEARS
        if year in metas
    }

    reference_metas = {
        year: metas[year]
        for year in REFERENCE_YEARS
        if year in metas
    }

    if not check_metas:
        raise FileNotFoundError("2021/2022 SAV metadata를 하나도 읽지 못했습니다.")

    if not reference_metas:
        raise FileNotFoundError("2023/2024/2025 기준 SAV metadata를 하나도 읽지 못했습니다.")

    raw_detail = build_raw_variable_detail_report(
        selected_columns=selected_columns,
        check_metas=check_metas,
        reference_metas=reference_metas,
    )

    feature_summary = summarize_feature_compatibility(raw_detail)

    family_code = build_family_code_report(
        check_metas=check_metas,
        reference_metas=reference_metas,
    )

    attention = build_attention_report(
        feature_summary=feature_summary,
        raw_detail=raw_detail,
        family_code=family_code,
    )

    save_outputs(
        output_dir=output_dir,
        selected_column_report=selected_column_report,
        feature_summary=feature_summary,
        raw_detail=raw_detail,
        family_code=family_code,
        attention=attention,
    )

    print_summary(
        sav_paths=sav_paths,
        feature_summary=feature_summary,
        raw_detail=raw_detail,
        family_code=family_code,
        attention=attention,
    )


if __name__ == "__main__":
    main()