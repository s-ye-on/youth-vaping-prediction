# 21년 22년 데이터 확인

from pathlib import Path

import pandas as pd


# ============================================================
# 1. 프로젝트 경로 설정
# ============================================================

PROJECT_ROOT = Path("/Users/choi-seung-yeon/PyCharmMiscProject")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compatibility_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 확인할 연도
# ============================================================

CHECK_YEARS = [2021, 2022]

# 기준 연도 후보.
# 2025 sav가 있으면 2025를 기준으로 value label 비교.
# 없으면 2024, 2023 순서로 사용.
REFERENCE_YEAR_CANDIDATES = [2025, 2024, 2023]


# ============================================================
# 3. 현재 모델링 데이터셋을 만들 때 사용한 원본 변수들
# ============================================================

GENERATED_COLS = {
    "YEAR": "원자료에 없으면 파일 연도에서 생성 가능",
}

REQUIRED_RAW_VARIABLES = {
    # target source
    "TC_EC_MN": "target source: 최근 30일 액상형 전자담배 사용일수",

    # basic
    "CITY": "지역",
    "CTYPE": "도시규모",
    "SEX": "성별",
    "AGE": "만 나이",
    "GRADE": "학년",
    "SCHOOL": "학교급",
    "STYPE": "학교유형",

    # subjective health / body image
    "PR_HT": "주관적 건강인지",
    "PR_BI": "주관적 체형인지",

    # mental health
    "M_STR": "스트레스 인지",
    "M_SAD": "슬픔/절망감 경험",

    # physical activity
    "PA_TOT": "신체활동",

    # smartphone
    "INT_SPWD_TM": "주중 스마트폰 사용 시간",
    "INT_SPWK_TM": "주말 스마트폰 사용 시간",

    # economic / residence / academic
    "E_SES": "경제상태",
    "E_RES": "거주형태",
    "E_S_RCRD": "주관적 학업성적",

    # diet
    "F_BR": "최근 7일 아침식사 빈도",
    "F_FRUIT": "최근 7일 과일 섭취 빈도",
    "F_FASTFOOD": "최근 7일 패스트푸드 섭취 빈도",

    # secondhand smoke
    "TC_SND_H": "최근 7일 가정 내 간접흡연",
    "TC_SND_P": "최근 7일 공공장소 간접흡연",

    # alcohol
    "AC_LT": "평생 음주 경험",
    "AC_FAGE": "처음 음주 경험 나이",
    "AC_DAYS": "최근 30일 음주 일수",

    # general cigarette
    "TC_LT": "평생 일반담배 흡연 경험",

    # family consent
    "A_FM": "가구조사 동의 여부",

    # family members
    "E_FM_F_1": "가족구성: 아버지",
    "E_FM_SF_2": "가족구성: 새아버지",
    "E_FM_M_3": "가족구성: 어머니",
    "E_FM_SM_4": "가족구성: 새어머니",
    "E_FM_GF_5": "가족구성: 할아버지",
    "E_FM_GM_6": "가족구성: 할머니",
    "E_FM_OBS_7": "가족구성: 형/누나/오빠/언니",
    "E_FM_YBS_8": "가족구성: 남동생/여동생",
    "E_FM_NO_9": "가족구성: 가족구성원 없음",

    # parent
    "E_EDU_F": "아버지 학력",
    "E_EDU_M": "어머니 학력",
    "E_KRN_F": "아버지 한국 출생 여부",
    "E_KRN_M": "어머니 한국 출생 여부",
}


# ============================================================
# 4. 후보 변수명
#    2021/2022에서 변수명이 다를 가능성이 있는 항목을 찾기 위한 후보
# ============================================================

ALTERNATIVE_CANDIDATES = {
    "E_S_RCRD": ["E_S_RCRD", "E_RCRD", "E_ACDM", "E_SCHOOL_RECORD", "E_GRADE_RECORD"],

    "TC_SND_H": ["TC_SND_H", "TC_SHS_H", "TC_SHS_HOME", "TC_HOME_SHS", "TC_HSHS"],
    "TC_SND_P": ["TC_SND_P", "TC_SHS_P", "TC_SHS_PUBLIC", "TC_PUBLIC_SHS", "TC_PSHS"],

    "F_BR": ["F_BR", "F_BR_FQ", "F_BREAKFAST", "F_BREAKFAST_FREQ"],
    "F_FRUIT": ["F_FRUIT", "F_FRUIT_FQ", "F_FRUIT_FREQ"],
    "F_FASTFOOD": ["F_FASTFOOD", "F_FASTFOOD_FQ", "F_FASTFOOD_FREQ"],
}


# ============================================================
# 5. pyreadstat import
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


# ============================================================
# 6. 파일 탐색
# ============================================================

def find_sav_file(year: int) -> Path:
    candidates = []

    patterns = [
        f"*{year}*.sav",
        f"kyrbs{year}.sav",
        f"KYRBS{year}.sav",
    ]

    ignore_parts = {
        ".git",
        ".idea",
        ".venv",
        "venv",
        "__pycache__",
    }

    for pattern in patterns:
        for path in PROJECT_ROOT.rglob(pattern):
            if any(part in ignore_parts for part in path.parts):
                continue
            candidates.append(path)

    unique_candidates = sorted(set(candidates))

    if not unique_candidates:
        raise FileNotFoundError(
            f"\n{year}년 sav 파일을 프로젝트 안에서 찾지 못했습니다.\n"
            f"탐색 위치: {PROJECT_ROOT}\n\n"
            f"파일명을 kyrbs{year}.sav 형태로 두거나, 코드의 PROJECT_ROOT를 확인하세요.\n"
        )

    if len(unique_candidates) > 1:
        print(f"\n[주의] {year}년 sav 후보가 여러 개 발견되었습니다. 첫 번째 파일을 사용합니다.")
        for idx, path in enumerate(unique_candidates, start=1):
            print(f"{idx}. {path}")

    return unique_candidates[0]


# ============================================================
# 7. metadata 읽기
# ============================================================

def read_sav_metadata(path: Path):
    pyreadstat = import_pyreadstat()

    # metadataonly=True로 데이터 전체를 읽지 않고 메타데이터 중심으로 확인
    df, meta = pyreadstat.read_sav(
        str(path),
        metadataonly=True,
        apply_value_formats=False,
        user_missing=True,
    )

    return meta


def get_column_label(meta, variable: str) -> str:
    labels = getattr(meta, "column_names_to_labels", {}) or {}
    return labels.get(variable, "")


def get_value_labels(meta, variable: str) -> dict:
    variable_value_labels = getattr(meta, "variable_value_labels", {}) or {}
    labels = variable_value_labels.get(variable, {}) or {}

    # 비교를 안정적으로 하기 위해 key를 문자열로 변환
    return {str(key): str(value) for key, value in labels.items()}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ").replace("\r", " ")


# ============================================================
# 8. 변수 존재 및 의미 비교
# ============================================================

def find_existing_variable(meta, expected_variable: str) -> tuple[str | None, str]:
    column_names = set(meta.column_names)

    if expected_variable in column_names:
        return expected_variable, "same_name"

    for candidate in ALTERNATIVE_CANDIDATES.get(expected_variable, []):
        if candidate in column_names:
            return candidate, "alternative_name"

    return None, "missing"


def labels_equal(label_a: str, label_b: str) -> bool:
    return normalize_text(label_a) == normalize_text(label_b)


def value_labels_equal(labels_a: dict, labels_b: dict) -> bool:
    return labels_a == labels_b


def build_compatibility_report(
    check_year: int,
    check_meta,
    reference_year: int | None,
    reference_meta,
) -> pd.DataFrame:
    rows = []

    for expected_variable, description in REQUIRED_RAW_VARIABLES.items():
        actual_variable, status = find_existing_variable(check_meta, expected_variable)

        check_label = ""
        check_value_labels = {}
        reference_label = ""
        reference_value_labels = {}

        if actual_variable is not None:
            check_label = get_column_label(check_meta, actual_variable)
            check_value_labels = get_value_labels(check_meta, actual_variable)

        if reference_meta is not None:
            ref_actual_variable, _ = find_existing_variable(reference_meta, expected_variable)

            if ref_actual_variable is not None:
                reference_label = get_column_label(reference_meta, ref_actual_variable)
                reference_value_labels = get_value_labels(reference_meta, ref_actual_variable)

        if actual_variable is None:
            compatibility = "MISSING"
        elif reference_meta is None:
            compatibility = "EXISTS_REFERENCE_NOT_AVAILABLE"
        else:
            same_label = labels_equal(check_label, reference_label)
            same_value_labels = value_labels_equal(check_value_labels, reference_value_labels)

            if same_label and same_value_labels:
                compatibility = "SAME_LABEL_AND_VALUES"
            elif same_label and not same_value_labels:
                compatibility = "SAME_LABEL_BUT_VALUE_LABEL_DIFFERS"
            elif not same_label and same_value_labels:
                compatibility = "LABEL_DIFFERS_BUT_VALUES_SAME"
            else:
                compatibility = "LABEL_AND_VALUE_LABEL_DIFFERS"

        rows.append(
            {
                "check_year": check_year,
                "expected_variable": expected_variable,
                "actual_variable": actual_variable if actual_variable is not None else "",
                "status": status,
                "description": description,
                "compatibility": compatibility,
                "check_label": normalize_text(check_label),
                "reference_year": reference_year if reference_year is not None else "",
                "reference_label": normalize_text(reference_label),
                "check_value_labels": check_value_labels,
                "reference_value_labels": reference_value_labels,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 9. 특수 검증: 가족구성 선택 코드 확인
# ============================================================

FAMILY_SELECTED_CODES = {
    "E_FM_F_1": 1,
    "E_FM_SF_2": 2,
    "E_FM_M_3": 3,
    "E_FM_SM_4": 4,
    "E_FM_GF_5": 5,
    "E_FM_GM_6": 6,
    "E_FM_OBS_7": 7,
    "E_FM_YBS_8": 8,
    "E_FM_NO_9": 9,
}


def build_family_code_report(year: int, meta) -> pd.DataFrame:
    rows = []

    for variable, expected_code in FAMILY_SELECTED_CODES.items():
        actual_variable, status = find_existing_variable(meta, variable)
        value_labels = {}

        if actual_variable is not None:
            value_labels = get_value_labels(meta, actual_variable)

        selected_label = value_labels.get(str(float(expected_code))) or value_labels.get(str(expected_code)) or ""

        rows.append(
            {
                "year": year,
                "expected_variable": variable,
                "actual_variable": actual_variable if actual_variable is not None else "",
                "status": status,
                "expected_selected_code": expected_code,
                "selected_code_label": selected_label,
                "all_value_labels": value_labels,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 10. 요약 출력
# ============================================================

def print_year_summary(report: pd.DataFrame, year: int) -> None:
    print("\n" + "=" * 70)
    print(f"{year}년 변수 호환성 요약")
    print("=" * 70)

    print("\n[status]")
    print(report["status"].value_counts(dropna=False))

    print("\n[compatibility]")
    print(report["compatibility"].value_counts(dropna=False))

    missing = report[report["compatibility"] == "MISSING"]

    if not missing.empty:
        print("\n[누락 변수]")
        for _, row in missing.iterrows():
            print(f"- {row['expected_variable']} : {row['description']}")

    diff = report[
        report["compatibility"].isin(
            [
                "SAME_LABEL_BUT_VALUE_LABEL_DIFFERS",
                "LABEL_DIFFERS_BUT_VALUES_SAME",
                "LABEL_AND_VALUE_LABEL_DIFFERS",
            ]
        )
    ]

    if not diff.empty:
        print("\n[라벨 또는 값 의미 확인 필요]")
        for _, row in diff.iterrows():
            print(
                f"- {row['expected_variable']} "
                f"(실제: {row['actual_variable']}) "
                f"=> {row['compatibility']}"
            )


def save_report(report: pd.DataFrame, filename: str) -> Path:
    output_path = OUTPUT_DIR / filename
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


# ============================================================
# 11. main
# ============================================================

def main() -> None:
    print("프로젝트 루트:", PROJECT_ROOT)
    print("결과 저장 폴더:", OUTPUT_DIR)

    metas = {}
    sav_paths = {}

    all_years_to_try = sorted(set(CHECK_YEARS + REFERENCE_YEAR_CANDIDATES))

    for year in all_years_to_try:
        try:
            sav_path = find_sav_file(year)
            sav_paths[year] = sav_path

            print(f"\n{year}년 sav metadata 읽는 중...")
            print("파일:", sav_path)

            metas[year] = read_sav_metadata(sav_path)
            print(f"{year}년 변수 수:", len(metas[year].column_names))

        except FileNotFoundError:
            print(f"\n[참고] {year}년 sav 파일 없음. 해당 연도는 건너뜁니다.")

    reference_year = None
    reference_meta = None

    for year in REFERENCE_YEAR_CANDIDATES:
        if year in metas:
            reference_year = year
            reference_meta = metas[year]
            break

    if reference_meta is None:
        print("\n[주의] 2023~2025 기준 sav를 찾지 못했습니다.")
        print("2021/2022 변수 존재 여부만 확인하고, 라벨 비교는 제한됩니다.")
    else:
        print(f"\n기준 연도: {reference_year}")

    all_reports = []
    all_family_reports = []

    for year in CHECK_YEARS:
        if year not in metas:
            continue

        report = build_compatibility_report(
            check_year=year,
            check_meta=metas[year],
            reference_year=reference_year,
            reference_meta=reference_meta,
        )

        family_report = build_family_code_report(
            year=year,
            meta=metas[year],
        )

        print_year_summary(report, year)

        report_path = save_report(report, f"variable_compatibility_{year}.csv")
        family_path = save_report(family_report, f"family_code_check_{year}.csv")

        print("\n저장 완료:")
        print("-", report_path)
        print("-", family_path)

        all_reports.append(report)
        all_family_reports.append(family_report)

    if all_reports:
        combined = pd.concat(all_reports, ignore_index=True)
        combined_path = save_report(combined, "variable_compatibility_2021_2022_combined.csv")
        print("\n통합 변수 호환성 리포트 저장:")
        print("-", combined_path)

    if all_family_reports:
        combined_family = pd.concat(all_family_reports, ignore_index=True)
        combined_family_path = save_report(combined_family, "family_code_check_2021_2022_combined.csv")
        print("\n통합 가족구성 코드 리포트 저장:")
        print("-", combined_family_path)

    print("\n완료")


if __name__ == "__main__":
    main()