# 가족 구성 변수 로직이 21~25 전체에서 실제로 작동하는지
# target 변수인 TC_EC_MN이 존재하고, 최근 30일 액상형 전자 담배 사용일수와 같은 의미인지

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/Users/choi-seung-yeon/PyCharmMiscProject")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compatibility_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025]

TARGET_RAW_VARIABLE = "TC_EC_MN"

FAMILY_MEMBER_SPECS = {
    "E_FM_F_1": ("live_with_father", 1, "아버지"),
    "E_FM_SF_2": ("live_with_stepfather", 2, "새아버지"),
    "E_FM_M_3": ("live_with_mother", 3, "어머니"),
    "E_FM_SM_4": ("live_with_stepmother", 4, "새어머니"),
    "E_FM_GF_5": ("live_with_grandfather", 5, "할아버지"),
    "E_FM_GM_6": ("live_with_grandmother", 6, "할머니"),
    "E_FM_OBS_7": ("live_with_older_sibling", 7, "형/누나/오빠/언니"),
    "E_FM_YBS_8": ("live_with_younger_sibling", 8, "남동생/여동생"),
    "E_FM_NO_9": ("live_with_no_family", 9, "가족구성원 없음"),
}

HOUSEHOLD_PRIVATE_VALUE = 8888


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
        raise FileNotFoundError(f"{year}년 SAV 파일을 찾지 못했습니다.")

    if len(candidates) > 1:
        print(f"\n[주의] {year}년 SAV 후보가 여러 개 있습니다. 첫 번째를 사용합니다.")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"{idx}. {candidate}")

    return candidates[0]


def normalize_value_key(value) -> str:
    if pd.isna(value):
        return "NaN"

    try:
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))
        return str(float_value)
    except (TypeError, ValueError):
        return str(value).strip()


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ").replace("\r", " ")


def get_variable_label(meta, variable: str) -> str:
    labels = getattr(meta, "column_names_to_labels", {}) or {}
    return normalize_text(labels.get(variable, ""))


def get_value_labels(meta, variable: str) -> dict[str, str]:
    variable_value_labels = getattr(meta, "variable_value_labels", {}) or {}
    labels = variable_value_labels.get(variable, {}) or {}

    return {
        normalize_value_key(key): normalize_text(value)
        for key, value in labels.items()
    }


def dict_to_string(value_labels: dict[str, str]) -> str:
    if not value_labels:
        return ""

    return " | ".join(
        f"{key}:{label}"
        for key, label in sorted(value_labels.items(), key=lambda x: x[0])
    )


def create_current_ecig_use(series: pd.Series) -> pd.Series:
    """
    기존 2023~2025에서 사용한 target 생성 규칙.
    TC_EC_MN:
    - 9999: 최근 30일 사용 경험 없음 또는 비해당 계열로 처리하여 0
    - 1: 최근 30일 0일 사용으로 처리하여 0
    - 2~7: 최근 30일 사용 있음으로 처리하여 1
    """
    return series.apply(lambda x: 1 if x in [2, 3, 4, 5, 6, 7] else 0)


def read_year_data(year: int) -> tuple[pd.DataFrame, object, Path]:
    pyreadstat = import_pyreadstat()

    sav_path = find_sav_file(year)

    usecols = [TARGET_RAW_VARIABLE] + list(FAMILY_MEMBER_SPECS.keys())

    print(f"\n{year}년 SAV 읽는 중...")
    print("파일:", sav_path)

    df, meta = pyreadstat.read_sav(
        str(sav_path),
        usecols=usecols,
        apply_value_formats=False,
        user_missing=True,
    )

    df["YEAR"] = year

    return df, meta, sav_path


def build_target_metadata_report(year: int, meta, df: pd.DataFrame, sav_path: Path) -> dict:
    exists = TARGET_RAW_VARIABLE in df.columns

    if not exists:
        return {
            "year": year,
            "sav_path": str(sav_path),
            "target_raw_variable": TARGET_RAW_VARIABLE,
            "exists": False,
            "variable_label": "",
            "value_labels": "",
            "raw_value_counts": "",
            "target_0_count": "",
            "target_1_count": "",
            "target_1_ratio": "",
            "decision": "사용 불가: TC_EC_MN 없음",
        }

    variable_label = get_variable_label(meta, TARGET_RAW_VARIABLE)
    value_labels = get_value_labels(meta, TARGET_RAW_VARIABLE)

    raw_counts = df[TARGET_RAW_VARIABLE].value_counts(dropna=False).sort_index()

    current_ecig_use = create_current_ecig_use(df[TARGET_RAW_VARIABLE])
    target_counts = current_ecig_use.value_counts(dropna=False).sort_index()

    target_0_count = int(target_counts.get(0, 0))
    target_1_count = int(target_counts.get(1, 0))
    total = target_0_count + target_1_count
    target_1_ratio = target_1_count / total if total else 0

    expected_values = {1, 2, 3, 4, 5, 6, 7, 9999}
    observed_values = {
        int(value)
        for value in df[TARGET_RAW_VARIABLE].dropna().unique()
        if float(value).is_integer()
    }

    has_expected_positive_values = bool(observed_values & {2, 3, 4, 5, 6, 7})
    has_expected_zero_values = bool(observed_values & {1, 9999})
    unknown_values = sorted(observed_values - expected_values)

    if unknown_values:
        decision = f"확인 필요: 예상 밖 코드값 존재 {unknown_values}"
    elif not has_expected_positive_values:
        decision = "확인 필요: 2~7 양성 코드가 관측되지 않음"
    elif not has_expected_zero_values:
        decision = "확인 필요: 1 또는 9999 음성 코드가 관측되지 않음"
    else:
        decision = "사용 가능: TC_EC_MN 존재, 2~7 양성 코드 관측됨"

    return {
        "year": year,
        "sav_path": str(sav_path),
        "target_raw_variable": TARGET_RAW_VARIABLE,
        "exists": True,
        "variable_label": variable_label,
        "value_labels": dict_to_string(value_labels),
        "raw_value_counts": " | ".join(
            f"{normalize_value_key(index)}:{int(count)}"
            for index, count in raw_counts.items()
        ),
        "target_0_count": target_0_count,
        "target_1_count": target_1_count,
        "target_1_ratio": round(target_1_ratio, 6),
        "decision": decision,
    }


def build_family_metadata_and_value_report(year: int, meta, df: pd.DataFrame, sav_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_rows = []
    dummy_summary_rows = []

    dummy_df = pd.DataFrame(index=df.index)

    for raw_variable, (dummy_variable, selected_code, expected_label) in FAMILY_MEMBER_SPECS.items():
        exists = raw_variable in df.columns

        if not exists:
            metadata_rows.append(
                {
                    "year": year,
                    "sav_path": str(sav_path),
                    "raw_variable": raw_variable,
                    "dummy_variable": dummy_variable,
                    "exists": False,
                    "expected_selected_code": selected_code,
                    "expected_label": expected_label,
                    "variable_label": "",
                    "value_labels": "",
                    "selected_count": "",
                    "private_8888_count": "",
                    "decision": "사용 불가: 변수 없음",
                }
            )
            continue

        variable_label = get_variable_label(meta, raw_variable)
        value_labels = get_value_labels(meta, raw_variable)

        selected_count = int((df[raw_variable] == selected_code).sum())
        private_count = int((df[raw_variable] == HOUSEHOLD_PRIVATE_VALUE).sum())

        raw_counts = df[raw_variable].value_counts(dropna=False).sort_index()

        dummy_df[dummy_variable] = (df[raw_variable] == selected_code).astype(int)

        if selected_count <= 0:
            decision = "확인 필요: 기대 선택 코드가 실제 데이터에 없음"
        else:
            decision = "사용 가능: 기대 선택 코드가 실제 데이터에 존재"

        metadata_rows.append(
            {
                "year": year,
                "sav_path": str(sav_path),
                "raw_variable": raw_variable,
                "dummy_variable": dummy_variable,
                "exists": True,
                "expected_selected_code": selected_code,
                "expected_label": expected_label,
                "variable_label": variable_label,
                "value_labels": dict_to_string(value_labels),
                "raw_value_counts": " | ".join(
                    f"{normalize_value_key(index)}:{int(count)}"
                    for index, count in raw_counts.items()
                ),
                "selected_count": selected_count,
                "private_8888_count": private_count,
                "decision": decision,
            }
        )

        dummy_summary_rows.append(
            {
                "year": year,
                "dummy_variable": dummy_variable,
                "raw_variable": raw_variable,
                "expected_selected_code": selected_code,
                "selected_count": selected_count,
                "private_8888_count": private_count,
            }
        )

    member_dummy_cols = [
        dummy_variable
        for dummy_variable, _, in []
    ]

    member_dummy_cols = [
        spec[0]
        for raw, spec in FAMILY_MEMBER_SPECS.items()
        if spec[0] != "live_with_no_family"
    ]

    if "live_with_no_family" in dummy_df.columns:
        conflict_mask = (
            (dummy_df["live_with_no_family"] == 1)
            & (dummy_df[member_dummy_cols].sum(axis=1) > 0)
        )

        conflict_df = dummy_df.loc[conflict_mask].copy()
        conflict_df.insert(0, "YEAR", year)
    else:
        conflict_df = pd.DataFrame()

    return pd.DataFrame(metadata_rows), pd.DataFrame(dummy_summary_rows), conflict_df


def compare_metadata_across_years(report: pd.DataFrame, key_col: str) -> pd.DataFrame:
    rows = []

    for key, group in report.groupby(key_col):
        labels = group.set_index("year")["variable_label"].to_dict()
        value_labels = group.set_index("year")["value_labels"].to_dict()

        reference_label = labels.get(2025) or labels.get(2024) or labels.get(2023) or ""
        reference_value_labels = value_labels.get(2025) or value_labels.get(2024) or value_labels.get(2023) or ""

        for _, row in group.iterrows():
            year = row["year"]

            same_label_as_reference = row["variable_label"] == reference_label
            same_value_labels_as_reference = row["value_labels"] == reference_value_labels

            rows.append(
                {
                    "year": year,
                    key_col: key,
                    "same_label_as_2023_2025_reference": same_label_as_reference,
                    "same_value_labels_as_2023_2025_reference": same_value_labels_as_reference,
                    "variable_label": row["variable_label"],
                    "reference_label": reference_label,
                    "value_labels": row["value_labels"],
                    "reference_value_labels": reference_value_labels,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    target_reports = []
    family_metadata_reports = []
    family_dummy_summaries = []
    family_conflicts = []

    for year in YEARS:
        df, meta, sav_path = read_year_data(year)

        target_reports.append(
            build_target_metadata_report(year, meta, df, sav_path)
        )

        family_metadata, family_summary, family_conflict = build_family_metadata_and_value_report(
            year, meta, df, sav_path
        )

        family_metadata_reports.append(family_metadata)
        family_dummy_summaries.append(family_summary)

        if not family_conflict.empty:
            family_conflicts.append(family_conflict)

    target_report = pd.DataFrame(target_reports)
    family_metadata_report = pd.concat(family_metadata_reports, ignore_index=True)
    family_dummy_summary = pd.concat(family_dummy_summaries, ignore_index=True)

    if family_conflicts:
        family_conflict_report = pd.concat(family_conflicts, ignore_index=True)
    else:
        family_conflict_report = pd.DataFrame()

    target_metadata_compare = compare_metadata_across_years(
        target_report,
        key_col="target_raw_variable",
    )

    family_metadata_compare = compare_metadata_across_years(
        family_metadata_report,
        key_col="raw_variable",
    )

    target_report_path = OUTPUT_DIR / "confirm_target_TC_EC_MN_2021_2025.csv"
    target_compare_path = OUTPUT_DIR / "confirm_target_TC_EC_MN_metadata_compare_2021_2025.csv"
    family_metadata_path = OUTPUT_DIR / "confirm_family_metadata_and_values_2021_2025.csv"
    family_compare_path = OUTPUT_DIR / "confirm_family_metadata_compare_2021_2025.csv"
    family_summary_path = OUTPUT_DIR / "confirm_family_dummy_summary_2021_2025.csv"
    family_conflict_path = OUTPUT_DIR / "confirm_family_no_family_conflict_2021_2025.csv"
    excel_path = OUTPUT_DIR / "confirm_family_and_target_2021_2025.xlsx"

    target_report.to_csv(target_report_path, index=False, encoding="utf-8-sig")
    target_metadata_compare.to_csv(target_compare_path, index=False, encoding="utf-8-sig")
    family_metadata_report.to_csv(family_metadata_path, index=False, encoding="utf-8-sig")
    family_metadata_compare.to_csv(family_compare_path, index=False, encoding="utf-8-sig")
    family_dummy_summary.to_csv(family_summary_path, index=False, encoding="utf-8-sig")
    family_conflict_report.to_csv(family_conflict_path, index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(excel_path) as writer:
            target_report.to_excel(writer, sheet_name="target_TC_EC_MN", index=False)
            target_metadata_compare.to_excel(writer, sheet_name="target_metadata_compare", index=False)
            family_metadata_report.to_excel(writer, sheet_name="family_metadata_values", index=False)
            family_metadata_compare.to_excel(writer, sheet_name="family_metadata_compare", index=False)
            family_dummy_summary.to_excel(writer, sheet_name="family_dummy_summary", index=False)
            family_conflict_report.to_excel(writer, sheet_name="family_conflict", index=False)

        print("\n엑셀 저장 완료:")
        print(excel_path)

    except ImportError:
        print("\n[주의] openpyxl이 없어 엑셀 저장은 건너뜁니다.")
        print("/Users/choi-seung-yeon/.virtualenvs/.venv/bin/python -m pip install openpyxl")

    print("\n" + "=" * 80)
    print("TC_EC_MN target 확인")
    print("=" * 80)
    print(
        target_report[
            [
                "year",
                "exists",
                "variable_label",
                "raw_value_counts",
                "target_0_count",
                "target_1_count",
                "target_1_ratio",
                "decision",
            ]
        ].to_string(index=False, max_colwidth=120)
    )

    print("\n" + "=" * 80)
    print("TC_EC_MN metadata 비교")
    print("=" * 80)
    print(
        target_metadata_compare[
            [
                "year",
                "target_raw_variable",
                "same_label_as_2023_2025_reference",
                "same_value_labels_as_2023_2025_reference",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("가족구성 더미 합계")
    print("=" * 80)
    print(
        family_dummy_summary[
            [
                "year",
                "dummy_variable",
                "raw_variable",
                "expected_selected_code",
                "selected_count",
                "private_8888_count",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("가족구성 metadata 비교 요약")
    print("=" * 80)
    print(
        pd.crosstab(
            family_metadata_compare["year"],
            [
                family_metadata_compare["same_label_as_2023_2025_reference"],
                family_metadata_compare["same_value_labels_as_2023_2025_reference"],
            ],
        )
    )

    print("\n" + "=" * 80)
    print("live_with_no_family 논리 충돌 수")
    print("=" * 80)
    print(len(family_conflict_report))

    print("\n저장 완료:")
    print(target_report_path)
    print(target_compare_path)
    print(family_metadata_path)
    print(family_compare_path)
    print(family_summary_path)
    print(family_conflict_path)


if __name__ == "__main__":
    main()