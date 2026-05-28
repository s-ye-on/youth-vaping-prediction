# 2021, 2022 가족구성 원본 변수들을 직접 읽고, 각 변수에서 우리가 기대한 선택 코드가 실제로 몇 명 나오는지 확인

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/Users/choi-seung-yeon/PyCharmMiscProject")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compatibility_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECK_YEARS = [2021, 2022]

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

OUTPUT_VALUE_COUNTS_CSV = OUTPUT_DIR / "family_value_counts_2021_2022.csv"
OUTPUT_DUMMY_SUMMARY_CSV = OUTPUT_DIR / "family_dummy_summary_2021_2022.csv"
OUTPUT_CONFLICT_CSV = OUTPUT_DIR / "family_no_family_conflict_2021_2022.csv"


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
    candidates = []

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


def read_family_columns(year: int) -> pd.DataFrame:
    pyreadstat = import_pyreadstat()

    sav_path = find_sav_file(year)
    usecols = list(FAMILY_MEMBER_SPECS.keys())

    print(f"\n{year}년 가족구성 변수 읽는 중...")
    print("파일:", sav_path)

    df, _ = pyreadstat.read_sav(
        str(sav_path),
        usecols=usecols,
        apply_value_formats=False,
        user_missing=True,
    )

    df["YEAR"] = year
    return df


def build_value_count_report(df: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []

    for raw_col, (dummy_col, selected_code, label) in FAMILY_MEMBER_SPECS.items():
        counts = df[raw_col].value_counts(dropna=False).sort_index()

        for value, count in counts.items():
            rows.append(
                {
                    "year": year,
                    "raw_variable": raw_col,
                    "dummy_variable": dummy_col,
                    "expected_selected_code": selected_code,
                    "expected_label": label,
                    "observed_value": value,
                    "count": int(count),
                    "is_expected_selected_code": value == selected_code,
                    "is_household_private_8888": value == HOUSEHOLD_PRIVATE_VALUE,
                }
            )

    return pd.DataFrame(rows)


def build_dummy_summary(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pd.DataFrame(index=df.index)

    rows = []

    for raw_col, (dummy_col, selected_code, label) in FAMILY_MEMBER_SPECS.items():
        result[dummy_col] = (df[raw_col] == selected_code).astype(int)

        selected_count = int(result[dummy_col].sum())
        private_count = int((df[raw_col] == HOUSEHOLD_PRIVATE_VALUE).sum())
        missing_count = int(df[raw_col].isna().sum())

        rows.append(
            {
                "year": year,
                "raw_variable": raw_col,
                "dummy_variable": dummy_col,
                "expected_selected_code": selected_code,
                "expected_label": label,
                "selected_count": selected_count,
                "household_private_8888_count": private_count,
                "missing_count": missing_count,
            }
        )

    member_cols = [
        dummy_col
        for _, (dummy_col, _, _) in FAMILY_MEMBER_SPECS.items()
        if dummy_col != "live_with_no_family"
    ]

    conflict = (
        (result["live_with_no_family"] == 1)
        & (result[member_cols].sum(axis=1) > 0)
    )

    conflict_df = result.loc[conflict].copy()
    conflict_df.insert(0, "YEAR", year)

    return pd.DataFrame(rows), conflict_df


def main() -> None:
    all_value_counts = []
    all_dummy_summaries = []
    all_conflicts = []

    for year in CHECK_YEARS:
        df = read_family_columns(year)

        value_count_report = build_value_count_report(df, year)
        dummy_summary, conflict_df = build_dummy_summary(df, year)

        all_value_counts.append(value_count_report)
        all_dummy_summaries.append(dummy_summary)
        all_conflicts.append(conflict_df)

        print(f"\n[{year} 가족구성 더미 합계]")
        print(dummy_summary[["dummy_variable", "selected_count"]].to_string(index=False))

        print(f"\n[{year} live_with_no_family 논리 충돌 수]")
        print(len(conflict_df))

    value_counts = pd.concat(all_value_counts, ignore_index=True)
    dummy_summaries = pd.concat(all_dummy_summaries, ignore_index=True)

    if all_conflicts:
        conflicts = pd.concat(all_conflicts, ignore_index=True)
    else:
        conflicts = pd.DataFrame()

    value_counts.to_csv(OUTPUT_VALUE_COUNTS_CSV, index=False, encoding="utf-8-sig")
    dummy_summaries.to_csv(OUTPUT_DUMMY_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    conflicts.to_csv(OUTPUT_CONFLICT_CSV, index=False, encoding="utf-8-sig")

    print("\n저장 완료")
    print(OUTPUT_VALUE_COUNTS_CSV)
    print(OUTPUT_DUMMY_SUMMARY_CSV)
    print(OUTPUT_CONFLICT_CSV)


if __name__ == "__main__":
    main()