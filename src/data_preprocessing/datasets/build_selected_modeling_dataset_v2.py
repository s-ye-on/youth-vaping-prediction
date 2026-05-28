# 데이터 전처리 v2

import numpy as np
import pandas as pd

from src.data_preprocessing.project_paths import PROCESSED_DATA_DIR

INPUT_FILE = PROCESSED_DATA_DIR / "selected_modeling_dataset.csv"

OUTPUT_DIR = PROCESSED_DATA_DIR
OUTPUT_FILE = OUTPUT_DIR / "selected_modeling_dataset_v2.csv"
OUTPUT_X_FILE = OUTPUT_DIR / "selected_modeling_X_v2.csv"
OUTPUT_Y_FILE = OUTPUT_DIR / "selected_modeling_y_v2.csv"

TARGET_COL = "current_ecig_use"

HEALTH_COL = "PR_HT"
BODY_IMAGE_COL = "PR_BI"
SMARTPHONE_WEEKDAY_COL = "INT_SPWD_TM"
SMARTPHONE_WEEKEND_COL = "INT_SPWK_TM"

SUBJECTIVE_UNHEALTHY_COL = "subjective_unhealthy_level"
SMARTPHONE_TOTAL_COL = "smartphone_total_time"


BODY_IMAGE_DUMMY_MAP = {
    1: "body_image_very_thin",
    2: "body_image_thin",
    3: "body_image_normal",
    4: "body_image_fat",
    5: "body_image_very_fat",
}


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            "\n필수 컬럼이 데이터에 없습니다.\n"
            f"없는 컬럼: {missing}\n\n"
            "해결 방법:\n"
            "1. 먼저 build_selected_modeling_dataset.py를 실행해서 "
            "selected_modeling_dataset.csv를 생성했는지 확인하세요.\n"
            "2. 컬럼명이 바뀌었다면 코드 상단의 변수명 상수를 수정하세요.\n"
        )


def assert_target_valid(df: pd.DataFrame) -> None:
    values = sorted(df[TARGET_COL].dropna().unique().tolist())

    if values != [0, 1]:
        raise ValueError(
            f"{TARGET_COL} 값이 0/1이 아닙니다. 현재 값: {values}"
        )


def convert_subjective_health(result: pd.DataFrame) -> pd.DataFrame:
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
    숫자가 클수록 주관적으로 건강하지 않다고 느끼는 방향이 된다.
    """
    result[SUBJECTIVE_UNHEALTHY_COL] = result[HEALTH_COL]

    return result.drop(columns=[HEALTH_COL])


def convert_body_image_to_one_hot(result: pd.DataFrame) -> pd.DataFrame:
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
    결측이 있는 행은 모든 체형 더미가 0이 되며,
    body_image_missing 컬럼으로 결측 여부를 따로 보존한다.
    """
    body_image = result[BODY_IMAGE_COL]

    result["body_image_missing"] = body_image.isna().astype(int)

    for code, new_col in BODY_IMAGE_DUMMY_MAP.items():
        result[new_col] = (body_image == code).astype(int)

    return result.drop(columns=[BODY_IMAGE_COL])


def combine_smartphone_time(result: pd.DataFrame) -> pd.DataFrame:
    """
    스마트폰 사용 시간 처리.

    INT_SPWD_TM = 주중 스마트폰 사용 시간, 분 단위
    INT_SPWK_TM = 주말 스마트폰 사용 시간, 분 단위

    처리:
    smartphone_total_time = 주중 + 주말

    주의:
    둘 중 하나라도 결측이면 합산값도 결측으로 둔다.
    결측을 0분으로 잘못 해석하지 않기 위함이다.
    """
    result[SMARTPHONE_TOTAL_COL] = result[
        [SMARTPHONE_WEEKDAY_COL, SMARTPHONE_WEEKEND_COL]
    ].sum(axis=1, min_count=2)

    result["smartphone_time_missing"] = result[
        [SMARTPHONE_WEEKDAY_COL, SMARTPHONE_WEEKEND_COL]
    ].isna().any(axis=1).astype(int)

    return result.drop(columns=[SMARTPHONE_WEEKDAY_COL, SMARTPHONE_WEEKEND_COL])


def move_target_to_last(result: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in result.columns if col != TARGET_COL]
    columns.append(TARGET_COL)
    return result[columns]


def print_change_summary(before: pd.DataFrame, after: pd.DataFrame) -> None:
    print("\n[변경 전 shape]")
    print(before.shape)

    print("\n[변경 후 shape]")
    print(after.shape)

    removed_cols = sorted(set(before.columns) - set(after.columns))
    added_cols = sorted(set(after.columns) - set(before.columns))

    print("\n[제거된 컬럼]")
    for col in removed_cols:
        print("-", col)

    print("\n[추가된 컬럼]")
    for col in added_cols:
        print("-", col)

    print("\n[target 분포]")
    print(after[TARGET_COL].value_counts(dropna=False))

    print("\n[target 비율]")
    print(after[TARGET_COL].value_counts(normalize=True, dropna=False).round(4))

    print("\n[주관적 건강인지 변환 확인]")
    print(after[SUBJECTIVE_UNHEALTHY_COL].value_counts(dropna=False).sort_index())

    print("\n[주관적 체형인지 one-hot 합계 확인]")
    body_cols = list(BODY_IMAGE_DUMMY_MAP.values()) + ["body_image_missing"]
    print(after[body_cols].sum().sort_index())

    print("\n[스마트폰 총 사용 시간 결측 확인]")
    print(after[[SMARTPHONE_TOTAL_COL, "smartphone_time_missing"]].isna().sum())

    print("\n[최종 결측 비율 상위]")
    missing_ratio = after.isna().mean().sort_values(ascending=False)
    missing_ratio = missing_ratio[missing_ratio > 0]

    if missing_ratio.empty:
        print("결측치가 있는 컬럼이 없습니다.")
    else:
        print(missing_ratio.head(30).round(4))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("selected_modeling_dataset.csv 읽는 중...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("입력 데이터 shape:", df.shape)

    required_cols = [
        TARGET_COL,
        HEALTH_COL,
        BODY_IMAGE_COL,
        SMARTPHONE_WEEKDAY_COL,
        SMARTPHONE_WEEKEND_COL,
    ]

    require_columns(df, required_cols)
    assert_target_valid(df)

    result = df.copy()

    result = convert_subjective_health(result)
    result = convert_body_image_to_one_hot(result)
    result = combine_smartphone_time(result)
    result = move_target_to_last(result)

    assert_target_valid(result)

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    X = result.drop(columns=[TARGET_COL])
    y = result[[TARGET_COL]]

    X.to_csv(OUTPUT_X_FILE, index=False, encoding="utf-8-sig")
    y.to_csv(OUTPUT_Y_FILE, index=False, encoding="utf-8-sig")

    print("\n저장 완료")
    print("selected dataset v2:", OUTPUT_FILE)
    print("X only v2:", OUTPUT_X_FILE)
    print("y only v2:", OUTPUT_Y_FILE)

    print_change_summary(df, result)

    print("\n최종 컬럼 목록:")
    for col in result.columns:
        print("-", col)


if __name__ == "__main__":
    main()
