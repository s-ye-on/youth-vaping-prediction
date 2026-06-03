from pathlib import Path

import pandas as pd

from src.data_preprocessing.project_paths import PROCESSED_MODELING_DIR, PROCESSED_REDUCED_DIR, PROJECT_ROOT

# ============================================================
# 0. 기본 설정
# ============================================================

PROCESSED_DIR = PROCESSED_MODELING_DIR
REDUCED_DIR = PROCESSED_REDUCED_DIR
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "feature_reduction"

REDUCED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "current_ecig_use"


# ============================================================
# 1. 입력 데이터셋
# ============================================================

INPUT_DATASETS = [
    {
        "label": "2023_2025_with_ever_cigarette_use",
        "input_file": "selected_modeling_dataset.csv",
        "output_file": "selected_modeling_dataset_reduced_low_importance_removed.csv",
        "output_x_file": "selected_modeling_X_reduced_low_importance_removed.csv",
        "output_y_file": "selected_modeling_y_reduced_low_importance_removed.csv",
    },
    {
        "label": "2023_2025_without_ever_cigarette_use",
        "input_file": "selected_modeling_dataset_without_ever_cigarette_use.csv",
        "output_file": "selected_modeling_dataset_without_ever_cigarette_use_reduced_low_importance_removed.csv",
        "output_x_file": "selected_modeling_X_without_ever_cigarette_use_reduced_low_importance_removed.csv",
        "output_y_file": "selected_modeling_y_without_ever_cigarette_use_reduced_low_importance_removed.csv",
    },
    {
        "label": "2021_2025_with_ever_cigarette_use",
        "input_file": "selected_modeling_dataset_2021_2025.csv",
        "output_file": "selected_modeling_dataset_2021_2025_reduced_low_importance_removed.csv",
        "output_x_file": "selected_modeling_X_2021_2025_reduced_low_importance_removed.csv",
        "output_y_file": "selected_modeling_y_2021_2025_reduced_low_importance_removed.csv",
    },
    {
        "label": "2021_2025_without_ever_cigarette_use",
        "input_file": "selected_modeling_dataset_2021_2025_without_ever_cigarette_use.csv",
        "output_file": "selected_modeling_dataset_2021_2025_without_ever_cigarette_use_reduced_low_importance_removed.csv",
        "output_x_file": "selected_modeling_X_2021_2025_without_ever_cigarette_use_reduced_low_importance_removed.csv",
        "output_y_file": "selected_modeling_y_2021_2025_without_ever_cigarette_use_reduced_low_importance_removed.csv",
    },
]


# ============================================================
# 2. 제거 / 유지 feature
# ============================================================

DROP_FEATURES = [
    "fruit_freq",
    "academic_performance",
    "mother_korean_birth",
    "live_with_younger_sibling",
    "live_with_older_sibling",
    "live_with_grandmother",
]

KEEP_FEATURES_BY_DECISION = [
    "subjective_unhealthy_level",
    "father_edu_unknown",
    "father_absent_by_edu",
]


# ============================================================
# 3. 검증 함수
# ============================================================

def validate_target_exists(df: pd.DataFrame, input_path: Path) -> None:
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"\nTarget 컬럼이 없습니다.\n"
            f"파일: {input_path}\n"
            f"필요 컬럼: {TARGET_COL}\n"
            f"현재 컬럼: {df.columns.tolist()}\n"
        )


def validate_keep_features_not_dropped() -> None:
    accidentally_dropped = [
        feature
        for feature in KEEP_FEATURES_BY_DECISION
        if feature in DROP_FEATURES
    ]

    if accidentally_dropped:
        raise ValueError(
            "\n유지하기로 한 feature가 DROP_FEATURES에 들어 있습니다.\n"
            f"문제 feature: {accidentally_dropped}\n"
        )


def build_reduced_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    existing_drop_features = [
        feature
        for feature in DROP_FEATURES
        if feature in df.columns
    ]

    missing_drop_features = [
        feature
        for feature in DROP_FEATURES
        if feature not in df.columns
    ]

    reduced_df = df.drop(columns=existing_drop_features)

    return reduced_df, existing_drop_features, missing_drop_features


def build_dataset_summary(
    label: str,
    original_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
    removed_features: list[str],
    missing_drop_features: list[str],
) -> dict:
    original_target_counts = original_df[TARGET_COL].value_counts(dropna=False).sort_index()
    reduced_target_counts = reduced_df[TARGET_COL].value_counts(dropna=False).sort_index()

    return {
        "dataset_label": label,
        "original_rows": len(original_df),
        "reduced_rows": len(reduced_df),
        "original_columns": len(original_df.columns),
        "reduced_columns": len(reduced_df.columns),
        "removed_feature_count": len(removed_features),
        "removed_features": ", ".join(removed_features),
        "drop_features_not_found": ", ".join(missing_drop_features),
        "target_0_count_original": int(original_target_counts.get(0, 0)),
        "target_1_count_original": int(original_target_counts.get(1, 0)),
        "target_0_count_reduced": int(reduced_target_counts.get(0, 0)),
        "target_1_count_reduced": int(reduced_target_counts.get(1, 0)),
        "target_distribution_same": original_target_counts.equals(reduced_target_counts),
    }


def save_reduced_outputs(
    reduced_df: pd.DataFrame,
    output_dataset_path: Path,
    output_x_path: Path,
    output_y_path: Path,
) -> None:
    X = reduced_df.drop(columns=[TARGET_COL])
    y = reduced_df[[TARGET_COL]]

    reduced_df.to_csv(output_dataset_path, index=False, encoding="utf-8-sig")
    X.to_csv(output_x_path, index=False, encoding="utf-8-sig")
    y.to_csv(output_y_path, index=False, encoding="utf-8-sig")


# ============================================================
# 4. 메인 실행
# ============================================================

def main() -> None:
    validate_keep_features_not_dropped()

    summaries = []

    print("\n" + "=" * 80)
    print("저중요도 feature 제거 데이터셋 생성 시작")
    print("=" * 80)

    print("\n[제거 대상 feature]")
    for feature in DROP_FEATURES:
        print("-", feature)

    print("\n[명시적 유지 feature]")
    for feature in KEEP_FEATURES_BY_DECISION:
        print("-", feature)

    for dataset_config in INPUT_DATASETS:
        label = dataset_config["label"]

        input_path = PROCESSED_DIR / dataset_config["input_file"]
        output_dataset_path = REDUCED_DIR / dataset_config["output_file"]
        output_x_path = REDUCED_DIR / dataset_config["output_x_file"]
        output_y_path = REDUCED_DIR / dataset_config["output_y_file"]

        if not input_path.exists():
            raise FileNotFoundError(
                f"\n입력 데이터셋을 찾지 못했습니다.\n"
                f"dataset_label: {label}\n"
                f"예상 경로: {input_path}\n"
            )

        print("\n" + "-" * 80)
        print(f"데이터셋 처리: {label}")
        print("입력:", input_path)

        original_df = pd.read_csv(input_path)
        validate_target_exists(original_df, input_path)

        reduced_df, removed_features, missing_drop_features = build_reduced_dataset(original_df)

        save_reduced_outputs(
            reduced_df=reduced_df,
            output_dataset_path=output_dataset_path,
            output_x_path=output_x_path,
            output_y_path=output_y_path,
        )

        summary = build_dataset_summary(
            label=label,
            original_df=original_df,
            reduced_df=reduced_df,
            removed_features=removed_features,
            missing_drop_features=missing_drop_features,
        )
        summaries.append(summary)

        print("원본 shape:", original_df.shape)
        print("축소 shape:", reduced_df.shape)
        print("제거된 feature:", removed_features)

        if missing_drop_features:
            print("현재 데이터셋에 없어 제거되지 않은 feature:", missing_drop_features)

        print("저장:", output_dataset_path)
        print("저장:", output_x_path)
        print("저장:", output_y_path)

    summary_df = pd.DataFrame(summaries)
    summary_path = OUTPUT_DIR / "reduced_feature_dataset_summary.csv"
    removed_features_path = OUTPUT_DIR / "removed_features.txt"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    with open(removed_features_path, "w", encoding="utf-8") as f:
        f.write("[Removed features]\n")
        for feature in DROP_FEATURES:
            f.write(f"- {feature}\n")

        f.write("\n[Explicitly kept features]\n")
        for feature in KEEP_FEATURES_BY_DECISION:
            f.write(f"- {feature}\n")

    print("\n" + "=" * 80)
    print("전체 저장 완료")
    print("=" * 80)
    print("요약:", summary_path)
    print("제거/유지 feature 기록:", removed_features_path)

    print("\n[요약]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
