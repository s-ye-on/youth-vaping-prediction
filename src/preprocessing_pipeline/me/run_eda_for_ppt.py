from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder


# ============================================================
# 0. 기본 설정
# ============================================================

PROJECT_ROOT = Path("/Users/choi-seung-yeon/PyCharmMiscProject")

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "selected_modeling_dataset_2021_2025.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_for_ppt"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "current_ecig_use"

# PPT에 근거로 쓰기 좋은 주요 변수
KEY_FEATURES = [
    "YEAR",
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
    "live_with_mother",
    "live_with_no_family",
    "father_edu_middle_or_less",
    "father_edu_high_school",
    "father_edu_college_or_more",
    "father_edu_unknown",
    "mother_edu_middle_or_less",
    "mother_edu_high_school",
    "mother_edu_college_or_more",
    "mother_edu_unknown",
    "father_korean_birth",
    "father_foreign_birth",
    "mother_korean_birth",
    "mother_foreign_birth",
]


# ============================================================
# 1. 유틸 함수
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터셋을 찾지 못했습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} 컬럼이 없습니다.")

    return df


def save_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total_count = len(df)

    target_counts = (
        df[TARGET_COL]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis(TARGET_COL)
        .reset_index(name="count")
    )

    target_counts["ratio"] = target_counts["count"] / total_count
    target_counts["ratio_percent"] = target_counts["ratio"] * 100

    output_path = TABLE_DIR / "target_distribution_total.csv"
    target_counts.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[1] 전체 target 분포")
    print("=" * 80)
    print(target_counts.to_string(index=False))
    print("저장:", output_path)

    return target_counts


def save_target_distribution_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if "YEAR" not in df.columns:
        print("\nYEAR 컬럼이 없어 연도별 target 분포는 건너뜁니다.")
        return pd.DataFrame()

    rows = []

    for year, group in df.groupby("YEAR"):
        total_count = len(group)
        positive_count = int((group[TARGET_COL] == 1).sum())
        negative_count = int((group[TARGET_COL] == 0).sum())

        rows.append({
            "YEAR": year,
            "total_count": total_count,
            "target_0_count": negative_count,
            "target_1_count": positive_count,
            "target_1_ratio": positive_count / total_count if total_count else 0,
            "target_1_ratio_percent": positive_count / total_count * 100 if total_count else 0,
        })

    result = pd.DataFrame(rows)

    output_path = TABLE_DIR / "target_distribution_by_year.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[1-추가] 연도별 target 분포")
    print("=" * 80)
    print(result.to_string(index=False))
    print("저장:", output_path)

    return result


def plot_target_distribution(target_counts: pd.DataFrame) -> None:
    plot_df = target_counts.copy()
    plot_df[TARGET_COL] = plot_df[TARGET_COL].astype(str)

    plt.figure(figsize=(7, 5))
    plt.bar(plot_df[TARGET_COL], plot_df["count"])
    plt.title("Target Distribution: current_ecig_use")
    plt.xlabel("current_ecig_use")
    plt.ylabel("Count")
    plt.tight_layout()

    output_path = FIGURE_DIR / "target_distribution_total.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("그래프 저장:", output_path)


def calculate_target_rate_by_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    if feature not in df.columns:
        return pd.DataFrame()

    result = (
        df.groupby(feature, dropna=False)[TARGET_COL]
        .agg(
            total_count="count",
            target_1_count="sum",
            target_1_ratio="mean",
        )
        .reset_index()
    )

    result["target_1_ratio_percent"] = result["target_1_ratio"] * 100
    result["feature"] = feature

    result = result[
        [
            "feature",
            feature,
            "total_count",
            "target_1_count",
            "target_1_ratio",
            "target_1_ratio_percent",
        ]
    ]

    result = result.sort_values("target_1_ratio", ascending=False)

    return result


def save_key_feature_target_rates(df: pd.DataFrame) -> pd.DataFrame:
    all_results = []

    existing_key_features = [feature for feature in KEY_FEATURES if feature in df.columns]
    missing_key_features = [feature for feature in KEY_FEATURES if feature not in df.columns]

    if missing_key_features:
        print("\n[주의] 데이터셋에 없는 KEY_FEATURES:")
        for feature in missing_key_features:
            print("-", feature)

    for feature in existing_key_features:
        result = calculate_target_rate_by_feature(df, feature)

        if result.empty:
            continue

        # 개별 저장
        safe_feature_name = feature.replace("/", "_").replace(" ", "_")
        output_path = TABLE_DIR / f"target_rate_by_{safe_feature_name}.csv"
        result.to_csv(output_path, index=False, encoding="utf-8-sig")

        all_results.append(result)

    if not all_results:
        raise ValueError("주요 변수별 target 비율을 계산하지 못했습니다.")

    combined = pd.concat(all_results, ignore_index=True)

    output_path = TABLE_DIR / "target_rate_by_key_features_long.csv"
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[2] 주요 변수별 전자담배 사용률")
    print("=" * 80)
    print("저장:", output_path)

    # 콘솔에는 발표에 쓰기 좋은 변수 몇 개만 미리보기
    preview_features = [
        "ever_cigarette_use",
        "ever_alcohol_use",
        "alcohol_days_30d_cat",
        "M_STR",
        "M_SAD",
        "family_info_private",
        "secondhand_smoke_home",
    ]

    for feature in preview_features:
        if feature not in df.columns:
            continue

        print("\n" + "-" * 80)
        print(f"[{feature}]")
        print("-" * 80)
        preview = calculate_target_rate_by_feature(df, feature)
        print(preview.to_string(index=False))

    return combined


def plot_key_feature_target_rates(df: pd.DataFrame) -> None:
    plot_features = [
        "ever_cigarette_use",
        "ever_alcohol_use",
        "alcohol_days_30d_cat",
        "M_STR",
        "M_SAD",
        "family_info_private",
        "secondhand_smoke_home",
        "academic_performance",
        "E_SES",
    ]

    for feature in plot_features:
        if feature not in df.columns:
            continue

        result = calculate_target_rate_by_feature(df, feature)
        if result.empty:
            continue

        value_col = feature

        # 핵심 수정:
        # Matplotlib 카테고리 축에서 str과 float/NaN이 섞이면 오류가 날 수 있으므로
        # x축 라벨을 모두 문자열로 통일합니다.
        plot_df = result.copy()
        plot_df["x_label"] = plot_df[value_col].astype("string").fillna("missing")

        # 정렬도 원래 값 기준으로 하다가 타입 혼합 오류가 날 수 있으므로
        # 문자열 라벨 기준으로 안정적으로 정렬합니다.
        plot_df = plot_df.sort_values("x_label")

        plt.figure(figsize=(9, 5))
        plt.bar(plot_df["x_label"], plot_df["target_1_ratio_percent"])
        plt.title(f"Current E-cigarette Use Rate by {feature}")
        plt.xlabel(feature)
        plt.ylabel("Target 1 Ratio (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        safe_feature_name = feature.replace("/", "_").replace(" ", "_")
        output_path = FIGURE_DIR / f"target_rate_by_{safe_feature_name}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()

        print("그래프 저장:", output_path)


def save_mutual_information(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [col for col in df.columns if col != TARGET_COL]

    X = df[feature_columns].copy()
    y = df[TARGET_COL].copy()

    # 혹시 target에 결측이 있으면 제거
    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    # mutual_info_classif는 숫자형 입력이 필요하므로,
    # object/category 컬럼은 OrdinalEncoder로 변환합니다.
    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in object_cols]

    X_encoded = X.copy()

    if object_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )

        X_encoded[object_cols] = encoder.fit_transform(X_encoded[object_cols])

    # 숫자형 결측 처리
    for col in numeric_cols:
        if X_encoded[col].isna().any():
            X_encoded[col] = X_encoded[col].fillna(-1)

    # 혹시 object 인코딩 뒤에도 결측이 있으면 처리
    X_encoded = X_encoded.fillna(-1)

    # 모든 변수를 discrete로 둡니다.
    # 현재 selected dataset은 대부분 설문 코드/더미/범주형이기 때문입니다.
    mi_scores = mutual_info_classif(
        X_encoded,
        y,
        discrete_features=True,
        random_state=42,
    )

    result = pd.DataFrame({
        "feature": feature_columns,
        "mutual_information": mi_scores,
    })

    result = result.sort_values("mutual_information", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    result = result[["rank", "feature", "mutual_information"]]

    output_path = TABLE_DIR / "mutual_information_ranking.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[3] Mutual Information 상위 변수")
    print("=" * 80)
    print(result.head(30).to_string(index=False))
    print("저장:", output_path)

    return result


def plot_mutual_information_top30(mi_result: pd.DataFrame) -> None:
    top_n = 30
    plot_df = mi_result.head(top_n).sort_values("mutual_information", ascending=True)

    plt.figure(figsize=(10, 9))
    plt.barh(plot_df["feature"], plot_df["mutual_information"])
    plt.title("Top 30 Features by Mutual Information")
    plt.xlabel("Mutual Information")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_path = FIGURE_DIR / "mutual_information_top30.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("그래프 저장:", output_path)


def save_feature_summary_for_ppt(
    df: pd.DataFrame,
    target_distribution: pd.DataFrame,
    target_rate_long: pd.DataFrame,
    mi_result: pd.DataFrame,
) -> None:
    feature_columns = [col for col in df.columns if col != TARGET_COL]

    summary_rows = []

    for feature in feature_columns:
        feature_target_rate = target_rate_long[target_rate_long["feature"] == feature]

        if feature_target_rate.empty:
            max_rate = None
            min_rate = None
            max_rate_value = None
            min_rate_value = None
        else:
            value_col = feature
            max_row = feature_target_rate.sort_values("target_1_ratio", ascending=False).iloc[0]
            min_row = feature_target_rate.sort_values("target_1_ratio", ascending=True).iloc[0]

            max_rate = max_row["target_1_ratio_percent"]
            min_rate = min_row["target_1_ratio_percent"]
            max_rate_value = max_row[value_col]
            min_rate_value = min_row[value_col]

        mi_row = mi_result[mi_result["feature"] == feature]

        if mi_row.empty:
            mi_rank = None
            mi_score = None
        else:
            mi_rank = int(mi_row.iloc[0]["rank"])
            mi_score = float(mi_row.iloc[0]["mutual_information"])

        summary_rows.append({
            "feature": feature,
            "n_unique": df[feature].nunique(dropna=False),
            "missing_count": int(df[feature].isna().sum()),
            "missing_ratio_percent": float(df[feature].isna().mean() * 100),
            "highest_target_rate_value": max_rate_value,
            "highest_target_rate_percent": max_rate,
            "lowest_target_rate_value": min_rate_value,
            "lowest_target_rate_percent": min_rate,
            "mutual_information_rank": mi_rank,
            "mutual_information": mi_score,
        })

    summary = pd.DataFrame(summary_rows)

    output_path = TABLE_DIR / "feature_eda_summary_for_ppt.csv"
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[추가] PPT용 feature EDA 요약")
    print("=" * 80)
    print("저장:", output_path)


# ============================================================
# 2. 실행
# ============================================================

def main() -> None:
    print("=" * 80)
    print("PPT용 EDA 생성 시작")
    print("=" * 80)

    df = load_dataset()

    print("데이터셋:", DATA_PATH)
    print("shape:", df.shape)
    print("target:", TARGET_COL)

    feature_columns = [col for col in df.columns if col != TARGET_COL]
    print("feature 수:", len(feature_columns))

    target_distribution = save_target_distribution(df)
    save_target_distribution_by_year(df)
    plot_target_distribution(target_distribution)

    target_rate_long = save_key_feature_target_rates(df)
    plot_key_feature_target_rates(df)

    mi_result = save_mutual_information(df)
    plot_mutual_information_top30(mi_result)

    save_feature_summary_for_ppt(
        df=df,
        target_distribution=target_distribution,
        target_rate_long=target_rate_long,
        mi_result=mi_result,
    )

    print("\n" + "=" * 80)
    print("완료")
    print("=" * 80)
    print("결과 폴더:", OUTPUT_DIR)
    print("\n주요 확인 파일:")
    print("1.", TABLE_DIR / "target_distribution_total.csv")
    print("2.", TABLE_DIR / "target_distribution_by_year.csv")
    print("3.", TABLE_DIR / "target_rate_by_key_features_long.csv")
    print("4.", TABLE_DIR / "mutual_information_ranking.csv")
    print("5.", TABLE_DIR / "feature_eda_summary_for_ppt.csv")
    print("6.", FIGURE_DIR / "target_distribution_total.png")
    print("7.", FIGURE_DIR / "mutual_information_top30.png")


if __name__ == "__main__":
    main()