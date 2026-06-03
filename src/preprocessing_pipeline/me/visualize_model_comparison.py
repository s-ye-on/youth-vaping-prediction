# 모델 비교 시각화

from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing.project_paths import PROJECT_ROOT


# ============================================================
# 0. 기본 경로 설정
# ============================================================

MODEL_RESULTS_DIR = PROJECT_ROOT / "outputs" / "model_results"
INPUT_METRICS_PATH = MODEL_RESULTS_DIR / "combined_model_metrics.csv"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / f"model_comparison_visualization_{RUN_TIMESTAMP}"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PLOTS_DIR = OUTPUT_ROOT / "plots"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 표시용 이름 매핑
# ============================================================

DATASET_LABEL_MAP = {
    "selected_modeling_dataset": "2023-2025\noriginal\nwith ever",
    "selected_modeling_dataset_without_ever_cigarette_use": "2023-2025\noriginal\nwithout ever",
    "selected_modeling_dataset_2021_2025": "2021-2025\noriginal\nwith ever",
    "selected_modeling_dataset_2021_2025_without_ever_cigarette_use": "2021-2025\noriginal\nwithout ever",
    "selected_modeling_dataset_reduced_low_importance_removed": "2023-2025\nreduced\nwith ever",
    "selected_modeling_dataset_without_ever_cigarette_use_reduced_low_importance_removed": "2023-2025\nreduced\nwithout ever",
    "selected_modeling_dataset_2021_2025_reduced_low_importance_removed": "2021-2025\nreduced\nwith ever",
    "selected_modeling_dataset_2021_2025_without_ever_cigarette_use_reduced_low_importance_removed": "2021-2025\nreduced\nwithout ever",
}

DATASET_ORDER = [
    "selected_modeling_dataset",
    "selected_modeling_dataset_reduced_low_importance_removed",
    "selected_modeling_dataset_2021_2025",
    "selected_modeling_dataset_2021_2025_reduced_low_importance_removed",
    "selected_modeling_dataset_without_ever_cigarette_use",
    "selected_modeling_dataset_without_ever_cigarette_use_reduced_low_importance_removed",
    "selected_modeling_dataset_2021_2025_without_ever_cigarette_use",
    "selected_modeling_dataset_2021_2025_without_ever_cigarette_use_reduced_low_importance_removed",
]

MODEL_LABEL_MAP = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
]


# ============================================================
# 2. 유틸 함수
# ============================================================

def load_metrics() -> pd.DataFrame:
    if not INPUT_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"\ncombined_model_metrics.csv 파일을 찾지 못했습니다.\n"
            f"예상 경로: {INPUT_METRICS_PATH}\n"
            f"먼저 run_model_pipelines.py를 실행해 주세요.\n"
        )

    df = pd.read_csv(INPUT_METRICS_PATH)

    required_columns = [
        "dataset",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision_pr_auc",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"\ncombined_model_metrics.csv에 필요한 컬럼이 없습니다.\n"
            f"누락 컬럼: {missing}\n"
            f"현재 컬럼: {df.columns.tolist()}\n"
        )

    df = df.copy()

    df["dataset_label"] = df["dataset"].map(DATASET_LABEL_MAP).fillna(df["dataset"])
    df["model_label"] = df["model"].map(MODEL_LABEL_MAP).fillna(df["model"])

    df["dataset_order"] = df["dataset"].apply(
        lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 999
    )
    df["model_order"] = df["model"].apply(
        lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999
    )

    df = df.sort_values(
        by=["dataset_order", "model_order"],
        ascending=True,
    ).reset_index(drop=True)

    return df


def save_table(df: pd.DataFrame, filename: str) -> Path:
    output_path = REPORTS_DIR / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def add_value_labels(ax, values, fmt="{:.3f}", rotation=0):
    for index, value in enumerate(values):
        if pd.isna(value):
            continue

        ax.text(
            index,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=rotation,
        )


def plot_metric_by_dataset_best_model(df: pd.DataFrame, metric: str, title: str, filename: str):
    best_df = (
        df.sort_values(metric, ascending=False)
        .groupby("dataset", as_index=False)
        .first()
    )

    best_df["dataset_order"] = best_df["dataset"].apply(
        lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 999
    )
    best_df = best_df.sort_values("dataset_order")

    x_labels = best_df["dataset_label"].tolist()
    values = best_df[metric].tolist()
    model_labels = best_df["model_label"].tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(values)), values)

    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(values) * 1.25 if values else 1)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}\n{model_labels[idx]}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path = PLOTS_DIR / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_metric_by_model_grouped(df: pd.DataFrame, metric: str, title: str, filename: str):
    pivot = df.pivot_table(
        index="dataset_label",
        columns="model_label",
        values=metric,
        aggfunc="first",
    )

    ordered_dataset_labels = [
        DATASET_LABEL_MAP[d]
        for d in DATASET_ORDER
        if DATASET_LABEL_MAP[d] in pivot.index
    ]

    ordered_model_labels = [
        MODEL_LABEL_MAP[m]
        for m in MODEL_ORDER
        if MODEL_LABEL_MAP[m] in pivot.columns
    ]

    pivot = pivot.reindex(index=ordered_dataset_labels, columns=ordered_model_labels)

    ax = pivot.plot(kind="bar", figsize=(13, 7))

    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(title="Model")
    ax.set_ylim(0, pivot.max().max() * 1.25)

    fig = ax.get_figure()
    fig.tight_layout()

    output_path = PLOTS_DIR / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_precision_recall_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        ax.scatter(row["recall"], row["precision"], s=80)
        ax.text(
            row["recall"] + 0.005,
            row["precision"] + 0.005,
            f"{row['model_label']}\n{row['dataset_label'].replace(chr(10), ' ')}",
            fontsize=8,
        )

    ax.set_title("Precision vs Recall by Dataset and Model")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = PLOTS_DIR / "03_precision_recall_scatter.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_metric_heatmap(df: pd.DataFrame, metric: str, title: str, filename: str):
    pivot = df.pivot_table(
        index="dataset_label",
        columns="model_label",
        values=metric,
        aggfunc="first",
    )

    ordered_dataset_labels = [
        DATASET_LABEL_MAP[d]
        for d in DATASET_ORDER
        if DATASET_LABEL_MAP[d] in pivot.index
    ]

    ordered_model_labels = [
        MODEL_LABEL_MAP[m]
        for m in MODEL_ORDER
        if MODEL_LABEL_MAP[m] in pivot.columns
    ]

    pivot = pivot.reindex(index=ordered_dataset_labels, columns=ordered_model_labels)

    fig, ax = plt.subplots(figsize=(11, 6))

    image = ax.imshow(pivot.values, aspect="auto")

    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.isna(value):
                text = ""
            else:
                text = f"{value:.3f}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
            )

    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()

    output_path = PLOTS_DIR / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def build_best_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric in [
        "average_precision_pr_auc",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "accuracy",
    ]:
        best_row = df.sort_values(metric, ascending=False).iloc[0]

        rows.append(
            {
                "selection_metric": metric,
                "best_dataset": best_row["dataset"],
                "best_dataset_label": best_row["dataset_label"].replace("\n", " "),
                "best_model": best_row["model"],
                "best_model_label": best_row["model_label"],
                "best_score": best_row[metric],
                "accuracy": best_row["accuracy"],
                "precision": best_row["precision"],
                "recall": best_row["recall"],
                "f1": best_row["f1"],
                "roc_auc": best_row["roc_auc"],
                "average_precision_pr_auc": best_row["average_precision_pr_auc"],
            }
        )

    return pd.DataFrame(rows)


def build_dataset_best_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, group in df.groupby("dataset"):
        best_pr_auc = group.sort_values("average_precision_pr_auc", ascending=False).iloc[0]
        best_f1 = group.sort_values("f1", ascending=False).iloc[0]

        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABEL_MAP.get(dataset, dataset).replace("\n", " "),
                "best_pr_auc_model": best_pr_auc["model"],
                "best_pr_auc": best_pr_auc["average_precision_pr_auc"],
                "best_pr_auc_f1": best_pr_auc["f1"],
                "best_f1_model": best_f1["model"],
                "best_f1": best_f1["f1"],
                "best_f1_pr_auc": best_f1["average_precision_pr_auc"],
            }
        )

    result = pd.DataFrame(rows)
    result["dataset_order"] = result["dataset"].apply(
        lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 999
    )
    result = result.sort_values("dataset_order").drop(columns=["dataset_order"])

    return result


def main():
    df = load_metrics()

    print("\n입력 파일:")
    print(INPUT_METRICS_PATH)

    print("\n출력 디렉토리:")
    print(OUTPUT_ROOT)

    metrics_copy_path = save_table(df, "combined_model_metrics_with_labels.csv")

    best_model_summary = build_best_model_summary(df)
    best_model_summary_path = save_table(best_model_summary, "best_model_by_metric_summary.csv")

    dataset_best_summary = build_dataset_best_summary(df)
    dataset_best_summary_path = save_table(dataset_best_summary, "best_model_by_dataset_summary.csv")

    plot_paths = []

    plot_paths.append(
        plot_metric_by_dataset_best_model(
            df,
            metric="average_precision_pr_auc",
            title="Best PR-AUC by Dataset",
            filename="01_best_pr_auc_by_dataset.png",
        )
    )

    plot_paths.append(
        plot_metric_by_dataset_best_model(
            df,
            metric="f1",
            title="Best F1 by Dataset",
            filename="02_best_f1_by_dataset.png",
        )
    )

    plot_paths.append(plot_precision_recall_scatter(df))

    plot_paths.append(
        plot_metric_by_dataset_best_model(
            df,
            metric="roc_auc",
            title="Best ROC-AUC by Dataset",
            filename="04_best_roc_auc_by_dataset.png",
        )
    )

    plot_paths.append(
        plot_metric_by_model_grouped(
            df,
            metric="average_precision_pr_auc",
            title="PR-AUC by Dataset and Model",
            filename="05_pr_auc_grouped_bar.png",
        )
    )

    plot_paths.append(
        plot_metric_by_model_grouped(
            df,
            metric="f1",
            title="F1 by Dataset and Model",
            filename="06_f1_grouped_bar.png",
        )
    )

    plot_paths.append(
        plot_metric_heatmap(
            df,
            metric="average_precision_pr_auc",
            title="Model Performance Heatmap: PR-AUC",
            filename="07_pr_auc_heatmap.png",
        )
    )

    plot_paths.append(
        plot_metric_heatmap(
            df,
            metric="f1",
            title="Model Performance Heatmap: F1",
            filename="08_f1_heatmap.png",
        )
    )

    print("\n저장된 report 파일:")
    print(metrics_copy_path)
    print(best_model_summary_path)
    print(dataset_best_summary_path)

    print("\n저장된 plot 파일:")
    for path in plot_paths:
        print(path)

    print("\n[요약: 지표별 최고 모델]")
    print(best_model_summary.to_string(index=False))

    print("\n[요약: 데이터셋별 최고 모델]")
    print(dataset_best_summary.to_string(index=False))


if __name__ == "__main__":
    main()
