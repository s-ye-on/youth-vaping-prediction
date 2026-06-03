from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 0. 기본 설정
# ============================================================

PROJECT_ROOT = Path("/Users/choi-seung-yeon/PyCharmMiscProject")

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"
REDUCED_DATA_DIR = DATA_DIR / "reduced"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / f"final_candidate_threshold_{RUN_TIMESTAMP}"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PLOTS_DIR = OUTPUT_ROOT / "plots"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "current_ecig_use"
RANDOM_STATE = 42

# 전체 100% 기준:
# train_valid 80%, test 20%
# train_valid 내부에서 train 75%, valid 25%
# 최종: train 60%, valid 20%, test 20%
TEST_SIZE = 0.2
VALID_SIZE_FROM_TRAIN_VALID = 0.25


# ============================================================
# 1. 비교할 최종 후보 데이터셋
# ============================================================

DATASET_CONFIGS = [
    {
        "dataset_key": "2021_2025_original_with_ever",
        "dataset_label": "2021-2025 original with ever_cigarette_use",
        "path": DATA_DIR / "selected_modeling_dataset_2021_2025.csv",
    },
    {
        "dataset_key": "2021_2025_reduced_with_ever",
        "dataset_label": "2021-2025 reduced with ever_cigarette_use",
        "path": REDUCED_DATA_DIR / "selected_modeling_dataset_2021_2025_reduced_low_importance_removed.csv",
    },
]


# ============================================================
# 2. 컬럼 분류
# ============================================================

NUMERIC_COLS = [
    "AGE",
    "INT_SPWD_TM",
    "INT_SPWK_TM",
]

CATEGORICAL_COLS = [
    "YEAR",
    "CITY",
    "CTYPE",
    "SEX",
    "GRADE",
    "SCHOOL",
    "STYPE",
    "M_STR",
    "M_SAD",
    "PA_TOT",
    "E_SES",
    "E_RES",
    "subjective_unhealthy_level",
    "breakfast_freq",
    "fruit_freq",
    "fastfood_freq",
    "secondhand_smoke_home",
    "secondhand_smoke_public",
    "academic_performance",
    "alcohol_start_age_cat",
    "alcohol_days_30d_cat",
]

BINARY_COLS = [
    "body_image_missing",
    "body_image_very_thin",
    "body_image_thin",
    "body_image_normal",
    "body_image_fat",
    "body_image_very_fat",
    "family_info_private",
    "ever_alcohol_use",
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
]


# ============================================================
# 3. 라이브러리 import
# ============================================================

def import_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError as e:
        raise ImportError(
            "\nxgboost가 설치되어 있지 않습니다.\n"
            "설치 명령어:\n"
            "/Users/choi-seung-yeon/.virtualenvs/.venv/bin/python -m pip install -U xgboost\n"
        ) from e


def import_lightgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError as e:
        raise ImportError(
            "\nlightgbm이 설치되어 있지 않습니다.\n"
            "설치 명령어:\n"
            "/Users/choi-seung-yeon/.virtualenvs/.venv/bin/python -m pip install -U lightgbm\n"
        ) from e


# ============================================================
# 4. 전처리 / 모델 생성
# ============================================================

def get_existing_columns(df: pd.DataFrame):
    numeric_cols = [col for col in NUMERIC_COLS if col in df.columns]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    binary_cols = [col for col in BINARY_COLS if col in df.columns]

    missing_numeric = [col for col in NUMERIC_COLS if col not in df.columns]
    missing_categorical = [col for col in CATEGORICAL_COLS if col not in df.columns]
    missing_binary = [col for col in BINARY_COLS if col not in df.columns]

    return numeric_cols, categorical_cols, binary_cols, missing_numeric, missing_categorical, missing_binary


def build_preprocessor(numeric_cols, categorical_cols, binary_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", StandardScaler(), numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    if binary_cols:
        transformers.append(
            ("binary", "passthrough", binary_cols)
        )

    return ColumnTransformer(transformers=transformers)


def calculate_scale_pos_weight(y: pd.Series) -> float:
    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())

    if positive_count == 0:
        raise ValueError("양성 클래스가 없습니다.")

    return negative_count / positive_count


def build_models(scale_pos_weight: float):
    XGBClassifier = import_xgboost()
    LGBMClassifier = import_lightgbm()

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    return models


# ============================================================
# 5. threshold 선택 / 평가
# ============================================================

def find_best_threshold_by_f1(y_valid, valid_proba):
    precision, recall, thresholds = precision_recall_curve(y_valid, valid_proba)

    # precision, recall은 thresholds보다 길이가 1 더 깁니다.
    precision_for_thresholds = precision[:-1]
    recall_for_thresholds = recall[:-1]

    f1_scores = (
        2 * precision_for_thresholds * recall_for_thresholds
        / (precision_for_thresholds + recall_for_thresholds + 1e-12)
    )

    best_index = int(np.nanargmax(f1_scores))

    return {
        "best_threshold": float(thresholds[best_index]),
        "valid_best_f1": float(f1_scores[best_index]),
        "valid_precision_at_best_threshold": float(precision_for_thresholds[best_index]),
        "valid_recall_at_best_threshold": float(recall_for_thresholds[best_index]),
    }


def evaluate_with_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision_pr_auc": average_precision_score(y_true, proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": classification_report(y_true, pred, zero_division=0),
    }


def get_positive_class_proba(model_pipeline, X):
    proba = model_pipeline.predict_proba(X)
    return proba[:, 1]


# ============================================================
# 6. 시각화
# ============================================================

def plot_confusion_matrix(cm_values, title, output_path):
    matrix = np.array([
        [cm_values["tn"], cm_values["fp"]],
        [cm_values["fn"], cm_values["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=13,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curve_for_model(y_true, proba, title, output_path):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"AP={ap:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curve_for_model(y_true, proba, title, output_path):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_threshold_curve(y_valid, valid_proba, best_threshold, title, output_path):
    precision, recall, thresholds = precision_recall_curve(y_valid, valid_proba)

    precision_for_thresholds = precision[:-1]
    recall_for_thresholds = recall[:-1]
    f1_scores = (
        2 * precision_for_thresholds * recall_for_thresholds
        / (precision_for_thresholds + recall_for_thresholds + 1e-12)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, precision_for_thresholds, label="Precision")
    ax.plot(thresholds, recall_for_thresholds, label="Recall")
    ax.plot(thresholds, f1_scores, label="F1")
    ax.axvline(best_threshold, linestyle="--", label=f"Best threshold={best_threshold:.4f}")

    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(model_pipeline, feature_names, model_name, title, output_path, top_n=25):
    model = model_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_title(title)
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_final_comparison(metrics_df, output_path):
    display_df = metrics_df.copy()
    display_df["candidate"] = display_df["dataset_key"] + "\n" + display_df["model"]

    fig, ax = plt.subplots(figsize=(13, 7))

    x = np.arange(len(display_df))
    width = 0.2

    ax.bar(x - width, display_df["average_precision_pr_auc"], width, label="PR-AUC")
    ax.bar(x, display_df["f1"], width, label="F1")
    ax.bar(x + width, display_df["precision"], width, label="Precision")

    ax.set_title("Final Candidate Comparison after Validation Threshold Tuning")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(display_df["candidate"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# 7. feature 이름 복원
# ============================================================

def get_feature_names(preprocessor, numeric_cols, categorical_cols, binary_cols):
    feature_names = []

    if numeric_cols:
        feature_names.extend(numeric_cols)

    if categorical_cols:
        encoder = preprocessor.named_transformers_["categorical"]
        encoded_names = encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(encoded_names)

    if binary_cols:
        feature_names.extend(binary_cols)

    return feature_names


# ============================================================
# 8. 단일 데이터셋 실행
# ============================================================

def run_for_dataset(config):
    dataset_key = config["dataset_key"]
    dataset_label = config["dataset_label"]
    dataset_path = config["path"]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"\n데이터셋 파일을 찾지 못했습니다.\n"
            f"dataset_key: {dataset_key}\n"
            f"path: {dataset_path}\n"
        )

    print("\n" + "=" * 80)
    print(f"데이터셋 실행: {dataset_key}")
    print(dataset_path)
    print("=" * 80)

    df = pd.read_csv(dataset_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} 컬럼이 없습니다: {dataset_path}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    numeric_cols, categorical_cols, binary_cols, missing_numeric, missing_categorical, missing_binary = get_existing_columns(X)

    print("\n[컬럼 분류]")
    print("numeric:", numeric_cols)
    print("categorical:", categorical_cols)
    print("binary:", binary_cols)

    if missing_categorical:
        print("없는 categorical:", missing_categorical)
    if missing_binary:
        print("없는 binary:", missing_binary)

    print("\n[target 분포]")
    print(y.value_counts().sort_index())
    print("\n[target 비율]")
    print(y.value_counts(normalize=True).sort_index().round(6))

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=VALID_SIZE_FROM_TRAIN_VALID,
        stratify=y_train_valid,
        random_state=RANDOM_STATE,
    )

    print("\n[split shape]")
    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("X_test :", X_test.shape)

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print("\nscale_pos_weight:", round(scale_pos_weight, 4))

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, binary_cols)
    models = build_models(scale_pos_weight)

    rows = []

    for model_name, model in models.items():
        print("\n" + "-" * 80)
        print(f"모델 학습: {dataset_key} / {model_name}")
        print("-" * 80)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        valid_proba = get_positive_class_proba(pipeline, X_valid)
        threshold_info = find_best_threshold_by_f1(y_valid, valid_proba)
        best_threshold = threshold_info["best_threshold"]

        test_proba = get_positive_class_proba(pipeline, X_test)

        default_metrics = evaluate_with_threshold(
            y_true=y_test,
            proba=test_proba,
            threshold=0.5,
        )

        tuned_metrics = evaluate_with_threshold(
            y_true=y_test,
            proba=test_proba,
            threshold=best_threshold,
        )

        row = {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "model": model_name,
            "scale_pos_weight": scale_pos_weight,
            "best_threshold_from_valid": best_threshold,
            "valid_best_f1": threshold_info["valid_best_f1"],
            "valid_precision_at_best_threshold": threshold_info["valid_precision_at_best_threshold"],
            "valid_recall_at_best_threshold": threshold_info["valid_recall_at_best_threshold"],

            "default_accuracy": default_metrics["accuracy"],
            "default_precision": default_metrics["precision"],
            "default_recall": default_metrics["recall"],
            "default_f1": default_metrics["f1"],

            "accuracy": tuned_metrics["accuracy"],
            "precision": tuned_metrics["precision"],
            "recall": tuned_metrics["recall"],
            "f1": tuned_metrics["f1"],
            "roc_auc": tuned_metrics["roc_auc"],
            "average_precision_pr_auc": tuned_metrics["average_precision_pr_auc"],
            "tn": tuned_metrics["tn"],
            "fp": tuned_metrics["fp"],
            "fn": tuned_metrics["fn"],
            "tp": tuned_metrics["tp"],
        }

        rows.append(row)

        safe_name = f"{dataset_key}_{model_name}"

        plot_confusion_matrix(
            cm_values=tuned_metrics,
            title=f"Confusion Matrix\n{dataset_key} / {model_name}",
            output_path=PLOTS_DIR / f"confusion_matrix_{safe_name}.png",
        )

        plot_precision_recall_curve_for_model(
            y_true=y_test,
            proba=test_proba,
            title=f"Precision-Recall Curve\n{dataset_key} / {model_name}",
            output_path=PLOTS_DIR / f"pr_curve_{safe_name}.png",
        )

        plot_roc_curve_for_model(
            y_true=y_test,
            proba=test_proba,
            title=f"ROC Curve\n{dataset_key} / {model_name}",
            output_path=PLOTS_DIR / f"roc_curve_{safe_name}.png",
        )

        plot_threshold_curve(
            y_valid=y_valid,
            valid_proba=valid_proba,
            best_threshold=best_threshold,
            title=f"Threshold Tuning on Validation Set\n{dataset_key} / {model_name}",
            output_path=PLOTS_DIR / f"threshold_curve_{safe_name}.png",
        )

        transformed_preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = get_feature_names(
            preprocessor=transformed_preprocessor,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            binary_cols=binary_cols,
        )

        plot_feature_importance(
            model_pipeline=pipeline,
            feature_names=feature_names,
            model_name=model_name,
            title=f"Feature Importance\n{dataset_key} / {model_name}",
            output_path=PLOTS_DIR / f"feature_importance_{safe_name}.png",
            top_n=25,
        )

        report_path = REPORTS_DIR / f"classification_report_{safe_name}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("[Default threshold = 0.5]\n")
            f.write(default_metrics["classification_report"])
            f.write("\n\n")
            f.write(f"[Tuned threshold from validation = {best_threshold:.6f}]\n")
            f.write(tuned_metrics["classification_report"])

        print("\n[Validation threshold]")
        print(f"best_threshold: {best_threshold:.6f}")
        print(f"valid_best_f1: {threshold_info['valid_best_f1']:.4f}")

        print("\n[Test metrics after threshold tuning]")
        print(f"precision: {tuned_metrics['precision']:.4f}")
        print(f"recall   : {tuned_metrics['recall']:.4f}")
        print(f"f1       : {tuned_metrics['f1']:.4f}")
        print(f"PR-AUC   : {tuned_metrics['average_precision_pr_auc']:.4f}")
        print(f"ROC-AUC  : {tuned_metrics['roc_auc']:.4f}")
        print("CM:")
        print(np.array([
            [tuned_metrics["tn"], tuned_metrics["fp"]],
            [tuned_metrics["fn"], tuned_metrics["tp"]],
        ]))

    return pd.DataFrame(rows)


# ============================================================
# 9. 메인
# ============================================================

def main():
    all_results = []

    print("\n출력 디렉토리:")
    print(OUTPUT_ROOT)

    for config in DATASET_CONFIGS:
        result_df = run_for_dataset(config)
        all_results.append(result_df)

    metrics_df = pd.concat(all_results, ignore_index=True)

    metrics_df = metrics_df.sort_values(
        by=["average_precision_pr_auc", "f1"],
        ascending=False,
    ).reset_index(drop=True)

    metrics_path = REPORTS_DIR / "final_candidate_threshold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    plot_final_comparison(
        metrics_df=metrics_df,
        output_path=PLOTS_DIR / "final_candidate_comparison.png",
    )

    best_by_pr_auc = metrics_df.sort_values("average_precision_pr_auc", ascending=False).iloc[0]
    best_by_f1 = metrics_df.sort_values("f1", ascending=False).iloc[0]

    summary_rows = [
        {
            "selection": "best_by_pr_auc",
            "dataset_key": best_by_pr_auc["dataset_key"],
            "model": best_by_pr_auc["model"],
            "threshold": best_by_pr_auc["best_threshold_from_valid"],
            "precision": best_by_pr_auc["precision"],
            "recall": best_by_pr_auc["recall"],
            "f1": best_by_pr_auc["f1"],
            "average_precision_pr_auc": best_by_pr_auc["average_precision_pr_auc"],
            "roc_auc": best_by_pr_auc["roc_auc"],
        },
        {
            "selection": "best_by_f1",
            "dataset_key": best_by_f1["dataset_key"],
            "model": best_by_f1["model"],
            "threshold": best_by_f1["best_threshold_from_valid"],
            "precision": best_by_f1["precision"],
            "recall": best_by_f1["recall"],
            "f1": best_by_f1["f1"],
            "average_precision_pr_auc": best_by_f1["average_precision_pr_auc"],
            "roc_auc": best_by_f1["roc_auc"],
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = REPORTS_DIR / "final_candidate_selection_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("최종 후보 비교 완료")
    print("=" * 80)

    print("\n전체 metrics 저장:")
    print(metrics_path)

    print("\n최종 후보 요약 저장:")
    print(summary_path)

    print("\n[상위 결과]")
    print(metrics_df.to_string(index=False))

    print("\n[선정 요약]")
    print(summary_df.to_string(index=False))

    print("\nplot 저장 위치:")
    print(PLOTS_DIR)


if __name__ == "__main__":
    main()