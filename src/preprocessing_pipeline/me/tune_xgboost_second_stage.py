# xgboost 두번째 하이퍼파라미터 튜닝

from pathlib import Path
from datetime import datetime
import csv
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 0. 기본 설정
# ============================================================

PROJECT_ROOT = Path("/Users/choi-seung-yeon/PyCharmMiscProject")

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
    / "selected_modeling_dataset_2021_2025.csv"
)

TARGET_COL = "current_ecig_use"
RANDOM_STATE = 42

TEST_SIZE = 0.2
VALID_SIZE_FROM_TRAIN_VALID = 0.25

# 처음은 30~50 추천
N_TRIALS = 50

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"xgboost_second_stage_tuning_{RUN_TIMESTAMP}"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 컬럼 설정
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
# 2. XGBoost 2차 탐색 공간
# ============================================================
# 1차 XGBoost best:
# n_estimators=300, max_depth=3, learning_rate=0.03,
# subsample=0.8, colsample_bytree=0.8,
# min_child_weight=1, gamma=0.5,
# reg_alpha=1, reg_lambda=2

XGBOOST_SECOND_STAGE_PARAM_SPACE = {
    "n_estimators": [250, 300, 400, 500, 700],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.015, 0.02, 0.03, 0.04, 0.05],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 2, 3, 5],
    "gamma": [0, 0.25, 0.5, 1, 2],
    "reg_alpha": [0.5, 1, 2, 3],
    "reg_lambda": [1, 2, 3, 5, 10],
}


# ============================================================
# 3. 함수
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


def get_existing_columns(df: pd.DataFrame):
    numeric_cols = [col for col in NUMERIC_COLS if col in df.columns]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    binary_cols = [col for col in BINARY_COLS if col in df.columns]

    return numeric_cols, categorical_cols, binary_cols


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


def sample_params(param_space: dict, rng: random.Random) -> dict:
    return {key: rng.choice(values) for key, values in param_space.items()}


def build_xgboost_model(params: dict, scale_pos_weight: float):
    XGBClassifier = import_xgboost()

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )


def get_positive_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    return proba[:, 1]


def find_best_threshold_by_f1(y_valid, valid_proba):
    precision, recall, thresholds = precision_recall_curve(y_valid, valid_proba)

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


def evaluate_at_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision_pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_pr_curve(y_test, proba, title, output_path):
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"AP={ap:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_threshold_curve(y_valid, valid_proba, output_path):
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
    ax.set_title("Validation threshold curve")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# 4. 튜닝 실행
# ============================================================

def run_second_stage_tuning(
    preprocessor,
    X_train,
    y_train,
    X_valid,
    y_valid,
    scale_pos_weight: float,
):
    rng = random.Random(RANDOM_STATE)

    rows = []
    pipelines_by_trial = {}

    tried_params = set()
    trial = 0

    print("\n" + "=" * 80)
    print("XGBoost 2차 튜닝 시작")
    print("=" * 80)

    while trial < N_TRIALS:
        params = sample_params(XGBOOST_SECOND_STAGE_PARAM_SPACE, rng)
        params_key = json.dumps(params, sort_keys=True)

        if params_key in tried_params:
            continue

        tried_params.add(params_key)
        trial += 1

        print("\n" + "-" * 80)
        print(f"[XGBoost 2차] trial {trial}/{N_TRIALS}")
        print(params)
        print("-" * 80)

        model = build_xgboost_model(
            params=params,
            scale_pos_weight=scale_pos_weight,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        valid_proba = get_positive_proba(pipeline, X_valid)

        valid_pr_auc = float(average_precision_score(y_valid, valid_proba))
        valid_roc_auc = float(roc_auc_score(y_valid, valid_proba))

        threshold_info = find_best_threshold_by_f1(
            y_valid=y_valid,
            valid_proba=valid_proba,
        )

        threshold = threshold_info["best_threshold"]

        valid_metrics_at_threshold = evaluate_at_threshold(
            y_true=y_valid,
            proba=valid_proba,
            threshold=threshold,
        )

        row = {
            "model": "xgboost",
            "trial": trial,
            "valid_pr_auc": valid_pr_auc,
            "valid_roc_auc": valid_roc_auc,
            "best_threshold_from_valid": threshold,
            "valid_precision_at_threshold": valid_metrics_at_threshold["precision"],
            "valid_recall_at_threshold": valid_metrics_at_threshold["recall"],
            "valid_f1_at_threshold": valid_metrics_at_threshold["f1"],
            "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        }

        rows.append(row)
        pipelines_by_trial[trial] = pipeline

        print(
            f"valid PR-AUC={valid_pr_auc:.4f}, "
            f"valid F1={valid_metrics_at_threshold['f1']:.4f}, "
            f"threshold={threshold:.4f}"
        )

    tuning_df = pd.DataFrame(rows)

    tuning_df_by_pr_auc = tuning_df.sort_values(
        by=["valid_pr_auc", "valid_f1_at_threshold"],
        ascending=False,
    ).reset_index(drop=True)

    tuning_df_by_f1 = tuning_df.sort_values(
        by=["valid_f1_at_threshold", "valid_pr_auc"],
        ascending=False,
    ).reset_index(drop=True)

    return tuning_df, tuning_df_by_pr_auc, tuning_df_by_f1, pipelines_by_trial


def evaluate_selected_model(
    selection_name: str,
    selected_row: dict,
    selected_pipeline: Pipeline,
    X_valid,
    y_valid,
    X_test,
    y_test,
):
    threshold = float(selected_row["best_threshold_from_valid"])

    valid_proba = get_positive_proba(selected_pipeline, X_valid)
    test_proba = get_positive_proba(selected_pipeline, X_test)

    default_metrics = evaluate_at_threshold(
        y_true=y_test,
        proba=test_proba,
        threshold=0.5,
    )

    tuned_metrics = evaluate_at_threshold(
        y_true=y_test,
        proba=test_proba,
        threshold=threshold,
    )

    result = {
        "selection_name": selection_name,
        "model": "xgboost",
        "selected_trial": int(selected_row["trial"]),
        "best_threshold_from_valid": threshold,

        "valid_pr_auc": float(selected_row["valid_pr_auc"]),
        "valid_f1_at_threshold": float(selected_row["valid_f1_at_threshold"]),
        "valid_precision_at_threshold": float(selected_row["valid_precision_at_threshold"]),
        "valid_recall_at_threshold": float(selected_row["valid_recall_at_threshold"]),

        "test_default_precision": default_metrics["precision"],
        "test_default_recall": default_metrics["recall"],
        "test_default_f1": default_metrics["f1"],

        "test_tuned_accuracy": tuned_metrics["accuracy"],
        "test_tuned_precision": tuned_metrics["precision"],
        "test_tuned_recall": tuned_metrics["recall"],
        "test_tuned_f1": tuned_metrics["f1"],
        "test_tuned_roc_auc": tuned_metrics["roc_auc"],
        "test_tuned_pr_auc_average_precision": tuned_metrics["average_precision_pr_auc"],
        "tn": tuned_metrics["tn"],
        "fp": tuned_metrics["fp"],
        "fn": tuned_metrics["fn"],
        "tp": tuned_metrics["tp"],

        "params_json": selected_row["params_json"],
    }

    plot_pr_curve(
        y_test=y_test,
        proba=test_proba,
        title=f"{selection_name} PR Curve",
        output_path=PLOTS_DIR / f"{selection_name}_pr_curve.png",
    )

    plot_threshold_curve(
        y_valid=y_valid,
        valid_proba=valid_proba,
        output_path=PLOTS_DIR / f"{selection_name}_validation_threshold_curve.png",
    )

    write_json(
        REPORTS_DIR / f"{selection_name}_selected_params.json",
        {
            "selection_name": selection_name,
            "selected_trial": int(selected_row["trial"]),
            "best_threshold_from_valid": threshold,
            "params": json.loads(selected_row["params_json"]),
            "validation": {
                "pr_auc": float(selected_row["valid_pr_auc"]),
                "f1_at_threshold": float(selected_row["valid_f1_at_threshold"]),
                "precision_at_threshold": float(selected_row["valid_precision_at_threshold"]),
                "recall_at_threshold": float(selected_row["valid_recall_at_threshold"]),
            },
            "test": {
                "precision": tuned_metrics["precision"],
                "recall": tuned_metrics["recall"],
                "f1": tuned_metrics["f1"],
                "pr_auc": tuned_metrics["average_precision_pr_auc"],
                "roc_auc": tuned_metrics["roc_auc"],
            },
        },
    )

    return result


# ============================================================
# 5. 메인
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("XGBoost 2차 하이퍼파라미터 튜닝")
    print("=" * 80)
    print("입력 데이터:", DATA_PATH)
    print("출력 폴더:", OUTPUT_DIR)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"입력 데이터셋을 찾지 못했습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} 컬럼이 없습니다.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    numeric_cols, categorical_cols, binary_cols = get_existing_columns(X)

    print("\n[컬럼 분류]")
    print("numeric:", numeric_cols)
    print("categorical:", categorical_cols)
    print("binary:", binary_cols)

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

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
    )

    tuning_df, tuning_df_by_pr_auc, tuning_df_by_f1, pipelines_by_trial = run_second_stage_tuning(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        scale_pos_weight=scale_pos_weight,
    )

    tuning_df.to_csv(
        REPORTS_DIR / "xgboost_second_stage_all_trials.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tuning_df_by_pr_auc.to_csv(
        REPORTS_DIR / "xgboost_second_stage_ranked_by_valid_pr_auc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tuning_df_by_f1.to_csv(
        REPORTS_DIR / "xgboost_second_stage_ranked_by_valid_f1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_by_pr_auc_row = tuning_df_by_pr_auc.iloc[0].to_dict()
    best_by_f1_row = tuning_df_by_f1.iloc[0].to_dict()

    final_results = []

    final_results.append(
        evaluate_selected_model(
            selection_name="best_by_valid_pr_auc",
            selected_row=best_by_pr_auc_row,
            selected_pipeline=pipelines_by_trial[int(best_by_pr_auc_row["trial"])],
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
        )
    )

    if int(best_by_f1_row["trial"]) != int(best_by_pr_auc_row["trial"]):
        final_results.append(
            evaluate_selected_model(
                selection_name="best_by_valid_f1",
                selected_row=best_by_f1_row,
                selected_pipeline=pipelines_by_trial[int(best_by_f1_row["trial"])],
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
            )
        )

    final_df = pd.DataFrame(final_results)
    final_df = final_df.sort_values(
        by=["test_tuned_pr_auc_average_precision", "test_tuned_f1"],
        ascending=False,
    ).reset_index(drop=True)

    final_df.to_csv(
        REPORTS_DIR / "xgboost_second_stage_final_test_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("XGBoost 2차 튜닝 완료")
    print("=" * 80)

    print("\n[validation PR-AUC 기준 상위 10개]")
    print(tuning_df_by_pr_auc.head(10).to_string(index=False))

    print("\n[validation F1 기준 상위 10개]")
    print(tuning_df_by_f1.head(10).to_string(index=False))

    print("\n[최종 test 평가]")
    print(final_df.to_string(index=False))

    print("\n저장 위치:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()