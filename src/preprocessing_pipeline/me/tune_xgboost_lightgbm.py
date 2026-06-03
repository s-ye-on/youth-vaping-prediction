# 하이퍼 파라미터 튜닝

from pathlib import Path
from datetime import datetime
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

# 너무 크게 잡으면 오래 걸립니다.
# 처음에는 15~20 정도 추천.
# 시간이 괜찮으면 30~50으로 늘리면 됩니다.
N_TRIALS_PER_MODEL = 20

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"hyperparameter_tuning_{RUN_TIMESTAMP}"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 컬럼 분류
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
# 2. 모델 import
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
# 3. 전처리 / 평가 함수
# ============================================================

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
    }


def sample_params(param_space: dict, rng: random.Random) -> dict:
    sampled = {}

    for key, values in param_space.items():
        sampled[key] = rng.choice(values)

    return sampled


def save_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# 4. 하이퍼파라미터 후보
# ============================================================

XGBOOST_PARAM_SPACE = {
    "n_estimators": [300, 400, 500, 700],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.07, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.5, 1, 2],
    "reg_alpha": [0, 0.1, 0.5, 1],
    "reg_lambda": [1, 2, 5, 10],
}

LIGHTGBM_PARAM_SPACE = {
    "n_estimators": [300, 400, 500, 700],
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.07, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_samples": [20, 40, 60, 100],
    "reg_alpha": [0, 0.1, 0.5, 1],
    "reg_lambda": [0, 1, 2, 5, 10],
}


# ============================================================
# 5. 모델 생성
# ============================================================

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


def build_lightgbm_model(params: dict, scale_pos_weight: float):
    LGBMClassifier = import_lightgbm()

    return LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


# ============================================================
# 6. 단일 모델 튜닝
# ============================================================

def tune_model(
    model_name: str,
    param_space: dict,
    build_model_func,
    preprocessor,
    X_train,
    y_train,
    X_valid,
    y_valid,
    scale_pos_weight: float,
    n_trials: int,
):
    rng = random.Random(RANDOM_STATE)

    rows = []
    best_row = None
    best_pipeline = None

    tried_params = set()

    print("\n" + "=" * 80)
    print(f"{model_name} 하이퍼파라미터 튜닝 시작")
    print("=" * 80)

    trial = 0

    while trial < n_trials:
        params = sample_params(param_space, rng)
        params_key = json.dumps(params, sort_keys=True)

        if params_key in tried_params:
            continue

        tried_params.add(params_key)
        trial += 1

        print("\n" + "-" * 80)
        print(f"[{model_name}] trial {trial}/{n_trials}")
        print(params)
        print("-" * 80)

        model = build_model_func(params, scale_pos_weight)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        valid_proba = get_positive_proba(pipeline, X_valid)

        valid_pr_auc = average_precision_score(y_valid, valid_proba)
        valid_roc_auc = roc_auc_score(y_valid, valid_proba)

        threshold_info = find_best_threshold_by_f1(y_valid, valid_proba)
        threshold = threshold_info["best_threshold"]

        valid_metrics_at_threshold = evaluate_at_threshold(
            y_true=y_valid,
            proba=valid_proba,
            threshold=threshold,
        )

        row = {
            "model": model_name,
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

        print(
            f"valid PR-AUC={valid_pr_auc:.4f}, "
            f"valid F1={valid_metrics_at_threshold['f1']:.4f}, "
            f"threshold={threshold:.4f}"
        )

        # 1순위 PR-AUC, 2순위 F1 기준
        if best_row is None:
            best_row = row
            best_pipeline = pipeline
        else:
            current_score = (row["valid_pr_auc"], row["valid_f1_at_threshold"])
            best_score = (best_row["valid_pr_auc"], best_row["valid_f1_at_threshold"])

            if current_score > best_score:
                best_row = row
                best_pipeline = pipeline

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        by=["valid_pr_auc", "valid_f1_at_threshold"],
        ascending=False,
    ).reset_index(drop=True)

    return result_df, best_row, best_pipeline


# ============================================================
# 7. 최종 test 평가
# ============================================================

def evaluate_best_model_on_test(
    model_name: str,
    best_row: dict,
    best_pipeline: Pipeline,
    X_test,
    y_test,
):
    threshold = best_row["best_threshold_from_valid"]
    test_proba = get_positive_proba(best_pipeline, X_test)

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
        "model": model_name,
        "best_trial": best_row["trial"],
        "selection_metric": "valid_pr_auc_first_valid_f1_second",
        "best_threshold_from_valid": threshold,

        "valid_pr_auc": best_row["valid_pr_auc"],
        "valid_f1_at_threshold": best_row["valid_f1_at_threshold"],

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
        "params_json": best_row["params_json"],
    }

    return result, test_proba


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


# ============================================================
# 8. 메인
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("XGBoost / LightGBM 하이퍼파라미터 튜닝 시작")
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

    tuning_jobs = [
        {
            "model_name": "xgboost",
            "param_space": XGBOOST_PARAM_SPACE,
            "build_model_func": build_xgboost_model,
        },
        {
            "model_name": "lightgbm",
            "param_space": LIGHTGBM_PARAM_SPACE,
            "build_model_func": build_lightgbm_model,
        },
    ]

    final_results = []

    for job in tuning_jobs:
        model_name = job["model_name"]

        tuning_df, best_row, best_pipeline = tune_model(
            model_name=model_name,
            param_space=job["param_space"],
            build_model_func=job["build_model_func"],
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            scale_pos_weight=scale_pos_weight,
            n_trials=N_TRIALS_PER_MODEL,
        )

        tuning_path = REPORTS_DIR / f"{model_name}_tuning_results.csv"
        tuning_df.to_csv(tuning_path, index=False, encoding="utf-8-sig")

        best_params_path = REPORTS_DIR / f"{model_name}_best_params.json"
        save_json(
            best_params_path,
            {
                "model": model_name,
                "best_trial": int(best_row["trial"]),
                "valid_pr_auc": float(best_row["valid_pr_auc"]),
                "valid_f1_at_threshold": float(best_row["valid_f1_at_threshold"]),
                "best_threshold_from_valid": float(best_row["best_threshold_from_valid"]),
                "params": json.loads(best_row["params_json"]),
            },
        )

        test_result, test_proba = evaluate_best_model_on_test(
            model_name=model_name,
            best_row=best_row,
            best_pipeline=best_pipeline,
            X_test=X_test,
            y_test=y_test,
        )

        final_results.append(test_result)

        plot_pr_curve(
            y_test=y_test,
            proba=test_proba,
            title=f"{model_name} tuned PR Curve",
            output_path=PLOTS_DIR / f"{model_name}_tuned_pr_curve.png",
        )

        print("\n" + "=" * 80)
        print(f"{model_name} best result on test")
        print("=" * 80)
        print(json.dumps(test_result, ensure_ascii=False, indent=2))

    final_df = pd.DataFrame(final_results)
    final_df = final_df.sort_values(
        by=["test_tuned_pr_auc_average_precision", "test_tuned_f1"],
        ascending=False,
    ).reset_index(drop=True)

    final_path = REPORTS_DIR / "tuned_final_test_results.csv"
    final_df.to_csv(final_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("튜닝 완료")
    print("=" * 80)
    print("최종 결과:")
    print(final_df.to_string(index=False))
    print("\n저장 위치:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()