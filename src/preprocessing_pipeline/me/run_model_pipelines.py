# 1. Logistic Regression
# 2. Random Forest
# 3. XGBoost

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path("/Users/choi-seung-yeon/PyCharmMiscProject")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_results"

TARGET_COL = "current_ecig_use"

DATASET_FILES = [
    DATA_DIR / "selected_modeling_dataset.csv",
    DATA_DIR / "selected_modeling_dataset_without_ever_cigarette_use.csv",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


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
# 2. 유틸
# ============================================================

def make_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_dataset_name(dataset_path: Path) -> str:
    return dataset_path.stem


def require_target_column(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        raise ValueError(f"target 컬럼이 없습니다: {TARGET_COL}")


def assert_no_leakage_columns(X: pd.DataFrame) -> None:
    leakage_cols = []

    for col in X.columns:
        if col == TARGET_COL:
            leakage_cols.append(col)

        if col.startswith("TC_EC"):
            leakage_cols.append(col)

        if col.startswith("TC_HTP"):
            leakage_cols.append(col)

    if leakage_cols:
        raise ValueError(
            "\n누수 위험 컬럼이 X에 포함되어 있습니다.\n"
            f"문제 컬럼: {leakage_cols}\n"
        )


def existing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def missing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col not in df.columns]


def print_column_check(df: pd.DataFrame) -> None:
    print("\n[컬럼 확인]")

    for group_name, cols in [
        ("NUMERIC_COLS", NUMERIC_COLS),
        ("CATEGORICAL_COLS", CATEGORICAL_COLS),
        ("BINARY_COLS", BINARY_COLS),
    ]:
        missing = missing_columns(df, cols)
        if missing:
            print(f"- {group_name} 중 현재 데이터에 없는 컬럼: {missing}")
        else:
            print(f"- {group_name}: 모두 존재")


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric_cols = existing_columns(df, NUMERIC_COLS)
    categorical_cols = existing_columns(df, CATEGORICAL_COLS)
    binary_cols = existing_columns(df, BINARY_COLS)

    used_cols = set(numeric_cols + categorical_cols + binary_cols)
    all_feature_cols = set(df.columns) - {TARGET_COL}

    unused_cols = sorted(all_feature_cols - used_cols)

    if unused_cols:
        print("\n[주의] 컬럼 목록에 분류되지 않은 feature가 있습니다.")
        print("이번 실행에서는 자동으로 categorical로 처리합니다.")
        for col in unused_cols:
            print("-", col)

        categorical_cols = categorical_cols + unused_cols

    return numeric_cols, categorical_cols, binary_cols


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        return 1.0

    return negative_count / positive_count


# ============================================================
# 3. 전처리 파이프라인
# ============================================================

def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_logistic_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
            ("binary", binary_pipeline, binary_cols),
        ],
        remainder="drop",
    )


def make_tree_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
            ("binary", binary_pipeline, binary_cols),
        ],
        remainder="drop",
    )


# ============================================================
# 4. 모델 파이프라인
# ============================================================

def make_logistic_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
) -> Pipeline:
    preprocessor = make_logistic_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def make_random_forest_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
) -> Pipeline:
    preprocessor = make_tree_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def make_xgboost_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
    scale_pos_weight: float,
) -> Pipeline | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("\n[xgboost 없음]")
        print("XGBoost를 실행하려면 아래 명령으로 설치하세요.")
        print("pip install xgboost")
        return None

    preprocessor = make_tree_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
    )

    classifier = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def make_model_pipelines(
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
    scale_pos_weight: float,
) -> dict[str, Pipeline]:
    pipelines: dict[str, Pipeline] = {}

    pipelines["logistic_regression"] = make_logistic_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
    )

    pipelines["random_forest"] = make_random_forest_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
    )

    xgb_pipeline = make_xgboost_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        scale_pos_weight=scale_pos_weight,
    )

    if xgb_pipeline is not None:
        pipelines["xgboost"] = xgb_pipeline

    return pipelines


# ============================================================
# 5. 평가
# ============================================================

def predict_positive_probability(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return 1 / (1 + np.exp(-scores))

    raise ValueError("이 모델은 predict_proba 또는 decision_function을 지원하지 않습니다.")


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    print(f"\n========== {model_name} 학습 시작 ==========")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = predict_positive_probability(model, X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision_pr_auc": average_precision_score(y_test, y_proba),
    }

    print("\n[주요 지표]")
    for key, value in metrics.items():
        if key == "model":
            continue
        print(f"{key}: {value:.4f}")

    print("\n[Confusion Matrix]")
    print(confusion_matrix(y_test, y_pred))

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def save_metrics(dataset_name: str, metrics_list: list[dict[str, Any]]) -> pd.DataFrame:
    metrics_df = pd.DataFrame(metrics_list)
    output_file = OUTPUT_DIR / f"{dataset_name}_model_metrics.csv"
    metrics_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n[모델 평가 결과 저장]")
    print(output_file)

    return metrics_df


# ============================================================
# 6. 데이터셋 하나 실행
# ============================================================

def run_for_dataset(dataset_path: Path) -> pd.DataFrame | None:
    if not dataset_path.exists():
        print(f"\n[건너뜀] 파일 없음: {dataset_path}")
        return None

    dataset_name = safe_dataset_name(dataset_path)

    print("\n\n============================================================")
    print(f"데이터셋 실행: {dataset_name}")
    print("파일:", dataset_path)
    print("============================================================")

    df = pd.read_csv(dataset_path, low_memory=False)

    require_target_column(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    assert_no_leakage_columns(X)
    print_column_check(df)

    numeric_cols, categorical_cols, binary_cols = get_feature_columns(df)

    print("\n[최종 컬럼 분류]")
    print("numeric:", numeric_cols)
    print("categorical:", categorical_cols)
    print("binary:", binary_cols)

    print("\n[target 분포]")
    print(y.value_counts(dropna=False))

    print("\n[target 비율]")
    print(y.value_counts(normalize=True, dropna=False).round(4))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n[train/test shape]")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print("\n[불균형 보정]")
    print("scale_pos_weight:", round(scale_pos_weight, 4))

    pipelines = make_model_pipelines(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        scale_pos_weight=scale_pos_weight,
    )

    metrics_list = []

    for model_name, model in pipelines.items():
        metrics = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        metrics_list.append(metrics)

    metrics_df = save_metrics(dataset_name, metrics_list)

    print("\n[모델 비교]")
    print(metrics_df.sort_values("average_precision_pr_auc", ascending=False))

    return metrics_df


# ============================================================
# 7. main
# ============================================================

def main() -> None:
    make_output_dir()

    all_metrics = []

    for dataset_file in DATASET_FILES:
        metrics_df = run_for_dataset(dataset_file)

        if metrics_df is None:
            continue

        metrics_df.insert(0, "dataset", safe_dataset_name(dataset_file))
        all_metrics.append(metrics_df)

    if not all_metrics:
        raise FileNotFoundError(
            "\n실행 가능한 데이터셋이 없습니다.\n"
            "data/processed 폴더에 selected_modeling_dataset.csv가 있는지 확인하세요.\n"
        )

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_output_file = OUTPUT_DIR / "combined_model_metrics.csv"
    combined_metrics.to_csv(combined_output_file, index=False, encoding="utf-8-sig")

    print("\n\n============================================================")
    print("전체 모델 비교 결과")
    print("============================================================")
    print(combined_metrics.sort_values("average_precision_pr_auc", ascending=False))

    print("\n전체 결과 저장:")
    print(combined_output_file)


if __name__ == "__main__":
    main()