# 하이퍼파라미터 튜닝된 lightGBM으로 shap

from pathlib import Path
from datetime import datetime
import json
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

# 1차 튜닝 LightGBM best threshold
BEST_THRESHOLD = 0.9117153552749151

# 1차 튜닝 LightGBM best params
BEST_LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "num_leaves": 15,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "min_child_samples": 40,
    "reg_alpha": 1,
    "reg_lambda": 10,
}

# SHAP 계산 샘플 수
# 너무 크게 잡으면 오래 걸릴 수 있습니다.
SHAP_SAMPLE_SIZE = 5000

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"final_lightgbm_shap_{RUN_TIMESTAMP}"
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
# 2. import 함수
# ============================================================

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


def import_shap():
    try:
        import shap
        return shap
    except ImportError as e:
        raise ImportError(
            "\nshap이 설치되어 있지 않습니다.\n"
            "설치 명령어:\n"
            "/Users/choi-seung-yeon/.virtualenvs/.venv/bin/python -m pip install -U shap\n"
        ) from e


# ============================================================
# 3. 전처리 / 평가 함수
# ============================================================

def get_existing_columns(df: pd.DataFrame):
    numeric_cols = [col for col in NUMERIC_COLS if col in df.columns]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    binary_cols = [col for col in BINARY_COLS if col in df.columns]

    return numeric_cols, categorical_cols, binary_cols


def make_one_hot_encoder():
    """
    scikit-learn 버전에 따라 OneHotEncoder 인자가 다릅니다.
    최신 버전은 sparse_output=False,
    구버전은 sparse=False를 사용합니다.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_cols, categorical_cols, binary_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", StandardScaler(), numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", make_one_hot_encoder(), categorical_cols)
        )

    if binary_cols:
        transformers.append(
            ("binary", "passthrough", binary_cols)
        )

    return ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=True,
    )


def calculate_scale_pos_weight(y: pd.Series) -> float:
    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())

    if positive_count == 0:
        raise ValueError("양성 클래스가 없습니다.")

    return negative_count / positive_count


def build_final_lightgbm_model(scale_pos_weight: float):
    LGBMClassifier = import_lightgbm()

    return LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **BEST_LIGHTGBM_PARAMS,
    )


def get_positive_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    return proba[:, 1]


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


def save_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_to_dense(matrix):
    """
    SHAP plot 저장을 위해 pandas DataFrame으로 만들 예정입니다.
    sparse matrix인 경우 dense array로 변환합니다.
    현재 SHAP_SAMPLE_SIZE=5000이므로 메모리 부담은 크지 않습니다.
    """
    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return np.asarray(matrix)


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    ColumnTransformer + OneHotEncoder 이후의 실제 feature name을 가져옵니다.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception as e:
        raise RuntimeError(
            "전처리 후 feature 이름을 가져오지 못했습니다. "
            "scikit-learn 버전을 확인해주세요."
        ) from e


# ============================================================
# 4. plot 함수
# ============================================================

def plot_confusion_matrix_simple(metrics: dict, output_path: Path):
    matrix = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_shap_bar_plot(shap_module, shap_values, X_shap_df, output_path: Path):
    plt.figure()
    shap_module.summary_plot(
        shap_values,
        X_shap_df,
        plot_type="bar",
        show=False,
        max_display=25,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_shap_beeswarm_plot(shap_module, shap_values, X_shap_df, output_path: Path):
    plt.figure()
    shap_module.summary_plot(
        shap_values,
        X_shap_df,
        show=False,
        max_display=25,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_lgbm_builtin_importance(model, feature_names: list[str], output_csv_path: Path, output_plot_path: Path):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "lgbm_importance_split": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        by="lgbm_importance_split",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    top_df = importance_df.head(25).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top_df["feature"], top_df["lgbm_importance_split"])
    ax.set_title("LightGBM Built-in Feature Importance")
    ax.set_xlabel("split importance")
    ax.set_ylabel("feature")

    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. SHAP 계산 함수
# ============================================================

def normalize_shap_values(raw_shap_values):
    """
    SHAP 버전 / LightGBM 반환 형태 차이를 방어합니다.

    가능한 형태:
    1. ndarray: (n_samples, n_features)
    2. list: [class0_values, class1_values]
    3. ndarray: (n_samples, n_features, n_outputs)

    우리는 positive class, 즉 current_ecig_use=1에 대한 값을 사용합니다.
    """
    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) == 2:
            return raw_shap_values[1]
        return raw_shap_values[0]

    raw_shap_values = np.asarray(raw_shap_values)

    if raw_shap_values.ndim == 3:
        # 보통 (n_samples, n_features, n_outputs) 형태면 마지막 class=1 사용
        return raw_shap_values[:, :, -1]

    if raw_shap_values.ndim == 2:
        return raw_shap_values

    raise ValueError(f"지원하지 않는 SHAP 값 shape입니다: {raw_shap_values.shape}")


def calculate_and_save_shap(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    shap_module = import_shap()

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test))

    X_shap_raw = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    y_shap = y_test.loc[X_shap_raw.index]

    X_shap_transformed = preprocessor.transform(X_shap_raw)
    X_shap_array = safe_to_dense(X_shap_transformed)

    feature_names = get_transformed_feature_names(preprocessor)

    if X_shap_array.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP 입력 feature 수와 feature name 수가 다릅니다. "
            f"X_shap_array.shape[1]={X_shap_array.shape[1]}, "
            f"len(feature_names)={len(feature_names)}"
        )

    X_shap_df = pd.DataFrame(
        X_shap_array,
        columns=feature_names,
        index=X_shap_raw.index,
    )

    print("\n[SHAP 계산]")
    print("SHAP sample size:", sample_size)
    print("Transformed SHAP shape:", X_shap_df.shape)

    explainer = shap_module.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_shap_df)
    shap_values = normalize_shap_values(raw_shap_values)

    if shap_values.shape != X_shap_df.shape:
        raise ValueError(
            f"SHAP 값 shape과 입력 shape이 다릅니다. "
            f"shap_values.shape={shap_values.shape}, "
            f"X_shap_df.shape={X_shap_df.shape}"
        )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(
        by="mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    shap_importance_df.to_csv(
        REPORTS_DIR / "shap_feature_importance_mean_abs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 샘플별 예측값도 저장
    shap_sample_proba = model.predict_proba(X_shap_df)[:, 1]
    shap_sample_pred = (shap_sample_proba >= BEST_THRESHOLD).astype(int)

    shap_sample_prediction_df = pd.DataFrame({
        "index": X_shap_raw.index,
        "y_true": y_shap.to_numpy(),
        "predicted_probability": shap_sample_proba,
        "predicted_label_at_best_threshold": shap_sample_pred,
    })

    shap_sample_prediction_df.to_csv(
        REPORTS_DIR / "shap_sample_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # plot 저장
    save_shap_bar_plot(
        shap_module=shap_module,
        shap_values=shap_values,
        X_shap_df=X_shap_df,
        output_path=PLOTS_DIR / "shap_summary_bar_top25.png",
    )

    save_shap_beeswarm_plot(
        shap_module=shap_module,
        shap_values=shap_values,
        X_shap_df=X_shap_df,
        output_path=PLOTS_DIR / "shap_beeswarm_top25.png",
    )

    save_lgbm_builtin_importance(
        model=model,
        feature_names=feature_names,
        output_csv_path=REPORTS_DIR / "lightgbm_builtin_feature_importance.csv",
        output_plot_path=PLOTS_DIR / "lightgbm_builtin_feature_importance_top25.png",
    )

    # 상위 30개만 별도 json 저장
    top30 = shap_importance_df.head(30).to_dict(orient="records")
    save_json(
        REPORTS_DIR / "shap_top30_features.json",
        {
            "note": "mean_abs_shap 기준 상위 30개 feature입니다. SHAP는 인과관계가 아니라 모델 예측 기여도를 의미합니다.",
            "sample_size": sample_size,
            "top30_features": top30,
        },
    )

    print("\n[SHAP 중요도 상위 30개]")
    print(shap_importance_df.head(30).to_string(index=False))

    return shap_importance_df


# ============================================================
# 6. 메인
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("최종 LightGBM + SHAP 해석")
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

    model = build_final_lightgbm_model(
        scale_pos_weight=scale_pos_weight,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("\n[최종 LightGBM 하이퍼파라미터]")
    print(json.dumps(BEST_LIGHTGBM_PARAMS, ensure_ascii=False, indent=2))
    print("BEST_THRESHOLD:", BEST_THRESHOLD)

    print("\n[모델 학습 시작]")
    pipeline.fit(X_train, y_train)
    print("[모델 학습 완료]")

    valid_proba = get_positive_proba(pipeline, X_valid)
    test_proba = get_positive_proba(pipeline, X_test)

    valid_metrics = evaluate_at_threshold(
        y_true=y_valid,
        proba=valid_proba,
        threshold=BEST_THRESHOLD,
    )

    test_metrics = evaluate_at_threshold(
        y_true=y_test,
        proba=test_proba,
        threshold=BEST_THRESHOLD,
    )

    default_test_metrics = evaluate_at_threshold(
        y_true=y_test,
        proba=test_proba,
        threshold=0.5,
    )

    final_report = {
        "dataset": str(DATA_PATH),
        "model": "LightGBM",
        "best_params_from_first_stage_tuning": BEST_LIGHTGBM_PARAMS,
        "best_threshold_from_validation": BEST_THRESHOLD,
        "random_state": RANDOM_STATE,
        "split": {
            "test_size": TEST_SIZE,
            "valid_size_from_train_valid": VALID_SIZE_FROM_TRAIN_VALID,
            "train_shape": list(X_train.shape),
            "valid_shape": list(X_valid.shape),
            "test_shape": list(X_test.shape),
        },
        "scale_pos_weight": scale_pos_weight,
        "valid_metrics_at_best_threshold": valid_metrics,
        "test_metrics_at_best_threshold": test_metrics,
        "test_metrics_at_default_0_5_threshold": default_test_metrics,
    }

    save_json(
        REPORTS_DIR / "final_lightgbm_model_report.json",
        final_report,
    )

    metrics_df = pd.DataFrame([
        {
            "dataset": "selected_modeling_dataset_2021_2025",
            "model": "lightgbm_first_stage_tuned",
            "threshold": BEST_THRESHOLD,
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "roc_auc": test_metrics["roc_auc"],
            "average_precision_pr_auc": test_metrics["average_precision_pr_auc"],
            "tn": test_metrics["tn"],
            "fp": test_metrics["fp"],
            "fn": test_metrics["fn"],
            "tp": test_metrics["tp"],
            "params_json": json.dumps(BEST_LIGHTGBM_PARAMS, ensure_ascii=False, sort_keys=True),
        }
    ])

    metrics_df.to_csv(
        REPORTS_DIR / "final_lightgbm_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_confusion_matrix_simple(
        metrics=test_metrics,
        output_path=PLOTS_DIR / "final_lightgbm_confusion_matrix.png",
    )

    calculate_and_save_shap(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n" + "=" * 80)
    print("최종 LightGBM + SHAP 완료")
    print("=" * 80)
    print("\n[test metrics at best threshold]")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    print("\n저장 위치:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()