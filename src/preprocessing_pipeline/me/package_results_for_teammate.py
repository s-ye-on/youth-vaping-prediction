from pathlib import Path
from datetime import datetime
import shutil
import zipfile
import csv


# ============================================================
# 0. 기본 경로 설정
# ============================================================

PROJECT_ROOT = Path("/Users/choi-seung-yeon/PyCharmMiscProject")

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_RESULTS_DIR = OUTPUTS_DIR / "model_results"
FEATURE_REDUCTION_DIR = OUTPUTS_DIR / "feature_reduction"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

SHARE_ROOT = OUTPUTS_DIR / f"share_with_teammate_{RUN_TIMESTAMP}"
TABLES_DIR = SHARE_ROOT / "01_tables"
THRESHOLD_DIR = SHARE_ROOT / "02_threshold_experiment"
PLOTS_DIR = SHARE_ROOT / "03_plots"
FEATURE_REDUCTION_SHARE_DIR = SHARE_ROOT / "04_feature_reduction"
NOTES_DIR = SHARE_ROOT / "05_notes"

ZIP_PATH = OUTPUTS_DIR / f"share_with_teammate_{RUN_TIMESTAMP}.zip"


# ============================================================
# 1. 최신 결과 폴더 설정
# ============================================================

FINAL_THRESHOLD_RESULT_DIR = OUTPUTS_DIR / "final_candidate_threshold_20260530_161020"

VISUALIZATION_GLOB_PATTERNS = [
    "model_comparison_visualization_*",
]


# ============================================================
# 2. 승연 최종 결과 요약값
# ============================================================

SEUNGYEON_FINAL_ROWS = [
    {
        "rank": 1,
        "dataset_key": "2021_2025_reduced_with_ever",
        "dataset_label": "2021-2025 reduced with ever_cigarette_use",
        "model": "xgboost",
        "threshold_from_validation": 0.924001,
        "validation_best_f1": 0.543390,
        "test_precision": 0.479048,
        "test_recall": 0.631513,
        "test_f1": 0.544815,
        "test_pr_auc_average_precision": 0.508111,
        "test_roc_auc": 0.964010,
        "tn": 50994,
        "fp": 1094,
        "fn": 587,
        "tp": 1006,
        "note": "PR-AUC 기준 승연 실험 최고",
    },
    {
        "rank": 2,
        "dataset_key": "2021_2025_original_with_ever",
        "dataset_label": "2021-2025 original with ever_cigarette_use",
        "model": "xgboost",
        "threshold_from_validation": 0.921907,
        "validation_best_f1": 0.542409,
        "test_precision": 0.479050,
        "test_recall": 0.645951,
        "test_f1": 0.550120,
        "test_pr_auc_average_precision": 0.507993,
        "test_roc_auc": 0.964501,
        "tn": 50969,
        "fp": 1119,
        "fn": 564,
        "tp": 1029,
        "note": "F1 기준 승연 실험 최고, 최종 메인 후보",
    },
    {
        "rank": 3,
        "dataset_key": "2021_2025_reduced_with_ever",
        "dataset_label": "2021-2025 reduced with ever_cigarette_use",
        "model": "lightgbm",
        "threshold_from_validation": 0.895586,
        "validation_best_f1": 0.526959,
        "test_precision": 0.459925,
        "test_recall": 0.619586,
        "test_f1": 0.527949,
        "test_pr_auc_average_precision": 0.501932,
        "test_roc_auc": 0.960879,
        "tn": 50929,
        "fp": 1159,
        "fn": 606,
        "tp": 987,
        "note": "승연 실험 LightGBM reduced 결과",
    },
    {
        "rank": 4,
        "dataset_key": "2021_2025_original_with_ever",
        "dataset_label": "2021-2025 original with ever_cigarette_use",
        "model": "random_forest",
        "threshold_from_validation": 0.349737,
        "validation_best_f1": 0.535450,
        "test_precision": 0.473805,
        "test_recall": 0.647207,
        "test_f1": 0.547095,
        "test_pr_auc_average_precision": 0.500592,
        "test_roc_auc": 0.961070,
        "tn": 50943,
        "fp": 1145,
        "fn": 562,
        "tp": 1031,
        "note": "XGBoost와 F1이 매우 근접한 안정적 대안",
    },
    {
        "rank": 5,
        "dataset_key": "2021_2025_original_with_ever",
        "dataset_label": "2021-2025 original with ever_cigarette_use",
        "model": "lightgbm",
        "threshold_from_validation": 0.895434,
        "validation_best_f1": 0.525492,
        "test_precision": 0.463078,
        "test_recall": 0.610169,
        "test_f1": 0.526544,
        "test_pr_auc_average_precision": 0.500055,
        "test_roc_auc": 0.961065,
        "tn": 50961,
        "fp": 1127,
        "fn": 621,
        "tp": 972,
        "note": "승연 실험 LightGBM original 결과",
    },
    {
        "rank": 6,
        "dataset_key": "2021_2025_reduced_with_ever",
        "dataset_label": "2021-2025 reduced with ever_cigarette_use",
        "model": "random_forest",
        "threshold_from_validation": 0.347812,
        "validation_best_f1": 0.535742,
        "test_precision": 0.456118,
        "test_recall": 0.662272,
        "test_f1": 0.540195,
        "test_pr_auc_average_precision": 0.498650,
        "test_roc_auc": 0.961756,
        "tn": 50830,
        "fp": 1258,
        "fn": 538,
        "tp": 1055,
        "note": "승연 실험 Random Forest reduced 결과",
    },
]


TEAMMATE_COMPARISON_ROWS = [
    {
        "comparison_item": "main_model",
        "teammate_result": "2021-2025 with ever_cigarette_use + LightGBM",
        "seungyeon_result": "2021-2025 original with ever_cigarette_use + XGBoost",
        "comment": "팀원은 LightGBM, 승연 실험은 XGBoost가 최종 후보",
    },
    {
        "comparison_item": "f1",
        "teammate_result": "LightGBM F1 = 0.5470",
        "seungyeon_result": "XGBoost F1 = 0.5501",
        "comment": "F1 기준으로는 승연 XGBoost가 근소하게 높음",
    },
    {
        "comparison_item": "pr_auc",
        "teammate_result": "LightGBM PR-AUC = 0.5402",
        "seungyeon_result": "XGBoost PR-AUC = 0.5080 / reduced XGBoost PR-AUC = 0.5081",
        "comment": "팀원 PR-AUC가 상당히 높으므로 실험 조건 확인 필요",
    },
    {
        "comparison_item": "precision",
        "teammate_result": "LightGBM Precision = 0.5123",
        "seungyeon_result": "XGBoost Precision = 0.4791",
        "comment": "Precision은 팀원 LightGBM이 더 높음",
    },
    {
        "comparison_item": "recall",
        "teammate_result": "LightGBM Recall = 0.5868",
        "seungyeon_result": "XGBoost Recall = 0.6460",
        "comment": "Recall은 승연 XGBoost가 더 높음",
    },
    {
        "comparison_item": "threshold_method",
        "teammate_result": "확인 필요",
        "seungyeon_result": "validation set에서 F1 최대 threshold 선택 후 test set 평가",
        "comment": "threshold를 test set에서 고르면 결과가 과대평가될 수 있음",
    },
    {
        "comparison_item": "preprocessing",
        "teammate_result": "확인 필요",
        "seungyeon_result": "ColumnTransformer + StandardScaler + OneHotEncoder + binary passthrough",
        "comment": "categorical 처리 방식에 따라 LightGBM 결과가 달라질 수 있음",
    },
    {
        "comparison_item": "imbalance_handling",
        "teammate_result": "확인 필요",
        "seungyeon_result": "scale_pos_weight = negative / positive, SMOTE 미사용",
        "comment": "SMOTE/SMOTENC 또는 class weight 차이 가능",
    },
]


QUESTIONS_FOR_TEAMMATE = [
    "LightGBM 최종 결과에 사용한 정확한 입력 데이터 파일명과 경로가 무엇인가요?",
    "해당 데이터셋의 rows, target positive count, feature count가 각각 몇 개인가요?",
    "train/validation/test split 비율과 random_state가 어떻게 되나요?",
    "threshold는 validation set에서 선택했나요, test set에서 선택했나요?",
    "LightGBM 하이퍼파라미터 전체 설정값은 무엇인가요?",
    "LightGBM에서 scale_pos_weight, is_unbalance, class_weight 중 무엇을 사용했나요?",
    "SMOTE/SMOTENC를 적용했다면 train set에만 적용했나요?",
    "categorical 변수는 OneHotEncoder, LightGBM categorical_feature, 숫자 코드 그대로 중 어떤 방식으로 처리했나요?",
    "PR-AUC 0.5402는 어떤 test set에서 계산된 값인가요?",
    "같은 split에서 XGBoost와 LightGBM을 동시에 비교했나요?",
]


# ============================================================
# 3. 유틸 함수
# ============================================================

def mkdirs() -> None:
    for directory in [
        SHARE_ROOT,
        TABLES_DIR,
        THRESHOLD_DIR,
        PLOTS_DIR,
        FEATURE_REDUCTION_SHARE_DIR,
        NOTES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def copy_file_if_exists(src: Path, dst_dir: Path, new_name: str | None = None) -> bool:
    if not src.exists():
        print(f"[SKIP] 파일 없음: {src}")
        return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (new_name if new_name else src.name)

    shutil.copy2(src, dst)
    print(f"[COPY] {src} -> {dst}")
    return True


def copy_dir_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"[SKIP] 폴더 없음: {src}")
        return False

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    print(f"[COPY DIR] {src} -> {dst}")
    return True


def find_latest_dir(parent: Path, patterns: list[str]) -> Path | None:
    candidates = []

    for pattern in patterns:
        candidates.extend(parent.glob(pattern))

    candidates = [path for path in candidates if path.is_dir()]

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("CSV로 저장할 rows가 비어 있습니다.")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[WRITE CSV] {path}")


def write_questions_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["no", "question"])
        writer.writeheader()

        for index, question in enumerate(QUESTIONS_FOR_TEAMMATE, start=1):
            writer.writerow(
                {
                    "no": index,
                    "question": question,
                }
            )

    print(f"[WRITE CSV] {path}")


def make_zip(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(src_dir.parent)
                zip_file.write(file_path, arcname)

    print(f"[ZIP] {zip_path}")


# ============================================================
# 4. 복사 함수
# ============================================================

def copy_model_result_tables() -> None:
    print("\n" + "=" * 80)
    print("1. model_results 표 파일 복사")
    print("=" * 80)

    files_to_copy = [
        "combined_model_metrics.csv",

        "selected_modeling_dataset_model_metrics.csv",
        "selected_modeling_dataset_without_ever_cigarette_use_model_metrics.csv",
        "selected_modeling_dataset_2021_2025_model_metrics.csv",
        "selected_modeling_dataset_2021_2025_without_ever_cigarette_use_model_metrics.csv",

        "selected_modeling_dataset_reduced_low_importance_removed_model_metrics.csv",
        "selected_modeling_dataset_without_ever_cigarette_use_reduced_low_importance_removed_model_metrics.csv",
        "selected_modeling_dataset_2021_2025_reduced_low_importance_removed_model_metrics.csv",
        "selected_modeling_dataset_2021_2025_without_ever_cigarette_use_reduced_low_importance_removed_model_metrics.csv",
    ]

    for filename in files_to_copy:
        copy_file_if_exists(
            src=MODEL_RESULTS_DIR / filename,
            dst_dir=TABLES_DIR / "model_results",
        )


def copy_feature_reduction_files() -> None:
    print("\n" + "=" * 80)
    print("2. feature reduction 요약 파일 복사")
    print("=" * 80)

    files_to_copy = [
        "reduced_feature_dataset_summary.csv",
        "removed_features.txt",
    ]

    for filename in files_to_copy:
        copy_file_if_exists(
            src=FEATURE_REDUCTION_DIR / filename,
            dst_dir=FEATURE_REDUCTION_SHARE_DIR,
        )


def copy_threshold_experiment_files() -> None:
    print("\n" + "=" * 80)
    print("3. threshold 실험 결과 폴더 복사")
    print("=" * 80)

    copy_dir_if_exists(
        src=FINAL_THRESHOLD_RESULT_DIR,
        dst=THRESHOLD_DIR / FINAL_THRESHOLD_RESULT_DIR.name,
    )


def copy_visualization_files() -> None:
    print("\n" + "=" * 80)
    print("4. 시각화 결과 폴더 복사")
    print("=" * 80)

    latest_visualization_dir = find_latest_dir(
        parent=OUTPUTS_DIR,
        patterns=VISUALIZATION_GLOB_PATTERNS,
    )

    if latest_visualization_dir is None:
        print("[SKIP] model_comparison_visualization_* 폴더를 찾지 못했습니다.")
        return

    copy_dir_if_exists(
        src=latest_visualization_dir,
        dst=PLOTS_DIR / latest_visualization_dir.name,
    )


# ============================================================
# 5. 공유용 정리 문서 생성
# ============================================================

def build_readme_text() -> str:
    lines = [
        "# 팀원 공유용 결과 정리",
        "",
        f"생성 시각: {RUN_TIMESTAMP}",
        "",
        "## 1. 폴더 구성",
        "",
        "```text",
        f"{SHARE_ROOT.name}/",
        "  01_tables/",
        "    model_results/",
        "    seungyeon_final_candidate_threshold_summary.csv",
        "    teammate_vs_seungyeon_comparison_checklist.csv",
        "    questions_for_teammate.csv",
        "",
        "  02_threshold_experiment/",
        "    final_candidate_threshold_.../",
        "      reports/",
        "      plots/",
        "",
        "  03_plots/",
        "    model_comparison_visualization_.../",
        "",
        "  04_feature_reduction/",
        "    reduced_feature_dataset_summary.csv",
        "    removed_features.txt",
        "",
        "  05_notes/",
        "    README_for_teammate.md",
        "    message_to_teammate.txt",
        "```",
        "",
        "## 2. 승연 실험 방식",
        "",
        "최종 후보 실험은 아래 방식으로 진행했습니다.",
        "",
        "| 구분 | 비율 | 용도 |",
        "|---|---:|---|",
        "| train | 60% | 모델 학습 |",
        "| validation | 20% | F1이 최대가 되는 threshold 선택 |",
        "| test | 20% | 최종 성능 평가 |",
        "",
        "threshold는 validation set에서 F1이 최대가 되는 값으로 선택했고, test set은 최종 평가에만 사용했습니다.",
        "",
        "## 3. 승연 최종 후보 결과 요약",
        "",
        "| 순위 | dataset | model | threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        "| 1 | 2021~2025 reduced with ever | XGBoost | 0.924001 | 0.4790 | 0.6315 | 0.5448 | **0.5081** | 0.9640 |",
        "| 2 | 2021~2025 original with ever | XGBoost | 0.921907 | 0.4791 | 0.6460 | **0.5501** | 0.5080 | **0.9645** |",
        "| 3 | 2021~2025 reduced with ever | LightGBM | 0.895586 | 0.4599 | 0.6196 | 0.5279 | 0.5019 | 0.9609 |",
        "| 4 | 2021~2025 original with ever | Random Forest | 0.349737 | 0.4738 | 0.6472 | 0.5471 | 0.5006 | 0.9611 |",
        "| 5 | 2021~2025 original with ever | LightGBM | 0.895434 | 0.4631 | 0.6102 | 0.5265 | 0.5001 | 0.9611 |",
        "| 6 | 2021~2025 reduced with ever | Random Forest | 0.347812 | 0.4561 | 0.6623 | 0.5402 | 0.4987 | 0.9618 |",
        "",
        "## 4. 승연 기준 해석",
        "",
        "- F1 기준 최고: `2021_2025_original_with_ever + XGBoost`",
        "- PR-AUC 기준 최고: `2021_2025_reduced_with_ever + XGBoost`",
        "- 두 XGBoost 결과의 PR-AUC 차이는 매우 작음",
        "- original XGBoost가 F1, Recall, ROC-AUC에서 더 좋아 최종 메인 후보로 더 자연스러움",
        "- Random Forest도 F1 0.5471로 매우 근접한 안정적 대안",
        "- LightGBM은 승연 코드 기준에서는 XGBoost/Random Forest보다 낮게 나옴",
        "",
        "## 5. 팀원 결과와 비교할 때 확인할 점",
        "",
        "팀원 결과에서는 LightGBM이 PR-AUC 0.5402로 가장 높게 나왔고,",
        "승연 결과에서는 XGBoost가 F1 기준으로 가장 높게 나왔습니다.",
        "",
        "따라서 아래 항목이 같은지 확인해야 합니다.",
        "",
        "1. 정확한 입력 데이터셋 파일",
        "2. rows / positive target count / feature count",
        "3. train/validation/test split 비율과 random_state",
        "4. threshold 선택 기준",
        "5. LightGBM 하이퍼파라미터",
        "6. categorical 변수 처리 방식",
        "7. scale_pos_weight / is_unbalance / class_weight / SMOTE 사용 여부",
        "8. PR-AUC 계산 대상 test set",
        "9. 같은 split에서 XGBoost와 LightGBM을 비교했는지 여부",
        "",
        "## 6. 결론",
        "",
        "승연 개인 재현 실험에서는 validation set 기준 threshold 조정 후",
        "`2021~2025 original with ever_cigarette_use + XGBoost`가 F1 기준 최종 후보로 가장 적절하게 나타났습니다.",
        "",
        "다만 팀원 결과에서는 LightGBM의 PR-AUC가 더 높게 보고되었으므로,",
        "최종 결론 전에 같은 데이터, 같은 split, 같은 전처리, 같은 threshold 선택 방식으로",
        "XGBoost와 LightGBM을 다시 비교하는 것이 가장 안전합니다.",
        "",
    ]

    return "\n".join(lines)


def build_message_text() -> str:
    lines = [
        "팀원에게 공유할 메시지 초안",
        "",
        "안녕하세요. 제가 별도로 같은 문제를 재현 실험해본 결과를 정리했습니다.",
        "",
        "제 실험은 2021~2025 원본/reduced 데이터셋을 대상으로 Random Forest, XGBoost, LightGBM을 비교했고,",
        "train 60%, validation 20%, test 20%로 나눈 뒤 validation set에서 F1이 최대가 되는 threshold를 선택하고 test set에서 최종 평가했습니다.",
        "",
        "제 결과에서는 F1 기준으로 2021~2025 original with ever_cigarette_use + XGBoost가 가장 좋았습니다.",
        "",
        "- threshold: 0.921907",
        "- Precision: 0.4791",
        "- Recall: 0.6460",
        "- F1: 0.5501",
        "- PR-AUC: 0.5080",
        "- ROC-AUC: 0.9645",
        "",
        "반면 팀원님 결과에서는 LightGBM이 PR-AUC 0.5402로 더 높게 나왔기 때문에,",
        "데이터셋 파일, split 방식, threshold 선택 방식, LightGBM 하이퍼파라미터,",
        "categorical 처리 방식이 같은지 확인해보고 싶습니다.",
        "",
        "정리 폴더에 제 결과표, threshold 실험 결과, 시각화 결과, 확인 질문을 같이 넣어두었습니다.",
        "",
    ]

    return "\n".join(lines)


def write_summary_files() -> None:
    print("\n" + "=" * 80)
    print("5. 공유용 정리 문서 생성")
    print("=" * 80)

    write_csv(
        path=TABLES_DIR / "seungyeon_final_candidate_threshold_summary.csv",
        rows=SEUNGYEON_FINAL_ROWS,
    )

    write_csv(
        path=TABLES_DIR / "teammate_vs_seungyeon_comparison_checklist.csv",
        rows=TEAMMATE_COMPARISON_ROWS,
    )

    write_questions_csv(
        path=TABLES_DIR / "questions_for_teammate.csv",
    )

    readme_path = NOTES_DIR / "README_for_teammate.md"
    readme_path.write_text(build_readme_text(), encoding="utf-8")
    print(f"[WRITE MD] {readme_path}")

    message_path = NOTES_DIR / "message_to_teammate.txt"
    message_path.write_text(build_message_text(), encoding="utf-8")
    print(f"[WRITE TXT] {message_path}")


# ============================================================
# 6. 메인
# ============================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("팀원 공유용 결과 폴더 생성 시작")
    print("=" * 80)

    mkdirs()

    copy_model_result_tables()
    copy_feature_reduction_files()
    copy_threshold_experiment_files()
    copy_visualization_files()
    write_summary_files()

    make_zip(
        src_dir=SHARE_ROOT,
        zip_path=ZIP_PATH,
    )

    print("\n" + "=" * 80)
    print("팀원 공유용 결과 폴더 생성 완료")
    print("=" * 80)
    print("공유 폴더:")
    print(SHARE_ROOT)
    print("\n공유 zip:")
    print(ZIP_PATH)

    print("\n팀원에게는 zip 파일을 보내면 됩니다.")


if __name__ == "__main__":
    main()