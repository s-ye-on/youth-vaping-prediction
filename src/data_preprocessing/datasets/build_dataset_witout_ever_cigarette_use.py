from pathlib import Path

import pandas as pd


TARGET_COL = "current_ecig_use"
REMOVE_COL = "ever_cigarette_use"

INPUT_DATASET_NAME = "selected_modeling_dataset.csv"

OUTPUT_DATASET_NAME = "selected_modeling_dataset_without_ever_cigarette_use.csv"
OUTPUT_X_NAME = "selected_modeling_X_without_ever_cigarette_use.csv"
OUTPUT_Y_NAME = "selected_modeling_y_without_ever_cigarette_use.csv"

IGNORE_DIR_NAMES = {
    ".git",
    ".idea",
    ".venv",
    "venv",
    "__pycache__",
    "v1_error",
    "v1_errors",
    "error",
    "errors",
}


def is_ignored_path(path: Path) -> bool:
    path_parts = set(path.parts)
    return any(ignore_name in path_parts for ignore_name in IGNORE_DIR_NAMES)


def find_project_root() -> Path:
    """
    스크립트 위치가 프로젝트 루트에 있지 않아도 프로젝트 루트를 최대한 찾는다.

    우선순위:
    1. 현재 스크립트 위치와 부모 폴더 중 data/ 또는 outputs/가 있는 곳
    2. pyproject.toml, requirements.txt, .idea 등이 있는 곳
    3. 그래도 못 찾으면 현재 스크립트가 있는 폴더
    """
    script_dir = Path(__file__).resolve().parent

    markers = [
        "data",
        "outputs",
        "pyproject.toml",
        "requirements.txt",
        ".idea",
    ]

    for candidate in [script_dir, *script_dir.parents]:
        for marker in markers:
            if (candidate / marker).exists():
                return candidate

    return script_dir


def find_input_dataset(project_root: Path) -> Path:
    """
    프로젝트 구조가 바뀌어도 selected_modeling_dataset.csv를 자동 탐색한다.

    v1_error 폴더는 무시한다.
    selected_modeling_dataset_without_ever_cigarette_use.csv 같은 파생 파일도 무시한다.
    selected_modeling_X.csv, selected_modeling_y.csv도 무시한다.
    """
    candidates = []

    for path in project_root.rglob(INPUT_DATASET_NAME):
        if is_ignored_path(path):
            continue

        filename = path.name.lower()

        if "without_ever_cigarette_use" in filename:
            continue

        if "_x" in filename or "_y" in filename:
            continue

        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "\nselected_modeling_dataset.csv 파일을 찾지 못했습니다.\n\n"
            f"탐색 시작 위치: {project_root}\n\n"
            "확인할 점:\n"
            "1. 먼저 build_selected_modeling_dataset.py를 실행했는지 확인하세요.\n"
            "2. selected_modeling_dataset.csv가 프로젝트 내부에 있는지 확인하세요.\n"
            "3. 파일명이 다르다면 INPUT_DATASET_NAME 값을 수정하세요.\n"
        )

    def score(path: Path) -> tuple[int, int]:
        path_text = str(path).lower()

        processed_score = 1 if "data/processed" in path_text or "data\\processed" in path_text else 0
        shorter_path_score = -len(path.parts)

        return processed_score, shorter_path_score

    candidates = sorted(candidates, key=score, reverse=True)

    if len(candidates) > 1:
        print("\n[주의] selected_modeling_dataset.csv 후보가 여러 개 발견되었습니다.")
        print("아래 후보 중 우선순위가 가장 높은 파일을 사용합니다.")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"{idx}. {candidate}")

    return candidates[0]


def require_columns(df: pd.DataFrame, columns: list[str], input_file: Path) -> None:
    missing = [col for col in columns if col not in df.columns]

    if not missing:
        return

    raise ValueError(
        "\n필수 컬럼이 데이터에 없습니다.\n"
        f"입력 파일: {input_file}\n"
        f"없는 컬럼: {missing}\n\n"
        "확인할 점:\n"
        "1. 최신 build_selected_modeling_dataset.py를 다시 실행했는지 확인하세요.\n"
        "2. selected_modeling_dataset.csv에 current_ecig_use, ever_cigarette_use가 있는지 확인하세요.\n"
        "3. 이미 ever_cigarette_use를 제거한 파일을 입력으로 잡은 것은 아닌지 확인하세요.\n"
    )


def assert_target_valid(df: pd.DataFrame) -> None:
    values = sorted(df[TARGET_COL].dropna().unique().tolist())

    if values != [0, 1]:
        raise ValueError(
            f"\n{TARGET_COL} 값이 0/1이 아닙니다.\n"
            f"현재 값: {values}\n"
        )


def move_target_to_last(df: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in df.columns if col != TARGET_COL]
    columns.append(TARGET_COL)
    return df[columns]


def make_without_ever_cigarette_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = df.drop(columns=[REMOVE_COL])
    result = move_target_to_last(result)
    return result


def print_summary(
    input_file: Path,
    output_dataset_file: Path,
    output_x_file: Path,
    output_y_file: Path,
    original: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    print("\n[입력 파일]")
    print(input_file)

    print("\n[저장 파일]")
    print("dataset:", output_dataset_file)
    print("X:", output_x_file)
    print("y:", output_y_file)

    print("\n[shape]")
    print("원본 dataset:", original.shape)
    print("제거 후 dataset:", result.shape)
    print("X:", (result.drop(columns=[TARGET_COL])).shape)
    print("y:", result[[TARGET_COL]].shape)

    print("\n[제거한 컬럼]")
    print("-", REMOVE_COL)

    print("\n[제거 여부 확인]")
    print(f"{REMOVE_COL} in original:", REMOVE_COL in original.columns)
    print(f"{REMOVE_COL} in result:", REMOVE_COL in result.columns)

    print("\n[target 분포]")
    print(result[TARGET_COL].value_counts(dropna=False))

    print("\n[target 비율]")
    print(result[TARGET_COL].value_counts(normalize=True, dropna=False).round(4))

    print("\n[결측 비율 상위]")
    missing_ratio = result.isna().mean().sort_values(ascending=False)
    missing_ratio = missing_ratio[missing_ratio > 0]

    if missing_ratio.empty:
        print("결측치가 있는 컬럼이 없습니다.")
    else:
        print(missing_ratio.head(30).round(4))

    print("\n[최종 컬럼 목록]")
    for col in result.columns:
        print("-", col)


def main() -> None:
    project_root = find_project_root()

    print("프로젝트 루트 추정:", project_root)

    input_file = find_input_dataset(project_root)

    print("selected_modeling_dataset.csv 읽는 중...")
    print("사용 입력 파일:", input_file)

    df = pd.read_csv(input_file, low_memory=False)

    require_columns(df, [TARGET_COL, REMOVE_COL], input_file)
    assert_target_valid(df)

    result = make_without_ever_cigarette_dataset(df)

    output_dir = input_file.parent

    output_dataset_file = output_dir / OUTPUT_DATASET_NAME
    output_x_file = output_dir / OUTPUT_X_NAME
    output_y_file = output_dir / OUTPUT_Y_NAME

    result.to_csv(output_dataset_file, index=False, encoding="utf-8-sig")

    X = result.drop(columns=[TARGET_COL])
    y = result[[TARGET_COL]]

    X.to_csv(output_x_file, index=False, encoding="utf-8-sig")
    y.to_csv(output_y_file, index=False, encoding="utf-8-sig")

    print("\n저장 완료")
    print_summary(
        input_file=input_file,
        output_dataset_file=output_dataset_file,
        output_x_file=output_x_file,
        output_y_file=output_y_file,
        original=df,
        result=result,
    )


if __name__ == "__main__":
    main()