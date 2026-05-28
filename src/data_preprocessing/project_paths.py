from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"


def sav_file(year: int) -> Path:
    """Return the preferred SAV path while supporting the current root layout."""
    raw_path = RAW_DATA_DIR / f"kyrbs{year}.sav"

    if raw_path.exists():
        return raw_path

    return PROJECT_ROOT / f"kyrbs{year}.sav"


def sav_files() -> dict[int, Path]:
    return {
        2023: sav_file(2023),
        2024: sav_file(2024),
        2025: sav_file(2025),
    }
