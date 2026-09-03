from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def prepare_raw_data_dir() -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR


if __name__ == "__main__":
    data_dir = prepare_raw_data_dir()
    print(f"Raw data directory: {data_dir}")