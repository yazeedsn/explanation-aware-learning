"""
Fetch the VinBigData/VinDr-CXR competition data into data/raw, then run the
shared-storage preprocessing pipeline into data/preprocessed.

Usage:
    python scripts/prepare_data.py
"""
from pathlib import Path
import typer

from .data_handlers import DataConfig, download_data, load_annotations, get_or_build_storage

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/preprocessed")

IMG_SIZE = 224
MAX_WORKERS = 8
DISEASES = (
    "Aortic enlargement",
    "Cardiomegaly",
    "Pleural thickening",
    "Pulmonary fibrosis",
)


def main(
    img_size: int = IMG_SIZE,
    raw_dir: str | Path = RAW_DIR,
    processed_dir: str | Path = PROCESSED_DIR,
    kaggle_token: str = None,
    workers: int = MAX_WORKERS
):
    raw_dir = Path(raw_dir)
    preprocessed_dir = Path(processed_dir)
    print(f"Fetching raw data into {raw_dir} ...")
    raw_path = download_data(raw_dir, kaggle_token)
    print(f"Raw data available at: {raw_path}")

    config = DataConfig(
        raw_dir=raw_path / "train",
        processed_dir=preprocessed_dir,
        csv_path=raw_path / "train.csv",
        img_size=img_size,
        diseases=DISEASES,
    )

    df = load_annotations(config.csv_path)

    print(f"Preprocessing into {preprocessed_dir} (cache tag={config.cache_tag}) ...")
    image_id_to_row, _, _ = get_or_build_storage(df, config, workers)

    print(f"Done. {len(image_id_to_row)} images cached at {preprocessed_dir}")


if __name__ == "__main__":
    typer.run(main)
