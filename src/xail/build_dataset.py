"""
Entry point: builds the processed VinDr-CXR dataset.
"""

from pathlib import Path

import typer

from .vinbig_prep import DataConfig, build_storage, download_data, load_annotations

DISEASES = ("Aortic enlargement", "Pulmonary fibrosis", "Pleural effusion")


def main(
    download: bool = False,
    kaggle_token: str = typer.Option(None, help="Kaggle API token; see download_data() for auth priority."),
    raw_dir: Path = Path("./data/raw"),
    processed_dir: Path = Path("./data/processed"),
    img_size: int = 224,
):
    if download:
        raw_dir = download_data(path=raw_dir, kaggle_api_token=kaggle_token)

    data_config = DataConfig(
        raw_dir=raw_dir / "train",
        processed_dir=processed_dir,
        csv_path=raw_dir / "train.csv",
        img_size=img_size,
        diseases=DISEASES,
    )

    df = load_annotations(data_config.csv_path)
    metadata = build_storage(df, data_config)
    print(metadata)


if __name__ == "__main__":
    typer.run(main)
