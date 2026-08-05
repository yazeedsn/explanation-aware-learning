from pathlib import Path
import os
import zipfile


def download_data(path: Path, kaggle_api_token: str | None = None) -> Path:
    """
    Download the VinBigData competition dataset.

    Authentication priority:
      1. KAGGLE_API_TOKEN argument
      2. Existing KAGGLE_API_TOKEN environment variable
      3. ~/.kaggle/access_token
      4. OAuth login
    """

    # if working in kaggle within a competition notebook
    kaggle_input = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection")
    if kaggle_input.exists():
        return kaggle_input

    # working on external environment
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if kaggle_api_token is not None:
        os.environ["KAGGLE_API_TOKEN"] = kaggle_api_token

    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        raise RuntimeError(
            "Kaggle authentication failed.\n"
            "Authenticate using one of:\n"
            "  • KAGGLE_API_TOKEN environment variable\n"
            "  • ~/.kaggle/access_token\n"
            "  • kaggle auth login"
        ) from e

    api.competition_download_files(
        "vinbigdata-chest-xray-abnormalities-detection",
        path=str(path),
        quiet=False,
    )
    zip_path = path / "vinbigdata-chest-xray-abnormalities-detection.zip"
    if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(path)
            zip_path.unlink()

    return path
