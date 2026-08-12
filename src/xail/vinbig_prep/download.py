"""
Fetches the raw VinBigData competition files, so `DataConfig.raw_dir` /
`csv_path` have something to point at before `storage.build_storage` runs.

"""

import os
import zipfile
from pathlib import Path
from typing import Optional

KAGGLE_COMPETITION = "vinbigdata-chest-xray-abnormalities-detection"
KAGGLE_INPUT_DIR = Path("/kaggle/input") / KAGGLE_COMPETITION


def _running_in_kaggle_notebook() -> bool:
    """True when the competition data is already mounted (i.e. we're
    running inside a Kaggle competition notebook) -- no download needed."""
    return KAGGLE_INPUT_DIR.exists()


def _authenticate(kaggle_api_token: Optional[str] = None):
    """Authenticates with the Kaggle API.

    Priority: `kaggle_api_token` argument -> existing `KAGGLE_API_TOKEN`
    env var -> `~/.kaggle/access_token` -> OAuth login. All but the first
    are handled internally by the `kaggle` package itself.
    """
    if kaggle_api_token is not None:
        os.environ["KAGGLE_API_TOKEN"] = kaggle_api_token

    try:
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

    return api


def _download_and_extract(api, path: Path) -> None:
    """Downloads the competition zip into `path` and extracts it in place."""
    api.competition_download_files(KAGGLE_COMPETITION, path=str(path), quiet=False)

    zip_path = path / f"{KAGGLE_COMPETITION}.zip"
    if not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path)
    zip_path.unlink()


def download_data(path: Path, kaggle_api_token: Optional[str] = None) -> Path:
    """Ensures the raw VinBigData competition files are available locally.

    Returns the directory containing `train/` and `train.csv` -- pass it
    straight to `DataConfig(raw_dir=.../"train", csv_path=.../"train.csv", ...)`.

    If already running in a Kaggle competition notebook, the mounted
    input directory is returned as-is and nothing is downloaded.
    Otherwise, downloads and extracts into `path` via the Kaggle API,
    authenticating with `kaggle_api_token` if given (see `_authenticate`
    for the full priority order).
    """
    if _running_in_kaggle_notebook():
        return KAGGLE_INPUT_DIR

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    api = _authenticate(kaggle_api_token)
    _download_and_extract(api, path)

    return path
