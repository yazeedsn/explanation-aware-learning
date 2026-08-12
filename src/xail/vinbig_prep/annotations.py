"""Loading and grouping the raw annotation CSV."""

from pathlib import Path

import pandas as pd

_BOX_COLUMNS = ["class_name", "x_min", "y_min", "x_max", "y_max"]


def load_annotations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} annotation rows across {df.image_id.nunique()} images")
    print("Number of NaNs per column:")
    print(df.isna().sum())
    return df


def build_image_universe(df: pd.DataFrame) -> list:
    """The full, ordered set of image_ids that will get a row in storage."""
    return sorted(df.image_id.unique().tolist())


def group_by_image(df: pd.DataFrame) -> dict:
    """image_id -> DataFrame of its bounding-box rows (class_name, x/y min/max).
    Images with no rows in `df` simply won't have a key here; callers
    should handle missing keys as "no annotations"."""
    return dict(list(df.groupby("image_id")[_BOX_COLUMNS]))
