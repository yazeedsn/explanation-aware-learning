"""Turns raw bounding boxes (original image pixel space) into per-disease
binary masks aligned to the resized image."""

import numpy as np
import pandas as pd

from .config import DataConfig


def build_disease_masks(annotations: pd.DataFrame, config: DataConfig, orig_h: int, orig_w: int) -> dict:
    """Returns {disease_name: (img_size, img_size) bool mask}, one entry
    per disease in `config.diseases` (all-zero if the image has no boxes
    for that disease)."""
    img_size = config.img_size
    scale_x, scale_y = img_size / orig_w, img_size / orig_h
    masks = {d: np.zeros((img_size, img_size), dtype=np.bool_) for d in config.diseases}

    for cls, x_min, y_min, x_max, y_max in annotations.itertuples(index=False):
        if cls not in config.disease_to_idx:
            continue
        x1 = max(int(x_min * scale_x), 0)
        x2 = min(int(x_max * scale_x), img_size)
        y1 = max(int(y_min * scale_y), 0)
        y2 = min(int(y_max * scale_y), img_size)
        if x2 <= x1 or y2 <= y1:
            continue
        masks[cls][y1:y2, x1:x2] = True

    return masks
