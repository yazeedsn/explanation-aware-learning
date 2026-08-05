import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import DataConfig
from .utils import load_dicom_as_array

def load_and_resize_image(raw_path: Path, img_size: int) -> tuple[np.ndarray, int, int]:
    """ Loads dicom file as a numpy array and resize it to image_size x image_size uint8 array
        args:
            raw_path (Path): raw data dicom file path
            img_size (int): target size for resizing
        returns:
            (image: np.ndarray, original image height: int, original image weidth: int)
    """
    array = load_dicom_as_array(raw_path)
    orig_h, orig_w = array.shape
    resized = cv2.resize(array, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False), orig_h, orig_w

def build_disease_masks(annotations: pd.DataFrame | pd.Series, config: DataConfig,
                         orig_h: int, orig_w: int) -> dict[str, np.ndarray]:
    """ Builds a mask for each disease in the given annotations.
        Returns a dict of each present disease (str) with its given mask (np.array)
    """
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

def build_image_universe(df: pd.DataFrame) -> list[str]:
    return sorted(df.image_id.unique().tolist())

def get_or_build_storage(df: pd.DataFrame, config: DataConfig, max_workers: int = 8):
    """
    Preprocess the data in images within the data frame using the given config.
    If a cache matching this exact config (diseases + img_size, via cache_tag)
    already exists on disk, reopens it instead of reprocessing.

    Returns (image_id_to_row, image_store, mask_store).
    """
    lookup_path = config.path("lookup").with_suffix(".json")
    images_path = config.path("images").with_suffix(".dat")
    masks_path = config.path("masks").with_suffix(".dat")

    if lookup_path.exists() and images_path.exists() and masks_path.exists():
        print(f"Reusing existing cache (tag={config.cache_tag})")
        with open(lookup_path) as f:
            image_id_to_row = json.load(f)
        n = len(image_id_to_row)
        s = config.img_size
        image_store = np.memmap(images_path, dtype=np.uint8, mode="r+", shape=(n, s, s))
        mask_store = np.memmap(masks_path, dtype=np.bool_, mode="r+",
                                shape=(n, config.num_classes, s, s))
        return image_id_to_row, image_store, mask_store

    print(f"Building new cache (tag={config.cache_tag})")
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    universe = build_image_universe(df)
    image_id_to_row = {img_id: i for i, img_id in enumerate(universe)}
    n, s = len(universe), config.img_size

    image_store = np.memmap(images_path, dtype=np.uint8, mode="w+", shape=(n, s, s))
    mask_store = np.memmap(masks_path, dtype=np.bool_, mode="w+",
                            shape=(n, config.num_classes, s, s))

    grouped = dict(list(df.groupby("image_id")[["class_name", "x_min", "y_min", "x_max", "y_max"]]))

    def decode_one(image_id: str):
        raw_path = config.raw_dir / f"{image_id}.dicom"
        image, orig_h, orig_w = load_and_resize_image(raw_path, config.img_size)
        masks = build_disease_masks(grouped[image_id], config, orig_h, orig_w)
        return image_id, image, masks

    # Parallelize the expensive part (DICOM decode); write sequentially on
    # the main thread to avoid concurrent writes into the same memmap file.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for image_id, image, masks in tqdm(
            executor.map(decode_one, universe), total=len(universe), desc="Preprocessing"
        ):
            row = image_id_to_row[image_id]
            image_store[row] = image
            for disease, mask in masks.items():
                mask_store[row, config.disease_to_idx[disease]] = mask

    image_store.flush()
    mask_store.flush()
    with open(lookup_path, "w") as f:
        json.dump(image_id_to_row, f)

    return image_id_to_row, image_store, mask_store
