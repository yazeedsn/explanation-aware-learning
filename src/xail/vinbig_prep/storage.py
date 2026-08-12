"""Allocates the on-disk image/mask stores and drives the (parallel
decode, sequential write) preprocessing loop. Writes `metadata.json` at
the end so the output directory is self-describing (see `metadata.py`).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .annotations import build_image_universe, group_by_image
from .config import DataConfig
from .dicom_io import load_and_resize_image
from .masks import build_disease_masks
from .metadata import DatasetMetadata, load_metadata, save_metadata, METADATA_FILENAME


def allocate_storage(processed_dir: Path, image_shape: tuple, mask_shape: tuple):
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    image_store = np.memmap(processed_dir / "images.dat", dtype=np.uint8, mode="w+", shape=image_shape)
    mask_store = np.memmap(processed_dir / "masks.dat", dtype=np.bool_, mode="w+", shape=mask_shape)
    return image_store, mask_store


def make_dicom_decoder(config: DataConfig, grouped: dict):
    """Default decode_fn: DICOM -> (resized image, per-disease masks).
    """
    empty_annotations = pd.DataFrame(columns=["class_name", "x_min", "y_min", "x_max", "y_max"])

    def decode_one(image_id: str):
        raw_path = config.raw_dir / f"{image_id}.dicom"
        image, orig_h, orig_w = load_and_resize_image(raw_path, config.img_size)
        annotations = grouped.get(image_id, empty_annotations)
        masks = build_disease_masks(annotations, config, orig_h, orig_w)
        return image_id, image, masks

    return decode_one


def build_storage(df: pd.DataFrame, config: DataConfig, decode_fn=None, max_workers: int = 8, overwrite: bool = False) -> DatasetMetadata:
    """
    Builds the processed dataset: memmap image/mask stores, a lookup
    table (image_id -> row), per-image/per-disease classification labels
    (derived from mask presence, so guaranteed consistent with the
    explanation-loss annotation masks), and `metadata.json` describing
    all of it.

    If `processed_dir` already holds a dataset:
      - matching this config (same diseases, same order, same img_size)
        and `overwrite=False` -> skipped, existing metadata is returned.
      - matching but `overwrite=True` -> rebuilt from scratch.
      - present but built with a *different* config -> raises, to avoid
        silently mixing incompatible data under one directory. Point
        `processed_dir` at a fresh directory for a different variant.

    Args:
        df: raw annotation DataFrame (as returned by `load_annotations`).
        config: `DataConfig` describing raw/processed locations and the
            image size / disease list to build.
        decode_fn: optional `image_id -> (image_id, image, masks_dict)`
            callable. Defaults to DICOM decoding via `make_dicom_decoder`.
            Overriding this is the intended extension point if the raw
            data format changes.
        max_workers: thread pool size for the (I/O-bound) decode step.
        overwrite: force a rebuild even if a matching dataset exists.
    """
    processed_dir = Path(config.processed_dir)
    existing_metadata_path = processed_dir / METADATA_FILENAME

    if existing_metadata_path.exists() and not overwrite:
        existing = load_metadata(processed_dir)
        if existing.matches(config.diseases, config.img_size):
            print(f"Reusing existing processed dataset at {processed_dir}")
            return existing
        raise ValueError(
            f"{processed_dir} already contains a processed dataset built with a different "
            f"config (diseases={existing.diseases}, image_size={existing.image_size}). "
            "Use a different processed_dir for this config, or pass overwrite=True."
        )

    print(f"Building processed dataset at {processed_dir}")
    universe = build_image_universe(df)
    lookup = {image_id: i for i, image_id in enumerate(universe)}
    n = len(universe)

    image_shape = (n, config.img_size, config.img_size)
    mask_shape = (n, config.num_classes, config.img_size, config.img_size)
    image_store, mask_store = allocate_storage(processed_dir, image_shape, mask_shape)
    labels = np.zeros((n, config.num_classes), dtype=np.bool_)

    if decode_fn is None:
        grouped = group_by_image(df)
        decode_fn = make_dicom_decoder(config, grouped)

    # Parallelize the expensive part (decode); write sequentially on the
    # main thread to avoid concurrent writes into the same memmap file.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for image_id, image, masks in tqdm(executor.map(decode_fn, universe), total=n, desc="Preprocessing"):
            row = lookup[image_id]
            image_store[row] = image
            for disease, mask in masks.items():
                disease_idx = config.disease_to_idx[disease]
                mask_store[row, disease_idx] = mask
                # A classification label is derived from the same mask
                # used for explanation supervision, so the two are always
                # consistent (e.g. a box that clips to nothing counts as
                # negative for both, rather than disagreeing).
                labels[row, disease_idx] = mask.any()

    image_store.flush()
    mask_store.flush()

    with open(processed_dir / "lookup.json", "w") as f:
        json.dump(lookup, f)
    np.save(processed_dir / "labels.npy", labels)

    metadata = DatasetMetadata(diseases=config.diseases, image_size=config.img_size, num_samples=n)
    save_metadata(metadata, processed_dir)
    return metadata
