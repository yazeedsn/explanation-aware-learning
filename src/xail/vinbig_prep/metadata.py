"""
metadata when creating and loading processed data.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from ..core.datasets import ProcessedDataset, DatasetMetadata

METADATA_FILENAME = "metadata.json"
MemmapMode = Literal["r", "r+", "w+", "c"]




def save_metadata(metadata: DatasetMetadata, processed_dir: Path) -> Path:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / METADATA_FILENAME
    with open(path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    return path


def load_metadata(processed_dir: Path) -> DatasetMetadata:
    path = Path(processed_dir) / METADATA_FILENAME
    with open(path) as f:
        return DatasetMetadata.from_dict(json.load(f))




def open_dataset(processed_dir: Path, mode: MemmapMode = "r") -> ProcessedDataset:
    """The only entry point a downstream consumer needs.

    No knowledge of image size, disease list/order, dtypes, or file
    layout is required -- it all comes from `metadata.json`, `lookup.json`,
    and `labels.npy` in `processed_dir`.

    Args:
        processed_dir: directory produced by `storage.build_storage`.
        mode: memmap open mode, e.g. "r" (read-only) or "r+".
    """
    processed_dir = Path(processed_dir)
    metadata = load_metadata(processed_dir)

    with open(processed_dir / metadata.lookup_filename) as f:
        lookup = json.load(f)

    image_store = np.memmap(
        processed_dir / metadata.images_filename,
        dtype=metadata.image_dtype, mode=mode, shape=metadata.image_shape,
    )
    mask_store = np.memmap(
        processed_dir / metadata.masks_filename,
        dtype=metadata.mask_dtype, mode=mode, shape=metadata.mask_shape,
    )
    labels = np.load(processed_dir / metadata.labels_filename)

    return ProcessedDataset(metadata=metadata, lookup=lookup, images=image_store, masks=mask_store, labels=labels)
