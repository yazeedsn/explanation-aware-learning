"""
metadata creating and loading.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

METADATA_FILENAME = "metadata.json"
MemmapMode = Literal["r", "r+", "w+", "c"]


@dataclass(frozen=True)
class DatasetMetadata:
    """Self-contained description of a processed dataset directory.

    `diseases` is the authoritative mask-channel order: `mask_store[row, i]`
    corresponds to `diseases[i]`. Consumers should always read the disease
    list from here rather than assuming/hardcoding an order.
    """

    diseases: tuple
    image_size: int
    num_samples: int
    image_dtype: str = "uint8"
    mask_dtype: str = "bool"
    images_filename: str = "images.dat"
    masks_filename: str = "masks.dat"
    lookup_filename: str = "lookup.json"
    schema_version: int = 1
    notes: str = ""

    @property
    def disease_to_idx(self) -> dict:
        return {d: i for i, d in enumerate(self.diseases)}

    @property
    def num_diseases(self) -> int:
        return len(self.diseases)

    @property
    def image_shape(self) -> tuple:
        return (self.num_samples, self.image_size, self.image_size)

    @property
    def mask_shape(self) -> tuple:
        return (self.num_samples, self.num_diseases, self.image_size, self.image_size)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetMetadata":
        d = dict(d)
        d["diseases"] = tuple(d["diseases"])
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def matches(self, diseases: tuple, image_size: int) -> bool:
        """Whether this metadata was built with the given disease list
        (order-sensitive) and image size."""
        return tuple(self.diseases) == tuple(diseases) and self.image_size == image_size


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


def open_dataset(processed_dir: Path, mode: MemmapMode = "r"):
    """The only entry point a downstream consumer needs.

    Returns (metadata, lookup, image_store, mask_store).
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
        dtype=metadata.image_dtype,
        mode=mode,
        shape=metadata.image_shape,
    )
    mask_store = np.memmap(
        processed_dir / metadata.masks_filename,
        dtype=metadata.mask_dtype,
        mode=mode,
        shape=metadata.mask_shape,
    )
    return metadata, lookup, image_store, mask_store
