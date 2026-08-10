"""Preprocessing configuration.

No cache-tag hashing: each run writes to `processed_dir` under fixed
filenames (see `metadata.py`). If you want to keep multiple preprocessed
variants (different image size, different disease list, etc.) around at
once, point them at different `processed_dir`s -- `build_storage` will
refuse to silently overwrite a mismatched existing dataset (see
`storage.build_storage`).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    csv_path: Path
    img_size: int
    diseases: tuple

    @property
    def disease_to_idx(self) -> dict:
        return {d: i for i, d in enumerate(self.diseases)}

    @property
    def num_classes(self) -> int:
        return len(self.diseases)
