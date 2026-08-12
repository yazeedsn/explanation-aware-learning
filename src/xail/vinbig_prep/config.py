"""Preprocessing configuration.

Each run writes to `processed_dir` under fixed
filenames (see `metadata.py`).
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
