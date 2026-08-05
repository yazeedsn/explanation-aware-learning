import hashlib
from pathlib import Path
from dataclasses import dataclass, field


# Data configuration class
@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    csv_path: Path
    img_size: int
    diseases: tuple[str, ...]
    disease_to_idx: dict[str, int] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "disease_to_idx", {d: i for i, d in enumerate(self.diseases)})

    @property
    def num_classes(self) -> int:
        return len(self.diseases)

    @property
    def cache_tag(self) -> str:
        """Hash of (disease list, img_size) - changing either invalidates
        old storage automatically."""
        key = "|".join(self.diseases) + f"|{self.img_size}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def path(self, name: str) -> Path:
        return self.processed_dir / f"{name}_{self.cache_tag}"
