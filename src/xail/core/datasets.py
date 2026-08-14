"""Datasets."""

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class MetadataDataset:
    """Self-contained description of a processed dataset directory.

    `diseases` is the authoritative mask-channel order: `mask_store[row, i]`
    corresponds to `diseases[i]`.
    """

    diseases: tuple
    image_size: int
    num_samples: int
    image_dtype: str = "uint8"
    mask_dtype: str = "bool"
    images_filename: str = "images.dat"
    masks_filename: str = "masks.dat"
    lookup_filename: str = "lookup.json"
    labels_filename: str = "labels.npy"
    schema_version: int = 2
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
    def from_dict(cls, d: dict) -> "MetadataDataset":
        d = dict(d)
        d["diseases"] = tuple(d["diseases"])
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def matches(self, diseases: tuple, image_size: int) -> bool:
        """Whether this metadata was built with the given disease list
        (order-sensitive) and image size."""
        return tuple(self.diseases) == tuple(diseases) and self.image_size == image_size


@dataclass(frozen=True)
class ProcessedDataset:
    """Contains all the information about the preprocessed data.
    Returned by `open_dataset()`

    `labels[row, i]` is True iff `masks[row, i]` has any annotated pixel.
    """

    metadata: MetadataDataset
    lookup: dict
    images: np.memmap
    masks: np.memmap
    labels: np.ndarray  # (num_samples, num_diseases) bool

    def row(self, image_id: str) -> int:
        return self.lookup[image_id]



class BinaryDiseaseDataset(Dataset):
    """One disease, binary labels, plus its per-pixel annotation mask."""

    def __init__(self, image_ids: list, labels: dict, dataset: ProcessedDataset, disease_idx: int):
        self.image_ids = list(image_ids)
        self.labels = torch.tensor([labels[i] for i in self.image_ids], dtype=torch.int64)
        self.dataset = dataset
        self.disease_idx = disease_idx

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        row = self.dataset.row(self.image_ids[idx])

        image = torch.from_numpy(self.dataset.images[row].copy()).float().div(255.0)
        image = image.unsqueeze(0).expand(3, -1, -1)  # grayscale -> 3-channel

        mask = torch.from_numpy(self.dataset.masks[row, self.disease_idx].copy()).float()

        return image, self.labels[idx], mask
