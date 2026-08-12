"""Datasets."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dataclasses import asdict, dataclass

from .config import TrainConfig

@dataclass(frozen=True)
class DatasetMetadata:
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
    def from_dict(cls, d: dict) -> "DatasetMetadata":
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

    metadata: DatasetMetadata
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


def build_dataloaders(
    task: dict, dataset: ProcessedDataset, disease_idx: int, train: TrainConfig,
) -> tuple:
    """Builds train/val/test loaders from a task dict produced by
    `DiseaseTaskBuilder.build()`."""

    def make_loader(ids, batch_size, shuffle, drop_last=False):
        ds = BinaryDiseaseDataset(ids, task["labels"], dataset, disease_idx)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=train.num_workers, persistent_workers=train.num_workers > 0,
            pin_memory=True,
        )

    # drop_last on the train loader only: BatchNorm needs >1 sample per
    # channel in train mode, so a size-1 remainder batch would crash it.
    # Not applied to val/test since model.eval() doesn't use batch stats,
    # and we don't want to silently drop evaluation samples.
    train_dl = make_loader(task["train_ids"], train.train_batch_size, shuffle=True, drop_last=True)
    val_dl = make_loader(task["val_ids"], train.eval_batch_size, shuffle=False)
    test_dl = make_loader(task["test_ids"], train.eval_batch_size, shuffle=False)
    return train_dl, val_dl, test_dl
