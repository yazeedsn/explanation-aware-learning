"""
Builds a balanced binary classification task for one disease directly from a `datasets.ProcessedDataset`.
"""

import random
import json
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .config import ExperimentConfig
from .datasets import ProcessedDataset, MetadataDataset, BinaryDiseaseDataset

METADATA_FILENAME = "metadata.json"
MemmapMode = Literal["r", "r+", "w+", "c"]

def save_metadata(metadata: MetadataDataset, processed_dir: Path) -> Path:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / METADATA_FILENAME
    with open(path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    return path


def load_metadata(processed_dir: Path) -> MetadataDataset:
    path = Path(processed_dir) / METADATA_FILENAME
    with open(path) as f:
        return MetadataDataset.from_dict(json.load(f))


def open_dataset(processed_dir: Path, mode: MemmapMode = "r") -> ProcessedDataset:
    """opens a processed datasets

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


class DiseaseTaskBuilder:
    """Builds a balanced binary classification task (image_ids + labels)
    for one disease, and produces a stratified train/val/test split."""

    def __init__(self, dataset: ProcessedDataset, config: ExperimentConfig):
        self.dataset = dataset
        self.config = config

    def classify_images(self, disease: str) -> tuple:
        if disease not in self.dataset.metadata.disease_to_idx:
            raise ValueError(
                f"'{disease}' is not in this processed dataset's disease list: "
                f"{self.dataset.metadata.diseases}"
            )
        idx = self.dataset.metadata.disease_to_idx[disease]
        all_ids = list(self.dataset.lookup.keys())
        pos_ids = [i for i in all_ids if self.dataset.labels[self.dataset.row(i), idx]]
        neg_ids = [i for i in all_ids if not self.dataset.labels[self.dataset.row(i), idx]]
        return pos_ids, neg_ids

    def build_binary_task(self, disease: str) -> tuple:
        rng = random.Random(self.config.seed)
        pos_ids, neg_ids = self.classify_images(disease)
        sampled_neg_ids = rng.sample(neg_ids, len(pos_ids))

        labels = {i: 1 for i in pos_ids} | {i: 0 for i in sampled_neg_ids}
        image_ids = pos_ids + sampled_neg_ids
        return labels, image_ids

    def split_task(self, image_ids: list, labels: dict) -> tuple:
        """Stratified train/val/test split -- keeps the task's 1:1
        balance consistent across all three splits."""
        label_list = [labels[i] for i in image_ids]
        non_train_fraction = self.config.val_fraction + self.config.test_fraction

        train_ids, non_train_ids = train_test_split(
            image_ids, test_size=non_train_fraction,
            random_state=self.config.seed, stratify=label_list,
        )
        non_train_labels = [labels[i] for i in non_train_ids]
        val_ids, test_ids = train_test_split(
            non_train_ids, test_size=self.config.test_fraction / non_train_fraction,
            random_state=self.config.seed, stratify=non_train_labels,
        )
        return train_ids, val_ids, test_ids

    def build(self, disease: str) -> dict:
        """Convenience: returns a dict with labels + id splits for a disease."""
        labels, image_ids = self.build_binary_task(disease)
        train_ids, val_ids, test_ids = self.split_task(image_ids, labels)
        return {
            "labels": labels,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        }


def build_dataloaders(
    task: dict, dataset: ProcessedDataset, disease_idx: int, config: ExperimentConfig,
) -> tuple:
    """Builds train/val/test loaders from a task dict produced by
    `DiseaseTaskBuilder.build()`."""

    def make_loader(ids, batch_size, shuffle, drop_last=False):
        ds = BinaryDiseaseDataset(ids, task["labels"], dataset, disease_idx)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=config.num_workers, persistent_workers=config.num_workers > 0,
            pin_memory=True,
        )

    # drop_last on the train loader only: BatchNorm needs >1 sample per
    # channel in train mode, so a size-1 remainder batch would crash it.
    # Not applied to val/test since model.eval() doesn't use batch stats,
    # and we don't want to silently drop evaluation samples.
    train_dl = make_loader(task["train_ids"], config.train_batch_size, shuffle=True, drop_last=True)
    val_dl = make_loader(task["val_ids"], config.eval_batch_size, shuffle=False)
    test_dl = make_loader(task["test_ids"], config.eval_batch_size, shuffle=False)
    return train_dl, val_dl, test_dl


class DiseaseDataModule:
    """Single entry point from a processed dataset directory to a
    disease-specific train/val/test DataLoader triple. Replaces the
    open_dataset -> DiseaseTaskBuilder -> disease_idx lookup ->
    build_dataloaders boilerplate with one object:

        data = DiseaseDataModule(config.data.processed_dir, config.disease, config.split, config.train)
        train_dl, val_dl, test_dl = data.dataloaders()

    Everything computed once at construction is kept around
    (`.dataset`, `.task`, `.disease_idx`, `.split_sizes`) in case a
    caller needs it beyond just the three loaders.
    """

    def __init__(self, disease: str, config: ExperimentConfig, verbose: bool = True):
        # Local import to avoid a circular import: datasets.py doesn't
        # import from data.py, so this is one-directional and safe, but
        # keeping it local documents that data.py's only reason to know
        # about datasets.py is this one class.

        self.disease = disease
        self.dataset = open_dataset(config.processed_dir)

        task_builder = DiseaseTaskBuilder(self.dataset, config)
        self.task = task_builder.build(disease)  # raises a clear ValueError if `disease` isn't in this dataset
        self.disease_idx = self.dataset.metadata.disease_to_idx[disease]

        if verbose:
            sizes = self.split_sizes
            print(f"{disease}: train={sizes['train_ids']}, val={sizes['val_ids']}, test={sizes['test_ids']}")

        self.train_dl, self.val_dl, self.test_dl = build_dataloaders(self.task, self.dataset, self.disease_idx, config)

    def dataloaders(self) -> tuple:
        """Returns (train_dl, val_dl, test_dl)."""
        return self.train_dl, self.val_dl, self.test_dl

    @property
    def split_sizes(self) -> dict:
        return {k: len(self.task[k]) for k in ("train_ids", "val_ids", "test_ids")}
