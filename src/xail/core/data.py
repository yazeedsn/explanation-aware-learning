"""
Builds a balanced binary classification task for one disease directly from a `datasets.ProcessedDataset`.
"""

import random

from sklearn.model_selection import train_test_split
from .datasets import ProcessedDataset

from .config import SplitConfig


class DiseaseTaskBuilder:
    """Builds a balanced binary classification task (image_ids + labels)
    for one disease, and produces a stratified train/val/test split."""

    def __init__(self, dataset: ProcessedDataset, split: SplitConfig):
        self.dataset = dataset
        self.split = split

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
        rng = random.Random(self.split.seed)
        pos_ids, neg_ids = self.classify_images(disease)
        sampled_neg_ids = rng.sample(neg_ids, len(pos_ids))

        labels = {i: 1 for i in pos_ids} | {i: 0 for i in sampled_neg_ids}
        image_ids = pos_ids + sampled_neg_ids
        return labels, image_ids

    def split_task(self, image_ids: list, labels: dict) -> tuple:
        """Stratified train/val/test split -- keeps the task's 1:1
        balance consistent across all three splits."""
        label_list = [labels[i] for i in image_ids]
        non_train_fraction = self.split.val_fraction + self.split.test_fraction

        train_ids, non_train_ids = train_test_split(
            image_ids, test_size=non_train_fraction,
            random_state=self.split.seed, stratify=label_list,
        )
        non_train_labels = [labels[i] for i in non_train_ids]
        val_ids, test_ids = train_test_split(
            non_train_ids, test_size=self.split.test_fraction / non_train_fraction,
            random_state=self.split.seed, stratify=non_train_labels,
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
