"""Training loop, metrics, and checkpointing, wrapped in a `Trainer`."""

from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ModelConfig, RunConfig, TrainConfig
from .losses import CombinedLoss
from .metrics import MetricBundle



class Trainer:
    """Runs the train/val loop for one binary disease-classification task
    and takes care of history, checkpointing, and logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: CombinedLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_config: TrainConfig,
        model_config: ModelConfig,
        run_config: RunConfig,
        disease: str,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_config = train_config
        self.device = train_config.resolve_device()
        self.non_blocking = self.device == "cuda"
        self.output_dir = run_config.disease_dir(disease)

        self.train_metrics = MetricBundle(model_config.num_classes, self.device)
        self.val_metrics = MetricBundle(model_config.num_classes, self.device)
        self.history = defaultdict(list)
        self.best_val_loss = float("inf")

    def _to_device(self, *tensors):
        return [t.to(self.device, non_blocking=self.non_blocking) for t in tensors]

    def train_epoch(self, train_dl: DataLoader, epoch: int, total_epochs: int):
        self.model.train()
        self.train_metrics.reset()


        pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
        for images, labels, masks in pbar:
            images, labels, masks = self._to_device(images, labels, masks)

            self.optimizer.zero_grad(set_to_none=True)
            logits, feature_map = self.model(images)
            loss, parts = self.loss_fn(logits, feature_map, labels, masks)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update(loss.detach(), logits.detach(), labels, cls_loss=parts["cls"],
                exp_loss=parts["exp"])
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{self.train_metrics.acc.compute().item():.3f}")

        return self.train_metrics.compute()

    @torch.inference_mode()
    def validate_epoch(self, val_dl: DataLoader):
        self.model.eval()
        self.val_metrics.reset()

        for images, labels, _masks in val_dl:
            images, labels = self._to_device(images, labels)
            logits, _feature_map = self.model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            self.val_metrics.update(loss, logits, labels)

        return self.val_metrics.compute()

    def _log_epoch(self, epoch: int):
        parts = [f"Epoch [{epoch + 1}]"]
        for name, values in self.history.items():
            parts.append(f"{name}={values[-1]:.4f}")
        print(" | ".join(parts))

    def _record(self, prefix: str, metrics: dict):
        for name, value in metrics.items():
            self.history[f"{prefix}_{name}"].append(value)

    def _save_checkpoint(self, name: str):
        torch.save(self.model.state_dict(), self.output_dir / name)

    def fit(self, train_dl: DataLoader, val_dl: DataLoader, epochs: int | None = None, verbose: bool = True) -> pd.DataFrame:
        epochs = epochs or self.train_config.epochs

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_dl, epoch, epochs)
            val_metrics = self.validate_epoch(val_dl)

            self._record("train", train_metrics)
            self._record("val", val_metrics)

            if verbose:
                self._log_epoch(epoch)

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint("best.pt")

            if self.scheduler:
                self.scheduler.step()

        self._save_checkpoint("last.pt")
        return self.save_history()

    def save_history(self) -> pd.DataFrame:
        hist_df = pd.DataFrame(self.history)
        hist_df.to_csv(self.output_dir / "history.csv", index=False)
        return hist_df
