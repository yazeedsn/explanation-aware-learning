"""
Everything is threaded through these configs, so
switching datasets / preprocessing / experiments means editing this file
(or constructing a new config in a script).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Where the preprocessed dataset lives."""
    processed_dir: Path = Path("./processed")


@dataclass
class SplitConfig:
    """Train/val/test split behavior for a single-disease binary task."""

    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42


@dataclass
class ModelConfig:
    """Backbone / classification head configuration."""

    dropout: float = 0.3
    num_classes: int = 2
    pretrained: bool = True
    feature_node: str = "features.denseblock4"
    logits_node: str = "classifier"


@dataclass
class ExplanationLossConfig:
    """Grad-CAM explanation loss configuration. Set `alpha=0.0` to fall
    back to a pure BCE/cross-entropy baseline without disabling the code
    path (the loss term is just weighted to zero)."""

    enabled: bool = True
    alpha: float = 1.0
    quantile: float = 0.5
    temperature: float = 0.1
    score_mode: str = "alg"  # "alg" | "abs" | "sqr"
    use_probs: bool = False
    only_positive_samples: bool = True  # only supervise samples that have an annotation mask


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 2e-4
    weight_decay: float = 1e-4
    train_batch_size: int = 12
    eval_batch_size: int = 12
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RunConfig:
    """Where results for a given run/disease get written."""

    runs_dir: Path = Path("./runs")

    def disease_dir(self, disease: str) -> Path:
        path = self.runs_dir / disease.replace(" ", "_").lower()
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class ExperimentConfig:
    """Top-level bundle passed around the pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    explanation_loss: ExplanationLossConfig = field(default_factory=ExplanationLossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)
    disease: str = "Aortic enlargement"
