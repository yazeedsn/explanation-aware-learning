"""
Everything is threaded through these configs, so
switching datasets / preprocessing / experiments means editing this file
(or constructing a new config in a script).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    processed_dir: Path = Path("./processed")           # path of processed data to load
    checkpoint_dir: Path = Path("./checkpoints")        # path to save model checkpoints, figures, plots.
    epochs: int = 10                                    # total number of traing epochs
    lr: float = 2e-4                                    # learning rate for the optimizer
    weight_decay: float = 1e-4                          # weight_decay used training regulization
    train_batch_size: int = 12                          # batch_size for the training dataloader.
    eval_batch_size: int = 12                           # batch_size for validation and test dataloaders
    num_workers: int = 4                                # number of workers for the dataloaders
    val_fraction: float = 0.1                           # portion of the validation split
    test_fraction: float = 0.1                          # portion of the test split
    seed: int = 42                                      # random seed for the experiment
    device: str = "auto"                                # device to use for training
    dropout: float = 0.3                                # dropout before the final classification layer.
    pretrained: bool = True                             # use a Densenet model
    feature_node: str = "features.denseblock4"          # name of the layer to use for getting the feature maps
    logits_node: str = "classifier"                     # name of the layer to use for getting the logits
    enabled: bool = True                                # use explanation loss in the training
    alpha: int = 1                                      # weight of the explanation loss
    quantile: float = 0.5                               # top largest #qunatile of the gradients to consider in the explanation loss
    temperature: float = 0.5                            # temperature for the soft masking the gradients.
    score_mode: str = "sqr"                             # alg: z1 - z0, abs: |z1 - z0|, sqr: (z1-z0)^2 z1, z0 is the logits of positive class and negative class, respectivly.
    use_probs: bool = False                             # use probabilities instead of logits (applies sigmoid to logits)
    only_positive_samples: bool = True                  # apply explination to positive class only (recommendation: always keep True)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
