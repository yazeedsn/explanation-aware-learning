"""Plotting for training history. Reads whatever metric columns are
present (train_X / val_X pairs)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _paired_metric_names(columns: list) -> list:
    """Finds metric names that have both a train_<name> and val_<name>
    column, preserving order of first appearance."""
    names = []
    for col in columns:
        if col.startswith("train_"):
            name = col[len("train_"):]
            if f"val_{name}" in columns and name not in names:
                names.append(name)
    return names


def plot_history(hist, output_dir: Path, show: bool = True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hist = pd.DataFrame(hist) if isinstance(hist, dict) else hist
    epochs = hist["epoch"] if "epoch" in hist else range(1, len(hist) + 1)

    for name in _paired_metric_names(list(hist.columns)):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, hist[f"train_{name}"], label="Train")
        ax.plot(epochs, hist[f"val_{name}"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name.capitalize())
        ax.set_title(f"Training and Validation {name.capitalize()}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)
