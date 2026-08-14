"""Grad-CAM explanation loss and the combined training objective
L = CE + alpha * L_exp."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig


class ExplanationLoss(nn.Module):
    """
    Grad-CAM-guided explanation loss: penalizes saliency mass that falls
    outside expert-annotated target regions.
    """

    def __init__(self, quantile: float = 0.5, score_mode: str = "alg", use_probs: bool = False, temperature: float = 0.5 ,eps: float = 1e-8):
        super().__init__()
        if score_mode not in ("alg", "abs", "sqr"):
            raise ValueError(f"Unknown score mode: {score_mode}")
        self.quantile = quantile
        self.score_mode = score_mode
        self.use_probs = use_probs
        self.temperature = temperature
        self.eps = eps

    def classification_score(self, logits: torch.Tensor) -> torch.Tensor:
        values = logits.softmax(dim=1) if self.use_probs else logits
        diff = values[:, 1] - values[:, 0]
        if self.score_mode == "alg":
            return diff
        elif self.score_mode == "abs":
            return diff.abs()
        else:  # "sqr"
            return diff.square()

    def gradcam_gradients(self, scores: torch.Tensor, feature_map: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        return torch.autograd.grad(outputs=scores.sum(), inputs=feature_map, create_graph=create_graph)[0]

    def gradcam_weights(self, gradients: torch.Tensor) -> torch.Tensor:
        return gradients.mean(dim=(-2, -1), keepdim=True)

    def gradcam_heatmap(self, weights: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        heatmap = (weights * feature_map).sum(dim=1)
        return F.relu(heatmap)

    def minmax_normalize(self, heatmap: torch.Tensor) -> torch.Tensor:
        min_v = heatmap.amin(dim=(-2, -1), keepdim=True)
        max_v = heatmap.amax(dim=(-2, -1), keepdim=True)
        return (heatmap - min_v) / (max_v - min_v + self.eps)


    def soft_mask(self, heatmap):
        threshold = torch.quantile(
            heatmap.flatten(1), self.quantile, dim=1
        )
        return torch.sigmoid(
            (heatmap - threshold[:, None, None]) / self.temperature
        )

    def get_heatmaps(self, scores: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        gradients = self.gradcam_gradients(scores, feature_map)
        weights = self.gradcam_weights(gradients)
        heatmap = self.gradcam_heatmap(weights, feature_map)
        heatmap = self.minmax_normalize(heatmap)
        return heatmap

    def per_sample_loss(self, logits: torch.Tensor, feature_map: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Same computation as `forward`, but unreduced (per-sample), so
        callers can select/weight a subset of the batch *after* Grad-CAM
        has been computed. Grad-CAM must run on the full, unsliced batch
        that `logits`/`feature_map` came from -- slicing before this call
        disconnects the score from the graph that produced it."""
        scores = self.classification_score(logits)
        heatmap = self.get_heatmaps(scores, feature_map)
        heatmap = self.soft_mask(heatmap)
        if target_masks.shape[-2:] != heatmap.shape[-2:]:
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(), size=heatmap.shape[-2:], mode="nearest",
            ).squeeze(1)

        inside = (heatmap * target_masks).sum(dim=(-2, -1))
        total = heatmap.sum(dim=(-2, -1))
        return 1.0 - inside / (total + self.eps)

    def forward(self, logits: torch.Tensor, feature_map: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Mean fraction of selected saliency mass outside target regions.
        See `per_sample_loss` if you need to select a subset of the batch
        (e.g. only annotated samples) -- do the selection on its output,
        not on `logits`/`feature_map` before calling this.
        """
        return self.per_sample_loss(logits, feature_map, target_masks).mean()


class CombinedLoss(nn.Module):
    """L_total = CrossEntropy(logits, labels) + alpha * L_exp

    Wraps `ExplanationLoss` and controls whether/how it's applied:
      - `config.enabled=False` or `alpha=0` -> pure classification loss
        (equivalent to the paper's "Pure BCE" baseline).
      - `config.only_positive_samples=True` -> the explanation term is
        computed only over samples whose annotation mask is non-empty,
        matching the paper (explanation supervision only applies to
        disease-positive, box-annotated samples).
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.explanation_loss = ExplanationLoss(
            quantile=config.quantile, score_mode=config.score_mode, use_probs=config.use_probs, temperature=config.temperature
        )

    def forward(self, logits, feature_map, labels, target_masks) -> tuple:
        """Returns (total_loss, {"cls": ..., "exp": ...}) so callers can
        log each component separately."""
        cls_loss = F.cross_entropy(logits, labels)

        if not self.config.enabled or self.config.alpha == 0.0:
            zero = torch.zeros((), device=logits.device)
            return cls_loss, {"cls": cls_loss.detach(), "exp": zero}

        # Grad-CAM must see the full batch (gradients are computed against
        # the feature map that actually produced `logits`); selection of
        # which samples count toward the loss happens after, on the
        # per-sample loss values, not on logits/feature_map themselves.
        per_sample = self.explanation_loss.per_sample_loss(logits, feature_map, target_masks)

        if self.config.only_positive_samples:
            has_annotation = target_masks.flatten(1).any(dim=1)
            if not has_annotation.any():
                zero = torch.zeros((), device=logits.device)
                return cls_loss, {"cls": cls_loss.detach(), "exp": zero}
            per_sample = per_sample[has_annotation]

        exp_loss = per_sample.mean()
        total = cls_loss + self.config.alpha * exp_loss
        return total, {"cls": cls_loss.detach(), "exp": exp_loss.detach()}
