import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchmetrics.aggregation import MeanMetric

from.losses import ExplanationLoss

class MetricBundle:
    """A small named collection of torchmetrics, updated/reset/read together.
    Extend this dict if you want extra metrics -- the trainer and history
    logging don't need to change."""

    def __init__(self, num_classes: int, device: str):
        self.loss = MeanMetric().to(device)
        self.acc = MulticlassAccuracy(num_classes=num_classes).to(device)
        self.auroc = MulticlassAUROC(num_classes=num_classes).to(device)
        self.cls_loss = MeanMetric().to(device)    # optinal
        self.exp_loss = MeanMetric().to(device)    # optinal

    def reset(self):
        for m in (self.loss, self.acc, self.auroc, self.cls_loss, self.exp_loss):
            m.reset()

    def update(self, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor,
        cls_loss: float | None = None, exp_loss: float | None = None):
        self.loss.update(loss, weight=labels.size(0))
        self.acc.update(logits, labels)
        self.auroc.update(logits, labels)
        if cls_loss is not None:
            self.cls_loss.update(cls_loss, weight=labels.size(0))
        if exp_loss is not None:
            self.exp_loss.update(exp_loss, weight=labels.size(0))

    def compute(self) -> dict:
        return {
            "loss": self.loss.compute().item(),
            "acc": self.acc.compute().item(),
            "auc": self.auroc.compute().item(),
            "cls_loss": self.cls_loss.compute().item(),
            "exp_loss": self.exp_loss.compute().item()
        }



class ExplanationMetricBundle:
    """
    Computes, per batch, three explanation-quality metrics (matching the
    paper's Eqs. 7-9) over disease-positive, annotated samples only
    (same selection rule as `CombinedLoss(only_positive_samples=True)`):

      - top_saliency_precision (Eq. 7): fraction of the top-`quantile`%
        saliency mask (the same hard top-k mask used internally by
        `ExplanationLoss`) that falls inside the annotation.
      - all_saliency_precision (Eq. 8): fraction of the *full*,
        unthresholded normalized saliency mass that falls inside the
        annotation -- captures overall concentration, not just the peak.
      - annotation_coverage (Eq. 9): whether the annotated region is
        acknowledged at all -- see the coverage caveat below.

    Coverage caveat: Eq. 9 in the paper is defined per individual
    bounding box. This implementation only has access to the per-disease
    *union* mask (see `vinbig_prep.masks.build_disease_masks`), so
    coverage here is approximated at the union-mask level: a sample
    counts as "covered" if the fraction of its top-k saliency pixels
    that fall inside the union mask is >= `coverage_threshold`.

    Uses the same score_mode/quantile/use_probs formulation as whatever
    `ExplanationLoss`/`CombinedLoss` config you trained with, so pass
    matching values if you want metrics that reflect the trained model's
    actual training-time saliency behavior.
    """

    def __init__(
        self,
        device,
        quantile: float = 0.5,
        temperature: float = 0.5,
        score_mode: str = "alg",
        use_probs: bool = False,
        coverage_threshold: float = 0.01,
        eps: float = 1e-8,
    ):
        self.explainer = ExplanationLoss(quantile=quantile, score_mode=score_mode, temperature=temperature, use_probs=use_probs, eps=eps)
        self.coverage_threshold = coverage_threshold
        self.eps = eps

        self.top_precision = MeanMetric().to(device)
        self.all_precision = MeanMetric().to(device)
        self.coverage = MeanMetric().to(device)

    def reset(self):
        self.top_precision.reset()
        self.all_precision.reset()
        self.coverage.reset()

    def update(self, logits: torch.Tensor, feature_map: torch.Tensor, target_masks: torch.Tensor):
        """
        Args:
            logits, feature_map: outputs of a *grad-enabled* forward pass
                (see module docstring -- must not come from
                torch.inference_mode()/torch.no_grad()).
            target_masks: (B, H, W) per-disease annotation masks, same
                tensor you'd pass to `ExplanationLoss`/`CombinedLoss`.
        """
        has_annotation = target_masks.flatten(1).any(dim=1)
        if not has_annotation.any():
            return  # nothing in this batch to evaluate explanation quality on

        # Grad-CAM must run on the full, unsliced batch (gradients are
        # computed against the feature map that actually produced
        # `logits`) -- selecting the annotated subset happens after, on
        # the resulting heatmap, not on logits/feature_map beforehand.
        scores = self.explainer.classification_score(logits)
        gradients = self.explainer.gradcam_gradients(scores, feature_map)
        weights = self.explainer.gradcam_weights(gradients)
        heatmap = self.explainer.gradcam_heatmap(weights, feature_map)
        heatmap = self.explainer.minmax_normalize(heatmap).detach()
        soft_mask = self.explainer.soft_mask(heatmap)

        heatmap = heatmap[has_annotation]
        soft_mask = soft_mask[has_annotation]
        target_masks = target_masks[has_annotation].float()

        if target_masks.shape[-2:] != heatmap.shape[-2:]:
            target_masks = F.interpolate(
                target_masks.unsqueeze(1), size=heatmap.shape[-2:], mode="nearest",
            ).squeeze(1)

        # Eq. 7: Top Saliency Precision
        top_num = (soft_mask * target_masks).sum(dim=(-2, -1))
        top_den = soft_mask.sum(dim=(-2, -1)) + self.eps
        top_precision = top_num / top_den

        # Eq. 8: All Saliency Precision
        all_num = (heatmap * target_masks).sum(dim=(-2, -1))
        all_den = heatmap.sum(dim=(-2, -1)) + self.eps
        all_precision = all_num / all_den

        # Eq. 9: Annotation coverage (approximated on the union mask; see class docstring)
        mask_area = target_masks.sum(dim=(-2, -1)) + self.eps
        salient_fraction_within_mask = (soft_mask * target_masks).sum(dim=(-2, -1)) / mask_area
        covered = (salient_fraction_within_mask >= self.coverage_threshold).float()

        self.top_precision.update(top_precision)
        self.all_precision.update(all_precision)
        self.coverage.update(covered)

    def compute(self) -> dict:
        return {
            "top_saliency_precision": self.top_precision.compute().item(),
            "all_saliency_precision": self.all_precision.compute().item(),
            "annotation_coverage": self.coverage.compute().item(),
        }
