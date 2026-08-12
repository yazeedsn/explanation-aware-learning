"""Model definition. A DenseNet121 classifier that also exposes its
last conv feature map, needed by the Grad-CAM explanation loss."""

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.models.feature_extraction import create_feature_extractor

from .config import ModelConfig


class DenseNet121WithFeatureMap(nn.Module):
    """DenseNet121 with a fresh binary classification head, returning
    both logits and the final-block feature map in one forward pass."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if config.pretrained else None
        backbone = densenet121(weights=weights)

        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential( # pyright: ignore[reportAttributeAccessIssue]
            nn.Dropout(config.dropout),
            nn.Linear(in_features, config.num_classes),
        )

        self.model = create_feature_extractor(
            backbone,
            return_nodes={config.feature_node: "feature_map", config.logits_node: "logits"},
        )

    def forward(self, x):
        outputs = self.model(x)
        return outputs["logits"], outputs["feature_map"]


def build_model(config: ModelConfig) -> nn.Module:
    model = DenseNet121WithFeatureMap(config)
    for param in model.parameters():  # fine-tune end-to-end, as in the paper
        param.requires_grad = True
    return model
