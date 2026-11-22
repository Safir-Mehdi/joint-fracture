"""
DenseNet121 — dense CNN (2017) for fracture classification.

Wraps torchvision.models.densenet121, replaces the final classifier
to output num_classes logits, and exposes the head via BaseModel.classifier_head.

Params: ~7.98M (full), ~2,050 head-only after freeze.

Note: smallest pretrained model in the comparison — fastest to train.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from src.models.base_model import BaseModel


class DenseNetModel(BaseModel):
    """
    DenseNet121 wrapper for fracture classification.

    Args:
        num_classes: number of output classes (2 for fractured vs normal)
        pretrained:  if True, loads ImageNet weights
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(
            num_classes=num_classes,
            model_name="densenet121",
            input_size=224,
        )

        # Load backbone (with or without ImageNet weights)
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = densenet121(weights=weights)

        # Replace final classifier → new head for num_classes
        # DenseNet's head is a single Linear layer named `classifier`
        in_features = self.backbone.classifier.in_features   # 1024
        self.backbone.classifier = nn.Linear(in_features, num_classes)

        # Expose head for BaseModel.freeze_backbone()
        self.classifier_head = self.backbone.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        return self.backbone(x)