"""
AlexNet — classic 2012 CNN baseline for fracture classification.

Wraps torchvision.models.alexnet, replaces the final classifier layer
to output num_classes logits, and exposes the head via BaseModel.classifier_head.

Params: ~61M (full), ~8,194 head-only after freeze.
"""

import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

from src.models.base_model import BaseModel


class AlexNetModel(BaseModel):
    """
    AlexNet wrapper for fracture classification.

    Args:
        num_classes: number of output classes (2 for fractured vs normal)
        pretrained:  if True, loads ImageNet weights
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(
            num_classes=num_classes,
            model_name="alexnet",
            input_size=224,
        )

        # Load backbone (with or without ImageNet weights)
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = alexnet(weights=weights)

        # Replace final classifier layer → new head for num_classes
        # AlexNet's head is classifier[6] (the last Linear in the Sequential)
        in_features = self.backbone.classifier[6].in_features   # 4096
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)

        # Expose head for BaseModel.freeze_backbone()
        self.classifier_head = self.backbone.classifier[6]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        return self.backbone(x)