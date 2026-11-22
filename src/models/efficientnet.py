"""
EfficientNet-B3 — efficient CNN (2019) for fracture classification.

Wraps torchvision.models.efficientnet_b3, replaces the final classifier
to output num_classes logits, and exposes the head via BaseModel.classifier_head.

Params: ~12.2M (full), ~3,074 head-only after freeze.

Note: EfficientNet-B3 expects 300×300 input by default (not 224).
      Configs can override input_size if GPU memory is tight.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from src.models.base_model import BaseModel


class EfficientNetModel(BaseModel):
    """
    EfficientNet-B3 wrapper for fracture classification.

    Args:
        num_classes: number of output classes (2 for fractured vs normal)
        pretrained:  if True, loads ImageNet weights
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(
            num_classes=num_classes,
            model_name="efficientnet_b3",
            input_size=300,          # ⚠️ B3 standard is 300, not 224
        )

        # Load backbone (with or without ImageNet weights)
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)

        # Replace final classifier layer → new head for num_classes
        # EfficientNet's head is classifier[1] (index 0 is Dropout)
        in_features = self.backbone.classifier[1].in_features   # 1536
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        # Expose head for BaseModel.freeze_backbone()
        self.classifier_head = self.backbone.classifier[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        return self.backbone(x)