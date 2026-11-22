# src/models/convnext.py
"""
ConvNeXt-Tiny — modern CNN (2022) for fracture classification.

Wraps torchvision.models.convnext_tiny, replaces the final classifier
to output num_classes logits, and exposes the head via BaseModel.classifier_head.

Params: ~28.6M (full), ~1,538 head-only after freeze.

Note: ConvNeXt borrows design ideas from Transformers (LayerNorm, GELU,
      depthwise convs) while staying purely convolutional. Strong baseline
      for "modern CNN" in the comparison.
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from src.models.base_model import BaseModel


class ConvNeXtModel(BaseModel):
    """
    ConvNeXt-Tiny wrapper for fracture classification.

    Args:
        num_classes: number of output classes (2 for fractured vs normal)
        pretrained:  if True, loads ImageNet weights
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(
            num_classes=num_classes,
            model_name="convnext_tiny",
            input_size=224,
        )

        # Load backbone (with or without ImageNet weights)
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)

        # Replace final classifier layer → new head for num_classes
        # ConvNeXt head is classifier[2] (LayerNorm → Flatten → Linear)
        in_features = self.backbone.classifier[2].in_features   # 768
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

        # Expose head for BaseModel.freeze_backbone()
        self.classifier_head = self.backbone.classifier[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        return self.backbone(x)