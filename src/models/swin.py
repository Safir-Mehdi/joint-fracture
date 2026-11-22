"""
Swin-Tiny — hierarchical Vision Transformer (2021) for fracture classification.

Wraps torchvision.models.swin_t, replaces the final head
to output num_classes logits, and exposes the head via BaseModel.classifier_head.

Params: ~28.3M (full), ~1,538 head-only after freeze.

Note: Swin uses shifted-window attention → hierarchical features
      (like a CNN) + global attention (like ViT). Uses standard 224 input.
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

from src.models.base_model import BaseModel


class SwinModel(BaseModel):
    """
    Swin-Tiny wrapper for fracture classification.

    Args:
        num_classes: number of output classes (2 for fractured vs normal)
        pretrained:  if True, loads ImageNet weights
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(
            num_classes=num_classes,
            model_name="swin_tiny",
            input_size=224,
        )

        # Load backbone (with or without ImageNet weights)
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = swin_t(weights=weights)

        # Replace final head → new head for num_classes
        # Swin's head is a single Linear layer: backbone.head
        in_features = self.backbone.head.in_features   # 768
        self.backbone.head = nn.Linear(in_features, num_classes)

        # Expose head for BaseModel.freeze_backbone()
        self.classifier_head = self.backbone.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        return self.backbone(x)