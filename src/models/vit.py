"""
ViT-B/16 — Vision Transformer (2020) for fracture classification.

Wraps torchvision.models.vit_b_16, replaces the final heads.head
to output num_classes logits, and exposes the head via BaseModel.classifier_head.

Params: ~85.8M (full), ~1,538 head-only after freeze.

⚠️ Note: ViT-B/16 pretrained weights expect 384×384 input (not 224).
         This model uses input_size=384 by default.

Pretrained weights: IMAGENET1K_SWAG_E2E_V1 — trained on 3.6B images at 384×384.
Best transfer-learning baseline for fracture classification.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from src.models.base_model import BaseModel


class ViTModel(BaseModel):
    """
    ViT-B/16 wrapper for fracture classification.

    Args:
        num_classes: number of output classes (2 for fractured vs normal)
        pretrained:  if True, loads ImageNet SWAG-E2E weights (384×384)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(
            num_classes=num_classes,
            model_name="vit_b16",
            input_size=384,          # ⚠️ ViT-B/16 pretrained at 384×384
        )

        # Load backbone (with or without ImageNet weights)
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        self.backbone = vit_b_16(weights=weights)

        # Replace final classifier head → new head for num_classes
        # ViT's head is a nested Linear: backbone.heads.head
        in_features = self.backbone.heads.head.in_features   # 768
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

        # Expose head for BaseModel.freeze_backbone()
        self.classifier_head = self.backbone.heads.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        return self.backbone(x)