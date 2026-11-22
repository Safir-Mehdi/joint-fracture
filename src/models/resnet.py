"""
ResNet50 — ImageNet-pretrained baseline for fracture classification.

Wraps torchvision.models.resnet50, replaces the final fc layer to output
num_classes logits, and exposes the head via BaseModel.classifier_head.

Params: ~25.6M (full), ~2M head only after freeze.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from src.models.base_model import BaseModel

class ResNetModel(BaseModel):
    """
    ResNet50 wrapper for fracture classification.

    Args:
        num_classes: number of output classes (2 for fractured vs normal)
        pretrained:  if True, loads ImageNet weights
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(
            num_classes=num_classes,
            model_name="resnet50",
            input_size=224
        )

        # Load backbone with or without ImageNet weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Replace final fc layer → new head with num_classes outputs
        in_features = self.backbone.fc.in_features      # 2048 for resnet50
        self.backbone.fc = nn.Linear(in_features=in_features,out_features=num_classes)

        # Expose head for BaseModel.freeze_backbone()
        self.classifier_head = self.backbone.fc

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        return self.backbone(x)