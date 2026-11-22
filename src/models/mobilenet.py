"""
MobileNetV3-Large — lightweight CNN (2019) for fracture classification.

Wraps torchvision.models.mobilenet_v3_large, replaces the final classifier
to output num_classes logits, and exposes the head via BaseModel.classifier_head.

Params: ~5.48M (full), ~2,562 head-only after freeze.

Note: smallest and fastest model in the comparison — great for
      "accuracy vs efficiency" analysis and edge deployment discussion.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from src.models.base_model import BaseModel


class MobileNetModel(BaseModel):
    """
    MobileNetV3-Large wrapper for fracture classification.

    Args:
        num_classes: number of output classes (2 for fractured vs normal)
        pretrained:  if True, loads ImageNet weights
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(
            num_classes=num_classes,
            model_name="mobilenet_v3",
            input_size=224,
        )

        # Load backbone (with or without ImageNet weights)
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)

        # Replace final classifier layer → new head for num_classes
        # MobileNetV3 head is classifier[3] (Linear → Hardswish → Dropout → Linear)
        in_features = self.backbone.classifier[3].in_features   # 1280
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

        # Expose head for BaseModel.freeze_backbone()
        self.classifier_head = self.backbone.classifier[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        return self.backbone(x)