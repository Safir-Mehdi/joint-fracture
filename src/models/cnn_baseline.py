"""
Small from-scratch CNN baseline for fracture classification.

Architecture:
    4 × [Conv → BN → ReLU → MaxPool]  →  AdaptiveAvgPool  →  Dropout  →  Linear

No pretrained weights — this is a true baseline showing what a simple
CNN can learn on the fracture dataset without transfer learning.

Params: ~1.2M
"""

import warnings
import torch
import torch.nn as nn
from src.models.base_model import BaseModel

class CNNBaseline(BaseModel):
    """
    Simple 4-block CNN.

    Args:
        num_classes: number of output classes (2 for fractured vs normal)
        pretrained:  ignored for this model (no pretrained weights exist)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(num_classes=num_classes, model_name="cnn_baseline", input_size=224)

        if pretrained:
            warnings.warn(
                "cnn_baseline has no pretrained weights — 'pretrained' is ignored.",
                stacklevel=2,
            )

        # Feature extractor — 4 blocks
        self.features = nn.Sequential(
            self._block(3,   32),      # (B, 32, 112, 112)
            self._block(32,  64),      # (B, 64,  56,  56)
            self._block(64,  128),     # (B, 128, 28,  28)
            self._block(128, 256),     # (B, 256, 14,  14)
        )

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))        # (B, 256, 1, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier_head = nn.Linear(256, num_classes)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        """One conv block: Conv → BN → ReLU → MaxPool."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_classes)."""
        x = self.features(x)          # (B, 256, 14, 14)
        x = self.pool(x)              # (B, 256, 1, 1)
        x = torch.flatten(x, 1)       # (B, 256)
        x = self.dropout(x)
        x = self.classifier_head(x)   # (B, num_classes)
        return x