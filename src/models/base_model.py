"""
Abstract base class for all fracture-classification models.

Subclass contract:
    1. Call super().__init__(num_classes, model_name, input_size)
    2. Set self.classifier_head = <final Linear layer>
    3. Implement forward(x) -> logits of shape (B, num_classes)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for fracture-classification models.

    Subclasses must:
        1. Call ``super().__init__(num_classes, model_name, input_size)``.
        2. Assign the final classification layer to ``self.classifier_head``.
        3. Implement ``forward(x)`` returning logits of shape ``(B, num_classes)``.

    Attributes:
        num_classes: Number of target output classes.
        model_name: Identifier name for the architecture.
        input_size: Expected spatial resolution (height/width) of input images.
        classifier_head: Final linear classification layer set by subclasses.
    """

    def __init__(
            self,
            num_classes: int = 2,
            model_name: str = "unnamed",
            input_size: int = 224
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.input_size = input_size
        self.classifier_head: Optional[nn.Module] = None

    # ── Abstract API ──────────────────────────────────────────
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass logits of shape ``(B, num_classes)``."""
        raise NotImplementedError

    # ── Parameter counting ────────────────────────────────────
    def get_num_params(self,trainable_only: bool = True) -> int:
        """Return the total number of model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ── Freeze / unfreeze ─────────────────────────────────────
    def freeze_backbone(self) -> None:
        """Freeze all backbone weights, leaving only ``classifier_head`` trainable."""
        if self.classifier_head is None:
            raise RuntimeError(
                f"{self.__class__.__name__} did not set self.classifier_head."
            )

        for p in self.parameters():
            p.requires_grad = False
        for p in self.classifier_head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        """Enable gradient computation for all model parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_by_name(self, pattern: str) -> int:
        """Unfreeze params whose name contains `pattern`.
        Use after freeze_backbone() to selectively unfreeze deeper layers:
            model.freeze_backbone()
            model.unfreeze_by_name("layer4")   # unfreeze ResNet's last block

        Returns:
            Number of parameters unfrozen."""
        n = 0
        for name, param in self.named_parameters():
            if pattern in name and not param.requires_grad:
                param.requires_grad = True
                n += param.numel()
        return n

    # ── AdamW parameter groups ────────────────────────────────
    def get_parameter_groups(self, weight_decay: float = 0.01) -> List[Dict]:
        decay, no_decay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]

    # ── Nice printing ─────────────────────────────────────────
    def __repr__(self) -> str:
        """Return a concise summary string of model metadata and parameter counts."""
        total = self.get_num_params(trainable_only=False)
        trainable = self.get_num_params(trainable_only=True)
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"num_classes={self.num_classes}, "
            f"input_size={self.input_size}, "
            f"params={total/1e6:.2f}M, "
            f"trainable={trainable/1e6:.2f}M)"
        )   