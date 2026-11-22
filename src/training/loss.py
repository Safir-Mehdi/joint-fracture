"""
Loss functions for fracture classification.

Supported:
    - focal          (default) — handles 2.5:1 class imbalance
    - cross_entropy  — baseline for ablation

Usage:
    from src.training.loss import build_loss

    # Without class weights
    loss_fn = build_loss(cfg)

    # With class weights computed from training data
    loss_fn = build_loss(cfg, class_weights=[1.73, 0.70])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Focal Loss ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance (Lin et al., 2017).

    Down-weights easy examples so training focuses on hard,
    misclassified ones. Standard choice for imbalanced medical data.

    Args:
        gamma:  focusing parameter (default 2.0). Higher → more focus on hard examples.
        alpha:  class balancing weight (scalar or tensor of shape [num_classes]).
                If None, no alpha weighting — use class_weights via `weight` instead.
        weight: per-class weights (like nn.CrossEntropyLoss(weight=...)).
    """

    def __init__(
            self,
            gamma: float = 2.0,
            alpha: float | None = None,
            weight: torch.Tensor | None = None
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if alpha is not None and not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("weight", weight) if weight is not None else None
        self.weight = weight    # ← also set as attr for easy access

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes) — raw scores (not softmaxed)
            targets: (B,) — integer class labels
        """

        # Per-sample CE (no reduction yet)
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")

        # p_t = probability of the true class
        pt = torch.exp(-ce)

        # Focal modulation
        focal = (1.0 - pt) ** self.gamma * ce

        # Optional alpha weighting
        if self.alpha is not None:
            focal = self.alpha * focal

        return focal.mean()

# ── Factory ────────────────────────────────────────────────
def build_loss(
        cfg: dict,
        class_weights: list[float] | None = None,
) -> nn.Module:
    """
    Build a loss function from config.

    Args:
        cfg: merged config (reads cfg["loss"])
        class_weights: optional per-class weights.
                       If provided, converted to tensor for the loss.

    Returns:
        nn.Module loss function.

    Raises:
        ValueError: if loss name is unknown or params are invalid.
    """

    loss_cfg = cfg["loss"]
    name = loss_cfg.get("name", "focal")

    # Convert class weights (if any) to tensor
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # ── Focal Loss ──
    if name == "focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        alpha = loss_cfg.get("focal_alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha, weight=weight_tensor)

    # ── Cross-Entropy ──
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight_tensor)

    raise ValueError(
        f"Unknown loss '{name}'. Available: focal, cross_entropy"
    )