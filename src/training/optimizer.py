"""
Optimizer and scheduler factory.

Usage:
    from src.training.optimizer import build_optimizer

    optimizer, scheduler = build_optimizer(model, cfg)
    # In training loop:
    #   if isinstance(scheduler, ReduceLROnPlateau):
    #       scheduler.step(val_loss)
    #   else:
    #       scheduler.step()

Supported optimizers:
    - adamw  (default) — AdamW with decoupled weight decay
    - adam             — classic Adam
    - sgd              — SGD with momentum

Supported schedulers:
    - cosine_annealing   (default) — smooth cosine LR
    - step               — step decay every N epochs
    - reduce_on_plateau  — reduce when metric stops improving
    - none               — constant LR
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from torch.optim import Optimizer
from src.models.base_model import BaseModel
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR
)
# ── Optimizer factory ──────────────────────────────────────
def _build_optimizer_inner(
        model: BaseModel,
        name: str,
        lr: float,
        weight_decay: float,
        momentum: float = 0.9
) -> Optimizer:
    """Internal — build just the optimizer (no scheduler)."""
    param_groups = model.get_parameter_groups(weight_decay=weight_decay)

    if name == "adamw":
        return torch.optim.AdamW(params=param_groups, lr=lr)

    if name == "adam":
        return torch.optim.Adam(params=param_groups, lr=lr)

    if name == "sgd":
        return torch.optim.SGD(params=param_groups, lr=lr, momentum=momentum)
    raise ValueError(
        f"Unknown optimizer '{name}'. Available: adamw, adam, sgd"
    )

# ── Warmup + cosine ────────────────────────────────────────
def _warmup_cosine_lambda(
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0,
):
    """
    LR multiplier as a function of epoch:
        epoch < warmup_epochs: linear 0 → 1
        epoch >= warmup_epochs: cosine 1 → min_lr_ratio
    """
    def _lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return _lambda

def _build_scheduler(
        optimizer: Optimizer,
        name: str,
        cfg: dict,
) -> Optional[LRScheduler]:
    """Internal — build scheduler from config."""
    if name == "none":
        return None

    sched_cfg = cfg.get("scheduler", {})
    total_epochs = cfg["training"]["epochs"]
    if name == "cosine_annealing":
        warmup = sched_cfg.get("warmup_epochs", 0)
        min_lr = sched_cfg.get("min_lr", 0.0)
        base_lr = cfg["optimizer"]["lr"]
        min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

        if warmup > 0:
            return LambdaLR(
                optimizer,
                lr_lambda=_warmup_cosine_lambda(warmup, total_epochs, min_lr_ratio),
            )

        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    if name == "step":
        step_size = sched_cfg.get("step_size", 10)
        gamma = sched_cfg.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "reduce_on_plateau":
        factor = sched_cfg.get("factor", 0.1)
        patience = sched_cfg.get("patience", 5)
        return ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)

    raise ValueError(
        f"Unknown scheduler '{name}'. "
        f"Available: cosine_annealing, step, reduce_on_plateau, none"
    )

# ── Public API ─────────────────────────────────────────────
def build_optimizer(
    model: BaseModel,
    cfg: dict,
) -> tuple[Optimizer, Optional[LRScheduler]]:
    """
    Build optimizer + scheduler from config.

    Args:
        model: BaseModel instance (uses model.get_parameter_groups()).
        cfg:   merged config. Reads:
                 cfg["optimizer"]["name"], ["lr"], ["weight_decay"]
                 cfg["scheduler"]["name"]
                 cfg["training"]["epochs"]

    Returns:
        (optimizer, scheduler_or_None)
    """

    opt_cfg = cfg["optimizer"]
    opt_name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    momentum = float(opt_cfg.get("momentum", 0.9))

    if lr <= 0:
        raise ValueError(f"optimizer.lr must be > 0, got {lr}")

    optimizer = _build_optimizer_inner(
        model=model,
        name=opt_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    sched_name = cfg.get("scheduler", {}).get("name", "cosine_annealing").lower()
    scheduler = _build_scheduler(optimizer, sched_name, cfg)

    return optimizer, scheduler