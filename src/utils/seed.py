"""
Centralized reproducibility setup.

Call set_seed() ONCE at the start of every training / evaluation script,
before creating the model, dataloaders, or anything random.
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed all randomness sources so runs are reproducible.

    Args:
        seed: integer seed (default 42)
        deterministic: enforce deterministic CUDA ops (slight slowdown,
                       but required for reproducible research)
    """

    # Python & OS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(a=seed)


    # Numpy
    np.random.seed(seed=seed)

    # PyTorch CPU + GPU
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)

    # cuDNN deterministic
    if deterministic:
        # CUDA uses deterministic ops (same result every run — if True)
        torch.backends.cudnn.deterministic = True
        # Don't auto-tune algorithms — if False (auto-tuning is non-deterministic)
        torch.backends.cudnn.benchmark = False

    # Optional: for full determinism on some cuBLAS ops
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"