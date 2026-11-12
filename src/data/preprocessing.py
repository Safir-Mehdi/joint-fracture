"""
Deterministic transforms applied to ALL splits (train/val/test).
Only: resize → tensor → normalize. Nothing random.

Note: Random augmentation lives in augmentation.py.
"""

import yaml
from pathlib import Path
from torchvision import transforms
from typing import Optional, List

CONFIG_PATH = "configs/base_config.yaml"

def load_data_stats(config_path=CONFIG_PATH) -> tuple[list, list]:
    """Return (mean, std) from base_config.yaml — kept for backwards-compat."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg["data"]["mean"], cfg["data"]["std"]

def get_base_transform(
        image_size: int,
        mean: Optional[List[float]]=None,
        std: Optional[List[float]]=None) -> transforms.Compose:
    """
    Creates a pipeline to resize, convert to tensor, and normalize an image.

    Loads `mean` and `std` from `base_config.yaml` if not provided.

    Args:
        image_size: Target width and height in pixels (e.g., 224).
        mean: RGB mean values for normalization.
        std: RGB standard deviation values for normalization.

    Returns:
        transforms.Compose: The image preprocessing pipeline.
    """
    if mean is None or std is None:
        mean, std = load_data_stats()

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])