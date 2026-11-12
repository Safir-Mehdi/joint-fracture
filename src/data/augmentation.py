"""
Image augmentation pipeline for the training dataset.

Purpose:
    Builds a PyTorch `transforms.Compose` pipeline (`get_train_transform`) that
    applies random image augmentation (flips, rotation, brightness, contrast)
    from config, followed by tensor conversion and normalization.

Usage:
    from src.data.augmentation import get_train_transform

    train_tf = get_train_transform(image_size=224)
"""

from torchvision import transforms
from src.utils.config_loader import load_config
from src.data.preprocessing import load_data_stats      # The config for mean and std, could be load directly from configs/base_config.yaml.

def get_train_transform(image_size: int) -> transforms.Compose:
    """Applies random image augmentation for the training set using config params."""
    cfg = load_config(training=True)
    aug = cfg["augmentation"]["train"]
    mean, std = load_data_stats()

    return transforms.Compose([ 
        transforms.Resize((image_size, image_size)),
        # ── random augmentations ──
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=aug["brightness"],
                contrast=aug["contrast"],
            )
        ], p=aug["color_jitter_p"]),
        # ── base transforms ──
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])