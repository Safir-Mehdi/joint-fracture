"""
YAML config loader — single place to read project config.

Usage:
    from src.utils.config_loader import load_config
    cfg = load_config()                                  # base only
    cfg = load_config("base_config.yaml")                # same
    cfg = load_config(model="resnet50")                  # base + model merged
    cfg = load_config(model="resnet50", training=True)   # base + model + training
"""

import yaml
from src.utils.path import get_path
from pathlib import Path

CONFIG_DIR = get_path('configs')

def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(base: str = "base_config.yaml",
        model: str | None = None,
        training: bool = False) -> dict:
    """
    Load and merge config YAML files.

    Merge order (later overrides earlier):
        1. configs/base_config.yaml
        2. configs/models/<model>.yaml   (if `model` is given)
        3. configs/training/default_train.yaml  (if `training=True`)
    """
    merged = _read_yaml(CONFIG_DIR / base)

    if model:
        model_path = CONFIG_DIR / "models" / f"{model}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_path}")
        merged.update(_read_yaml(model_path))

    if training:
        train_path = CONFIG_DIR / "training" / "default_train.yaml"
        if train_path.exists():
            merged.update(_read_yaml(train_path))

    return merged