"""
Model factory — config-driven only.

Single source of truth: models are ALWAYS built from a merged config dict
(see src/utils/config_loader.py). This guarantees reproducibility —
same config = same model, every time.

Usage:
    from src.utils.config_loader import load_config
    from src.models.model_factory import build_model

    cfg = load_config(model="resnet50")      # merges YAMLs
    model = build_model(cfg)                 # reads cfg["model"]

The cfg must contain:
    cfg["model"]["name"]         → "resnet50"
    cfg["model"]["num_classes"]  → 2
    cfg["model"]["pretrained"]   → True
"""

from typing import Type
from src.models.base_model import BaseModel
from src.models.cnn_baseline import CNNBaseline
from src.models.alexnet import AlexNetModel
from src.models.resnet import ResNetModel
from src.models.vgg import VGGModel
from src.models.densenet import DenseNetModel
from src.models.efficientnet import EfficientNetModel
from src.models.mobilenet import MobileNetModel
from src.models.convnext import ConvNeXtModel
from src.models.vit import ViTModel
from src.models.swin import SwinModel


# ── Custom exception ─────────────────────────────────────────
class ModelFactoryError(ValueError):
    """Raised when a model cannot be built from the given config."""
    pass


MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "cnn_baseline": CNNBaseline,
    "alexnet": AlexNetModel,
    "resnet50": ResNetModel,
    "vgg16": VGGModel,
    "densenet121":  DenseNetModel,
    "efficientnet_b3": EfficientNetModel,
    "mobilenet_v3": MobileNetModel,
    "convnext_tiny": ConvNeXtModel,
    "vit_b16": ViTModel,
    "swin_tiny": SwinModel,
}

def list_models() -> list[str]:
    """Return available model names (sorted)."""
    return sorted(MODEL_REGISTRY.keys())

def _format_available_models() -> str:
    """Format available models as an indented bullet list for error messages."""
    return "\n".join(f"-- {name}" for name in list_models())

def build_model(cfg: dict) -> BaseModel:
    """
    Build a model from a merged config dict.

    Args:
        cfg: merged config (must have cfg["model"]["name"], ["num_classes"], ["pretrained"])

    Returns:
        Instantiated model.

    Raises:
        ModelFactoryError: if cfg is malformed or the model name is unknown.
    """

    # ── 1. Validate 'model' section exists ──
    if not isinstance(cfg, dict):
        raise ModelFactoryError(
            f"Expected cfg to be a dict, got {type(cfg).__name__}. "
            "Call load_config() first."
        )

    if "model" not in cfg:
        raise ModelFactoryError(
            "Config is missing the required 'model' section. "
            "Expected: cfg['model'] = {'name': ..., 'num_classes': ..., 'pretrained': ...}"
        )

    model_cfg = cfg["model"]
    if not isinstance(model_cfg, dict):
        raise ModelFactoryError(
            f"cfg['model'] must be a dict, got {type(model_cfg).__name__}. "
            "Check the 'model:' block in your YAML file."
        )

    # ── 2. Validate 'name' key ──
    if "name" not in model_cfg:
        raise ModelFactoryError(
            "cfg['model'] is missing the required 'name' key. "
            f"Please add a model name. Available models:\n{_format_available_models()}"
        )

    name = model_cfg["name"]
    if not isinstance(name, str) or not name.strip():
        raise ModelFactoryError(
            f"cfg['model']['name'] must be a non-empty string, got {name!r}."
        )

    # ── 3. Validate name is registered ──
    if name not in MODEL_REGISTRY:
        raise ModelFactoryError(
            f"Unknown model '{name}'. Available models:\n{_format_available_models()}"
        )

    # ── 4. Build ──
    model_class = MODEL_REGISTRY[name]
    return model_class(
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True)
    )