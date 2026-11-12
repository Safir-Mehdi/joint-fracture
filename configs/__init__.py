"""
Config package — re-exports load_config for convenient importing.

Usage:
    configs/
        ├── __init__.py                       ← re-exports load_config
        ├── base_config.yaml                  ← load_config()              (default)
        ├── models/
        │   ├── cnn_baseline.yaml             ← load_config(model="cnn_baseline")
        │   ├── resnet50.yaml                 ← load_config(model="resnet50")
        │   ├── vgg16.yaml                    ← load_config(model="vgg16")
        │   ├── efficientnet_b0.yaml          ← load_config(model="efficientnet_b0")
        │   ├── densenet121.yaml              ← load_config(model="densenet121")
        │   ├── vit_b16.yaml                  ← load_config(model="vit_b16")
        │   └── yolov8.yaml                   ← load_config(model="yolov8")
        └── training/
            ├── default_train.yaml            ← load_config(training=True)
            └── hyperparameter_search.yaml    ← ***(read separately by the tuning script, not here)***
"""

from src.utils.config_loader import load_config

__all__ = ["load_config"]