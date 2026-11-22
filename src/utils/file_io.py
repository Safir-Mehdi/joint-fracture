import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_run_dir(
    base_dir: str | Path = "experiments",
    model_name: str | None = None,
) -> Path:
    """
    Create a timestamped run folder with subfolders.

    Returns:
        experiments/run_YYYYMMDD_HHMMSS[_modelname]/
        ├── logs/
        ├── checkpoints/
        └── plots/
    """
    base = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    if model_name:
        run_id += f"_{model_name}"

    run_dir = ensure_dir(base / run_id)
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "plots")
    return run_dir


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Save object as JSON, creating parent dirs."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(obj: Any, path: str | Path) -> None:
    """Save object as YAML, creating parent dirs."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def load_yaml(path: str | Path) -> Any:
    """Load YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)