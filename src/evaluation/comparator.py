"""
Compare all trained runs by reading experiments/run_*/metrics*.json.

Two APIs:
    - compare_all_runs(experiments_dir, results_dir) → (df_val, df_test)
    - print_summary(df)                              → terminal summary

Usage:
    from src.evaluation.comparator import compare_all_runs

    df_val, df_test = compare_all_runs()
    print(df_test)

Outputs:
    results/comparison_val.csv
    results/comparison_val.md
    results/comparison_test.csv
    results/comparison_test.md
"""

import json
from pathlib import Path

import pandas as pd

from src.utils.file_io import ensure_dir, load_yaml
from src.utils.logger import get_logger

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
# 7 metrics to show in the paper's markdown table
PAPER_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
]

# Sort by this metric (descending — higher is better)
SORT_BY = "f1"


# ═══════════════════════════════════════════════════════════════
# Load one run
# ═══════════════════════════════════════════════════════════════

def _load_run(run_dir: Path) -> dict | None:
    """
    Load a single run.

    Returns dict:
        {
            "run_id": "run_20241123_100000_resnet50",
            "model_name": "resnet50",
            "val": {accuracy: ..., f1: ..., ...},
            "test": {...} or None
        }
    or None if run is incomplete (missing config.yaml or metrics.json).
    """

    config_path = run_dir / "config.yaml"
    val_path = run_dir / "metrics.json"
    test_path = run_dir / "metrics_test.json"

    if not config_path.exists() or not val_path.exists():
        return None