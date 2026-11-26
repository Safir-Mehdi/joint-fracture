"""
Plot confusion matrices for trained models.

Reads predictions_test.json from each run's folder.
Saves:
    - One PNG per model (results/figures/confusion_matrices/<model>.png)
    - One combined grid PNG (results/figures/confusion_matrices_all.png)

Usage:
    from src.evaluation.confusion_matrix import plot_all_confusion_matrices

    plot_all_confusion_matrices()
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.utils.file_io import ensure_dir, load_yaml
from src.utils.logger import get_logger

CLASS_NAMES = ["Fractured", "Normal"]
COLORS = {0: "crimson", 1: "seagreen"}

# ═══════════════════════════════════════════════════════════════
# Single confusion matrix
# ═══════════════════════════════════════════════════════════════
def plot_single_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    model_name: str,
    save_path: Path,
    class_names: list[str] = CLASS_NAMES,
):
    """
    Plot one confusion matrix as a heatmap with counts + percentages.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = cm.sum()
    cm_pct = (cm / total * 100) if total > 0 else cm.astype(float)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold")

    # Cell annotations: count + percentage
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color=color, fontsize=14, fontweight="bold",
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return cm


# ═══════════════════════════════════════════════════════════════
# Load predictions from a run
# ═══════════════════════════════════════════════════════════════
def _load_predictions(run_dir: Path) -> dict | None:
    """Load predictions_test.json from a run folder."""
    pred_path = run_dir / "predictions_test.json"
    if not pred_path.exists():
        return None
    try:
        return json.loads(pred_path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# Grid of all confusion matrices
# ═══════════════════════════════════════════════════════════════
def plot_all_confusion_matrices(
    experiments_dir: str | Path = "experiments",
    results_dir: str | Path = "results",
    class_names: list[str] = CLASS_NAMES,
):
    """
    Scan experiments/, generate one matrix per model + a combined grid.

    Skips runs without predictions_test.json.
    If multiple runs exist per model, uses the most recent.
    """

    logger = get_logger("confusion_matrix")
    experiments_dir = Path(experiments_dir)
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures" / "confusion_matrices"
    ensure_dir(fig_dir)

    logger.info(f"Scanning: {experiments_dir}")
    run_dirs = sorted(
        [p for p in experiments_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    )
    logger.info(f"Found {len(run_dirs)} run(s)")

    # ── Collect predictions per model (latest run wins) ──
    per_model: dict[str, dict] = {}
    for rd in run_dirs:
        preds = _load_predictions(rd)
        if preds is None:
            logger.info(f"  ⚠️  Skipping {rd.name} — no predictions_test.json")
            continue

        # Model name from config
        cfg_path = rd / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = load_yaml(cfg_path)
        model_name = cfg.get("model", {}).get("name", "unknown")

        # Save latest (sorted means last one wins)
        per_model[model_name] = {
            "run_dir": rd,
            "y_true": preds["y_true"],
            "y_pred": preds["y_pred"],
            "y_prob": preds.get("y_prob"),
        }
        logger.info(f"  ✅ {rd.name:40s}  model={model_name:18s}")

        if not per_model:
            logger.warning("No runs with predictions found.")
            return

        # ── 1. Plot per-model matrices ──
        logger.info(f"\nPlotting {len(per_model)} per-model confusion matrices...")
        for model_name, data in per_model.items():
            out = fig_dir / f"{model_name}.png"
            plot_single_confusion_matrix(
                data["y_true"], data["y_pred"], model_name, out, class_names,
            )
            logger.info(f"  Saved: {out}")

        # ── 2. Combined grid ──
    n = len(per_model)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_name, data) in enumerate(sorted(per_model.items())):
        ax = axes[idx]
        cm = confusion_matrix(data["y_true"], data["y_pred"], labels=[0, 1])
        total = cm.sum()
        cm_pct = (cm / total * 100) if total > 0 else cm.astype(float)

        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_title(model_name, fontsize=10, fontweight="bold")

        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_pct[i, j]
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color=color, fontsize=9,
                )

    # Hide unused axes
    for idx in range(len(per_model), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Confusion Matrices — All Models (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    grid_path = results_dir / "figures" / "confusion_matrices_all.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"\nSaved combined grid: {grid_path}")

    # ── Return summary ──
    print(f"\n{'=' * 60}")
    print(f"Generated {len(per_model)} confusion matrices")
    print(f"  Individual PNGs: {fig_dir}")
    print(f"  Combined grid:   {grid_path}")
    print(f"{'=' * 60}")