"""
Plot ROC and Precision-Recall curves for trained models.

Reads predictions_test.json from each run's folder.
Saves:
    - One PNG per model (results/figures/roc_curves/<model>.png)
    - One overlaid ROC curve PNG (results/figures/roc_curves_all.png)
    - One overlaid PR curve PNG (results/figures/pr_curves_all.png)

Usage:
    from src.evaluation.roc_curve import plot_all_curves
    plot_all_curves()
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils.file_io import ensure_dir, load_yaml
from src.utils.logger import get_logger


# ═══════════════════════════════════════════════════════════════
# Load predictions
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
# Single-model ROC + PR
# ═══════════════════════════════════════════════════════════════
def plot_single_curves(
    y_true: list[int],
    y_prob: list[float],
    model_name: str,
    save_path: Path,
    positive_class: int = 0,
):
    """
    Plot ROC + PR curves side by side for a single model.
    """
    # Binary target: 1 = positive class (fractured)
    y_bin = np.array([1 if t == positive_class else 0 for t in y_true])
    y_prob_arr = np.array(y_prob, dtype=float)

    # ── ROC ──
    fpr, tpr, _ = roc_curve(y_bin, y_prob_arr)
    roc_auc_val = roc_auc_score(y_bin, y_prob_arr)

    # ── PR ──
    precision, recall, _ = precision_recall_curve(y_bin, y_prob_arr)
    pr_auc_val = average_precision_score(y_bin, y_prob_arr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    axes[0].plot(fpr, tpr, color="crimson", lw=2,
                 label=f"AUC = {roc_auc_val:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title(f"{model_name} — ROC Curve", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # PR
    axes[1].plot(recall, precision, color="seagreen", lw=2,
                 label=f"AP = {pr_auc_val:.4f}")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title(f"{model_name} — Precision-Recall Curve",
                      fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return roc_auc_val, pr_auc_val


# ═══════════════════════════════════════════════════════════════
# Overlaid ROC + PR across all models
# ═══════════════════════════════════════════════════════════════
def plot_all_curves(
    experiments_dir: str | Path = "experiments",
    results_dir: str | Path = "results",
    positive_class: int = 0,
):
    """
    Scan experiments/, generate:
        - Per-model ROC+PR PNG
        - Overlaid ROC PNG (all models)
        - Overlaid PR PNG (all models)
    """
    logger = get_logger("roc_curve")
    experiments_dir = Path(experiments_dir)
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures" / "roc_curves"
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
            logger.info(f"  ⚠️  Skipping {rd.name} — no predictions")
            continue

        cfg_path = rd / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = load_yaml(cfg_path)
        model_name = cfg.get("model", {}).get("name", "unknown")

        per_model[model_name] = {
            "run_dir": rd,
            "y_true": preds["y_true"],
            "y_prob": preds["y_prob"],
        }
        logger.info(f"  ✅ {rd.name:40s}  model={model_name}")

    if not per_model:
        logger.warning("No runs with predictions found.")
        return

    # ── 1. Per-model ROC+PR ──
    logger.info(f"\nPlotting {len(per_model)} per-model curves...")
    model_aucs = {}
    for model_name, data in sorted(per_model.items()):
        out = fig_dir / f"{model_name}.png"
        roc_val, pr_val = plot_single_curves(
            data["y_true"], data["y_prob"], model_name, out, positive_class,
        )
        model_aucs[model_name] = {"roc_auc": roc_val, "pr_auc": pr_val}
        logger.info(
            f"  Saved: {out}  "
            f"(ROC-AUC={roc_val:.4f}  PR-AUC={pr_val:.4f})"
        )

    # ── 2. Overlaid ROC ──
    logger.info("\nPlotting overlaid ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")

    for idx, (model_name, data) in enumerate(sorted(per_model.items())):
        y_bin = np.array([1 if t == positive_class else 0 for t in data["y_true"]])
        y_prob = np.array(data["y_prob"], dtype=float)
        fpr, tpr, _ = roc_curve(y_bin, y_prob)
        roc_auc_val = roc_auc_score(y_bin, y_prob)
        ax.plot(
            fpr, tpr, lw=2, color=cmap(idx % 10),
            label=f"{model_name} (AUC={roc_auc_val:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models (Test Set)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = results_dir / "figures" / "roc_curves_all.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {roc_path}")

    # ── 3. Overlaid PR ──
    logger.info("Plotting overlaid PR curves...")
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, (model_name, data) in enumerate(sorted(per_model.items())):
        y_bin = np.array([1 if t == positive_class else 0 for t in data["y_true"]])
        y_prob = np.array(data["y_prob"], dtype=float)
        precision, recall, _ = precision_recall_curve(y_bin, y_prob)
        pr_auc_val = average_precision_score(y_bin, y_prob)
        ax.plot(
            recall, precision, lw=2, color=cmap(idx % 10),
            label=f"{model_name} (AP={pr_auc_val:.4f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models (Test Set)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = results_dir / "figures" / "pr_curves_all.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {pr_path}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Generated curves for {len(per_model)} model(s)")
    print(f"  Per-model PNGs: {fig_dir}")
    print(f"  Overlaid ROC:   {roc_path}")
    print(f"  Overlaid PR:    {pr_path}")
    print(f"\nAUC summary:")
    print(f"{'Model':<20}{'ROC-AUC':>12}{'PR-AUC':>12}")
    print("-" * 44)
    for m, v in sorted(model_aucs.items(), key=lambda kv: -kv[1]["roc_auc"]):
        print(f"{m:<20}{v['roc_auc']:>12.4f}{v['pr_auc']:>12.4f}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════
# Convenience alias
# ═══════════════════════════════════════════════════════════════
def plot_all_roc_curves(*args, **kwargs):
    """Alias for plot_all_curves (matches name in some docs)."""
    return plot_all_curves(*args, **kwargs)