from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Loss Curves ─────────────────────────────────────
def plot_loss_curves(
    history: dict[str, list[float]],
    plots_dir: Path,
    filename: str = "loss_curve.png",
) -> None:
    """Plot and save a single-figure training and validation loss curve."""
    train_loss = history.get("train_loss")
    val_loss = history.get("val_loss")
    epochs = history["epoch"]

    # Curated color palette
    train_color = "#2563EB"  # Royal Blue
    val_color = "#E11D48"    # Crimson Rose
    best_color = "#059669"   # Emerald Green

    plt.figure(figsize=(9, 5.2), dpi=300, facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("#FAFAFA")

    # Training & validation lines with hollow markers and soft fills
    plt.plot(
        epochs,
        train_loss,
        label="Training Loss",
        color=train_color,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        zorder=3,
    )
    plt.fill_between(epochs, train_loss, color=train_color, alpha=0.06)

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        color=val_color,
        linewidth=2.2,
        marker="s",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        zorder=4,
    )
    plt.fill_between(epochs, val_loss, color=val_color, alpha=0.06)

    # Best validation loss marker & guide line
    best_idx = min(range(len(val_loss)), key=val_loss.__getitem__)
    best_epoch = epochs[best_idx]
    best_val = val_loss[best_idx]

    plt.axvline(best_epoch, color=best_color, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    plt.scatter(
        [best_epoch],
        [best_val],
        color=best_color,
        s=75,
        zorder=5,
        edgecolors="white",
        linewidth=1.5,
        label=f"Best Val ({best_val:.4f} @ Ep {best_epoch})",
    )

    # Titles & axis labels
    plt.title(
        "Model Training & Validation Loss",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        loc="left",
        pad=14,
    )
    plt.xlabel("Epoch", fontsize=11, fontweight="medium", color="#374151", labelpad=8)
    plt.ylabel("Loss", fontsize=11, fontweight="medium", color="#374151", labelpad=8)

    # Clean ticks, grid, and spines
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tick_params(colors="#4B5563", labelsize=9.5)
    plt.grid(True, which="major", linestyle="--", linewidth=0.7, color="#D1D5DB", alpha=0.8, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")

    # Legend styling
    legend = plt.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7EB",
        fontsize=9.5,
        loc="upper right",
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(1.0)

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

# ── F1 Curve ─────────────────────────────────────
def plot_f1_curve(
    history: dict[str, list[float]],
    plots_dir: Path,
    filename: str = "f1_curve.png",
) -> None:
    """Plot and save a single-figure validation F1 curve."""
    val_f1 = history.get("val_f1")

    if not val_f1:
        return

    epochs = history["epoch"]

    # Curated color palette matching the loss plot
    f1_color = "#E11D48"    # Crimson Rose
    best_color = "#059669"  # Emerald Green

    plt.figure(figsize=(9, 5.2), dpi=300, facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("#FAFAFA")

    # Validation F1 curve with hollow markers and soft area fill
    plt.plot(
        epochs,
        val_f1,
        label="Validation F1",
        color=f1_color,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        zorder=3,
    )
    plt.fill_between(epochs, val_f1, color=f1_color, alpha=0.07)

    # Highlight the highest F1 score checkpoint
    best_idx = max(range(len(val_f1)), key=val_f1.__getitem__)
    best_epoch = epochs[best_idx]
    best_f1 = val_f1[best_idx]

    plt.axvline(best_epoch, color=best_color, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    plt.scatter(
        [best_epoch],
        [best_f1],
        color=best_color,
        s=75,
        zorder=4,
        edgecolors="white",
        linewidth=1.5,
        label=f"Best F1 ({best_f1:.4f} @ Ep {best_epoch})",
    )

    # Titles & axis labels
    plt.title(
        "Model Validation F1 Score",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        loc="left",
        pad=14,
    )
    plt.xlabel("Epoch", fontsize=11, fontweight="medium", color="#374151", labelpad=8)
    plt.ylabel("F1 Score", fontsize=11, fontweight="medium", color="#374151", labelpad=8)

    # Clean ticks, bounded Y-axis, grid, and spines
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(0.0, 1.05)
    plt.tick_params(colors="#4B5563", labelsize=9.5)
    plt.grid(True, which="major", linestyle="--", linewidth=0.7, color="#D1D5DB", alpha=0.8, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")

    # Legend styling (placed bottom-right so it doesn't cover upward F1 curves)
    legend = plt.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7EB",
        fontsize=9.5,
        loc="lower right",
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(1.0)

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()