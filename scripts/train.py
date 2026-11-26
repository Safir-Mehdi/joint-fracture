"""
Train a single model from the command line.

Usage:
    python scripts/train.py --model resnet50
    python scripts/train.py --model cnn_baseline --epochs 5 --device cpu
    python scripts/train.py --model resnet50 --run-dir experiments/custom_run

The script:
    1. Loads config (base + training + model)
    2. Sets seed for reproducibility
    3. Builds dataloaders (train + val)
    4. Builds model, applies fine-tune strategy
    5. Builds loss + optimizer + scheduler
    6. Runs Trainer.fit()
    7. Prints final metrics
"""

import argparse
import sys
import traceback
from pathlib import Path

# ── Make `src/` importable when running from anywhere ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils.config_loader import load_config
from src.utils.seed import set_seed
from src.utils.file_io import create_run_dir, ensure_dir
from src.utils.logger import get_logger, close_handlers
from src.data.preprocessing import get_base_transform
from src.data.augmentation import get_train_transform
from src.data.dataloader import build_dataloader
from src.models.model_factory import build_model
from src.training.loss import build_loss
from src.training.optimizer import build_optimizer
from src.training.trainer import Trainer

# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fracture-classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Model name (must have a config in configs/models/)")

    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg['training']['epochs']")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                        help="Device preference to train on, CPU or CUDA!")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Explicit run directory, by default it saves to experiments/run_YYYYMMDD_HHMMSS/*. But it could set it up by your own choice.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override cfg['seed']")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Fine-tune strategy
# ═══════════════════════════════════════════════════════════════
def apply_fine_tune_strategy(model, cfg: dict, logger) -> None:
    """Apply freeze/unfreeze based on cfg['model']['fine_tune_mode']."""
    mcfg = cfg["model"]
    mode = mcfg.get("fine_tune_mode", "full")

    if mode == "full":
        model.unfreeze_all()
    elif mode == "head_only":
        model.freeze_backbone()
    elif mode == "partial":
        model.freeze_backbone()
        for pattern in mcfg.get("unfreeze_patterns", []):
            model.unfreeze_by_name(pattern)
    else:
        raise ValueError(
            f"Unknown fine_tune_mode '{mode}'. "
            f"Expected: full | partial | head_only"
        )

    trainable = model.get_num_params(trainable_only=True)
    total = model.get_num_params(trainable_only=False)
    logger.info(
        f"  Fine-tune mode: {mode} "
        f"→ {trainable:,} / {total:,} trainable "
        f"({100 * trainable / total:.1f}%)"
    )
    # print(
    #     f"  Fine-tune mode: {mode} "
    #     f"→ {trainable:,} / {total:,} trainable "
    #     f"({100 * trainable / total:.1f}%)"
    # )


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main() -> int:
    args = parse_args()

    cfg = load_config(model=args.model, training=True)

    # ═══════════════════════════════════════════════════════
    # PHASE 1 — Load config & setup run dir BEFORE logging
    # ═══════════════════════════════════════════════════════
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed

    # ── Determine run_dir early ──
    if args.run_dir:
        run_dir = Path(args.run_dir)
        ensure_dir(run_dir / "logs")
        ensure_dir(run_dir / "checkpoints")
        ensure_dir(run_dir / "plots")
    else:
        run_dir = create_run_dir("experiments", model_name=args.model)

    # ═══════════════════════════════════════════════════════
    # PHASE 2 — Create the script-level logger
    # ═══════════════════════════════════════════════════════
    close_handlers("train_script")   # clear stale handlers from prior runs
    logger = get_logger(
        "train_script",
        log_file=run_dir / "logs" / "train_script.log",
        level=args.log_level,
    )

    logger.info("=" * 60)
    logger.info(f"Training model: {args.model}")
    logger.info("=" * 60)
    logger.info(f"  Run dir: {run_dir}")
    logger.info(f"  Seed: {cfg['seed']}")

    # print("=" * 60)
    # print(f"Training model: {args.model}")
    # print("=" * 60)
    # print(f"  Run dir: {run_dir}")
    # print(f"  Seed: {cfg['seed']}")

    if args.epochs is not None:
        logger.info(f"  Epochs override: {args.epochs}")
        # print(f"  Epochs override: {args.epochs}")

    # ── Device ──
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"  Device: {device}")
    # print(f"  Device: {device}")

    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        # print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Seed ──
    set_seed(cfg["seed"])

    # ═══════════════════════════════════════════════════════
    # PHASE 3 — Data
    # ═══════════════════════════════════════════════════════
    image_size = cfg["model"]["image_size"]
    logger.info(f"  Image size: {image_size}")
    logger.info(f"  Batch size: {cfg['dataloader']['batch_size']}")

    # print(f"  Image size: {image_size}")
    # print(f"  Batch size: {cfg['dataloader']['batch_size']}")

    train_tf = get_train_transform(image_size)
    val_tf = get_base_transform(image_size)

    train_loader = build_dataloader(cfg, "data/raw/train", train_tf, mode="train")
    val_loader = build_dataloader(cfg, "data/raw/val", val_tf, mode="val")

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches:   {len(val_loader)}")

    # print(f"  Train batches: {len(train_loader)}")
    # print(f"  Val batches:   {len(val_loader)}")

    # ═══════════════════════════════════════════════════════
    # PHASE 4 — Model
    # ═══════════════════════════════════════════════════════
    logger.info("Building model...")
    # print("Building model...")

    model = build_model(cfg)

    logger.info(f"  Model: {model}")
    # print(f"  Model: {model}")

    apply_fine_tune_strategy(model, cfg, logger)

    # ═══════════════════════════════════════════════════════
    # PHASE 5 — Loss + Optimizer
    # ═══════════════════════════════════════════════════════
    loss_fn = build_loss(cfg, class_weights=None)
    logger.info(f"  Loss: {loss_fn}")
    # print(f"  Loss: {loss_fn}")

    optimizer, scheduler = build_optimizer(model, cfg)
    logger.info(f"  Optimizer: {type(optimizer).__name__}")
    logger.info(f"  Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")

    # print(f"  Optimizer: {type(optimizer).__name__}")
    # print(f"  Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")

    # ═══════════════════════════════════════════════════════
    # PHASE 6 — Trainer (uses same run_dir)
    # ═══════════════════════════════════════════════════════
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        run_dir=run_dir,      # ← explicitly pass
    )

    # ═══════════════════════════════════════════════════════
    # PHASE 7 — Fit
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    # print("=" * 60)
    # print("Starting training...")
    # print("=" * 60)

    final = trainer.fit()

    # ═══════════════════════════════════════════════════════
    # PHASE 8 — Summary
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info("=" * 60)
    logger.info(f"  Run dir:      {run_dir}")
    logger.info(f"  Val F1:       {final.get('val_f1', 0):.4f}")
    logger.info(f"  Val AUC:      {final.get('val_auc', 0):.4f}")
    logger.info(f"  Sensitivity:  {final.get('val_sensitivity', 0):.4f}")
    logger.info(f"  Specificity:  {final.get('val_specificity', 0):.4f}")
    logger.info(f"  Accuracy:     {final.get('val_accuracy', 0):.4f}")

    # print("=" * 60)
    # print("Training complete")
    # print("=" * 60)
    # print(f"  Run dir:      {run_dir}")
    # print(f"  Val F1:       {final.get('val_f1', 0):.4f}")
    # print(f"  Val AUC:      {final.get('val_auc', 0):.4f}")
    # print(f"  Sensitivity:  {final.get('val_sensitivity', 0):.4f}")
    # print(f"  Specificity:  {final.get('val_specificity', 0):.4f}")
    # print(f"  Accuracy:     {final.get('val_accuracy', 0):.4f}")

    # ── Close loggers (Windows file lock) ──
    close_handlers("train_script")

    return 0


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(130)
    except Exception:
        print(f"\n\n❌ Training failed with error:\n")
        traceback.print_exc()
        sys.exit(1)