"""
Evaluate trained models on the test set.

Logs go to BOTH:
    - Terminal (console)
    - experiments/evaluation_runs/batch_YYYYMMDD_HHMMSS.log

Each run also has its own per-run log: <run_dir>/logs/evaluate.log
(from src/evaluation/evaluator.py)

Usage:
    # Evaluate ONE run
    python scripts/evaluate.py --run-dir experiments/run_xxx_resnet50

    # Evaluate ALL runs in experiments/
    python scripts/evaluate.py --all

    # Re-evaluate even if metrics_test.json exists
    python scripts/evaluate.py --all --force

    # Use a different checkpoint
    python scripts/evaluate.py --run-dir experiments/run_xxx --checkpoint last.pth

    # Run on CPU
    python scripts/evaluate.py --all --device cpu
"""

import argparse
import sys
import traceback
from pathlib import Path
from datetime import datetime

# ── Make `src/` importable ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_run
from src.utils.file_io import ensure_dir
from src.utils.logger import get_logger, close_handlers


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained fracture-classification models on the test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode — mutually exclusive
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run-dir", type=str, default=None,
        help="Path to a single run folder (e.g. experiments/run_xxx_resnet50)",
    )
    mode.add_argument(
        "--all", action="store_true",
        help="Evaluate every run in experiments/ that hasn't been evaluated yet",
    )

    parser.add_argument(
        "--experiments-dir", type=str, default="experiments",
        help="Folder containing run_*/ subfolders (used with --all)",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best.pth",
        help="Checkpoint filename in <run_dir>/checkpoints/",
    )
    parser.add_argument(
        "--test-dir", type=str, default="data/raw/test",
        help="Path to the test set folder",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate even if metrics_test.json already exists",
    )
    parser.add_argument("--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity"
    )
    return parser.parse_args()

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def _is_already_evaluated(run_dir: Path) -> bool:
    """A run is 'already evaluated' if both metrics_test.json and predictions_test.json exist."""
    return (
        (run_dir / "metrics_test.json").exists()
        and (run_dir / "predictions_test.json").exists()
    )

def _collect_runs(args) -> list[Path]:
    """Return the list of run directories to evaluate."""
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        return [run_dir]

    # --all mode
    exp_dir = Path(args.experiments_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiments dir not found: {exp_dir}")

    run_dirs = sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* folders found in {exp_dir}")

    return run_dirs


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main() -> int:
    args = parse_args()

    # ── 1. Collect runs ──
    run_dirs = _collect_runs(args)

    # ── 2. Set up batch log ──
    batch_log_dir = Path(args.experiments_dir) / "evaluation_runs"
    ensure_dir(batch_log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = batch_log_dir / f"batch_{timestamp}.log"

    close_handlers("evaluate_batch")  # clear stale handlers from prior runs
    logger = get_logger(
        "evaluate_batch",
        log_file=log_file,
        level=args.log_level,
    )

    # ── 3. Header ──
    mode_name = "single run" if args.run_dir else "all runs"
    logger.info("=" * 60)
    logger.info(f"Evaluation batch — {mode_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Log file:  {log_file}")
    logger.info(f"Device:    {args.device or 'auto'}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test dir:  {args.test_dir}")
    logger.info(f"Force:     {args.force}")
    logger.info(f"Found {len(run_dirs)} run(s)")
    logger.info("=" * 60)

    # ── 4. Evaluate each run ──
    results: list[dict] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for i, run_dir in enumerate(run_dirs, 1):
        name = run_dir.name
        logger.info(f"[{i}/{len(run_dirs)}] {name}")

        # Skip if already evaluated (unless --force)
        if not args.force and _is_already_evaluated(run_dir):
            logger.info(f"  ⏭️  Already evaluated — skipping (use --force to redo)")
            skipped.append(name)
            continue

        # Evaluate
        try:
            metrics = evaluate_run(
                run_dir=run_dir,
                device=args.device,
                checkpoint_name=args.checkpoint,
                test_data_dir=args.test_dir,
                save=True,
            )

            results.append({
                "run_id": name,
                "model": metrics.get("_model_name", "unknown"),
                "f1": metrics.get("f1", 0.0),
                "auc": metrics.get("auc", 0.0),
                "sensitivity": metrics.get("sensitivity", 0.0),
                "specificity": metrics.get("specificity", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
            })

        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")
            failed.append((name, str(e)))

    # ── 5. Summary ──
    logger.info(f"\n{'=' * 60}")
    logger.info("📊 Evaluation Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"  ✅ Evaluated: {len(results)}")
    logger.info(f"  ⏭️  Skipped:   {len(skipped)}")
    logger.info(f"  ❌ Failed:    {len(failed)}")

    if results:
        logger.info(f"\nResults:")
        logger.info(f"  {'Run':<40}{'F1':>8}{'AUC':>8}{'Sens':>8}{'Spec':>8}")
        logger.info(f"  {'-' * 72}")
        for r in sorted(results, key=lambda x: -x["f1"]):
            logger.info(
                f"  {r['run_id']:<40}"
                f"{r['f1']:>8.4f}"
                f"{r['auc']:>8.4f}"
                f"{r['sensitivity']:>8.4f}"
                f"{r['specificity']:>8.4f}"
            )

    if skipped:
        logger.info("")
        logger.info(" ⏭️ Skipped (already evaluated):")
        for s in skipped:
            logger.info(f"  - {s}")

    if failed:
        logger.info("")
        logger.info(" ❌ Failed:")
        for name, err in failed:
            logger.info(f"  - {name}: {err}")

    logger.info("=" * 60)
    if failed:
        logger.info("❌ Evaluation completed with failures")
    else:
        logger.info("✅ Evaluation complete")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 60)

    # ── 6. Close logger (release Windows file lock) ──
    close_handlers("evaluate_batch")

    return 1 if failed else 0

# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
        sys.exit(130)
    except Exception:
        print(f"\n\n❌ Evaluation failed:\n")
        traceback.print_exc()
        sys.exit(1)