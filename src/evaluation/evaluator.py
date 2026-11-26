"""
Evaluate a trained model on the test set.

Two APIs:
    1. Evaluator(model, test_loader, device).evaluate()  →  metrics dict
    2. evaluate_run(run_dir)                             →  metrics dict

Usage:
    # High-level — evaluate a full run
    from src.evaluation.evaluator import evaluate_run
    metrics = evaluate_run("experiments/run_20241123_100000_resnet50")
    print(f"Test F1: {metrics['f1']:.4f}")

    # Low-level — evaluate a custom model + loader
    from src.evaluation.evaluator import Evaluator
    ev = Evaluator(model, test_loader, device="cuda")
    metrics = ev.evaluate()
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.utils.config_loader import load_config
from src.utils.file_io import load_yaml, save_json
from src.utils.logger import get_logger

# ═══════════════════════════════════════════════════════════════
# Evaluator class
# ═══════════════════════════════════════════════════════════════
class Evaluator:
    """
    Evaluate a trained model on a dataloader.

    Args:
        model:       nn.Module (already built, no checkpoint loaded)
        test_loader: DataLoader for the test set
        device:      "cuda" | "cpu"
        pos_class:   positive class for sensitivity (default 0 = fractured)
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
        pos_class: int = 0,
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.pos_class = pos_class

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Run inference on the test set and return all 18 metrics.

        Returns:
            dict with keys: accuracy, precision, recall, f1,
                            precision_fractured, precision_normal,
                            recall_fractured, recall_normal,
                            f1_fractured, f1_normal, auc,
                            sensitivity, specificity,
                            tn, fp, fn, tp, loss
        """
        self.model.eval()
        all_targets: list[int] = []
        all_preds: list[int] = []
        all_probs: list[float] = []

        total = 0

        for images, labels in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            # y_prob = probability of the POSITIVE class (fractured = class 0)
            all_probs.extend(probs[:, self.pos_class].cpu().numpy().tolist())
            total += labels.size(0)

        metrics = compute_metrics(
            y_true=all_targets,
            y_pred=all_preds,
            y_prob=all_probs,
            positive_class=self.pos_class,
        )
        metrics["num_samples"] = total

        self.predictions = {
            "y_true": all_targets,
            "y_pred": all_preds,
            "y_prob": all_probs,
        }
        
        return metrics


# ═══════════════════════════════════════════════════════════════
# High-level helper — evaluate a full run
# ═══════════════════════════════════════════════════════════════
def evaluate_run(
    run_dir: str | Path,
    device: Optional[str] = None,
    checkpoint_name: str = "best.pth",
    test_data_dir: str = "data/raw/test",
    save: bool = True,
) -> dict:
    """
    Evaluate a completed run in one call.

    Steps:
        1. Load config from <run_dir>/config.yaml
        2. Build model via model_factory
        3. Load checkpoint from <run_dir>/checkpoints/<checkpoint_name>
        4. Build test dataloader (deterministic)
        5. Run Evaluator.evaluate()
        6. Save <run_dir>/metrics_test.json & predictions_test.json

    Args:
        run_dir:         path to experiments/run_xxx_<model>/
        device:          override device (default: auto cuda/cpu)
        checkpoint_name: which checkpoint to load (default "best.pth")
        test_data_dir:   folder containing test split (default "data/raw/test")
        save:            if True, save metrics_test.json in run_dir

    Returns:
        metrics dict.

    Raises:
        FileNotFoundError: if run_dir or checkpoint missing.
    """
    from src.models.model_factory import build_model
    from src.data.preprocessing import get_base_transform
    from src.data.dataloader import build_dataloader

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    # ── Logger ──
    logger = get_logger(
        "evaluator",
        log_file=run_dir / "logs" / "evaluate.log",
    )
    logger.info("=" * 60)
    logger.info(f"Evaluating run: {run_dir}")
    logger.info("=" * 60)

    # ── 1. Load config ──
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_yaml(config_path)
    model_name = cfg["model"]["name"]
    logger.info(f"  Model: {model_name}")

    # ── 2. Device ──
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")

    # ── 3. Build model ──
    model = build_model(cfg)
    logger.info(f"  Model built: {model}")

    # ── 4. Load checkpoint ──
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    logger.info(f"  Checkpoint loaded: {ckpt_path}")

    # ── 5. Build test dataloader ──
    image_size = cfg["model"]["image_size"]
    test_tf = get_base_transform(image_size)
    test_loader = build_dataloader(
        cfg, test_data_dir, test_tf, mode="test",
    )
    logger.info(f"  Test batches: {len(test_loader)}")
    logger.info(f"  Test samples: {len(test_loader.dataset)}")

    # ── 6. Evaluate ──
    evaluator = Evaluator(model, test_loader, device=device, pos_class=0)
    metrics = evaluator.evaluate()

    # ── 7. Save ──
    if save:
        # For metrics
        out_path = run_dir / "metrics_test.json"
        save_json(metrics, out_path)
        logger.info(f"  Saved: {out_path}")

        # For predicted values  
        save_json(evaluator.predictions, run_dir / "predictions_test.json")
        logger.info(f"  Saved predictions: {run_dir / 'predictions_test.json'}")

    # ── 8. Log summary ──
    logger.info("-" * 60)
    logger.info("Test metrics:")
    logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:    {metrics['precision']:.4f}")
    logger.info(f"  Recall:       {metrics['recall']:.4f}")
    logger.info(f"  F1:           {metrics['f1']:.4f}")
    logger.info(f"  AUC:          {metrics['auc']:.4f}")
    logger.info(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity:  {metrics['specificity']:.4f}")
    logger.info(f"  Confusion:    TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    logger.info("=" * 60)

    return metrics