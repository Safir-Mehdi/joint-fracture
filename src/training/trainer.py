"""
Main training loop for fracture classification.

Usage:
    from src.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device="cuda",
    )
    final_metrics = trainer.fit()
"""
import torch
import time
import torch.nn as nn
from  pathlib import Path
from typing import Optional, Dict, List, Tuple
from torch.utils.data import DataLoader
from src.evaluation.metrics import compute_metrics
from src.training.callback import build_callbacks
from src.utils.file_io import create_run_dir, save_json, save_yaml, ensure_dir
from src.utils.plots import plot_loss_curves, plot_f1_curve
from src.utils.logger import get_logger

class Trainer:
     """
    Main training loop.

    Responsibilities:
        - Creates a timestamped run folder (experiments/run_xxx_model/)
        - Saves config.yaml for reproducibility
        - Logs to console + file
        - Runs training + validation per epoch
        - Applies grad clipping and AMP
        - Drives callbacks (early stopping, checkpointing, logging, ...)
        - Saves metrics.json, history.json, and training plots
    """

     def __init__(
               self,
               model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
               cfg: dict,
               device: str = "cuda",
               run_dir: Optional[str | Path] = None
     ): 
          # ── Core components ────────────────────────────────
          self.model = model
          self.train_loader = train_loader
          self.val_loader = val_loader
          self.loss_fn = loss_fn
          self.optimizer = optimizer
          self.scheduler = scheduler
          self.cfg = cfg
          self.device = device
          # ── Training settings ─────────────────────────────
          self.epochs = cfg["training"]["epochs"]
          self.grad_clip = cfg["training"]["grad_clip"]
          self.use_amp = cfg["training"]["use_amp"]
          self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

          # ── Run folder ────────────────────────────────────
          model_name = cfg["model"]["name"]
          if run_dir is not None:
              self.run_dir = Path(run_dir)
              ensure_dir(self.run_dir / "logs")
              ensure_dir(self.run_dir / "checkpoints")
              ensure_dir(self.run_dir / "plots")
          else:
              self.run_dir = create_run_dir("experiments", model_name=model_name)
          save_yaml(cfg, self.run_dir/"config.yaml")

          # ── Logger ────────────────────────────────────────
          self.logger = get_logger(
               __name__,
               log_file=self.run_dir / "logs" / "train.log"
          )

          self.logger.info(f"Run dir: {self.run_dir}")
          self.logger.info(f"Model:   {model_name}")
          self.logger.info(f"Device:  {device}  |  AMP: {self.use_amp}")

          # ── Terminal ───
          print(f"Run dir: {self.run_dir}")
          print(f"Model:   {model_name}")
          print(f"Device:  {device}  |  AMP: {self.use_amp}")

          # ── Callbacks ─────────────────────────────────────
          self.callbacks = build_callbacks(
               cfg=cfg,
               model=self.model,
               run_dir=self.run_dir,
               scheduler=scheduler
          )
          self.logger.info(f"Callbacks: {[type(cb).__name__ for cb in self.callbacks]}")

          # ── Terminal ───
          print(f"Callbacks: {[type(cb).__name__ for cb in self.callbacks]}")

          # ── History (for plots) ───────────────────────────
          self.history: Dict[str, List] = {}
    
    
    # ═══════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════

     def fit(self, epochs: Optional[int] = None) -> dict:
        """
        Run the full training loop.

            Args:
                epochs: override cfg["training"]["epochs"] if given.

            Returns:
                Final metrics dict (from the last epoch).
        """

        n_epochs = epochs or self.epochs
        self.logger.info(f"Starting training for {n_epochs} epochs")
        start_time = time.time()

        final_metrics: dict = {}

        for epoch in range(n_epochs):
            # ── 1. Train one epoch ──
            train_loss, train_acc = self._train_one_epoch()

            # ── 2. Validate ──
            val_metrics = self._validate()

            # ── 3. Assemble metrics dict ──
            metrics = self._assemble_metrics(train_loss, train_acc, val_metrics)

            # ── 4. Update internal history ──
            for k, v in {"epoch": epoch, **metrics}.items():
                self.history.setdefault(k, []).append(v)

            # ── 5. Run callbacks ──
            stop = False
            for cb in self.callbacks:
                result = cb.on_epoch_end(epoch, metrics)
                if result == "stop":
                    stop = True

            final_metrics = metrics

            if stop:
                self.logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                break


        # ── Training complete ──
        total_time = time.time() - start_time
        self.logger.info(f"Training finished in {total_time/60:.1f} min")
        print(f"Training finished in {total_time/60:.1f} min")

        # ── on_train_end callbacks ──
        for cb in self.callbacks:
            cb.on_train_end(final_metrics)

        # ── Save artifacts ──
        save_json(final_metrics, self.run_dir / "metrics.json")
        save_json(self.history, self.run_dir / "history.json")

        if self.cfg.get("logging", {}).get("save_plots", False):
            self._save_plots()

        self.logger.info(f"Metrics saved: {self.run_dir / 'metrics.json'}")
        print(f"Metrics saved: {self.run_dir / 'metrics.json'}")
        return final_metrics


     # ═══════════════════════════════════════════════════════
     # Internal methods
     # ═══════════════════════════════════════════════════════

     def _train_one_epoch(self) -> Tuple[float, float]:
         """Train for one epoch. Returns (avg_loss, accuracy)."""
         self.model.train()
         running_loss = 0.0
         correct = 0
         total = 0

         for images, labels in self.train_loader:
             images = images.to(self.device, non_blocking=True)
             labels = labels.to(self.device, non_blocking=True) 

             self.optimizer.zero_grad(set_to_none=True)

             # ── Forward + backward (with optional AMP) ──
             if self.use_amp:
                 with torch.cuda.amp.autocast():
                     logits = self.model(images)
                     loss = self.loss_fn(logits, labels)

                 self.scaler.scale(loss).backward()
                 self.scaler.unscale_(self.optimizer)
                 if self.grad_clip > 0:
                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                 self.scaler.step(self.optimizer)
                 self.scaler.update()
             else:
                 logits = self.model(images)
                 loss = self.loss_fn(logits, labels)
                 loss.backward()

                 if self.grad_clip > 0:
                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                 self.optimizer.step()
             # ── Track stats ──
             running_loss += loss.item() * images.size(0)
             preds = logits.argmax(dim=1)
             correct += (preds == labels).sum().item()
             total += labels.size(0)

         avg_loss = running_loss / max(total, 1)
         acc = correct / max(total, 1)

         return avg_loss, acc

     @torch.no_grad()
     def _validate(self) -> dict:
         """Run validation and compute full metrics."""
         self.model.eval()
         running_loss = 0.0
         total = 0

         all_targets = []
         all_preds = []
         all_probs = []

         for images, labels in self.val_loader:
             images = images.to(self.device, non_blocking=True)
             labels = labels.to(self.device, non_blocking=True)

             logits = self.model(images)
             loss = self.loss_fn(logits, labels)

             running_loss += loss.item() * images.size(0)
             total += labels.size(0)

             probs = torch.softmax(logits, dim=1)
             preds = logits.argmax(dim=1)

             all_targets.extend(labels.cpu().numpy())
             all_preds.extend(preds.cpu().numpy())
             # y_prob = probability of POSITIVE class (fractured = class 0)
             all_probs.extend(probs[:, 0].cpu().numpy())

         avg_loss = running_loss / max(total, 1)
        
         # ── Full metrics via compute_metrics ──
         metrics = compute_metrics(
             y_true=all_targets,
             y_pred=all_preds,
             y_prob=all_probs,
             positive_class=0,
         )
         metrics["loss"] = round(avg_loss, 4)
         return metrics

     def _assemble_metrics(
        self,
        train_loss: float,
        train_acc: float,
        val_metrics: dict,
    ) -> dict:
        """Merge train + val metrics + current LR into one dict."""
        metrics = {
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        for key, value in val_metrics.items():
            metrics[f"val_{key}"] = value
        return metrics

     def _save_plots(self) -> None:
        """Save loss and F1 curve plots from history."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available — skipping plots")
            print("matplotlib not available — skipping plots")
            return

        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if not self.history.get("epoch"):
            return

        # ── Loss curves ──
        plot_loss_curves(
            history=self.history,
            plots_dir= plots_dir,
            filename="loss_curve.png"
        )

        # epochs = self.history["epoch"]

        # if "train_loss" in self.history and "val_loss" in self.history:
        #     plt.figure(figsize=(8, 5))
        #     plt.plot(epochs, self.history["train_loss"], label="train", marker="o", markersize=3)
        #     plt.plot(epochs, self.history["val_loss"], label="val", marker="s", markersize=3)
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Loss")
        #     plt.title("Loss curves")
        #     plt.legend()
        #     plt.grid(alpha=0.3) 
        #     plt.tight_layout()
        #     plt.savefig(plots_dir / "loss_curve.png", dpi=150)
        #     plt.close()

        # ── F1 curve ──
        plot_f1_curve(
            history=self.history,
            plots_dir=plots_dir,
            filename="f1_curve.png"
        )

        # if "val_f1" in self.history:
        #     plt.figure(figsize=(8, 5))
        #     plt.plot(epochs, self.history["val_f1"], label="val F1", color="crimson", marker="o", markersize=3)
        #     plt.xlabel("Epoch")
        #     plt.ylabel("F1")
        #     plt.title("Validation F1")
        #     plt.legend()
        #     plt.grid(alpha=0.3)
        #     plt.tight_layout()
        #     plt.savefig(plots_dir / "f1_curve.png", dpi=150)
        #     plt.close()

        self.logger.info(f"Plots saved to {plots_dir}")
        print(f"Plots saved to {plots_dir}")
