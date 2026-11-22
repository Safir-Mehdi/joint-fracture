
import csv
import time
import torch
import torch.nn as nn
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class BaseCallback(ABC):
    """Abstract base for all callbacks."""
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: dict) -> Optional[str]:
        """Called at end of each epoch. Return 'stop' to end training."""
        raise NotImplementedError

    def on_train_end(self, metrics: dict) -> None:
        """Called once at end of training. Override to clean up."""
        pass


class HistoryLogger(BaseCallback):
    """Records metrics per epoch into a dict — for later plotting."""

    def __init__(self):
        self.history: Dict[str, list] = {}

    def on_epoch_end(self, epoch, metrics):
        self.history.setdefault("epoch", []).append(epoch)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.history.setdefault(key, []).append(float(value))
        return None

    def on_train_end(self, metrics: dict) -> None:
        pass


class CSVLogger(BaseCallback):
    """Appends one row per epoch to a CSV file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False
        self._file = None

    def on_epoch_end(self, epoch: int, metrics: dict) -> Optional[str]:
        if self._file is None:
            self._file = self.path.open("w", newline="")
            self._writer = csv.writer(self._file)

        row = {"epoch": epoch, **metrics}
        if not self._header_written:
            self._writer.writerow(row.keys())
            self._header_written = True
        self._writer.writerow(row.values())
        self._file.flush()
        return None

    def on_train_end(self, metrics: dict) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class ModelCheckpoint(BaseCallback):
    """Saves model when monitored metric improves."""

    def __init__(self, model, path, monitor="val_f1", mode="max", min_delta=0.001):
        self.model = model
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta

        self.best = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1

    def on_epoch_end(self, epoch: int, metrics: dict) -> Optional[str]:
        if self.monitor not in metrics:
            return None
        v = metrics[self.monitor]

        improved = (
            v > self.best + self.min_delta
            if self.mode == "max"
            else v < self.best - self.min_delta
        )
        if improved:
            self.best = v
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.path)
        return None


class EarlyStopping(BaseCallback):
    """Stops training when monitored metric stops improving."""

    def __init__(self, monitor="val_f1", patience=10, mode="max", min_delta=0.001):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = -1

    def on_epoch_end(self, epoch: int, metrics: dict) -> Optional[str]:
        if self.monitor not in metrics:
            return None
        v = metrics[self.monitor]

        improved = (
            v > self.best + self.min_delta
            if self.mode == "max"
            else v < self.best - self.min_delta
        )
        if improved:
            self.best = v
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return "stop"
        return None


class TensorBoardLogger(BaseCallback):
    """Writes metrics to TensorBoard."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def on_epoch_end(self, epoch: int, metrics: dict) -> Optional[str]:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
        return None

    def on_train_end(self, metrics: dict) -> None:
        self.writer.close()


class ProgressPrinter(BaseCallback):
    """Prints a clean per-epoch summary line."""

    def on_epoch_end(self, epoch: int, metrics: dict) -> Optional[str]:
        parts = [f"Epoch {epoch:03d}"]
        if "train_loss" in metrics:
            parts.append(f"train_loss={metrics['train_loss']:.4f}")
        if "val_loss" in metrics:
            parts.append(f"val_loss={metrics['val_loss']:.4f}")
        if "val_f1" in metrics:
            parts.append(f"val_f1={metrics['val_f1']:.4f}")
        if "lr" in metrics:
            parts.append(f"lr={metrics['lr']:.2e}")
        print(" | ".join(parts))
        return None


class TimeTracker(BaseCallback):
    """Records time per epoch."""

    def __init__(self):
        self.epoch_times: list[float] = []
        self._last_time = None
        self.total_time = 0.0

    def on_epoch_end(self, epoch: int, metrics: dict) -> Optional[str]:
        now = time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            self.epoch_times.append(dt)
            self.total_time += dt
        self._last_time = now
        return None

    def on_train_end(self, metrics: dict) -> None:
        if self.epoch_times:
            avg = sum(self.epoch_times) / len(self.epoch_times)
            print(f"⏱  Avg epoch time: {avg:.1f}s | Total: {self.total_time/60:.1f} min")


class LRSchedulerCallback(BaseCallback):
    """Steps the LR scheduler each epoch."""

    def __init__(self, scheduler, monitor="val_loss"):
        self.scheduler = scheduler
        self.monitor = monitor  # only used by ReduceLROnPlateau

    def on_epoch_end(self, epoch: int, metrics: dict) -> Optional[str]:
        if self.scheduler is None:
            return None

        if isinstance(self.scheduler, ReduceLROnPlateau):
            metric_val = metrics.get(self.monitor)
            if metric_val is not None:
                self.scheduler.step(metric_val)
        else:
            self.scheduler.step()
        return None


# ── Factory ────────────────────────────────────
def build_callbacks(
        cfg: dict,
        model: nn.Module,
        run_dir: str | Path,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
        ) -> list[BaseCallback]:
    """
    Build all callbacks in the correct order.

    Returns:
        List of callbacks. Order: scheduler → loggers → printer → time →
        checkpoint → early stopping.
    """
    run_dir = Path(run_dir)
    cb_cfg = cfg.get("early_stopping", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    log_cfg = cfg.get("logging", {})

    callbacks: list[BaseCallback] = []

    # 1. Scheduler FIRST
    if scheduler is not None:
        callbacks.append(LRSchedulerCallback(scheduler))

    # 2. Loggers
    callbacks.append(HistoryLogger())
    callbacks.append(CSVLogger(run_dir / "metrics.csv"))

    if log_cfg.get("tensorboard", False):
        callbacks.append(TensorBoardLogger(run_dir / "logs" / "tensorboard"))

    # 3. Printer + TimeTracker
    callbacks.append(ProgressPrinter())
    callbacks.append(TimeTracker())

    # 4. Checkpoint (save best)
    if ckpt_cfg.get("save_best_only", True):
        callbacks.append(
            ModelCheckpoint(
                model=model,
                path=run_dir / "checkpoints" / "best.pth",
                monitor=ckpt_cfg.get("monitor", "val_f1"),
                mode=ckpt_cfg.get("mode", "max"),
                min_delta=0.001,
            )
        )

    # 5. Early stopping (last — may return "stop")
    if cb_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=cb_cfg.get("monitor", "val_f1"),
                patience=cb_cfg.get("patience", 10),
                mode=cb_cfg.get("mode", "max"),
                min_delta=cb_cfg.get("min_delta", 0.001),
            )
        )

    return callbacks