"""
DataLoader factory — one place to build loaders for train / val / test.

Handles the 2.5:1 class imbalance (386 fractured / 948 normal) via
WeightedRandomSampler for the TRAIN split only.

    ⚠️ Notice: You can tune or add these parameter accordingly using configs/train/default_train.yaml and can bring you own dataset to the DatasetLoader:
    Usage:
        from src.data.dataset import <YourOwnDataset>
    
    Take dataset.py as a reference.
"""

import torch
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataset import JointFractureDataset

def _make_weighted_sampler(dataset: JointFractureDataset) -> WeightedRandomSampler:
    """
    Build a sampler that oversamples the minority class so each batch is
    roughly balanced (≈50/50) instead of the raw 29/71 split (Like for this case).
    """

    labels = [label for _, label in dataset._inner.samples]
    counts = Counter(labels)
    class_weights = {cls: 1.0/count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,   # required for oversampling
    )

def build_dataloader(cfg: dict, root_dir: str, transform, mode: str = "train") -> DataLoader:
    """
    Build a DataLoader for a split folder.

    Args:
        cfg: merged config dict (from load_config(training=True))
        root_dir: e.g. "data/raw/train"
        transform: torchvision transform pipeline
        mode: "train" | "val" | "test" — controls shuffle & sampler behavior
    """

    if mode not in {"train", "val", "test"}:
        raise ValueError(f"mode must be 'train', 'val', or 'test' (got '{mode}')")

    dataset = JointFractureDataset(root=root_dir, transform=transform)

    dl_cfg = cfg["dataloader"]
    batch_size = dl_cfg["batch_size"]
    num_workers = dl_cfg.get("num_workers", 4)
    use_sampler = dl_cfg.get("use_weighted_sampler", False) and mode == "train"

    if use_sampler:
        loader_kwargs = {
            "batch_size": batch_size,
            "sampler": _make_weighted_sampler(dataset),
            # Because WeightedRandomSampler already shuffles — passing shuffle=True too crashes PyTorch
            "shuffle": False,
        }
    else:
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": (mode == "train"),
        }

    return DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )