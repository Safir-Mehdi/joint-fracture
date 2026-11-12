import pytest
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from src.data.dataloader import build_dataloader
from src.utils.seed import set_seed

# Minimal config for tests — no YAML I/O needed
TEST_CFG = {
    "dataloader": {
        "batch_size": 8,
        "num_workers": 0,
        "use_weighted_sampler": True,
    }
}

TRAIN_ROOT = "data/raw/train"
VAL_ROOT   = "data/raw/val"

def _simple_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

def test_build_dataloader_returns_dataloader():
    dl = build_dataloader(TEST_CFG, TRAIN_ROOT, _simple_transform(), mode="train")
    assert isinstance(dl, DataLoader)

def test_batch_shapes():
    dl = build_dataloader(TEST_CFG, TRAIN_ROOT, _simple_transform(), mode="train")
    images, labels = next(iter(dl))
    assert images.shape[1:] == (3, 224, 224)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

def test_val_is_deterministic():
    set_seed(42)
    dl1 = build_dataloader(TEST_CFG, VAL_ROOT, _simple_transform(), mode="val")
    dl2 = build_dataloader(TEST_CFG, VAL_ROOT, _simple_transform(), mode="val")

    b1 = next(iter(dl1))[1].tolist()
    b2 = next(iter(dl2))[1].tolist()
    assert b1 == b2, "Val mode must be deterministic"

def test_weighted_sampler_balances_batches():
    """Over many batches, class ratio should be much closer to 50/50 than 29/71."""
    set_seed(42)
    dl = build_dataloader(TEST_CFG, TRAIN_ROOT, _simple_transform(), mode="train")

    total = 0
    fractured = 0
    for _ in range(10):
        _, labels = next(iter(dl))
        total += len(labels)
        fractured += (labels == 0).sum().item()

    ratio = fractured / total
    # Raw ratio is ~0.29; a working sampler pulls it toward 0.5
    assert 0.35 < ratio < 0.65, f"Sampler not balancing: ratio={ratio:.2f}"

def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        build_dataloader(TEST_CFG, TRAIN_ROOT, _simple_transform(), mode="wrong")