"""
JointFractureDataset — thin wrapper around torchvision ImageFolder.

Expected folder structure:
    data/raw/train/0/*.png    ← class 0
    data/raw/train/1/*.png    ← class 1
    data/raw/val/0/*.png
    data/raw/val/1/*.png
    data/raw/test/0/*.png
    data/raw/test/1/*.png

What this file does:
    - Opens one image at a time
    - Returns (image_tensor, label)
    - Applies the transform passed in (from preprocessing/augmentation)
"""

from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class JointFractureDataset(Dataset):
    """
    A thin wrapper around torchvision's ImageFolder.

    Args:
        root (str | Path): folder for ONE split, e.g. "data/raw/train"
        transform (callable, optional): transform pipeline (built in preprocessing.py / augmentation.py)
    """
    # Class-id -> human-readable name
    CLASS_NAMES ={0: "fractured", 1: "normal"}
    def __init__(self, root: str | Path, transform=None):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        
        self._inner = ImageFolder(root=str(root), transform=transform)
        self.transform = transform

        self.classes = self._inner.classes                  # Returns labels ['0', '1']
        self.class_to_idx = self._inner.class_to_idx        # Returns labels with index {'0': 0, '1': 1}

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int):
        # ImageFolder handles: open image → apply transform → return tensor
        image, label = self._inner[idx]
        return image, label

    def summary(self) -> dict:
        """Return {class_name: count} for a quick sanity check."""
        counts = Counter(label for _, label in self._inner.samples)

        return{
            self.CLASS_NAMES.get(cls_id, str(cls_id)): counts[cls_id]
            for cls_id in sorted(counts.keys())
        }

    def __repr__(self) -> str:
        return (
            f"FractureDataset(len={len(self)}, "
            f"classes={self.classes}, "
            f"summary={self.summary()})"
        )