"""
Purpose: We calculate Mean and Std to normalize pixel values so the neural network learns faster and more accurately. So, it the raw pixel brightness don't confuse the model and slow down training.

One-time utility: compute mean & std per channel (RGB) for a dataset folder.

Usage (from project root):
    python -m src.utils.dataset_stats --root data/raw/train

Then hardcode the printed numbers into configs/base_config.yaml which latter be use in src/data/preprocessing.py.
"""

import argparse
import numpy as np
import yaml
from PIL import Image
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def compute_mean_std(root:str | Path) -> tuple[list,list]:
     """
    Compute per-channel mean and std over all images in `root` (recursively).

    Returns:
        mean (list[float, float, float]) — RGB order
        std  (list[float, float, float]) — RGB order
    """

     root = Path(root)
     if not root.exists():
          raise FileNotFoundError(f"Folder not found: {root}")

     image = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

     if not image:
          raise ValueError(f"No images found under: {root}")

     # Accumulators — sum and sum-of-squares per channel
     channel_sum = np.zeros(3, dtype=np.float64)
     channel_sq_sum = np.zeros(3, dtype=np.float64)
     pixel_count = 0

     for img_path in tqdm(image, desc="Computing mean/std"):
        img = Image.open(img_path).convert("RGB")           # force 3 channels
        arr = np.asarray(img, dtype=np.float32) / 255.0     # 0-1 range

        channel_sum += arr.sum(axis=(0, 1))
        channel_sq_sum += (arr ** 2).sum(axis=(0, 1))
        pixel_count += arr.shape[0] * arr.shape[1]

     mean = channel_sum / pixel_count
     std = np.sqrt((channel_sq_sum / pixel_count) - (mean ** 2))

     return mean.tolist(), std.tolist()

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--root", type=str, default="data/raw/train", help="path to the folder that containing training images")
     parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="YAML config file to write mean/std into")

     args = parser.parse_args()

     # 1. Compute
     mean, std = compute_mean_std(args.root)
     mean_rounded = [round(x, 8) for x in mean]
     std_rounded  = [round(x, 8) for x in std]

     # 2. Load existing config (or start fresh)
     config_path = Path(args.config)
     if config_path.exists():
          with config_path.open("r", encoding="utf-8") as f:
               cfg = yaml.safe_load(f) or {}
     else:
          config_path.parent.mkdir(parents=True, exist_ok=True)
          cfg = {}

     # 3. Update the data section (preserve everything else)
     cfg.setdefault("data", {})
     cfg["data"]["mean"] = mean_rounded
     cfg["data"]["std"] = std_rounded

     # 4. write back7
     with config_path.open("w", encoding="utf-8") as f:
          yaml.safe_dump(cfg, f, sort_keys=False)

     # 5. Print summary
     print("\n" + "=" * 60)
     print(f"Dataset: {args.root}")
     print(f"Mean (RGB): {mean_rounded}")
     print(f"Std  (RGB): {std_rounded}")
     print("=" * 60)
     print(f"\n✅ Saved to: {config_path}")
     print(f"   → data.mean = {mean_rounded}")
     print(f"   → data.std  = {std_rounded}")
     print()