# tests/test_dataset.py
from src.data.dataset import JointFractureDataset

def test_dataset_loads():
    ds = JointFractureDataset("data/raw/train")
    assert len(ds) > 0

def test_dataset_returns_image_and_label():
    ds = JointFractureDataset("data/raw/train")
    img, label = ds[0]
    assert label in (0, 1)
    assert img is not None

def test_summary_counts_match_length():
    ds = JointFractureDataset("data/raw/train")
    assert sum(ds.summary().values()) == len(ds)

# import sys
# print("\n--- PYTHON SEARCH PATHS ---", *sys.path, sep="\n")