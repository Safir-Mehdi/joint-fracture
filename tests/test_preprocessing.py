"""
Unit tests for the image preprocessing pipeline in `src/data/preprocessing.py`.

Purpose:
    Validates configuration loading, tensor resizing, data types, and pixel
    normalization behavior using synthetic images.

Usage:
    Run all tests from the project root directory:
        pytest tests/test_preprocessing.py -v

Requirements:
    Requires `configs/base_config.yaml` to exist for default stats testing.
"""

import pytest
import torch
from PIL import Image
from src.data.preprocessing import get_base_transform, load_data_stats

def test_load_data_stats_returns_mean_and_std():
    """Verify load_data_stats returns 3-element float lists for mean and std."""
    mean, std = load_data_stats()
    assert isinstance(mean, list) and len(mean) == 3
    assert isinstance(std, list) and len(std) == 3
    assert all(isinstance(x, float) for x in mean)
    assert all(isinstance(x, float) for x in std)

@pytest.mark.parametrize("size", [224, 384])
def test_get_base_transform_output_shape(size):
    """Verify the transform outputs a float32 tensor with shape (3, size, size)."""
    tf = get_base_transform(image_size=size)
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))  # synthetic
    tensor = tf(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, size, size)
    assert tensor.dtype == torch.float32

def test_get_base_transform_respects_custom_mean_std():
    """Verify if mean/std passed explicitly, they should be used (not read from config)."""
    tf = get_base_transform(image_size=64, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    tensor = tf(img)

    # 128/255 ≈ 0.502, normalized: (0.502 - 0.5) / 0.5 ≈ 0.004  → near 0
    assert torch.allclose(tensor, torch.full((3, 64, 64), 0.004),
                          atol=0.02)
