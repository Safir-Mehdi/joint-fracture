"""
Unit tests for the training image augmentation pipeline in `src/data/augmentation.py`.

Purpose:
    Validates that `get_train_transform` outputs properly shaped float32
    tensors and applies random augmentations across repeated calls.

Usage:
    pytest tests/test_augmentation.py -v
"""

import torch
import pytest
from PIL import Image
from src.data.augmentation import get_train_transform

@pytest.mark.parametrize("size", [224, 384])
def test_train_transform_output_shape(size):
    """Verify the training transform outputs a float32 tensor with shape (3, size, size)."""
    tf = get_train_transform(image_size=size)
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    tensor = tf(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, size, size)
    assert tensor.dtype == torch.float32

def test_train_transform_is_random():
    """
    Two calls on the SAME image must produce DIFFERENT tensors.
    
        ⚠️ Note: the randomness test could theoretically fail if all random ops happen to produce zero change (e.g., flip on a symmetric image, rotation lands on 0°, jitter lands on 0). With a flat grey image and multiple ops, this is extremely unlikely but not impossible.

        Safer randomness check: run 5 times, ensure not all identical
    """
    tf = get_train_transform(image_size=224)
    img = Image.new("RGB", (512, 512), color=(100, 100, 100))

    a = tf(img)
    b = tf(img)

    # Highly unlikely to be identical across 10+ random ops
    assert not torch.equal(a, b), "Augmentation is not random!"