import pytest
import torch
import torch.nn as nn
from src.models.base_model import BaseModel
from src.models.cnn_baseline import CNNBaseline


# ── Fixtures ─────────────────────────────────────────────
@pytest.fixture
def model():
    """Fresh model for each test — no shared state."""
    return CNNBaseline(num_classes=2, pretrained=False)


# ── Tests ────────────────────────────────────────────────
def test_instantiation(model):
    """Model instantiates without error."""
    assert model is not None
    assert model.num_classes == 2
    assert model.model_name == "cnn_baseline"
    assert model.input_size == 224


def test_inherits_from_base_model(model):
    """Model respects the BaseModel contract."""
    assert isinstance(model, BaseModel)


def test_forward_shape(model):
    """Forward pass returns logits of shape (B, num_classes)."""
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    assert y.shape == (4, 2), f"Expected (4, 2), got {y.shape}"
    assert y.dtype == torch.float32


def test_param_count():
    """Architecture should have exactly 389,410 params."""
    model = CNNBaseline(num_classes=2, pretrained=False)
    total = model.get_num_params(trainable_only=False)
    assert total == 389_410, f"Expected 389,410 params, got {total:,}"


def test_pretrained_true_warns():
    """pretrained=True should warn but not crash."""
    with pytest.warns(UserWarning, match="ignored"):
        _ = CNNBaseline(pretrained=True)


def test_classifier_head_is_linear(model):
    """classifier_head must be a Linear(256, num_classes) for freeze to work."""
    assert isinstance(model.classifier_head, nn.Linear)
    assert model.classifier_head.in_features == 256
    assert model.classifier_head.out_features == 2


def test_freeze_backbone_leaves_only_head_trainable(model):
    """After freeze, only classifier_head params (514) should be trainable."""
    model.freeze_backbone()
    assert model.get_num_params(trainable_only=True) == 514


def test_forward_with_batch_size_one(model):
    """Edge case: single-image batch should work."""
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 2)