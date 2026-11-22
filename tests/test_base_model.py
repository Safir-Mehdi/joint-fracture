"""Unit tests for ``src.models.base_model.BaseModel``.

Verifies abstract base class enforcement, parameter counting, backbone
freezing/unfreezing, and AdamW weight-decay parameter grouping.

    To Test: pytest tests/test_base_model.py -v
"""

import pytest
import torch.nn as nn

from src.models.base_model import BaseModel


# ── A minimal valid subclass for testing ─────────────────
class _DummyModel(BaseModel):
    def __init__(self):
        super().__init__(num_classes=2, model_name="dummy", input_size=224)
        self.backbone = nn.Linear(10, 10)
        self.classifier_head = nn.Linear(10, 2)

    def forward(self, x):
        return self.classifier_head(self.backbone(x))


# ── A broken subclass — forgets to set classifier_head ──
class _NoHeadModel(BaseModel):
    def __init__(self):
        super().__init__(num_classes=2, model_name="no_head", input_size=224)
        self.backbone = nn.Linear(10, 2)

    def forward(self, x):
        return self.backbone(x)


# ── Tests ────────────────────────────────────────────────
def test_cannot_instantiate_base_model():
    """BaseModel is abstract — cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseModel()


def test_subclass_must_implement_forward():
    """A subclass without forward() is still abstract."""
    class Incomplete(BaseModel):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_valid_subclass_works():
    """A valid subclass instantiates and reports params correctly."""
    model = _DummyModel()
    # backbone: 10*10 + 10 = 110 params
    # head:     10*2  +  2 =  22 params
    # total = 132
    assert model.get_num_params(trainable_only=False) == 132
    assert model.get_num_params(trainable_only=True) == 132


def test_freeze_backbone_freezes_only_head():
    """After freeze_backbone(), only classifier_head params are trainable."""
    model = _DummyModel()
    model.freeze_backbone()
    trainable = model.get_num_params(trainable_only=True)
    assert trainable == 22, f"Expected 22 trainable params (head only), got {trainable}"


def test_unfreeze_all_restores_training():
    """After unfreeze_all(), every param is trainable again."""
    model = _DummyModel()
    model.freeze_backbone()
    model.unfreeze_all()
    assert model.get_num_params(trainable_only=True) == 132


def test_freeze_backbone_raises_if_no_classifier_head():
    """freeze_backbone() should raise a clear error if classifier_head is None."""
    model = _NoHeadModel()
    with pytest.raises(RuntimeError, match="did not set self.classifier_head"):
        model.freeze_backbone()


def test_get_parameter_groups_splits_correctly():
    """Weights (ndim>=2) → decay group; biases (ndim==1) → no_decay group."""
    model = _DummyModel()
    groups = model.get_parameter_groups(weight_decay=0.01)

    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0

    # _DummyModel params:
    #   backbone.weight (10x10), backbone.bias (10)
    #   head.weight     (2x10),  head.bias     (2)
    # decay    = backbone.weight + head.weight = 2 params
    # no_decay = backbone.bias   + head.bias   = 2 params
    assert len(groups[0]["params"]) == 2
    assert len(groups[1]["params"]) == 2

def test_unfreeze_by_name_selectively_unfreezes():
    """unfreeze_by_name should unfreeze only params matching the pattern."""
    model = _DummyModel()          # has backbone + classifier_head
    model.freeze_backbone()         # all frozen except head
    frozen_before = model.get_num_params(trainable_only=True)
    assert frozen_before == 22      # head only

    # Unfreeze params whose name contains "backbone"
    n = model.unfreeze_by_name("backbone")
    assert n == 110                 # backbone weight (100) + bias (10)
    assert model.get_num_params(trainable_only=True) == 132


def test_unfreeze_by_name_no_match_returns_zero():
    """If pattern matches nothing, return 0 and don't crash."""
    model = _DummyModel()
    model.freeze_backbone()
    n = model.unfreeze_by_name("nonexistent_layer")
    assert n == 0
    assert model.get_num_params(trainable_only=True) == 22   # head only, unchanged