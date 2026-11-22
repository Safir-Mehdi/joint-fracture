
"""Unit tests for ``src.models.model_factory``.

Verifies model instantiation, registry listing, config defaults/overrides,
and validation errors for malformed configurations.

    To execute:
                pytest tests/test_model_factory.py -v      — For all instances
                pytest "tests/test_model_factory.py::test_all_registered_models_instantiate[name]" -v    — For single instance name -> efficientnet_b3
"""

import pytest

from src.models.base_model import BaseModel
from src.models.cnn_baseline import CNNBaseline
from src.models.model_factory import (
    MODEL_REGISTRY,
    ModelFactoryError,
    build_model,
    list_models,
)
from src.models.resnet import ResNetModel


# ── Fixtures ─────────────────────────────────────────────
@pytest.fixture
def cnn_cfg():
    """Minimal valid config for cnn_baseline (pretrained=False → fast)."""
    return {"model": {"name": "cnn_baseline", "num_classes": 2, "pretrained": False}}


@pytest.fixture
def resnet_cfg():
    """Minimal valid config for resnet50 (pretrained=False → fast)."""
    return {"model": {"name": "resnet50", "num_classes": 2, "pretrained": False}}


# ── Happy path ───────────────────────────────────────────
def test_build_cnn_baseline(cnn_cfg):
    """Valid config builds the correct class."""
    model = build_model(cnn_cfg)
    assert isinstance(model, CNNBaseline)
    assert isinstance(model, BaseModel)


def test_build_resnet(resnet_cfg):
    """Valid config builds ResNetModel."""
    model = build_model(resnet_cfg)
    assert isinstance(model, ResNetModel)


def test_list_models_returns_registered_names():
    """list_models() returns all registry keys, sorted."""
    names = list_models()
    assert names == sorted(MODEL_REGISTRY.keys())
    assert "cnn_baseline" in names
    assert "resnet50" in names


# ── Overrides and defaults ───────────────────────────────
def test_num_classes_override():
    """Explicit num_classes is passed through."""
    cfg = {"model": {"name": "cnn_baseline", "num_classes": 5, "pretrained": False}}
    model = build_model(cfg)
    assert model.num_classes == 5
    assert model.classifier_head.out_features == 5


def test_defaults_when_keys_missing():
    """Missing num_classes/pretrained → defaults (2 / True)."""
    cfg = {"model": {"name": "cnn_baseline"}}   # no num_classes, no pretrained
    model = build_model(cfg)
    assert model.num_classes == 2


# ── Error paths ──────────────────────────────────────────
def test_unknown_model_raises():
    cfg = {"model": {"name": "resnet999"}}
    with pytest.raises(ModelFactoryError, match="Unknown model 'resnet999'"):
        build_model(cfg)


def test_missing_model_section_raises():
    with pytest.raises(ModelFactoryError, match="missing the required 'model' section"):
        build_model({})


def test_model_section_not_dict_raises():
    with pytest.raises(ModelFactoryError, match="cfg\\['model'\\] must be a dict"):
        build_model({"model": "resnet50"})


def test_missing_name_raises():
    with pytest.raises(ModelFactoryError, match="missing the required 'name' key"):
        build_model({"model": {}})


def test_empty_name_raises():
    with pytest.raises(ModelFactoryError, match="non-empty string"):
        build_model({"model": {"name": ""}})


def test_cfg_not_dict_raises():
    with pytest.raises(ModelFactoryError, match="Expected cfg to be a dict"):
        build_model(None)   # type: ignore[arg-type]


# ── Integration — every registered model must instantiate ──
@pytest.mark.parametrize("name", list(MODEL_REGISTRY.keys()))
def test_all_registered_models_instantiate(name):
    """Every model in the registry builds with the default config (pretrained=False)."""
    cfg = {"model": {"name": name, "num_classes": 2, "pretrained": False}}
    model = build_model(cfg)
    assert isinstance(model, BaseModel)
    assert model.num_classes == 2
    assert model.get_num_params() > 0