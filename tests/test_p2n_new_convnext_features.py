"""Offline Nano preprocessing, weight coverage, freeze and restoration tests.

Any on-disk checkpoints produced here are synthetic test weights, never a
substitute for acceptance of the user's pinned pretrained snapshot.
"""
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from oat.perception.convnext_feature_encoder import (
    CONVNEXT_MODEL_NAME, ConvNeXtFeatureEncoder, IMAGE_MEAN, IMAGE_STD,
    TIMM_VERSION, _load_feature_state, _read_local_state, _resolve_weight_file,
)


class TinyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale16 = nn.Parameter(torch.linspace(0.5, 1.5, 320))
        self.scale32 = nn.Parameter(torch.linspace(0.5, 1.5, 640))

    def forward(self, pixels):
        return [F.avg_pool2d(pixels, 16).mean(1, keepdim=True) * self.scale16[None, :, None, None],
                F.avg_pool2d(pixels, 32).mean(1, keepdim=True) * self.scale32[None, :, None, None]]


def injected(**kwargs):
    return ConvNeXtFeatureEncoder(backbone=TinyFeatures(), **kwargs)


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(old, 2))
    yield
    torch.set_num_threads(old)


def test_raw_preprocessing_uses_full_nonsquare_field_once_in_fp32():
    model = injected().eval()
    rgb = torch.zeros(1, 17, 29, 3, dtype=torch.uint8)
    rgb[:, 0] = 255
    rgb[:, -1] = 127
    rgb[..., 1] = 64
    expected = F.interpolate(rgb.float().permute(0, 3, 1, 2) / 255,
                             size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
    expected = (expected - torch.tensor(IMAGE_MEAN)[None, :, None, None]) / torch.tensor(IMAGE_STD)[None, :, None, None]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model.preprocess(rgb)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("input_range,scale", [("float_0_1", 1), ("float_0_255", 255)])
def test_explicit_float_ranges_match_uint8_without_batch_range_guessing(input_range, scale):
    floats, byte = injected(input_range=input_range).eval(), injected().eval()
    rgb = torch.full((1, 17, 29, 3), scale / 255)
    torch.testing.assert_close(floats.preprocess(rgb), byte.preprocess(torch.ones_like(rgb, dtype=torch.uint8)))
    with pytest.raises(ValueError, match="range"):
        floats.preprocess(torch.full_like(rgb, scale + 1))
    with pytest.raises(ValueError, match="range"):
        floats.preprocess(torch.full_like(rgb, float("nan")))
    with pytest.raises(TypeError, match="floating"):
        floats.preprocess(torch.zeros_like(rgb, dtype=torch.uint8))


def test_default_raw_schema_rejects_float_or_malformed_images():
    model = injected()
    with pytest.raises(TypeError, match="explicit"):
        model(torch.ones(1, 20, 20, 3))
    with pytest.raises(ValueError, match="NHWC|raw RGB"):
        model(torch.zeros(1, 3, 20, 20, dtype=torch.uint8))


def test_frozen_eval_features_allow_trainable_projection_backward_and_deepcopy():
    import copy
    model = injected().train()
    rgb = torch.randint(256, (2, 32, 32, 3), dtype=torch.uint8)
    f16, f32 = model(rgb)
    assert f16.shape == (2, 320, 14, 14)
    assert f32.shape == (2, 640, 7, 7)
    assert not f16.requires_grad and not f32.requires_grad
    assert not f16.is_inference() and not f32.is_inference()
    adapter16, adapter32 = nn.Conv2d(320, 3, 1), nn.Conv2d(640, 3, 1)
    (adapter16(f16).square().mean() + adapter32(f32).square().mean()).backward()
    assert adapter16.weight.grad is not None and adapter32.weight.grad is not None
    assert model.training and not model.backbone.training
    assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())
    ema = copy.deepcopy(model).train()
    assert not ema.backbone.training
    model.backbone.train()
    model.maintain_frozen_backbone_mode()
    assert not model.backbone.training


def test_deterministic_preprocessing_disables_both_augmentations():
    model = injected().train()
    rgb = torch.randint(256, (3, 32, 32, 3), dtype=torch.uint8)
    torch.manual_seed(7)
    assert not torch.equal(model.preprocess(rgb), model.preprocess(rgb))
    deterministic = model.preprocess(rgb, deterministic=True)
    model.eval()
    torch.testing.assert_close(model.preprocess(rgb), deterministic, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs,match", [
    ({"frozen": False}, "frozen"), ({"image_size": 128}, "image_size"),
    ({"feature_stages": (1, 3)}, "feature_stages"), ({"interpolation": "bilinear"}, "bicubic"),
    ({"backbone_config": {"features_only": True}}, "backbone_config"),
    ({"input_range": "auto"}, "input_range"), ({"mean": (0, 0)}, "mean and std"),
])
def test_fixed_production_contract_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        injected(**kwargs)


def test_missing_local_weights_and_unpinned_revision_fail_before_random_fallback(tmp_path):
    with pytest.raises(ValueError, match="explicit local model_path"):
        ConvNeXtFeatureEncoder()
    with pytest.raises(FileNotFoundError, match="do not exist"):
        ConvNeXtFeatureEncoder(model_path=str(tmp_path / "absent.pt"))
    torch.save({"weight": torch.zeros(1)}, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="40-character"):
        ConvNeXtFeatureEncoder(model_path=str(tmp_path / "model.pt"))
    with pytest.raises(ValueError, match="provenance"):
        ConvNeXtFeatureEncoder(construction_mode="restore")
    with pytest.raises(ValueError, match="injected"):
        injected().export_config()


def test_loader_only_excludes_explicit_head_keys_and_requires_all_feature_parameters():
    model = nn.Module()
    model.add_module("stem_0", nn.Linear(3, 4))
    state = {"stem.0.weight": model.stem_0.weight.detach().clone(),
             "stem.0.bias": model.stem_0.bias.detach().clone(),
             "head.norm.weight": torch.ones(640), "head.norm.bias": torch.zeros(640),
             "head.fc.weight": torch.zeros(1000, 640), "head.fc.bias": torch.zeros(1000)}
    metadata = _load_feature_state(model, state)
    assert len(metadata["excluded_head_keys"]) == 4
    assert metadata["feature_state_keys"] == 2
    with pytest.raises(ValueError, match="missing=.*stem_0.bias"):
        _load_feature_state(model, {k: v for k, v in state.items() if k != "stem.0.bias"})
    with pytest.raises(ValueError, match="unexpected=.*head.surprise"):
        _load_feature_state(model, dict(state, **{"head.surprise": torch.zeros(1)}))
    with pytest.raises(ValueError, match="classification head shape"):
        _load_feature_state(model, dict(state, **{"head.fc.bias": torch.zeros(21841)}))
    with pytest.raises(ValueError, match="Feature shape mismatch"):
        _load_feature_state(model, dict(state, **{"stem.0.weight": torch.zeros(4, 2)}))
    with pytest.raises(ValueError, match="Multiple checkpoint keys"):
        _load_feature_state(model, dict(state, **{"stem_0.weight": state["stem.0.weight"]}))


def test_official_fb_prefix_remapping_and_wrapped_local_state(tmp_path):
    model = nn.Module()
    model.add_module("stem_0", nn.Conv2d(3, 80, 4, stride=4))
    official = {"module.downsample_layers.0.0.weight": model.stem_0.weight.detach(),
                "module.downsample_layers.0.0.bias": model.stem_0.bias.detach(),
                "module.norm.weight": torch.ones(640), "module.norm.bias": torch.zeros(640),
                "module.head.weight": torch.ones(1000, 640), "module.head.bias": torch.zeros(1000)}
    path = tmp_path / "official.pt"
    torch.save({"model": official}, path)
    state, container = _read_local_state(path)
    assert container == "model"
    result = _load_feature_state(model, state)
    assert result["source_format"] == "facebook" and len(result["excluded_head_keys"]) == 4
    assert _resolve_weight_file(str(tmp_path))[0] == path
    torch.save(official, tmp_path / "second.pt")
    with pytest.raises(ValueError, match="unambiguous"):
        _resolve_weight_file(str(tmp_path))


def test_failed_state_load_does_not_unblock_restore():
    model = injected()
    model._restore_ready = False
    state = dict(model.state_dict())
    state.pop("backbone.scale16")
    model.load_state_dict(state, strict=False)
    with pytest.raises(RuntimeError, match="complete policy state_dict"):
        model(torch.zeros(1, 32, 32, 3, dtype=torch.uint8))
    state = dict(model.state_dict())
    state["backbone.scale16"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(state, strict=True)
    assert not model._restore_ready
    model.load_state_dict(model.state_dict(), strict=True)
    assert model._restore_ready


def test_real_timm_architecture_fresh_coverage_and_offline_restore(tmp_path, monkeypatch):
    timm = pytest.importorskip("timm")
    if timm.__version__ != TIMM_VERSION:
        pytest.skip(f"requires pinned timm {TIMM_VERSION}")
    original_create = timm.create_model

    def local_only_create(*args, **kwargs):
        assert kwargs.get("pretrained") is False, "no external weight loading is permitted"
        return original_create(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", local_only_create)
    source = original_create(CONVNEXT_MODEL_NAME, pretrained=False, features_only=True, out_indices=(2, 3))
    # Synthetic state verifies full feature coverage and real implementation shape.
    # The classification names also exercise timm -> FeatureListNet remapping.
    state = {}
    for name, value in source.state_dict().items():
        name = name.replace("stem_0.", "stem.0.").replace("stem_1.", "stem.1.")
        for stage in range(4):
            name = name.replace(f"stages_{stage}.", f"stages.{stage}.")
        state[name] = value
    path = tmp_path / "synthetic_test_weights.pt"
    torch.save(state, path)
    fresh = ConvNeXtFeatureEncoder(model_path=str(path), revision="a" * 40).eval()
    assert sum(p.numel() for p in fresh.backbone.parameters()) == 14_981_520
    assert fresh.metadata["feature_state_keys"] == len(state)
    assert len(fresh.metadata["weight_sha256"]) == 64
    assert fresh.metadata["timm_version"] == TIMM_VERSION
    rgb = torch.randint(256, (1, 128, 128, 3), dtype=torch.uint8)
    expected = fresh(rgb)
    assert expected[0].shape == (1, 320, 14, 14) and expected[1].shape == (1, 640, 7, 7)
    config = fresh.export_config()
    assert "model_path" not in config and config["construction_mode"] == "restore"
    saved = fresh.state_dict()
    path.unlink()
    restored = ConvNeXtFeatureEncoder(**config, model_path="/moved/source/is/not/required").eval()
    with pytest.raises(RuntimeError, match="complete policy state_dict"):
        restored(rgb)
    restored.load_state_dict(saved, strict=True)
    actual = restored(rgb)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert restored.export_config() == config
    incompatible = dict(config)
    incompatible["metadata"] = dict(config["metadata"], timm_version="0.0.0")
    with pytest.raises(ValueError, match="pinned timm"):
        ConvNeXtFeatureEncoder(**incompatible)
