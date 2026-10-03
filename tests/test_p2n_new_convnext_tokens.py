"""CPU contracts for fusion, camera/time layout, gradients and factory dispatch."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from oat.perception.convnext_token_obs_encoder import (
    ConvNeXtTokenObservationEncoder, DualScaleFusion,
)
from oat.perception.obs_encoder_factory import (
    build_observation_encoder, normalize_observation_encoder_config,
    maintain_frozen_backbone_mode,
)


META = {"obs": {"front": {"shape": [16, 16, 3], "type": "rgb"},
                "wrist": {"shape": [16, 16, 3], "type": "rgb"},
                "state": {"shape": [7], "type": "state"}},
        "action": {"shape": [7]}}


class FeatureStub(nn.Module):
    """Explicit injected features; never presented as pretrained weights."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 640, 1).requires_grad_(False).eval()
        self.last_images = None

    def maintain_frozen_backbone_mode(self):
        self.backbone.requires_grad_(False).eval()

    def forward(self, images):
        self.last_images = images.detach().clone()
        with torch.no_grad():
            values = self.backbone(images.permute(0, 3, 1, 2).float() / 255)
            return (F.adaptive_avg_pool2d(values[:, :320], (14, 14)),
                    F.adaptive_avg_pool2d(values, (7, 7)))


def encoder(**kwargs):
    return ConvNeXtTokenObservationEncoder(META, convnext_encoder=FeatureStub(), **kwargs)


def obs(batch=2):
    return {key: (torch.randint(0, 256, (batch, 2, *info["shape"]), dtype=torch.uint8)
                  if info["type"] == "rgb" else torch.randn(batch, 2, *info["shape"]))
            for key, info in META["obs"].items()}


def test_production_visual_and_proprio_shapes_and_actual_parameter_counts():
    model = encoder().eval()
    with torch.no_grad():
        visual, proprio = model(obs(1))
    assert visual.shape == (1, 256, 768)
    assert proprio.shape == (1, 2, 768)
    assert model.output_feature_dim() == 768
    assert model.num_queries == model.resampler.num_queries == 64
    counts = model.parameter_counts()
    assert counts["fusion_resampler_output"] == sum(
        counts[key] for key in ("fusion", "resampler", "output_projection"))
    assert 2_600_000 < counts["fusion_resampler_output"] < 2_900_000


def test_batch_frame_camera_order_and_post_projection_embeddings():
    model = encoder(n_emb=8, visual_resampler_dim=8, visual_resampler_heads=2,
                    visual_resampler_ffn_dim=16, num_queries=2, resampler_depth=1).eval()
    sample = obs()
    for camera_index, camera in enumerate(model.rgb_ports):
        for b in range(2):
            for frame in range(2):
                sample[camera][b, frame].fill_(b * 4 + frame * 2 + camera_index)
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.zero_()
        model.camera_embedding[0].fill_(1)
        model.camera_embedding[1].fill_(2)
        model.frame_embedding[0].fill_(10)
        model.frame_embedding[1].fill_(20)
        visual, _ = model(sample)
    assert model.convnext_encoder.last_images[:, 0, 0, 0].tolist() == list(range(8))
    assert visual[0, :, 0].tolist() == [11, 11, 12, 12, 21, 21, 22, 22]
    torch.testing.assert_close(visual[0], visual[1])


def test_fusion_normalizes_channels_per_position_and_both_scales_receive_gradients():
    torch.manual_seed(42)
    model = DualScaleFusion(8)
    f16 = torch.randn(2, 320, 14, 14)
    f32 = torch.randn(2, 640, 7, 7)
    actual = model(f16, f32)
    pre = (model.projection16(f16) + F.interpolate(model.projection32(f32), (14, 14),
           mode="bilinear", align_corners=False)) / (2 ** 0.5)
    expected = pre * torch.rsqrt(pre.square().mean(dim=1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual, expected.flatten(2).transpose(1, 2))
    (actual * torch.randn_like(actual)).sum().backward()
    assert model.projection16.weight.grad.abs().sum() > 0
    assert model.projection32.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("bf16", [False, True])
def test_adapter_gradients_frozen_features_and_mode_restoration(bf16):
    model = encoder(n_emb=16, visual_resampler_dim=16, visual_resampler_heads=2,
                    visual_resampler_ffn_dim=32, num_queries=4, resampler_depth=1).train()
    assert model.training and model.convnext_encoder.training
    assert not model.convnext_encoder.backbone.training
    sample = obs(1)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        visual, proprio = model(sample)
        loss = visual.float().square().mean() + proprio.float().square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    model.eval().train()
    maintain_frozen_backbone_mode(model)
    assert not model.convnext_encoder.backbone.training


def test_schema_errors_and_nonfinite_state_are_not_silently_accepted():
    model = encoder(n_emb=8, visual_resampler_dim=8, visual_resampler_heads=2,
                    visual_resampler_ffn_dim=16, num_queries=2, resampler_depth=1)
    sample = obs(1)
    with pytest.raises(KeyError, match="wrist"):
        model({key: value for key, value in sample.items() if key != "wrist"})
    sample["wrist"] = sample["wrist"].float()
    with pytest.raises(ValueError, match="dtype"):
        model(sample)
    sample = obs(1)
    sample["state"].fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        model(sample)


def test_factory_has_strict_whitelist_and_legacy_dino_explanation():
    legacy = {"shape_meta": META, "dino": {"rgb_range": "uint8"}}
    normalized = normalize_observation_encoder_config(legacy)
    assert normalized["encoder_type"] == "dinov3_tokens"
    assert normalized["schema_version"] == 1
    assert "encoder_type" not in legacy
    with pytest.raises(ValueError, match="target"):
        normalize_observation_encoder_config(dict(legacy, _target_="arbitrary.executable"))
    with pytest.raises(ValueError, match="disagree"):
        normalize_observation_encoder_config(dict(legacy, encoder_type="convnextv2_tokens",
            _target_="oat.perception.token_obs_encoder.TokenObservationEncoder"))
    with pytest.raises(ValueError, match="DINO"):
        normalize_observation_encoder_config(dict(legacy, encoder_type="convnextv2_tokens"))
    with pytest.raises(ValueError, match="explicit"):
        normalize_observation_encoder_config({"shape_meta": META})
    with pytest.raises(ValueError, match="schema_version"):
        normalize_observation_encoder_config(dict(legacy, schema_version=100))


def test_factory_restore_drops_source_path_without_mutating_input(monkeypatch):
    import oat.perception.obs_encoder_factory as factory
    calls = []
    monkeypatch.setattr(factory, "ConvNeXtTokenObservationEncoder", lambda **cfg: calls.append(cfg))
    cfg = dict(encoder_type="convnextv2_tokens", shape_meta=META,
               convnext={"model_path": "/missing/source", "revision": "a" * 40})
    original = copy.deepcopy(cfg)
    build_observation_encoder(cfg, construction_mode="restore")
    assert cfg == original
    assert "model_path" not in calls[0]["convnext"]
    assert calls[0]["convnext"]["construction_mode"] == "restore"
    assert calls[0]["visual_resampler_dim"] == 256
    assert calls[0]["n_emb"] == 768
    assert calls[0]["fusion"]["normalization"] == "rmsnorm_channels"
