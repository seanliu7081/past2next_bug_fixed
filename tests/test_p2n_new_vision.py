"""Offline vision contracts: preprocessing, grouping, freezing and restoration."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from oat.perception.dinov3_patch_encoder import DINOv3PatchEncoder
from oat.perception.token_obs_encoder import TokenObservationEncoder
from oat.perception.visual_resampler import VisualResampler


PROCESSOR = dict(do_rescale=True, rescale_factor=1 / 255,
                 do_normalize=True, image_mean=[0.485, 0.456, 0.406],
                 image_std=[0.229, 0.224, 0.225])


class MockDINO(nn.Module):
    """Explicit tiny backbone; prefix sentinels detect accidental CLS/register use."""
    def __init__(self, width=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width, patch_size=16, num_register_tokens=4)
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, width))
        self.last_pixels = None

    def forward(self, pixel_values):
        self.last_pixels = pixel_values
        patches = F.avg_pool2d(pixel_values, 16).mean(1).flatten(1)[..., None] * self.scale
        prefix = patches.new_full((patches.shape[0], 5, patches.shape[-1]), -100)
        return SimpleNamespace(last_hidden_state=torch.cat((prefix, patches), dim=1))


def tiny_dino(**kwargs):
    width = kwargs.pop("width", 16)
    return DINOv3PatchEncoder(backbone=MockDINO(width), processor_config=PROCESSOR,
                              image_size=kwargs.pop("image_size", 32), **kwargs)


def tiny_encoder(**kwargs):
    schema = {"obs": {"front": {"shape": [32, 32, 3], "type": "rgb"},
                      "wrist": {"shape": [32, 32, 3], "type": "rgb"},
                      "state": {"shape": [3], "type": "state"},
                      "task_uid": {"shape": [1], "type": "state"}}}
    return TokenObservationEncoder(schema, n_emb=16, dino_encoder=tiny_dino(),
                                   n_head=4, ffn_dim=32, **kwargs)


def observations():
    return dict(front=torch.randint(256, (2, 2, 32, 32, 3), dtype=torch.uint8),
                wrist=torch.randint(256, (2, 2, 32, 32, 3), dtype=torch.uint8),
                state=torch.randn(2, 2, 3), task_uid=torch.zeros(2, 2, 1))


def test_standard_patch_shape_excludes_prefix_and_backbone_is_frozen():
    model = tiny_dino(width=384, image_size=224).train()
    patches = model(torch.full((2, 128, 128, 3), 255, dtype=torch.uint8))
    assert patches.shape == (2, 196, 384)
    assert not (patches == -100).any()
    assert model.training and not model.backbone.training
    assert not patches.requires_grad
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


@pytest.mark.parametrize("rgb_range,scale", [("0_1", 1.0), ("0_255", 255.0)])
def test_float_rgb_declared_range_matches_byte_processing_exactly_once(rgb_range, scale):
    model = tiny_dino(rgb_range=rgb_range).eval()
    byte_rgb = torch.full((1, 32, 32, 3), 255, dtype=torch.uint8)
    float_rgb = torch.full((1, 32, 32, 3), scale)
    processed = model.preprocess(float_rgb)
    torch.testing.assert_close(processed, model.preprocess(byte_rgb))
    expected = (torch.ones(3) - torch.tensor(PROCESSOR["image_mean"])) / torch.tensor(PROCESSOR["image_std"])
    torch.testing.assert_close(processed[0, :, 0, 0], expected)


def test_float_and_geometry_contracts_reject_ambiguous_inputs():
    model = tiny_dino()
    with pytest.raises(TypeError, match="explicit"):
        model(torch.zeros(1, 32, 32, 3))
    with pytest.raises(ValueError, match="Nonsquare"):
        model(torch.zeros(1, 24, 32, 3, dtype=torch.uint8))
    model = tiny_dino(rgb_range="0_1")
    with pytest.raises(ValueError, match="range"):
        model(torch.full((1, 32, 32, 3), 1.1))
    with pytest.raises(ValueError, match="range"):
        model(torch.full((1, 32, 32, 3), float("nan")))
    assert tiny_dino(geometry="stretch")(torch.zeros(1, 24, 32, 3, dtype=torch.uint8)).shape == (1, 4, 16)


def test_photometric_augmentation_is_disabled_for_evaluation_and_deterministic_paths():
    model = tiny_dino(brightness=0.2, contrast=0.2).train()
    rgb = torch.randint(256, (2, 32, 32, 3), dtype=torch.uint8)
    first, second = model.preprocess(rgb), model.preprocess(rgb)
    assert not torch.equal(first, second)
    deterministic = model.preprocess(rgb, deterministic=True)
    model.eval()
    torch.testing.assert_close(model.preprocess(rgb), deterministic)
    torch.testing.assert_close(model(rgb), model(rgb), rtol=0, atol=0)


def test_multi_camera_frame_layout_and_default_256_visual_tokens():
    model, obs = tiny_encoder(), observations()
    model.eval()
    visual, proprio = model(obs)
    assert visual.shape == (2, 256, 16)
    assert proprio.shape == (2, 2, 16)
    # Independently reconstruct each image: flattened layout must be frame, camera, slot.
    for frame in range(2):
        for camera, port in enumerate(model.rgb_ports):
            expected = model.resampler(model.patch_projection(model.dino_encoder(obs[port][:, frame])))
            expected = expected + model.camera_embedding[camera] + model.frame_embedding[frame]
            offset = (frame * 2 + camera) * 64
            torch.testing.assert_close(visual[:, offset:offset + 64], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_trainable_adapters_receive_gradients_and_dino_remains_frozen(checkpointing):
    model = tiny_encoder(num_queries=4, activation_checkpointing=checkpointing).train()
    visual, proprio = model(observations())
    (visual.square().mean() + proprio.square().mean()).backward()
    assert not model.dino_encoder.backbone.training
    for name, parameter in model.named_parameters():
        if name.startswith("dino_encoder."):
            assert parameter.grad is None
        elif parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_camera_and_state_ports_fail_explicitly():
    model, obs = tiny_encoder(num_queries=4), observations()
    del obs["wrist"]
    with pytest.raises(KeyError, match="wrist"):
        model(obs)
    obs = observations()
    obs["state"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        model(obs)


def test_spatial_order_affects_resampling_and_query_slots_remain_distinct():
    model = VisualResampler(n_emb=16, n_head=4, ffn_dim=32, num_queries=4, grid_size=2).eval()
    patches = torch.randn(2, 4, 16)
    output = model(patches)
    assert output.shape == (2, 4, 16)
    assert not torch.equal(output[:, 0], output[:, 1])
    assert not torch.allclose(model(patches.flip(1)), output)


def test_missing_fresh_backbone_never_silently_constructs_random_weights(tmp_path):
    with pytest.raises(ValueError, match="local pretrained_path"):
        DINOv3PatchEncoder()
    with pytest.raises(FileNotFoundError, match="config.json"):
        DINOv3PatchEncoder(pretrained_path=str(tmp_path))


def test_offline_restore_constructs_without_external_files_or_pretrained_calls(monkeypatch):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    def forbid_download(*args, **kwargs):
        raise AssertionError("Offline restore must not call from_pretrained")

    monkeypatch.setattr(DINOv3ViTModel, "from_pretrained", forbid_download)
    model = DINOv3PatchEncoder(load_mode="restore",
                               config=DINOv3ViTConfig(num_register_tokens=4).to_dict(),
                               processor_config=PROCESSOR, revision="a" * 40,
                               weight_sha256="b" * 64, pretrained_path="/moved/unavailable")
    with pytest.raises(RuntimeError, match="complete policy state_dict"):
        model(torch.zeros(1, 32, 32, 3, dtype=torch.uint8))
    # Simulates strict state application from an artifact (no pretrained source).
    model.load_state_dict(model.state_dict(), strict=True)
    assert model._restore_ready
    # Exercise the installed Transformers implementation with synthetic weights;
    # this validates its output contract, not pretrained DINO acceptance.
    patches = model(torch.zeros(1, 32, 32, 3, dtype=torch.uint8))
    assert patches.shape == (1, 196, 384)
    assert torch.isfinite(patches).all()
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    exported = model.export_config()
    assert exported["load_mode"] == "restore"
    assert "pretrained_path" not in exported
    assert exported["revision"] == "a" * 40
    assert exported["processor_config"] == PROCESSOR
