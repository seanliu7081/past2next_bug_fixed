"""Actual original-CNN equivalence, normalization, ordering and schema checks."""
import copy

import hydra
import pytest
import torch
from torch import nn

from oat.model.common.context_batch import ContextBatch as LegacyContextBatch
from oat.model.common.context_batch import Segment as LegacySegment
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.common.original_fused_context_batch import ContextBatch, Segment
from oat.perception.crop_randomizer import CropRandomizer
from oat.perception.fused_obs_encoder import FusedObservationEncoder
from oat.perception.original_fused_obs_adapter import (
    OriginalFusedObservationAdapter, normalize_original_obs_config,
)


def shape_meta():
    # Deliberately interleaved and nonalphabetical: neither dictionary sorting
    # nor interleaving cameras with state is the original fused output order.
    return dict(obs={
        "z_camera": dict(shape=[128, 128, 3], type="rgb"),
        "position": dict(shape=[3], type="state"),
        "a_camera": dict(shape=[128, 128, 3], type="rgb"),
        "rotation": dict(shape=[6], type="state"),
        "gripper": dict(shape=[1], type="state"),
        "task_uid": dict(shape=[1], type="state"),
    }, action=dict(shape=[7]))


def normalizer(meta):
    result = LinearNormalizer()
    for key, attr in meta["obs"].items():
        rgb = attr["type"] == "rgb"
        count = 1 if rgb else attr["shape"][0]
        lo, hi = (0.0, 255.0) if rgb else (-2.0, 6.0)
        result[key] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((count,), 2.0 / (hi - lo)),
            offset=torch.full((count,), -1.0 - 2.0 * lo / (hi - lo)),
            input_stats_dict=dict(
                min=torch.full((count,), lo), max=torch.full((count,), hi),
                mean=torch.full((count,), (lo + hi) / 2), std=torch.ones(count),
            ),
        )
    return result


def observations(meta):
    values = {}
    for key, attr in meta["obs"].items():
        size = (1, 2, *attr["shape"])
        values[key] = (torch.randint(0, 256, size, dtype=torch.uint8)
                       if attr["type"] == "rgb" else torch.randn(size) + 3.0)
    return values


def original(meta):
    return FusedObservationEncoder(
        shape_meta=meta,
        vision_encoder=dict(
            _target_="oat.perception.robomimic_vision_encoder.RobomimicRgbEncoder",
            crop_shape=[112, 112], eval_fixed_crop=True,
            use_group_norm=True, share_rgb_model=False,
        ),
        state_encoder=dict(
            _target_="oat.perception.state_encoder.ProjectionStateEncoder", out_dim=None,
        ),
    )


@pytest.fixture
def adapter():
    model = OriginalFusedObservationAdapter(shape_meta(), embed_dim=24)
    model.set_normalizer(normalizer(shape_meta()))
    return model


def test_same_seed_preserves_original_initialization_and_exact_eval_output():
    meta = shape_meta()
    torch.manual_seed(13)
    reference = original(meta)
    torch.manual_seed(13)
    model = OriginalFusedObservationAdapter(meta, embed_dim=24)
    for key, expected in reference.state_dict().items():
        torch.testing.assert_close(model.fused_encoder.state_dict()[key], expected, rtol=0, atol=0)
    stats = normalizer(meta)
    reference.set_normalizer(stats)
    model.set_normalizer(stats)
    reference.eval()
    model.eval()
    obs = observations(meta)
    with torch.no_grad():
        torch.testing.assert_close(model.encode_fused(obs), reference(obs), rtol=0, atol=0)
    assert model.fused_feature_dim == 139
    assert model.output_feature_dim() == 24
    assert isinstance(model.fused_encoder.state_encoder.state_proj, nn.Identity)


def test_feature_content_keeps_camera_and_state_order_and_normalizes_once(adapter):
    adapter.eval()
    obs = observations(shape_meta())
    encoder = adapter.fused_encoder
    with torch.no_grad():
        vision = encoder.vision_encoder(obs)
        states = encoder.state_encoder(obs)
        fused = adapter.encode_fused(dict(reversed(list(obs.items()))))
    expected_states = torch.cat([normalizer(shape_meta())[key].normalize(obs[key])
                                 for key in adapter.state_ports], dim=-1)
    torch.testing.assert_close(states, expected_states, rtol=0, atol=0)
    torch.testing.assert_close(fused, torch.cat([vision, expected_states], dim=-1), rtol=0, atol=0)
    assert adapter.rgb_ports == ["z_camera", "a_camera"]
    assert adapter.state_ports == ["position", "rotation", "gripper", "task_uid"]
    torch.testing.assert_close(fused[..., -1:], expected_states[..., -1:])
    # Match each 64-D slice against the actual original camera network, so
    # wrong camera ordering cannot pass solely via agreement with fusion code.
    visual = encoder.vision_encoder
    with torch.no_grad():
        for index, key in enumerate(adapter.rgb_ports):
            images = visual.normalizer[key].normalize(obs[key]).reshape(2, 128, 128, 3).permute(0, 3, 1, 2)
            crop = visual.encoder.obs_randomizers[key]
            feature = visual.encoder.activation(visual.encoder.obs_nets[key](crop.forward_in(images)))
            expected = crop.forward_out(feature).reshape(1, 2, 64)
            torch.testing.assert_close(fused[..., index * 64:(index + 1) * 64], expected, rtol=0, atol=0)


def test_training_crop_rng_matches_original_and_eval_is_deterministic(adapter):
    reference = copy.deepcopy(adapter.fused_encoder)
    obs = observations(shape_meta())
    adapter.train()
    reference.train()
    with torch.no_grad():
        torch.manual_seed(100)
        expected = reference(obs)
        torch.manual_seed(100)
        actual = adapter.encode_fused(obs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.manual_seed(101)
        assert not torch.equal(actual, adapter.encode_fused(obs))
        adapter.eval()
        torch.manual_seed(102)
        first = adapter.encode_fused(obs)
        torch.manual_seed(103)
        torch.testing.assert_close(first, adapter.encode_fused(obs), rtol=0, atol=0)
    assert all(not module.training for module in adapter.modules() if isinstance(module, CropRandomizer))


def test_both_camera_cnns_bridge_and_frame_embeddings_receive_gradient(adapter):
    adapter.train()
    output = adapter(observations(shape_meta()))
    assert output.shape == (1, 2, 24)
    output.square().mean().backward()
    visual = adapter.fused_encoder.vision_encoder.encoder
    for key in adapter.rgb_ports:
        weights = next(module.weight for module in visual.obs_nets[key].backbone.modules()
                       if isinstance(module, nn.Conv2d))
        assert weights.grad is not None and torch.isfinite(weights.grad).all()
        assert weights.grad.abs().sum() > 0
    assert visual.obs_nets["z_camera"] is not visual.obs_nets["a_camera"]
    for parameter in (adapter.obs_projection.weight, adapter.obs_projection.bias, adapter.frame_embedding):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    for module in adapter.modules():
        if isinstance(module, LinearNormalizer):
            assert all(not parameter.requires_grad and parameter.grad is None for parameter in module.parameters())


def test_missing_rgb_and_state_statistics_are_rejected(adapter):
    stats = normalizer(shape_meta())
    del stats.params_dict["task_uid"]
    with pytest.raises(ValueError, match="task_uid"):
        adapter.set_normalizer(stats)
    stats = normalizer(shape_meta())
    del stats.params_dict["a_camera"]
    with pytest.raises(ValueError, match="a_camera"):
        adapter.set_normalizer(stats)
    del adapter.fused_encoder.vision_encoder.normalizer.params_dict["z_camera"]
    with pytest.raises(ValueError, match="z_camera"):
        adapter.encode_fused(observations(shape_meta()))


def test_export_rebuild_strict_restore_retains_stats_and_metadata(adapter):
    restored = hydra.utils.instantiate(adapter.export_config())
    restored.load_state_dict(adapter.state_dict(), strict=True)
    adapter.eval()
    restored.eval()
    obs = observations(shape_meta())
    with torch.no_grad():
        torch.testing.assert_close(restored(obs), adapter(obs), rtol=0, atol=0)
    metadata = restored.export_metadata()
    assert metadata["fused_feature_dim"] == 139
    assert metadata["observation_tokens"] == 2
    assert metadata["fusion_order"] == adapter.rgb_ports + adapter.state_ports
    assert metadata["dependency_versions"]["robomimic"]
    assert all(len(digest) == 64 for digest in metadata["source_hashes"].values())
    assert not any("resampler" in name for name, _ in restored.named_parameters())


def test_unsupported_recipe_and_observation_extensions_fail_explicitly():
    for config in ({"pretrained": True}, {"state_out_dim": 768}, {"num_queries": 64}):
        with pytest.raises(ValueError):
            normalize_original_obs_config(config)
    assert normalize_original_obs_config({"crop_shape": [76, 76]})["crop_shape"] == [76, 76]
    meta = shape_meta()
    meta["obs"]["language"] = dict(type="text", shape=[1])
    with pytest.raises(ValueError, match="RGB\\+state"):
        OriginalFusedObservationAdapter(meta)


def test_new_context_accepts_fused_and_leaves_old_schema_unchanged():
    assert [int(value) for value in LegacySegment] == [0, 1, 2, 3, 4]
    assert [int(value) for value in Segment] == [0, 1, 2, 3, 4, 5]
    memory = torch.randn(2, 11, 24)
    valid = torch.ones(2, 11, dtype=torch.bool)
    segments = torch.tensor([5, 5] + [2] * 7 + [3] * 2)
    context = ContextBatch(memory, valid, segments)
    assert context.validate_variant("p2n_new") is context
    with pytest.raises(ValueError, match="unknown segment"):
        LegacyContextBatch(memory, valid, segments).validate()
    old_segments = segments.clone()
    old_segments[:2] = int(LegacySegment.VISUAL)
    LegacyContextBatch(memory, valid, old_segments).validate_variant("p2n_new")
    valid[:, :2] = False
    with pytest.raises(ValueError, match="visible observation"):
        context.validate()


def test_closed_gate_masks_only_summaries_and_preserves_padding_rules():
    memory = torch.randn(2, 15, 24)
    valid = torch.ones(2, 15, dtype=torch.bool)
    valid[0, 2] = False
    memory[0, 2] = float("nan")
    memory[0, 11:] = float("nan")
    segments = torch.tensor([5, 5] + [2] * 7 + [3] * 2 + [4] * 4)
    gate = torch.tensor([[float("-inf")], [-0.1]])
    context = ContextBatch(memory, valid, segments, torch.zeros(2, 24),
                           torch.zeros(2, 24), torch.ones(2, 1), gate)
    context.validate_variant("p2n_state_gate_new", num_summary_tokens=4)
    bias = context.attention_bias()[:, 0, 0]
    assert torch.isneginf(bias[0, 2]) and torch.isneginf(bias[0, 11:]).all()
    torch.testing.assert_close(bias[:, :2], torch.zeros(2, 2))
    torch.testing.assert_close(bias[1, 11:], gate[1].expand(4))
    clean = context.sanitized_memory()
    assert torch.isfinite(clean).all()
    assert not clean[0, 11:].count_nonzero()


@pytest.mark.parametrize("mutation", ["missing_offset", "missing_std", "bad_width", "nonfinite", "zero_scale"])
def test_strict_restore_rejects_corrupt_dynamic_normalizer_fields(adapter, mutation):
    state = adapter.state_dict()
    prefix = "fused_encoder.state_encoder.normalizer.params_dict.position."
    if mutation == "missing_offset":
        del state[prefix + "offset"]
    elif mutation == "missing_std":
        del state[prefix + "input_stats.std"]
    elif mutation == "bad_width":
        for key in list(state):
            if key.startswith(prefix):
                state[key] = torch.ones(2)
    elif mutation == "nonfinite":
        state[prefix + "input_stats.mean"] = torch.full((3,), float("nan"))
    else:
        state[prefix + "scale"] = torch.zeros(3)
    with pytest.raises(ValueError, match="position"):
        adapter.load_state_dict(state, strict=True)
