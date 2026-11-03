"""CPU regression tests for native VLM history attention, caching, and loading."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from oat.model.vla.gemma_joint import GemmaJoint
from oat.model.vla.gemma_vlm import GemmaVLM
from oat.model.vla.layout import build_prefix_layout
from oat.model.vla.lora import inject_lora
from oat.model.vla.native_ar_layout import (build_native_ar_layout, native_decode_view,
                                           native_prefill_view)
from oat.model.vla.pi05_checkpoint import (EXPERT_PREFIX, LM_HEAD, _vlm_targets,
                                          load_pi05_vlm)
from oat.model.vla.specs import GemmaSpec


def _model(kv_heads=1, checkpointing=False):
    torch.manual_seed(45)
    spec = GemmaSpec(32, 3, 64, 4, kv_heads, 8, vocab_size=101)
    return GemmaVLM(spec, frozen_dtype=torch.float32, activation_checkpointing=checkpointing)


def _inputs():
    observation = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=torch.bool)
    history = torch.tensor([[1, 0, 0, 1, 0, 1, 1, 1, 1], [0] * 9], dtype=torch.bool)
    layout = build_native_ar_layout(observation, history, 8)
    generator = torch.Generator().manual_seed(21)
    embeds = torch.randn(2, 21, 32, generator=generator)
    return embeds, layout


def test_native_mask_boundaries_and_fixed_history_positions():
    _, layout = _inputs()
    visible = layout.bias[:, 0] == 0
    assert not visible[:, :4, 4:].any()  # observation cannot read history or actions
    assert not visible[:, 4:13, 13:].any()  # history cannot read action labels
    assert visible[0, 4, 7] and visible[0, 7, 4]  # history is bidirectional
    assert not visible[:, 13:, 5].any()  # invalid history is never an action key
    assert visible[:, 13, 13].all() and not visible[:, 13, 14:].any()
    assert visible[:, 20, 13:21].all()
    assert visible.diagonal(dim1=-2, dim2=-1).all()  # padding queries remain safe
    assert torch.equal(layout.positions[:, 13], torch.tensor([11, 12]))
    no_history = build_native_ar_layout(layout.valid[:, :4], torch.zeros(2, 9, dtype=torch.bool), 8)
    assert torch.equal(layout.positions, no_history.positions)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_teacher_forcing_equals_full_context_cache(kv_heads):
    model = _model(kv_heads).eval()
    embeds, layout = _inputs()
    with torch.no_grad():
        expected = model.native_forward(embeds, layout).hidden[:, 13:]
        prefill = native_prefill_view(layout, 1)
        out = model.native_forward(embeds[:, :14], prefill, use_cache=True)
        seen = [out.hidden[:, -1:]]
        assert out.cache.length == 14
        for index in range(1, 8):
            out = model.native_forward(embeds[:, 13 + index:14 + index], native_decode_view(layout, index),
                                       cache=out.cache, use_cache=True)
            seen.append(out.hidden)
            assert out.cache.length == 14 + index
            assert out.cache.kv[0][0].shape[2] == out.cache.length
        torch.testing.assert_close(torch.cat(seen, dim=1), expected, rtol=2e-5, atol=2e-6)


def test_context_only_prefill_then_bos_has_same_result():
    model = _model().eval()
    embeds, layout = _inputs()
    with torch.no_grad():
        context = model.native_forward(embeds[:, :13], native_prefill_view(layout, 0), use_cache=True)
        bos = model.native_forward(embeds[:, 13:14], native_decode_view(layout, 0),
                                   cache=context.cache, use_cache=True)
        joint = model.native_forward(embeds[:, :14], native_prefill_view(layout, 1))
    torch.testing.assert_close(bos.hidden, joint.hidden[:, -1:], rtol=2e-5, atol=2e-6)


def test_no_future_action_or_invalid_history_leakage():
    model = _model().eval()
    embeds, layout = _inputs()
    with torch.no_grad():
        expected = model(embeds, layout).hidden
        changed = embeds.clone()
        changed[:, 17:] += torch.randn_like(changed[:, 17:]) * 7
        future = model(changed, layout).hidden
        torch.testing.assert_close(future[:, :17], expected[:, :17], rtol=0, atol=0)
        assert not torch.allclose(future[:, 17:], expected[:, 17:])
        changed = embeds.clone()
        changed[~layout.valid] += torch.randn_like(changed[~layout.valid]) * 9
        masked = model(changed, layout).hidden
        torch.testing.assert_close(masked[layout.valid], expected[layout.valid], rtol=0, atol=0)
        assert torch.isfinite(masked).all()
        changed = embeds.clone()
        changed[:, :4] += torch.randn_like(changed[:, :4])
        observation_change = model(changed, layout).hidden
        assert not torch.allclose(observation_change[0, 4], expected[0, 4])


def test_observation_hidden_matches_legacy_prefix():
    model = _model().eval()
    embeds, layout = _inputs()
    spec = model.vlm_spec
    legacy = GemmaJoint(spec, replace(spec, vocab_size=None), "segment", frozen_dtype=torch.float32,
                        activation_checkpointing=False).eval()
    legacy.vlm.load_state_dict(model.vlm.state_dict())
    old_layout = build_prefix_layout(layout.valid[:, :4], torch.empty(2, 0, dtype=torch.bool))
    with torch.no_grad():
        native = model(embeds, layout).hidden[:, :4]
        old = legacy.prefix_forward(embeds[:, :4], old_layout).hidden
    torch.testing.assert_close(native, old, rtol=2e-5, atol=2e-6)
    assert not hasattr(model, "expert")


def test_final_layer_lora_receives_action_loss_gradients():
    model = _model()
    inject_lora(model.vlm, rank=4, alpha=4)
    embeds, layout = _inputs()
    hidden = model(embeds, layout).hidden[:, 13:]
    (hidden * torch.randn_like(hidden)).sum().backward()
    last = model.vlm.layers[-1]
    for path in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        module = last.get_submodule(path)
        assert module.lora_B.grad is not None and module.lora_B.grad.abs().sum() > 0, path


def test_activation_checkpointing_preserves_backward():
    eager, checkpointed = _model(), _model(checkpointing=True)
    embeds, layout = _inputs()
    inputs = [embeds.clone().requires_grad_(), embeds.clone().requires_grad_()]
    outputs = [model(value, layout).hidden[:, 13:] for model, value in zip((eager, checkpointed), inputs)]
    weights = torch.randn_like(outputs[0])
    for output in outputs:
        (output * weights).sum().backward()
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(inputs[0].grad, inputs[1].grad)


def test_bf16_compute_keeps_lora_masters_fp32_and_cache_consistent():
    spec = GemmaSpec(32, 3, 64, 4, 1, 8, vocab_size=101)
    model = GemmaVLM(spec, frozen_dtype=torch.bfloat16, activation_checkpointing=True)
    inject_lora(model.vlm, rank=4, alpha=4)
    embeds, layout = _inputs()
    embeds.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        full = model(embeds.to(torch.bfloat16), layout).hidden
    (full[:, 13:].float() * torch.randn_like(full[:, 13:].float())).sum().backward()
    assert full.dtype == torch.bfloat16
    assert torch.isfinite(embeds.grad).all()
    for name, parameter in model.named_parameters():
        if ".lora_" in name:
            assert parameter.dtype == torch.float32
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        out = model(embeds[:, :14].to(torch.bfloat16), native_prefill_view(layout, 1), use_cache=True)
        seen = [out.hidden[:, -1:]]
        for index in range(1, 8):
            out = model(embeds[:, 13 + index:14 + index].to(torch.bfloat16), native_decode_view(layout, index),
                        cache=out.cache, use_cache=True)
            seen.append(out.hidden)
    # Different GEMM shapes can round differently in bf16; compare at its precision.
    torch.testing.assert_close(torch.cat(seen, dim=1), full[:, 13:], rtol=2e-2, atol=2e-2)


def test_cache_rejects_changed_validity_or_wrong_offset():
    model = _model().eval()
    embeds, layout = _inputs()
    with torch.no_grad():
        prefill = model(embeds[:, :14], native_prefill_view(layout, 1), use_cache=True)
        bad = native_decode_view(layout, 1)
        validity = bad.valid.clone()
        validity[0, 4] = False
        with pytest.raises(ValueError, match="validity"):
            model(embeds[:, 14:15], replace(bad, valid=validity), cache=prefill.cache)
        with pytest.raises(ValueError, match="lengths"):
            model(embeds[:, 14:15], replace(bad, query_start=0), cache=prefill.cache)


def test_vlm_only_checkpoint_loading_is_strict(tmp_path):
    from safetensors.torch import save_file

    model = _model()
    vision = nn.Module()
    vision.vision = nn.Linear(3, 4)
    vision.projector = nn.Linear(4, 32)
    targets = _vlm_targets(vision, model)
    tensors = {key: torch.randn_like(value) for key, value in targets.items()}
    tensors[EXPERT_PREFIX + "layers.0.self_attn.q_proj.weight"] = torch.zeros(4, 4)
    tensors[EXPERT_PREFIX + "layers.0.input_layernorm.dense.weight"] = torch.zeros(12, 4)
    tensors["action_in_proj.weight"] = torch.zeros(4, 7)
    path = tmp_path / "native_loader.safetensors"
    save_file(tensors, str(path))
    report = load_pi05_vlm(path, siglip=vision, joint=model)
    assert report.mode == "vlm" and report.c0 is None and report.t0 is None
    assert set(report.consumed) == set(targets)
    assert set(report.dropped) == set(tensors) - set(targets)
    for key, destination in targets.items():
        torch.testing.assert_close(destination, tensors[key], rtol=0, atol=0)
    tensors[EXPERT_PREFIX + "unexpected.weight"] = torch.zeros(1)
    save_file(tensors, str(path))
    with pytest.raises(KeyError, match="Unexpected"):
        load_pi05_vlm(path, siglip=vision, joint=model)
    del tensors[EXPERT_PREFIX + "unexpected.weight"]
    tensors[LM_HEAD] = torch.zeros(1, 1)
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="shape"):
        load_pi05_vlm(path, siglip=vision, joint=model)
    del tensors[LM_HEAD]
    save_file(tensors, str(path))
    with pytest.raises(KeyError, match="missing"):
        load_pi05_vlm(path, siglip=vision, joint=model)
