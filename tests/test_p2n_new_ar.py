"""CPU contracts for the new decoder and typed conditioning memory."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from oat.model.common.context_batch import ContextBatch, Segment
from oat.model.autoregressive.modern_transformer_cache import ModernAutoregressiveModel, RMSNorm


def make_model(**kwargs):
    torch.manual_seed(12)
    return ModernAutoregressiveModel(vocab_size=17, max_seq_len=5, n_layer=2,
                                    n_emb=24, n_head=3, ffn_dim=40, **kwargs)


def make_context(gate=False):
    torch.manual_seed(34)
    segments = [Segment.VISUAL, Segment.PROPRIO, Segment.RAW_ACTION,
                Segment.RAW_ACTION, Segment.ACTION_DIFF]
    if gate:
        segments += [Segment.HISTORY_SUMMARY] * 2
    memory = torch.randn(2, len(segments), 24)
    valid = torch.ones(2, len(segments), dtype=torch.bool)
    valid[:, 2] = False
    context = ContextBatch(memory, valid, torch.tensor(segments))
    if gate:
        context.observation_summary = torch.randn(2, 24)
        context.history_summary_pool = torch.randn(2, 24)
        context.history_valid_fraction = torch.ones(2, 1)
        context.history_log_gate = F.logsigmoid(torch.ones(2, 1))
    return context


@pytest.mark.parametrize("gate", [False, True])
def test_cached_logits_match_full_forward_and_chunked_prefill(gate):
    model = make_model().eval()
    context = make_context(gate)
    tokens = torch.tensor([[16, 1, 2, 3, 4], [16, 4, 3, 2, 1]])
    expected = model(tokens, context)
    outputs = []
    logits, cache = model.prefill(tokens[:, :1], context)
    outputs.append(logits)
    for index in range(1, tokens.shape[1]):
        logits, cache = model.decode(tokens[:, index:index + 1], cache)
        outputs.append(logits)
    torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-6, rtol=1e-5)
    prefill, cache = model.prefill(tokens[:, :2], context)
    tail, cache = model.decode(tokens[:, 2:], cache)
    torch.testing.assert_close(torch.cat((prefill, tail), 1), expected, atol=1e-6, rtol=1e-5)
    assert cache.position == 5
    assert all(not value.requires_grad for pair in cache.self_key_values + cache.cross_key_values
               for value in pair)
    assert not any("cache" in key for key in model.state_dict())


def test_future_tokens_cannot_change_earlier_logits():
    model, context = make_model().eval(), make_context()
    tokens = torch.tensor([[16, 1, 2, 3, 4], [16, 4, 3, 2, 1]])
    altered = tokens.clone()
    altered[:, 3:] = 8
    torch.testing.assert_close(model(tokens, context)[:, :3], model(altered, context)[:, :3])


def test_padding_and_closed_summary_do_not_leak_but_raw_actions_remain_visible():
    model = make_model().eval()
    context = make_context(True)
    context.history_log_gate = torch.full((2, 1), float("-inf"))
    tokens = torch.tensor([[16, 1], [16, 2]])
    baseline = model(tokens, context)
    changed = context.memory.clone()
    changed[:, 2] = float("nan")
    changed[:, 5:] = float("nan")
    isolated = replace(context, memory=changed)
    torch.testing.assert_close(model(tokens, isolated), baseline)
    assert torch.isfinite(baseline).all()
    bias = context.attention_bias()
    assert not torch.isnan(bias).any()
    assert torch.isneginf(bias[..., 5:]).all()
    assert (bias[..., 3:5] == 0).all()
    torch.testing.assert_close(model.generate(isolated), model.generate(context))
    changed = changed.clone()
    changed[:, 3] = torch.randn(2, 24) * 20
    assert not torch.allclose(model(tokens, replace(context, memory=changed)), baseline)


def test_gate_logsigmoid_is_stable_and_receives_gradients():
    model, context = make_model(), make_context(True)
    gate_logits = torch.tensor([[-1000.0], [0.2]], requires_grad=True)
    context.history_log_gate = F.logsigmoid(gate_logits)
    context.memory.requires_grad_()
    loss = model(torch.tensor([[16, 1], [16, 2]]), context).square().mean()
    loss.backward()
    assert torch.isfinite(gate_logits.grad).all()
    assert gate_logits.grad[1].abs().sum() > 0
    assert context.memory.grad[:, 2].count_nonzero() == 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.head.weight is model.tok_emb.weight
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_generation_always_excludes_bos_and_builds_cross_cache_once(temperature):
    model, context = make_model().eval(), make_context()
    calls = [0 for _ in model.blocks]
    hooks = []
    for index, block in enumerate(model.blocks):
        def record(module, args, output, index=index):
            calls[index] += 1
        hooks.append(block.cross_attn.kv_proj.register_forward_hook(record))
    generated = model.generate(context, temperature=temperature, top_k=4)
    assert generated.shape == (2, 5)
    assert (generated < model.bos_id).all()
    assert calls == [1, 1]
    for hook in hooks:
        hook.remove()
    # A second session must reflect its own context; no model cache survives.
    other = replace(context, memory=torch.randn_like(context.memory))
    reference_logits, _ = model.prefill(torch.full((2, 1), model.bos_id), other)
    torch.testing.assert_close(reference_logits, model(torch.full((2, 1), model.bos_id), other))


def test_context_validates_observation_and_variant_contracts():
    base = make_context()
    base.validate_variant("p2n_new", num_summary_tokens=0)
    gate = make_context(True)
    gate.validate_variant("p2n_state_gate_new", num_summary_tokens=2)
    with pytest.raises(ValueError, match="must not contain"):
        gate.validate_variant("p2n_new")
    with pytest.raises(ValueError, match="requires history"):
        base.validate_variant("p2n_state_gate_new")
    with pytest.raises(ValueError, match="visible observation"):
        replace(base, valid_mask=torch.zeros_like(base.valid_mask)).validate()
    with pytest.raises(ValueError, match="bool"):
        replace(base, valid_mask=base.valid_mask.float()).validate()
    selected, valid = base.segment_memory(Segment.RAW_ACTION)
    assert selected.shape == (2, 2, 24)
    assert valid.tolist() == [[False, True], [False, True]]


def test_activation_checkpointing_preserves_outputs_gradients_and_rng():
    model = make_model(dropout=0.1).train()
    checkpointed = make_model(dropout=0.1, activation_checkpointing=True).train()
    checkpointed.load_state_dict(model.state_dict())
    tokens, context = torch.tensor([[16, 1], [16, 2]]), make_context(True)
    torch.manual_seed(67)
    expected = model(tokens, context)
    expected.square().mean().backward()
    torch.manual_seed(67)
    actual = checkpointed(tokens, context)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    for parameter, other in zip(model.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(parameter.grad, other.grad)


def test_bfloat16_cache_and_self_past_then_training_backward():
    model, context = make_model(), make_context(True)
    tokens = torch.tensor([[16, 1, 2], [16, 2, 1]])
    model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(tokens, context)
        start, cache = model.prefill(tokens[:, :1], context)
        rest, _ = model.decode(tokens[:, 1:], cache)
        torch.testing.assert_close(torch.cat((start, rest), 1), expected, atol=3e-3, rtol=3e-2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model.generate(context)
        model.train()
        model(tokens, context).float().square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert RMSNorm(8)(torch.randn(2, 8, dtype=torch.bfloat16)).dtype == torch.bfloat16
