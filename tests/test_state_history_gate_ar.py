"""History gates must affect attention mass consistently in training and KV decoding."""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from oat.model.autoregressive.transformer_cache_history_gate import (
    HistoryGatedAutoregressiveModel,
)


HISTORY_TOKENS = 2
BOS_ID = 10


@pytest.fixture
def models_and_inputs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        base = AutoregressiveModel(
            vocab_size=11, max_seq_len=12, max_cond_len=7, cond_dim=7,
            n_layer=2, n_head=2, n_emb=16, p_drop_emb=0, p_drop_attn=0,
        ).eval()
        # Nonzero positions expose off-by-one errors hidden by zero initialization.
        with torch.no_grad():
            base.tok_pos_emb.normal_(std=0.2)
            base.cond_pos_emb.normal_(std=0.2)
        gated = HistoryGatedAutoregressiveModel.from_base(
            base, history_token_count=HISTORY_TOKENS,
        ).eval()
        cond = torch.randn(3, 7, 7)
    prefix = torch.tensor([[BOS_ID, 2, 4], [BOS_ID, 6, 1], [BOS_ID, 3, 7]])
    return base, gated, prefix, cond


def precision_context(mixed_precision):
    return (torch.autocast("cpu", dtype=torch.bfloat16)
            if mixed_precision else contextlib.nullcontext())


def assert_logits_close(actual, expected, mixed_precision=False):
    torch.testing.assert_close(
        actual, expected,
        rtol=0.02 if mixed_precision else 1e-5,
        atol=0.02 if mixed_precision else 2e-6,
    )


@pytest.mark.parametrize("use_explicit_zero", [False, True])
def test_open_gate_preserves_base_logits_and_greedy_generation(
    models_and_inputs, use_explicit_zero,
):
    base, gated, prefix, cond = models_and_inputs
    log_gate = torch.zeros(3, 1) if use_explicit_zero else None
    with torch.no_grad():
        assert_logits_close(
            gated(prefix, cond, history_log_gate=log_gate), base(prefix, cond),
        )
    expected = base.generate(prefix, cond, 4, temperature=0, bos_id=BOS_ID)
    actual = gated.generate(
        prefix, cond, 4, temperature=0, bos_id=BOS_ID,
        history_log_gate=log_gate,
    )
    torch.testing.assert_close(actual, expected)


def test_closed_gate_equals_removing_history_and_ignores_changed_summaries(
    models_and_inputs,
):
    base, gated, prefix, cond = models_and_inputs
    closed = torch.full((3, 1), -torch.inf)
    altered = cond.clone()
    altered[:, -HISTORY_TOKENS:] = 30 * altered[:, -HISTORY_TOKENS:] + 17
    with torch.no_grad():
        expected = base(prefix, cond[:, :-HISTORY_TOKENS])
        actual = gated(prefix, cond, history_log_gate=closed)
        assert_logits_close(actual, expected)
        assert_logits_close(
            gated(prefix, altered, history_log_gate=closed), expected,
        )
        # Closing the new summaries must leave the preceding condition usable.
        changed_existing = cond.clone()
        changed_existing[:, -HISTORY_TOKENS - 1] += 9
        assert not torch.allclose(
            actual, gated(prefix, changed_existing, history_log_gate=closed),
        )
    expected_tokens = base.generate(
        prefix, cond[:, :-HISTORY_TOKENS], 4, temperature=0, bos_id=BOS_ID,
    )
    for conditioning in (cond, altered):
        actual_tokens = gated.generate(
            prefix, conditioning, 4, temperature=0, bos_id=BOS_ID,
            history_log_gate=closed,
        )
        torch.testing.assert_close(actual_tokens, expected_tokens)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_gate_receives_finite_nonzero_training_gradient(
    models_and_inputs, mixed_precision,
):
    _, gated, prefix, cond = models_and_inputs
    gate_logits = torch.nn.Parameter(torch.tensor([[-0.7], [0.4], [1.2]]))
    conditioning = cond.clone().requires_grad_()
    with precision_context(mixed_precision):
        log_gate = F.logsigmoid(gate_logits)
        if mixed_precision:
            log_gate = log_gate.to(torch.bfloat16)
        logits = gated(prefix, conditioning, history_log_gate=log_gate)
        targets = torch.tensor([[1, 3, 5], [2, 4, 6], [3, 5, 7]])
        loss = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert gate_logits.grad is not None
    assert torch.isfinite(gate_logits.grad).all()
    assert (gate_logits.grad.abs() > 1e-8).all()
    assert conditioning.grad is not None
    history_gradient = conditioning.grad[:, -HISTORY_TOKENS:]
    assert torch.isfinite(history_gradient).all()
    assert (history_gradient.abs().sum(dim=(1, 2)) > 1e-8).all()


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_cached_logits_and_tokens_match_full_prefix_decoding_with_mixed_gates(
    models_and_inputs, mixed_precision,
):
    _, gated, prefix, cond = models_and_inputs
    log_gate = torch.tensor([[0.0], [-0.8], [-torch.inf]])
    if mixed_precision:
        log_gate = log_gate.to(torch.bfloat16)
    cached_logits = []
    handle = gated.head.register_forward_hook(
        lambda module, args, output: cached_logits.append(output.detach().clone()),
    )
    try:
        with precision_context(mixed_precision):
            cached = gated.generate(
                prefix, cond, 4, temperature=0, bos_id=BOS_ID,
                history_log_gate=log_gate,
            )
    finally:
        handle.remove()
    assert len(cached_logits) == 4
    full = prefix.clone()
    with torch.no_grad(), precision_context(mixed_precision):
        for cached_step in cached_logits:
            logits = gated(full, cond, history_log_gate=log_gate)[:, -1:, :]
            assert_logits_close(cached_step, logits, mixed_precision)
            logits = logits.clone()
            logits[..., BOS_ID] = -torch.inf
            full = torch.cat((full, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
    torch.testing.assert_close(cached, full)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_mixed_batch_gates_are_independent_and_closed_row_matches_base(
    models_and_inputs, mixed_precision,
):
    base, gated, prefix, cond = models_and_inputs
    log_gate = torch.tensor([[0.0], [-0.8], [-torch.inf]])
    if mixed_precision:
        log_gate = log_gate.to(torch.bfloat16)
    with torch.no_grad(), precision_context(mixed_precision):
        together = gated(prefix, cond, history_log_gate=log_gate)
        assert torch.isfinite(together).all()
        for index in range(3):
            separate = gated(
                prefix[index:index + 1], cond[index:index + 1],
                history_log_gate=log_gate[index:index + 1],
            )
            assert_logits_close(together[index:index + 1], separate, mixed_precision)
        assert_logits_close(together[:1], base(prefix[:1], cond[:1]), mixed_precision)
        assert_logits_close(
            together[2:], base(prefix[2:], cond[2:, :-HISTORY_TOKENS]), mixed_precision,
        )


@pytest.mark.parametrize("temperature,top_k", [(0, None), (1, None), (1, 1), (1, 99)])
def test_gated_generation_suppresses_bos_before_sampling_and_cache_update(
    models_and_inputs, temperature, top_k,
):
    _, gated, prefix, cond = models_and_inputs
    cached_inputs = []

    def prefer_bos(module, args, output):
        scores = torch.full_like(output, -torch.inf)
        scores[..., BOS_ID] = 100
        scores[..., 1] = 0
        return scores

    def record_input(module, args):
        cached_inputs.append(args[0].detach().clone())

    head_handle = gated.head.register_forward_hook(prefer_bos)
    input_handle = gated.tok_emb.register_forward_pre_hook(record_input)
    try:
        actual = gated.generate(
            prefix, cond, 4, temperature=temperature, top_k=top_k,
            bos_id=BOS_ID, history_log_gate=torch.tensor([[0.0], [-0.8], [-torch.inf]]),
        )
    finally:
        head_handle.remove()
        input_handle.remove()
    torch.testing.assert_close(actual[:, :prefix.shape[1]], prefix)
    torch.testing.assert_close(actual[:, prefix.shape[1]:], torch.ones(3, 4, dtype=torch.long))
    assert len(cached_inputs) == 4
    torch.testing.assert_close(cached_inputs[0], prefix)
    for cached_input in cached_inputs[1:]:
        torch.testing.assert_close(cached_input, torch.ones(3, 1, dtype=torch.long))
