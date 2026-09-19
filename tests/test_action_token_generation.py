"""BOS is a conditioning token, never a generated action or cached action input."""

import pytest
import torch

from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from test_history_training import make_batch, make_policy


def prefer_bos(model, bos_id, action_id=1):
    inputs = []

    def logits(module, args, output):
        scores = torch.full_like(output, -torch.inf)
        scores[..., action_id] = 0
        scores[..., bos_id] = 100
        return scores

    def record_inputs(module, args):
        inputs.append(args[0].detach().clone())

    model.head.register_forward_hook(logits)
    model.tok_emb.register_forward_pre_hook(record_inputs)
    return inputs


def make_generator():
    return AutoregressiveModel(
        vocab_size=5, max_seq_len=4, max_cond_len=2, cond_dim=8,
        n_layer=1, n_head=2, n_emb=8, p_drop_emb=0, p_drop_attn=0,
    ).eval()


@pytest.mark.parametrize("temperature,top_k", [(0, None), (1, None), (1, 1), (1, 99)])
def test_bos_is_masked_before_selection_and_never_enters_action_cache(temperature, top_k):
    model = make_generator()
    inputs = prefer_bos(model, bos_id=4)
    prefix = torch.full((2, 1), 4, dtype=torch.long)

    tokens = model.generate(
        prefix, cond=torch.zeros(2, 2, 8), max_new_tokens=3,
        temperature=temperature, top_k=top_k, bos_id=4,
    )

    torch.testing.assert_close(tokens[:, :1], prefix)
    torch.testing.assert_close(tokens[:, 1:], torch.ones(2, 3, dtype=torch.long))
    assert len(inputs) == 3
    torch.testing.assert_close(inputs[0], prefix)
    for cached_input in inputs[1:]:
        torch.testing.assert_close(cached_input, torch.ones(2, 1, dtype=torch.long))


def test_generic_generation_without_bos_constraint_is_unchanged():
    model = make_generator()
    prefer_bos(model, bos_id=4)

    tokens = model.generate(
        torch.tensor([[4]]), cond=torch.zeros(1, 2, 8),
        max_new_tokens=3, temperature=0,
    )

    torch.testing.assert_close(tokens, torch.full((1, 4), 4, dtype=torch.long))


def test_distinct_eos_still_terminates_generation_with_bos_masked():
    model = make_generator()
    prefer_bos(model, bos_id=4, action_id=1)

    tokens = model.generate(
        torch.tensor([[4]]), cond=torch.zeros(1, 2, 8), max_new_tokens=3,
        temperature=0, eos_id=1, bos_id=4,
    )

    torch.testing.assert_close(tokens, torch.tensor([[4, 1]]))


@pytest.mark.parametrize("path", ["inference", "self_past"])
@pytest.mark.parametrize("temperature", [0, 1])
def test_policy_paths_decode_valid_tokens_without_posthoc_remapping(monkeypatch, path, temperature):
    policy = make_policy()
    policy.temperature = policy.self_past_temperature = temperature
    policy.topk = policy.self_past_topk = 1
    policy.train(path == "self_past")
    inputs = prefer_bos(policy.model, policy.bos_id)
    decoded = []

    def detokenize(tokens):
        decoded.append(tokens.detach().clone())
        return torch.zeros(tokens.shape[0], 16, 7, device=tokens.device)

    monkeypatch.setattr(policy.action_tokenizer, "detokenize", detokenize)
    batch = make_batch()
    if path == "inference":
        result = policy.predict_action(batch["obs"], past_actions=batch["past_action"])
        assert result["action"].shape == (2, 8, 7)
    else:
        result = policy._generate_prev_past({
            "prev_obs": batch["obs"], "prev_past_action": batch["past_action"],
        })
        assert result.shape == (2, 7, 7)
        assert policy.training and policy.model.training
        assert not policy.action_tokenizer.training

    assert len(decoded) == 1
    torch.testing.assert_close(decoded[0], torch.ones(2, 2, dtype=torch.long))
    torch.testing.assert_close(inputs[0], torch.full((2, 1), policy.bos_id, dtype=torch.long))
    for cached_input in inputs[1:]:
        torch.testing.assert_close(cached_input, torch.ones(2, 1, dtype=torch.long))
