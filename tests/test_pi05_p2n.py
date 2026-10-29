"""pi05_p2n (Pi05_past with the Past2Next self-past AR head over 8 OAT tokens) on tiny CPU models.

Uses the repo-local recovered OAT tokenizer and FAILS (never skips) when an asset is missing, so a green run
always exercised the policy. Shares the fixtures of tests/test_pi05_past.py.
Run: /venv/oat/bin/python -m pytest tests/test_pi05_p2n.py -q   (-m "not slow" for the fast subset)
"""
import json
import os
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import torch

from oat.model.vla.flow_head import flow_condition
from oat.model.vla.gemma_joint import AdaRMSNorm, SegmentRMSNorm
from oat.model.vla.layout import NEG, SEG_AR, SEG_RAWDIFF, decode_view, prefill_view
from oat.policy.p2n_vla_common import P2NVLACommonPolicy
from oat.policy.pi05_p2n import Pi05P2NPolicy, build_ar_suffix_layout
from oat.policy.pi05_past import Pi05PastPolicy, UNREACHABLE_LAST_LAYER
from test_pi05_past import (REPO, SHAPE_META, SPM, TOKENIZER, _normalizer, _records, cpu_accelerate,  # noqa: F401
                            make_batch, make_libero_like_zarr)

STORED_PI05_PAST_CONFIG = REPO / "output/training/pi05_past_libero10_seed42/p2n_vla_resolved.yaml"
TINY = dict(shape_meta=SHAPE_META, model_size="tiny", tokenizer_checkpoint=str(TOKENIZER), spm_path=str(SPM),
            activation_checkpointing=False, self_past_warmup_steps=0, self_past_ramp_steps=0, lambda_ki=0.0,
            lora_rank=4, lora_alpha=4.0)
# The config keys pi05_p2n may change relative to Pi05_past: naming, the flow knobs and snapshot retention.
HEAD_KEYS = {"name", "variant", "policy._target_", "policy.variant", "policy.flow_num_steps",
             "policy.flow_time_beta", "training.keep_snapshots", "logging.tags", "logging.group"}


# --------------------------------------------------------------------------------------------- helpers
def build(seed=0, cls=Pi05P2NPolicy, **kwargs):
    torch.manual_seed(seed)
    policy = cls(**dict(TINY, **kwargs))
    policy.set_normalizer(_normalizer())
    return policy


def joined_kv(policy, batch):
    with torch.no_grad():
        prefix_out, layout, _ = policy.vlm_pass(policy.build_prefix(batch["obs"], train_aug=False), None)
        kv, valid = policy._expert_kv(prefix_out, layout, batch["past_action"], batch["past_action_valid"])
    return prefix_out, layout, kv, valid


def _break_symmetries(policy, seed=3):
    """The tiny init hides head bugs: every SegmentRMSNorm segment row is identical (the pi05 loader also writes one
    fold into all rows), cond_pos_emb is zero, and the tied tok_emb makes each token row copy its own input, so greedy
    decoding emits one repeated token. Perturb the modulations and the slot embedding so the rows differ."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in policy.joint.expert.modules():
            if isinstance(module, SegmentRMSNorm):
                module.modulation.add_(0.2 * torch.randn(module.modulation.shape, generator=generator).to(
                    module.modulation.device))
        policy.cond_pos_emb.copy_(0.5 * torch.randn(policy.cond_pos_emb.shape, generator=generator))
    return policy


def _record_decoding(policy, forced):
    """Replace ``_sample`` with a recorder of every decoding step's logits that feeds back ``forced`` instead."""
    seen = []

    def sample(logits, temperature, topk, bos_id):
        seen.append(logits.clone())
        return forced[:, len(seen) - 1]

    policy._sample = sample
    return seen


def _original_sample(logits, temperature, top_k, bos_id):
    """AutoregressiveModel.generate's next-token rule (oat/model/autoregressive/transformer_cache.py)."""
    logits = logits.clone()[:, None]
    logits[..., bos_id] = -float("Inf")
    if temperature > 0:
        logits = logits.squeeze(1) / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)[:, 0]
    return torch.argmax(logits.squeeze(1), dim=-1)


def _flat(config, prefix=""):
    out = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _resolved(cfg):
    flat = _flat(OmegaConf.to_container(cfg, resolve=True))
    flat.pop("logging.name")  # ${now:...}
    return flat


# -------------------------------------------------------------------------------------- construction
def test_construction_scope_groups_and_frozen_keys():
    from oat.workspace.train_p2n_vla import validate_parameter_partition
    policy = build()
    assert policy.VARIANT == "pi05_p2n" and policy.VARIANT_CODE == 14 and policy.joint.norm_mode == "segment"
    assert policy.flow_heads is None and policy.n_cond_tokens == 9 and policy.max_seq_len == 8
    assert policy.flow_num_steps is None and policy.flow_time_beta is None
    assert not hasattr(policy, "type_embedding") and not hasattr(policy, "action_time_embedding")
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    assert "tok_emb.weight" in trainable and any(n.startswith("siglip.") for n in trainable)
    assert not any(n.startswith("ki_table.") for n in trainable), "KI table must be frozen with lambda_ki=0"
    last = len(policy.joint.vlm.layers) - 1
    dead = {f"joint.vlm.layers.{last}.{path}.{ab}" for path in UNREACHABLE_LAST_LAYER for ab in ("lora_A", "lora_B")}
    assert set(policy.dead_lora) == dead and not (dead & trainable)
    optimizer = policy.get_optimizer()
    named, clip = validate_parameter_partition(policy, optimizer)
    by_id = {id(p): n for n, p in named}
    groups = {g["name"]: {by_id[id(p)] for p in g["params"]} for g in optimizer.param_groups}
    assert all(n.startswith(("joint.expert.", "siglip.")) for n in groups["pretrained"])
    assert "tok_emb.weight" in groups["new"]
    assert all(".lora_" in n or n.startswith(("acc_proj", "jerk_proj", "raw_proj", "cond_pos_emb", "tok_emb"))
               for n in groups["new"])
    clip_names = {k: {by_id[id(p)] for p in v} for k, v in clip.items()}
    assert "tok_emb.weight" in clip_names["ar"]
    assert all(n.startswith("siglip.") or ".lora_" in n for n in clip_names["ki"])
    assert not any(k.startswith("siglip.") for k in policy.frozen_base_keys())
    assert all(t.dtype == torch.float32 for _, t in named)
    # Apart from the head (flow heads and adaRMS Dense out; tok_emb and folded modulations in), the trainable
    # scope is exactly Pi05_past's.
    past_trainable = {n for n, p in build(cls=Pi05PastPolicy).named_parameters() if p.requires_grad}

    def head(name):
        return name.startswith(("flow_heads.", "tok_emb.")) or ".dense." in name or name.endswith(".modulation")

    assert {n for n in trainable if not head(n)} == {n for n in past_trainable if not head(n)}
    assert any(n.endswith(".modulation") for n in trainable) and not any(".dense." in n for n in trainable)
    config = policy.export_config()
    assert "flow_num_steps" not in config and "flow_time_beta" not in config
    assert config["adarms_t0"] == config["cond_t0"] == 0.6


def test_head_settings_are_validated():
    for kwargs, match in ((dict(flow_num_steps=10), "no flow head"), (dict(flow_time_beta=(1.5, 1.0)), "no flow head"),
                          (dict(adarms_t0=0.5), "adarms_t0"), (dict(temperature=-0.1), "temperature"),
                          (dict(self_past_temperature=float("nan")), "self_past_temperature"),
                          (dict(topk=0), "topk"), (dict(self_past_topk=True), "self_past_topk"),
                          (dict(kv_grad_scale=0.0), "without")):
        with pytest.raises(ValueError, match=match):
            build(**kwargs)
    build(kv_grad_scale=0.0, train_vision=False, train_lora=False)  # a frozen VLM is allowed
    assert build(cond_t0=0.5).cond_t0 == 0.5  # adarms_t0 follows cond_t0 when unset


def test_null_self_past_sampler_means_the_deployment_sampler():
    policy = build()  # the config's samplers: greedy deployment, T=1 / top-k 10 self-past
    assert (policy.temperature, policy.topk, policy.self_past_temperature, policy.self_past_topk) == (0.0, 10, 1.0, 10)
    policy = build(temperature=0.0, topk=7, self_past_temperature=None, self_past_topk=None)
    assert (policy.self_past_temperature, policy.self_past_topk) == (0.0, 7)  # never Pi05_past's 1.0/10 placeholders
    assert build(temperature=0.5, self_past_temperature=None).self_past_temperature == 0.5


# ---------------------------------------------------------------------------------- condition fidelity
def test_condition_path_equals_pi05_past_at_the_fold():
    """With Pi05_past's adaRMS Dense folded at cond_t0 and every shared tensor copied, the condition K/V that the
    head reads are Pi05_past's: same tokens, same self-only pass, same position, only the norm parametrization."""
    past = build(cls=Pi05PastPolicy).eval()
    p2n = build(seed=1).eval()
    with torch.no_grad():
        for module in past.joint.expert.modules():  # make the fold non-trivial
            if isinstance(module, AdaRMSNorm):
                module.dense.weight.normal_(std=0.02)
                module.dense.bias.normal_(std=0.2)
        for layer in (past.flow_heads.time_mlp_in, past.flow_heads.time_mlp_out):
            layer.weight.normal_(std=0.2)
        state = p2n.state_dict()
        shared = {k: v for k, v in past.state_dict().items()
                  if k in state and state[k].shape == v.shape and not k.startswith("_")}
        p2n.load_state_dict({**state, **shared}, strict=True)
        c0 = flow_condition(past.flow_heads, torch.tensor([past.cond_t0]))[0]
        dense = [m for m in past.joint.expert.modules() if isinstance(m, AdaRMSNorm)]
        folded = [m for m in p2n.joint.expert.modules() if isinstance(m, SegmentRMSNorm)]
        assert len(dense) == len(folded) == 2 * past.joint.expert_spec.depth + 1
        for a, b in zip(dense, folded):
            b.modulation[SEG_RAWDIFF] = a.dense.weight @ c0 + a.dense.bias
        batch = make_batch()
        assert any(k.startswith(("acc_proj", "raw_proj", "cond_pos_emb")) for k in shared)
        tokens = past.history_tokens(batch["past_action"], batch["past_action_valid"])
        torch.testing.assert_close(p2n.history_tokens(batch["past_action"], batch["past_action_valid"]), tokens,
                                   rtol=0, atol=0)
        pos0 = torch.tensor([30, 31, 29])
        for (k1, v1), (k2, v2) in zip(past.history_kv(tokens, pos0), p2n.history_kv(tokens, pos0)):
            torch.testing.assert_close(k2, k1, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(v2, v1, rtol=1e-6, atol=1e-6)
        # The joined [VLM | condition] keys the head reads are Pi05_past's as well.
        (_, layout_a, kv_a, valid_a), (_, layout_b, kv_b, valid_b) = joined_kv(past, batch), joined_kv(p2n, batch)
        assert torch.equal(valid_a, valid_b) and layout_a.nonki_len == layout_b.nonki_len
        for (k1, v1), (k2, v2) in zip(kv_a, kv_b):
            torch.testing.assert_close(k2, k1, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(v2, v1, rtol=1e-6, atol=1e-6)


def test_condition_is_a_static_memory_and_masking_it_recovers_the_no_history_head():
    policy = build().eval()
    batch = make_batch()
    prefix_out, layout, kv, valid = joined_kv(policy, batch)
    assert kv[0][0].shape[2] == layout.nonki_len + 9 and valid[:, layout.nonki_len:].all()
    plain = tuple((k.detach(), v.detach()) for k, v in prefix_out.kv)
    targets = policy.encode_targets(batch["action"])
    with torch.no_grad():
        with_history, _ = policy.ar_logits(kv, prefix_out.pos0, valid, targets)
        masked = valid.clone()
        masked[:, layout.nonki_len:] = False
        no_history, _ = policy.ar_logits(kv, prefix_out.pos0, masked, targets)
        plain_logits, _ = policy.ar_logits(plain, prefix_out.pos0, layout.nonki_valid, targets)
    torch.testing.assert_close(no_history, plain_logits, rtol=1e-5, atol=1e-5)
    assert (with_history - plain_logits).abs().max() > 1e-4


def test_ar_layout_is_causal_after_the_prefix_and_matches_the_cached_views():
    pos0 = torch.tensor([5, 3])
    prefix_valid = torch.tensor([[1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 0, 0, 0, 1]], dtype=torch.bool)
    layout = build_ar_suffix_layout(pos0, prefix_valid, 4)
    assert layout.positions.tolist() == [[5, 6, 7, 8], [3, 4, 5, 6]]
    assert layout.seg_ids.tolist() == [SEG_AR] * 4 and layout.n_cond == 0 and layout.nonki_len == 7
    assert layout.bias.shape == (2, 1, 4, 11) and ((layout.bias == 0) | (layout.bias == NEG)).all()
    visible = layout.bias[:, 0] == 0
    assert torch.equal(visible[:, :, :7], prefix_valid[:, None, :].expand(2, 4, 7))
    assert torch.equal(visible[:, :, 7:], torch.ones(4, 4, dtype=torch.bool).tril().expand(2, 4, 4))
    bias, positions, seg_ids = prefill_view(layout, 1)
    assert torch.equal(bias, layout.bias[:, :, :1, :8]) and positions.tolist() == [[5], [3]]
    for index in range(1, 4):
        bias, position = decode_view(layout, index)
        assert torch.equal(bias, layout.bias[:, :, index:index + 1, :8 + index])
        assert torch.equal(position, layout.positions[:, index:index + 1])
    for n_ar in (0, True, 2.0):
        with pytest.raises(ValueError, match="n_ar"):
            build_ar_suffix_layout(pos0, prefix_valid, n_ar)


# ------------------------------------------------------------------------------- gradients / decoding
@pytest.mark.parametrize("history_mode", [None, "generated"])
@pytest.mark.parametrize("lambda_ki", [0.0, 1.0])
def test_every_trainable_parameter_gets_a_gradient(history_mode, lambda_ki):
    policy = build(lambda_ki=lambda_ki, self_past_p=1.0).train()
    loss = policy(make_batch(), history_mode=history_mode)
    loss.backward()
    missing = [n for n, p in policy.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing[:10]
    components = policy.last_loss_components
    assert torch.isfinite(loss) and components["loss_flow"] is None
    for key in ("loss_ar", "ar_token_acc", "hist_attention_mass", "self_past_hist_mse"):
        assert components[key] is not None, key
    assert (components["loss_ki"] is not None) == (lambda_ki > 0)
    assert components["self_past_rows"] == 2.0  # two rows have a previous window
    assert 0.0 < components["hist_attention_mass"] < 1.0
    if lambda_ki > 0:
        assert not policy.dead_lora and all(p.requires_grad for n, p in policy.named_parameters() if ".lora_" in n)
    # The token loss reaches SigLIP through the un-detached K/V even without KI, and the tied embedding trains.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in policy.named_parameters()
               if n.startswith("siglip."))
    assert policy.tok_emb.weight.grad.abs().sum() > 0


def test_insulated_regime_blocks_the_token_loss_gradient_into_the_vlm():
    policy = build(kv_grad_scale=0.0, lambda_ki=1.0).train()
    policy.lambda_ki = 0.0  # isolate the token loss: no KI term this step
    loss = policy(make_batch(), history_mode="expert")
    loss.backward()
    vision = [p.grad for n, p in policy.named_parameters() if n.startswith("siglip.")]
    assert all(g is None or g.abs().sum() == 0 for g in vision)
    assert policy.cond_pos_emb.grad is not None  # the condition still trains (its K/V are never detached)


def test_greedy_cached_decode_equals_teacher_forcing_argmax():
    policy = build().eval()
    batch = make_batch()
    prefix_out, _, kv, valid = joined_kv(policy, batch)
    with torch.no_grad():
        tokens = policy.ar_generate(kv, prefix_out.pos0, valid, 8, 0.0, 10)
        logits, _ = policy.ar_logits(kv, prefix_out.pos0, valid, tokens)
        assert torch.equal(policy.ar_generate(kv, prefix_out.pos0, valid, 3, 0.0, 10), tokens[:, :3])
    logits[..., policy.BOS_ID] = float("-inf")
    assert tokens.shape == (3, 8) and torch.equal(logits.argmax(-1), tokens)
    with pytest.raises(ValueError, match="n_tokens"):
        policy.ar_generate(kv, prefix_out.pos0, valid, 9, 0.0, 10)


def test_cached_decode_logits_equal_teacher_forcing_logits():
    """Each cached decoding step sees exactly what its teacher-forced row sees: inputs [BOS, z0..z6], RoPE pos0+k,
    the SEG_AR modulation, every valid VLM key plus the 9 condition keys, and causal token keys."""
    policy = _break_symmetries(build().eval())
    batch = make_batch()
    prefix_out, layout, kv, valid = joined_kv(policy, batch)
    forced = torch.randint(0, 5000, (3, 8), generator=torch.Generator().manual_seed(5))  # not one repeated token
    seen = _record_decoding(policy, forced)
    with torch.no_grad():
        assert torch.equal(policy.ar_generate(kv, prefix_out.pos0, valid, 8, 0.0, 10), forced)
        teacher, _ = policy.ar_logits(kv, prefix_out.pos0, valid, forced)
        masked = valid.clone()
        masked[:, layout.nonki_len:] = False
        no_history, _ = policy.ar_logits(kv, prefix_out.pos0, masked, forced)
    assert (teacher - no_history).abs().max() > 1e-3  # the history moves the logits measurably
    torch.testing.assert_close(torch.stack(seen, dim=1), teacher, rtol=1e-5, atol=1e-5)


def test_every_decoding_step_uses_the_requested_sampler():
    """Self-past decodes at T=1 / top-k 10, deployment greedily, and evaluator overrides reach every step."""
    policy = build(self_past_p=1.0).train()
    calls = []

    def spy(logits, temperature, topk, bos_id):
        calls.append((float(temperature), topk))
        return P2NVLACommonPolicy._sample(logits, temperature, topk, bos_id)

    policy._sample = spy
    batch = make_batch()
    policy(batch, history_mode="generated")  # rows 0 and 2 form one chunk: 8 decoding steps
    assert calls == [(1.0, 10)] * 8
    calls.clear()
    policy.eval()
    kwargs = dict(past_actions=batch["past_action"], past_action_valid=batch["past_action_valid"])
    policy.predict_action(batch["obs"], **kwargs)
    assert calls == [(0.0, 10)] * 8
    calls.clear()
    policy.predict_action(batch["obs"], use_k_tokens=3, temperature=0.7, topk=5, **kwargs)
    assert calls == [(0.7, 5)] * 3


@pytest.mark.parametrize("temperature,topk", [(1.0, 10), (0.7, 5), (0.0, 10)])
def test_token_sampler_is_the_original_past2next_rule(temperature, topk):
    logits = torch.randn(64, 5001, generator=torch.Generator().manual_seed(11)) * 3
    for seed in range(10):
        torch.manual_seed(seed)
        ours = P2NVLACommonPolicy._sample(logits, temperature, topk, 5000)
        torch.manual_seed(seed)
        assert torch.equal(ours, _original_sample(logits, temperature, topk, 5000))
        assert not (ours == 5000).any()


def test_self_past_uses_the_self_past_sampler_and_only_valid_rows():
    policy = build(self_past_p=1.0).train()
    calls = []
    original = policy._generate_actions

    def spy(obs, past, valid, n_tokens, temperature, topk):
        calls.append((past.shape[0], n_tokens, temperature, topk))
        return original(obs, past, valid, n_tokens, temperature, topk)

    policy._generate_actions = spy
    batch = make_batch()
    policy(batch, history_mode="generated")
    assert sum(c[0] for c in calls) == 2 and {c[1:] for c in calls} == {(8, 1.0, 10)}
    assert policy.last_loss_components["self_past_rows"] == 2.0
    # self_past_hist_mse is the normalized-action MSE over exactly the substituted (selected, valid) slots.
    torch.manual_seed(4)
    history = policy._maybe_self_past(batch, batch["past_action"], 1.0)
    substituted = (history != batch["past_action"]).any(-1)
    assert substituted.sum() == 14 and not substituted[1].any()  # rows 0 and 2, all 7 slots valid
    normalize = policy.action_normalizer["action"].normalize
    expected = (normalize(history) - normalize(batch["past_action"])).square().mean(-1)[substituted].mean()
    assert policy._last_self_past_mse == pytest.approx(float(expected), rel=1e-6)
    calls.clear()
    policy.eval()
    policy(batch, history_mode="expert")
    assert not calls and policy.last_loss_components["self_past_rows"] == 0.0
    assert policy.last_loss_components["self_past_hist_mse"] is None  # no stale value in expert mode
    policy.predict_action(batch["obs"], past_actions=batch["past_action"], past_action_valid=batch["past_action_valid"])
    assert calls[-1][1:] == (8, 0.0, 10)  # deployment decodes with the deployment sampler


def test_configured_history_follows_the_warmup_and_probability():
    """history_mode=None in training (the run's path): no self-past during the warmup, then each row with p."""
    policy = build(self_past_p=0.5, self_past_warmup_steps=2).train()
    batch = make_batch(batch=8, episode_step=(12,) * 8)
    torch.manual_seed(0)
    policy(batch)
    assert policy.last_loss_components["self_past_p"] == 0.0 and policy.last_loss_components["self_past_rows"] == 0.0
    assert policy.last_loss_components["self_past_hist_mse"] is None
    policy.set_self_past_step(2)
    rows = []
    for seed in range(4):
        torch.manual_seed(seed)
        policy(batch)
        assert policy.last_loss_components["self_past_p"] == 0.5
        rows.append(policy.last_loss_components["self_past_rows"])
    assert 0 < sum(rows) < 32, rows


def test_self_past_history_is_the_executed_part_of_the_previous_window_decode():
    """Selected valid slots hold actions 1..7 of the chunk decoded from (prev_obs, prev_past): the last 7 of the 8
    commands the previous window executed, as record_executed_actions stores at deployment."""
    policy = build(self_past_p=1.0, self_past_temperature=0.0).train()  # greedy self-past: a deterministic decode
    batch = make_batch()
    rows = torch.tensor([0, 2])  # the samples with a previous window
    policy.eval()
    with torch.no_grad():
        decoded = policy._generate_actions({k: v[rows] for k, v in batch["prev_obs"].items()},
                                           batch["prev_past_action"][rows], batch["prev_past_action_valid"][rows],
                                           8, 0.0, 10)
    policy.train()
    history = policy._maybe_self_past(batch, batch["past_action"], 1.0)
    expected = torch.cat((batch["prev_past_action"][rows], decoded[:, :8]), dim=1)[:, -7:]
    assert torch.equal(expected, decoded[:, 1:8]) and not torch.equal(decoded[:, :7], decoded[:, 1:8])
    torch.testing.assert_close(history[rows], expected, rtol=0, atol=1e-6)  # rows 0 and 2: all 7 slots valid
    assert torch.equal(history[1], batch["past_action"][1])


def test_stateful_rollout_protocol_and_deterministic_greedy_deployment():
    policy = build().eval()
    policy.reset()
    obs = make_batch(batch=2, episode_step=(0, 0))["obs"]
    out = policy.predict_action(obs)
    assert out["action"].shape == (2, 8, 7) and out["action_pred"].shape == (2, 16, 7)
    assert torch.isfinite(out["action_pred"]).all()
    policy.record_executed_actions(out["action"])
    assert policy._past_valid_buffer.all() and torch.equal(policy._past_buffer, out["action"][:, 1:])
    assert policy.predict_action(obs)["action"].shape == (2, 8, 7)
    batch = make_batch()
    kwargs = dict(past_actions=batch["past_action"], past_action_valid=batch["past_action_valid"])
    torch.manual_seed(1)
    first = policy.predict_action(batch["obs"], **kwargs)["action_pred"]
    torch.manual_seed(2)
    assert torch.equal(policy.predict_action(batch["obs"], **kwargs)["action_pred"], first)  # greedy: no randomness
    assert policy.predict_action(batch["obs"], use_k_tokens=3, **kwargs)["action_pred"].shape == (3, 16, 7)
    assert torch.isfinite(policy.predict_action(batch["obs"], temperature=1.0, **kwargs)["action_pred"]).all()


# ------------------------------------------------------------------------------- eval-only diagnostics
def test_history_interventions_change_only_what_the_head_reads():
    policy = build().eval()
    batch = make_batch()
    obs, past, valid = batch["obs"], batch["past_action"], batch["past_action_valid"]

    def predict(history, history_valid):
        return policy.predict_action(obs, past_actions=history, past_action_valid=history_valid)["action_pred"]

    expected = {"zero": predict(torch.zeros_like(past), torch.zeros_like(valid)),
                "swap": predict(past.roll(1, 0), valid.roll(1, 0))}
    for mode, reference in expected.items():
        policy.set_history_intervention(mode)
        torch.testing.assert_close(predict(past, valid), reference, rtol=0, atol=0)
    seen = []
    original = policy.history_tokens
    policy.history_tokens = lambda history, history_valid: (seen.append(history), original(history, history_valid))[1]
    policy.set_history_intervention("noise", noise_std=0.5)
    predict(past, valid)
    assert torch.equal(seen[-1][~valid], past[~valid]) and not torch.equal(seen[-1][valid], past[valid])
    # The execution buffer keeps the true commands; training and self-past never see the intervention.
    policy.reset()
    out = policy.predict_action(obs)
    policy.record_executed_actions(out["action"])
    assert torch.equal(policy._past_buffer, out["action"][:, 1:])
    policy.train()
    torch.manual_seed(9)
    intervened = policy(batch, history_mode="generated")
    policy.set_history_intervention(None)
    torch.manual_seed(9)
    assert torch.equal(intervened, policy(batch, history_mode="generated"))


# ------------------------------------------------------------------------------------ artifacts / arms
def test_export_restore_round_trip_and_variant_separation(tmp_path):
    policy = build().eval()
    path = tmp_path / "snapshot.ckpt"
    torch.save({"format": "p2n_vla_checkpoint_v1", "policy_config": policy.export_config(),
                "metadata": policy.artifact_metadata(),
                "state_dicts": {"model": policy.artifact_state_dict()}}, path)
    restored = Pi05P2NPolicy.from_checkpoint(str(path), weights="model")
    assert type(restored) is Pi05P2NPolicy and not hasattr(restored, "type_embedding")
    assert (restored.temperature, restored.self_past_temperature, restored.self_past_topk) == (0.0, 1.0, 10)
    batch = make_batch()
    kwargs = dict(past_actions=batch["past_action"], past_action_valid=batch["past_action_valid"])
    torch.testing.assert_close(restored.predict_action(batch["obs"], **kwargs)["action_pred"],
                               policy.predict_action(batch["obs"], **kwargs)["action_pred"], rtol=0, atol=0)
    assert [n for n, _ in restored.trainable_named_parameters()] == [n for n, _ in policy.trainable_named_parameters()]
    metadata = policy.artifact_metadata()
    assert "pi05_past" not in metadata and metadata["pi05_p2n"]["head"] == "past2next_ar_oat8"
    assert metadata["pi05_p2n"]["condition_tokens"] == 9 and "flow_num_steps" not in metadata["pi05_p2n"]
    assert {"oat/policy/pi05_p2n.py", "oat/policy/pi05_past.py"} <= set(metadata["source_sha256"])
    # Artifacts never cross variants (different heads, variant codes 13 vs 14).
    with pytest.raises((KeyError, ValueError)):
        build(cls=Pi05PastPolicy).load_artifact_state(policy.artifact_state_dict())
    with pytest.raises((KeyError, ValueError)):
        build().load_artifact_state(build(cls=Pi05PastPolicy).artifact_state_dict())


def test_no_past_control_arm():
    policy = build(use_past=False).train()
    assert policy.n_cond_tokens == 0 and policy.self_past_p == 0.0
    assert not any(hasattr(policy, n) for n in ("acc_proj", "raw_proj", "cond_pos_emb", "type_embedding"))
    loss = policy(make_batch(), history_mode="generated")
    loss.backward()
    assert policy.last_loss_components["hist_attention_mass"] is None
    assert not [n for n, p in policy.named_parameters() if p.requires_grad and p.grad is None]


# -------------------------------------------------------------------------------- launcher + workspace
def _compose(*overrides, variant="pi05_p2n"):
    import scripts.train_p2n_vla as launcher
    return launcher.compose_config(variant, "libero",
                                   ["policy.model_size=tiny", "policy.pi05_weights=null", *overrides])


def test_launcher_validates_the_pi05_p2n_config():
    cfg = _compose()
    policy = cfg.policy
    assert policy._target_ == "oat.policy.pi05_p2n.Pi05P2NPolicy" and policy.use_past is True
    assert policy.flow_num_steps is None and policy.flow_time_beta is None and policy.adarms_t0 == policy.cond_t0 == 0.6
    assert (policy.temperature, policy.topk, policy.self_past_temperature, policy.self_past_topk) == (0.0, 10, 1.0, 10)
    assert cfg.training.max_train_steps == 400 and cfg.training.max_optimizer_steps == 80000
    assert cfg.training.num_epochs == 800 and policy.self_past_ramp_steps == 0 and policy.lambda_ki == 0.0
    assert cfg.task.policy.lazy_eval is False and cfg.training.rollout_every == 100
    runner = cfg.task.policy.env_runner
    assert runner.protocol == "official" and runner.n_test == 500 and runner.n_parallel_envs == 10
    assert cfg.training.keep_snapshots == "all" and cfg.training.snapshot_every == 0
    assert _compose("policy.cond_t0=0.5").policy.adarms_t0 == 0.5  # the fold follows the condition's time
    for override, match in (("policy.flow_num_steps=10", "no flow head"), ("policy.adarms_t0=0.5", "adarms_t0"),
                            ("policy.temperature=-1", "temperature"), ("policy.topk=0", "topk"),
                            ("policy.self_past_topk=0", "self_past_topk")):
        with pytest.raises(ValueError, match=match):
            _compose(override)
    _compose("policy.self_past_temperature=null", "policy.self_past_topk=null")
    _compose("policy.use_past=false", "training.validate_generated_history=false")
    with pytest.raises(ValueError, match="no generated history"):
        _compose("policy.use_past=false")
    with pytest.raises(ValueError, match="without a loss"):
        _compose("policy.kv_grad_scale=0", "policy.lambda_ki=0")


def test_config_matches_pi05_past_except_the_head():
    """With the Pi05_past run's launcher overrides, pi05_p2n differs from Pi05_past, and from the run's stored
    resolved config when it exists, only in naming, the flow knobs and snapshot retention."""
    import scripts.train_p2n_vla as launcher
    overrides = [f"policy.tokenizer_checkpoint={TOKENIZER}", f"policy.spm_path={SPM}",
                 "training.gradient_accumulate_every=2", "logging.mode=online"]
    p2n = _resolved(launcher.compose_config("pi05_p2n", "libero", overrides))
    references = [_resolved(launcher.compose_config("pi05_past", "libero", overrides))]
    if STORED_PI05_PAST_CONFIG.is_file():
        references.append(_resolved(OmegaConf.load(STORED_PI05_PAST_CONFIG)))
    for reference in references:
        differing = {k for k in set(p2n) | set(reference) if p2n.get(k, "<missing>") != reference.get(k, "<missing>")}
        assert differing == HEAD_KEYS, sorted(differing ^ HEAD_KEYS)


def test_shell_launchers_route_the_variant_and_refuse_head_mismatches():
    def run(script, *args):
        return subprocess.run(["bash", str(REPO / script), *args], cwd=str(REPO), capture_output=True, text=True,
                              timeout=600)

    result = run("train_pi05_p2n.sh", "--devices", "0,1,2,3", "--dry-run")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Pi05_p2n: mode=dry-run" in result.stdout and "batch=4 x 4 GPUs x 2 = 32" in result.stdout
    assert "output/training/pi05_p2n_libero10_seed42" in result.stdout and "--variant pi05_p2n" in result.stderr
    for script, args, message in (("train_pi05_p2n.sh", ("--flow-steps", "5"), "pi05_past flow head only"),
                                  ("train_pi05_p2n.sh", ("--variant", "pi05_past"), "given twice"),
                                  ("train_pi05_past.sh", ("--temperature", "1"), "pi05_p2n token head only"),
                                  ("train_pi05_past.sh", ("--variant", "pi05_x"), "pi05_past or pi05_p2n")):
        result = run(script, "--devices", "0,1", *args, "--dry-run")
        assert result.returncode == 2 and message in result.stderr, (script, args, result.stderr[-1000:])


def tiny_run_config(zarr_path, *overrides):
    return _compose(
        "policy.activation_checkpointing=false", "policy.lora_rank=4", "policy.lora_alpha=4.0",
        "policy.self_past_warmup_steps=0", f"task.policy.dataset.zarr_path={zarr_path}", "training.num_demo=3",
        "task.policy.lazy_eval=true", "training.keep_snapshots=all",
        "training.num_epochs=2", "training.gradient_accumulate_every=1", "training.max_train_steps=2",
        "training.max_optimizer_steps=10", "training.lr_warmup_steps=1", "training.snapshot_every=1",
        "training.val_every=1", "training.checkpoint_every=1", "training.max_val_steps=1",
        "training.max_reconst_steps=1", "dataloader.batch_size=2", "dataloader.num_workers=0",
        "dataloader.persistent_workers=false", "dataloader.pin_memory=false", "val_dataloader.batch_size=2",
        "val_dataloader.num_workers=0", "val_dataloader.persistent_workers=false", "val_dataloader.pin_memory=false",
        "logging.mode=disabled", *overrides)


@pytest.mark.slow
def test_tiny_pi05_p2n_trains_validates_snapshots_and_reloads(tmp_path, cpu_accelerate, monkeypatch):
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    workspace = TrainP2NVLAWorkspace(tiny_run_config(zarr_path, "policy.self_past_p=1.0"),
                                     output_dir=str(tmp_path / "run"))
    workspace.run()
    assert workspace.completed_optimizer_steps == 4
    steps = _records(tmp_path / "run/logs.jsonl", "train_step")
    assert steps and all(r.get("train/loss_ar") is not None and r.get("train/loss_flow") is None for r in steps)
    assert any(r.get("train/self_past_hist_mse") is not None for r in steps)
    epochs = _records(tmp_path / "run/logs.jsonl", "epoch")
    assert any(r.get("val_loss_generated_history") is not None for r in epochs)
    snapshots = sorted((tmp_path / "run/snapshots").glob("upd-*_model.ckpt"))
    assert snapshots, "use_ema=false must write _model snapshots"
    restored = Pi05P2NPolicy.from_checkpoint(str(snapshots[-1]), weights="model")
    assert restored.VARIANT == "pi05_p2n" and restored.train_vision and restored.self_past_temperature == 1.0
    # The evaluator accepts the snapshot with the token-sampler overrides and a history intervention (dry run) ...
    import scripts.evaluate_p2n_vla as evaluate
    from scripts.pi05_past_offline_diagnostics import main as offline_diagnostics
    monkeypatch.chdir(REPO)
    for key, default in (("MUJOCO_GL", "egl"), ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
                         ("HF_HOME", "/workspace/.hf_home")):
        monkeypatch.setenv(key, os.environ.get(key, default))
    output = tmp_path / "eval"
    assert evaluate.main(["--snapshot", str(snapshots[-1]), "--weights", "model", "--output-dir", str(output),
                          "--device", "cpu", "--n-test", "10", "--history-intervention", "noise",
                          "--history-noise-std", "0.2", "--temperature", "1.0", "--topk", "10", "--use-k-tokens", "4",
                          "--threads", str(torch.get_num_threads()), "--dry-run"]) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "ready" and metadata["variant"] == "pi05_p2n"
    assert metadata["history_intervention"] == {"mode": "noise", "noise_std": 0.2}
    # ... and the offline diagnostics run on its held-out windows.
    report = offline_diagnostics(["--snapshot", str(snapshots[-1]), "--device", "cpu", "--batch-size", "4",
                                  "--max-windows", "8", "--output", str(tmp_path / "offline.json")])
    assert report["windows"] >= 2 and len(report["innovation_r2"]) == 16


@pytest.mark.slow
def test_tiny_pi05_p2n_resume_is_bitwise_identical_to_an_uninterrupted_run(tmp_path, cpu_accelerate):
    import shutil
    from accelerate.state import AcceleratorState
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")

    def run(output, *overrides):
        AcceleratorState._reset_state(True)
        TrainP2NVLAWorkspace(tiny_run_config(zarr_path, "policy.self_past_p=1.0", *overrides),
                             output_dir=str(output)).run()

    run(tmp_path / "full")
    run(tmp_path / "first", "training.num_epochs=1")
    shutil.copytree(tmp_path / "first", tmp_path / "resumed")
    latest = tmp_path / "resumed/checkpoints/latest.ckpt"
    run(tmp_path / "resumed", "training.resume=true", f"training.resume_checkpoint={latest}")
    for name in ("snapshots/upd-000004_model.ckpt", "checkpoints/latest.ckpt"):
        full, resumed = (torch.load(root / name, map_location="cpu", weights_only=False)["state_dicts"]["model"]
                         for root in (tmp_path / "full", tmp_path / "resumed"))
        assert sorted(full) == sorted(resumed)
        for key, value in full.items():
            torch.testing.assert_close(resumed[key], value, rtol=0, atol=0, msg=key)


@pytest.mark.slow
def test_tiny_pi05_p2n_rollouts_keep_every_evaluated_snapshot(tmp_path, cpu_accelerate):
    from test_p2n_vla_workspace import ScoredRolloutRunner
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    # keep_snapshots=all (the pi05_p2n default) with snapshot_every=0: each evaluation keeps the weights it scored.
    cfg = tiny_run_config(zarr_path, "task.policy.lazy_eval=false", "training.rollout_every=1",
                          "training.snapshot_every=0")
    OmegaConf.update(cfg, "task.policy.env_runner", {  # the real policy, a scripted simulator
        "_target_": "test_p2n_vla_workspace.ScoredRolloutRunner", "task_name": "libero10", "protocol": "official",
        "n_test": 4, "n_test_vis": 0, "test_start_seed": 3000, "init_state_offset": 0, "n_parallel_envs": 2,
        "max_episode_steps": 16}, merge=False)
    ScoredRolloutRunner.rates[:] = [0.75, 0.25]
    TrainP2NVLAWorkspace(cfg, output_dir=str(tmp_path / "run")).run()
    rollouts = _records(tmp_path / "run/logs.jsonl", "rollout")
    assert [record["rollout/success_rate"] for record in rollouts] == [0.75, 0.25]
    kept = sorted(path.name for path in (tmp_path / "run/snapshots").glob("upd-*.ckpt"))
    assert kept == [f"upd-{record['optimizer_step']:06d}_model.ckpt" for record in rollouts]
    assert not (tmp_path / "run/snapshots/best.json").exists() and (tmp_path / "run/checkpoints/latest.ckpt").is_file()


@pytest.mark.slow
def test_tiny_pi05_p2n_two_rank_gloo_ddp(tmp_path):
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    cfg = tiny_run_config(zarr_path, "policy.self_past_p=1.0")
    config_path = tmp_path / "worker.yaml"
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), config_path)
    output = tmp_path / "run"
    env = dict(os.environ, ACCELERATE_USE_CPU="true", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
               str(REPO / "scripts/train_p2n_vla.py"), "--worker-config", str(config_path), "--output", str(output)]
    result = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    report = json.loads((output / "training_report.json").read_text())
    assert report["world_size"] == 2 and report["ddp"]["find_unused_parameters"] is False
    epochs = _records(output / "logs.jsonl", "epoch")
    assert len(epochs) == 2 and all(r["replica_fingerprint"] is not None for r in epochs)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_trains_and_decodes_through_the_fp32_condition_path():
    def to_cuda(value):
        if isinstance(value, dict):
            return {key: to_cuda(item) for key, item in value.items()}
        return value.cuda() if isinstance(value, torch.Tensor) else value

    policy = build(frozen_dtype="bfloat16", activation_checkpointing=True, self_past_p=1.0).cuda().train()
    batch = to_cuda(make_batch())
    loss = policy(batch, history_mode="generated")
    loss.backward()
    assert torch.isfinite(loss) and policy.last_loss_components["self_past_rows"] == 2.0
    trainable = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    assert not [name for name, p in trainable if p.grad is None or not torch.isfinite(p.grad).all()]
    policy.eval()
    out = policy.predict_action(batch["obs"], past_actions=batch["past_action"],
                                past_action_valid=batch["past_action_valid"])
    assert out["action_pred"].dtype == torch.float32 and torch.isfinite(out["action_pred"]).all()
    with torch.no_grad(), policy._autocast():
        prefix_out, layout, _ = policy.vlm_pass(policy.build_prefix(batch["obs"], train_aug=False), None)
        kv, valid = policy._expert_kv(prefix_out, layout, batch["past_action"], batch["past_action_valid"])
        assert kv[0][0].dtype == prefix_out.kv[0][0].dtype == torch.bfloat16  # fp32 condition K/V cast at the join
        assert kv[0][0].shape[2] == layout.nonki_len + policy.n_cond_tokens
    # Cached bf16 decoding reads what teacher forcing reads (logit level, symmetry-broken weights).
    _break_symmetries(policy)
    forced = torch.randint(0, 5000, (3, 8), generator=torch.Generator().manual_seed(5)).cuda()
    seen = _record_decoding(policy, forced)
    with torch.no_grad(), policy._autocast():
        prefix_out, layout, _ = policy.vlm_pass(policy.build_prefix(batch["obs"], train_aug=False), None)
        kv, valid = policy._expert_kv(prefix_out, layout, batch["past_action"], batch["past_action_valid"])
        assert torch.equal(policy.ar_generate(kv, prefix_out.pos0, valid, 8, 0.0, 10), forced)
        teacher, _ = policy.ar_logits(kv, prefix_out.pos0, valid, forced)
    decoded = torch.stack(seen, dim=1)
    # bf16 kernels differ between the single-row cached and the full pass. Measured on an RTX 4090 as a fraction of
    # max|logit|: 0.0010 here, against 0.0066 for a one-position RoPE shift and 0.020 for ignoring the condition.
    gap = float((decoded - teacher).abs().max() / teacher.abs().max())
    assert gap < 0.003, gap
