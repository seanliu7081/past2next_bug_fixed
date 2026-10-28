"""Pi05_past (pi05 flow head + Past2Next condition) on tiny CPU models.

Uses the repo-local recovered OAT tokenizer and FAILS (never skips) when an asset is missing, so a green run
always exercised the policy. Run: /venv/oat/bin/python -m pytest tests/test_pi05_past.py -q
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.vla.flow_head import flow_condition
from oat.model.vla.layout import build_flow_suffix_layout
from oat.model.vla.lora import LoRALinear
from oat.policy.pi05_past import Pi05PastPolicy, UNREACHABLE_LAST_LAYER

REPO = Path(__file__).resolve().parents[1]
TOKENIZER = REPO / "data/pretrained/p2n_vla/oat_from_upd030000.ckpt"
SPM = REPO / "data/pretrained/p2n_vla/paligemma_tokenizer.model"

SHAPE_META = {
    "obs": {
        "agentview_rgb": {"shape": [128, 128, 3], "type": "rgb"},
        "robot0_eye_in_hand_rgb": {"shape": [128, 128, 3], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3], "type": "state"},
        "robot0_eef_quat": {"shape": [4], "type": "state"},
        "robot0_gripper_qpos": {"shape": [2], "type": "state"},
        "task_uid": {"shape": [1], "type": "state"},
    },
    "action": {"shape": [7]},
}


# --------------------------------------------------------------------------------------------- helpers
def _actions(*shape, generator):
    xyz = torch.rand(*shape, 3, generator=generator) * 1.8 - 0.9
    rot = torch.rand(*shape, 3, generator=generator) * 0.5 - 0.2
    grip = torch.where(torch.rand(*shape, 1, generator=generator) > 0.5, 1.0, -1.0)
    return torch.cat((xyz, rot, grip), dim=-1)


def _normalizer():
    generator = torch.Generator().manual_seed(7)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": _actions(512, generator=generator)}, last_n_dims=1, mode="limits")
    q01, q99 = -torch.ones(8) * 0.5, torch.ones(8) * 0.5
    scale = 2.0 / (q99 - q01 + 1e-6)
    normalizer["prompt_state"] = SingleFieldLinearNormalizer.create_manual(scale, -1.0 - q01 * scale,
                                                                           {"q01": q01, "q99": q99})
    return normalizer


def _obs(batch, generator):
    quat = torch.randn(batch, 1, 4, generator=generator)
    return {
        "agentview_rgb": torch.randint(0, 256, (batch, 1, 128, 128, 3), generator=generator, dtype=torch.uint8),
        "robot0_eye_in_hand_rgb": torch.randint(0, 256, (batch, 1, 128, 128, 3), generator=generator,
                                                dtype=torch.uint8),
        "robot0_eef_pos": torch.randn(batch, 1, 3, generator=generator) * 0.1,
        "robot0_eef_quat": quat / quat.norm(dim=-1, keepdim=True),
        "robot0_gripper_qpos": torch.rand(batch, 1, 2, generator=generator) * 0.08 - 0.04,
        "task_uid": torch.randint(30, 40, (batch, 1, 1), generator=generator),
    }


def make_batch(batch=3, episode_step=(12, 3, 9), seed=0):
    generator = torch.Generator().manual_seed(seed)
    steps = torch.tensor(episode_step)
    past_valid = torch.arange(7)[None] >= (7 - steps.clamp(max=7)[:, None])
    prev_valid = torch.arange(7)[None] >= (7 - (steps - 8).clamp(min=0, max=7)[:, None])
    return {
        "obs": _obs(batch, generator), "prev_obs": _obs(batch, generator),
        "action": _actions(batch, 16, generator=generator),
        "past_action": _actions(batch, 7, generator=generator) * past_valid[..., None],
        "past_action_valid": past_valid,
        "prev_past_action": _actions(batch, 7, generator=generator) * prev_valid[..., None],
        "prev_past_action_valid": prev_valid,
        "prev_window_valid": steps >= 8, "episode_step": steps,
    }


def build(seed=0, **kwargs):
    torch.manual_seed(seed)
    settings = dict(shape_meta=SHAPE_META, model_size="tiny", tokenizer_checkpoint=str(TOKENIZER),
                    spm_path=str(SPM), activation_checkpointing=False, self_past_warmup_steps=0,
                    self_past_ramp_steps=0, lambda_ki=0.0, lora_rank=4, lora_alpha=4.0)
    settings.update(kwargs)
    policy = Pi05PastPolicy(**settings)
    policy.set_normalizer(_normalizer())
    return policy


def _names(policy, predicate):
    return {name for name, parameter in policy.named_parameters() if predicate(name, parameter)}


def test_assets_present():
    assert TOKENIZER.is_file(), f"recovered OAT tokenizer missing: {TOKENIZER}"
    assert SPM.is_file(), f"PaliGemma SentencePiece model missing: {SPM}"


# -------------------------------------------------------------------------------------- construction
def test_default_scope_groups_and_frozen_keys():
    from oat.workspace.train_p2n_vla import validate_parameter_partition
    policy = build()
    assert policy.dtype == torch.float32 and policy.VARIANT_CODE == 13 and policy.n_cond_tokens == 9
    trainable = _names(policy, lambda n, p: p.requires_grad)
    assert any(n.startswith("siglip.") for n in trainable), "SigLIP must be trainable by default"
    assert not any(n.startswith("ki_table.") for n in trainable), "KI table must be frozen with lambda_ki=0"
    last = len(policy.joint.vlm.layers) - 1
    dead = {f"joint.vlm.layers.{last}.{path}.{ab}" for path in UNREACHABLE_LAST_LAYER for ab in ("lora_A", "lora_B")}
    assert set(policy.dead_lora) == dead and not (dead & trainable)
    lora = _names(policy, lambda n, p: ".lora_" in n)
    assert lora - dead <= trainable and len(lora - dead) == len(lora) - 10
    for module in (policy.joint.vlm.layers[last].self_attn.k_proj, policy.joint.vlm.layers[last].self_attn.v_proj):
        assert isinstance(module, LoRALinear) and module.lora_A.requires_grad  # last-layer K/V are read
    optimizer = policy.get_optimizer()
    named, clip = validate_parameter_partition(policy, optimizer)
    by_id = {id(p): n for n, p in named}
    groups = {g["name"]: {by_id[id(p)] for p in g["params"]} for g in optimizer.param_groups}
    assert all(n.startswith(("joint.expert.", "flow_heads.", "siglip.")) for n in groups["pretrained"])
    assert any(n.startswith("siglip.") for n in groups["pretrained"])
    assert all(".lora_" in n or n.startswith(("acc_proj", "jerk_proj", "raw_proj", "cond_pos_emb"))
               for n in groups["new"])
    clip_names = {k: {by_id[id(p)] for p in v} for k, v in clip.items()}
    assert all(n.startswith("siglip.") or ".lora_" in n for n in clip_names["ki"])
    assert all(n.startswith(("joint.expert.", "flow_heads.", "acc_proj", "jerk_proj", "raw_proj", "cond_pos_emb"))
               for n in clip_names["ar"])
    assert not any(k.startswith("siglip.") for k in policy.frozen_base_keys())
    assert all(t.dtype == torch.float32 for _, t in named)


def test_null_self_past_knobs_and_invalid_regimes():
    build(self_past_temperature=None, self_past_topk=None)  # the original yaml writes null
    with pytest.raises(ValueError, match="without"):
        build(kv_grad_scale=0.0, lambda_ki=0.0)
    build(kv_grad_scale=0.0, lambda_ki=0.0, train_vision=False, train_lora=False)  # frozen VLM is allowed
    with pytest.raises(ValueError, match="kv_grad_scale"):
        build(kv_grad_scale=1.5)
    with pytest.raises(ValueError, match="ki_init"):
        build(ki_init="fast")
    with pytest.raises(ValueError, match="cond_dropout"):
        build(cond_dropout=1.0)


# ---------------------------------------------------------------------------------- condition fidelity
def test_condition_tokens_follow_the_past2next_formula():
    policy = build().eval()
    batch = make_batch()
    past, valid = batch["past_action"].clone(), batch["past_action_valid"]
    tokens = policy.history_tokens(past, valid)
    # Past2NextPolicy._build_condition: normalize the (raw zero-padded) past, then acc/jerk from the last 3.
    normalized = policy.action_normalizer["action"].normalize(torch.where(valid[..., None], past, 0 * past)).float()
    acc = normalized[:, -1] - normalized[:, -2]
    jerk = normalized[:, -1] - 2 * normalized[:, -2] + normalized[:, -3]
    reference = torch.cat((policy.acc_proj(acc)[:, None], policy.jerk_proj(jerk)[:, None],
                           policy.raw_proj(normalized)), dim=1) + policy.cond_pos_emb
    assert tokens.shape == (3, 9, policy.expert_width)
    torch.testing.assert_close(tokens, reference)
    # Unavailable slots behave as raw zeros, whatever the tensor holds there (no masking, as the original).
    garbage = past.clone()
    garbage[~valid] = 123.0
    torch.testing.assert_close(policy.history_tokens(garbage, valid), tokens)
    zero_token = policy.raw_proj(policy.action_normalizer["action"].normalize(torch.zeros(1, 7)).float())[0]
    torch.testing.assert_close(tokens[1, 2], zero_token + policy.cond_pos_emb[2])  # row 1 has 3 valid of 7


def test_condition_is_a_static_memory_and_masking_it_recovers_plain_pi05():
    policy = build().eval()
    batch = make_batch()
    with torch.no_grad():
        prefix = policy.build_prefix(batch["obs"], train_aug=False)
        prefix_out, layout, _ = policy.vlm_pass(prefix, None)
        plain = tuple((k.detach(), v.detach()) for k, v in prefix_out.kv)
        tokens = policy.history_tokens(batch["past_action"], batch["past_action_valid"])
        cond_kv = policy.history_kv(tokens, prefix_out.pos0)
        assert len(cond_kv) == policy.joint.expert_spec.depth and cond_kv[0][0].shape[2] == 9
        joined, valid = policy._join_history(plain, cond_kv, layout.nonki_valid)
        x_t = torch.randn(3, 16, 32)
        t = torch.full((3,), 0.4)
        with_history, _ = policy._velocity(joined, prefix_out.pos0, valid, x_t, t)
        masked = valid.clone()
        masked[:, layout.nonki_len:] = False
        no_history, _ = policy._velocity(joined, prefix_out.pos0, masked, x_t, t)
        plain_velocity, _ = policy._velocity(plain, prefix_out.pos0, layout.nonki_valid, x_t, t)
        torch.testing.assert_close(no_history, plain_velocity, rtol=1e-5, atol=1e-5)
        assert (with_history - plain_velocity).abs().max() > 1e-4
        # Another history changes the velocity; the condition K/V do not depend on x_t or t.
        other = policy.history_kv(policy.history_tokens(batch["past_action"].flip(1), batch["past_action_valid"]),
                                  prefix_out.pos0)
        other_joined, _ = policy._join_history(plain, other, layout.nonki_valid)
        changed, _ = policy._velocity(other_joined, prefix_out.pos0, valid, x_t, t)
        assert (changed - with_history).abs().max() > 1e-5
        again = policy.history_kv(tokens, prefix_out.pos0)
        for (k1, v1), (k2, v2) in zip(cond_kv, again):
            assert torch.equal(k1, k2) and torch.equal(v1, v2)


# ------------------------------------------------------------------------------- gradients / self-past
@pytest.mark.parametrize("history_mode", [None, "generated"])
@pytest.mark.parametrize("lambda_ki", [0.0, 1.0])
def test_every_trainable_parameter_gets_a_gradient(history_mode, lambda_ki):
    policy = build(lambda_ki=lambda_ki, self_past_p=1.0).train()
    loss = policy(make_batch(), history_mode=history_mode)
    loss.backward()
    missing = [n for n, p in policy.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing[:10]
    assert torch.isfinite(loss)
    components = policy.last_loss_components
    assert components["loss_flow"] is not None and components["hist_attention_mass"] is not None
    assert (components["loss_ki"] is not None) == (lambda_ki > 0)
    assert components["self_past_rows"] == 2.0  # two rows have a previous window
    if lambda_ki > 0:
        assert not policy.dead_lora and all(p.requires_grad for n, p in policy.named_parameters() if ".lora_" in n)
    # The action loss reaches SigLIP through the un-detached K/V even without KI.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in policy.named_parameters()
               if n.startswith("siglip."))


def test_insulated_regime_blocks_the_action_gradient_into_the_vlm():
    policy = build(kv_grad_scale=0.0, lambda_ki=1.0).train()
    policy.lambda_ki = 0.0  # isolate the flow loss: no KI term this step
    loss = policy(make_batch(), history_mode="expert")
    loss.backward()
    vision = [p.grad for n, p in policy.named_parameters() if n.startswith("siglip.")]
    assert all(g is None or g.abs().sum() == 0 for g in vision)


def test_self_past_uses_the_deployment_sampler_and_only_valid_rows():
    policy = build(self_past_p=1.0).train()
    calls = []
    original = policy._generate_actions

    def spy(obs, past, valid, n_tokens, temperature, topk):
        calls.append(past.shape[0])
        return original(obs, past, valid, n_tokens, temperature, topk)

    policy._generate_actions = spy
    policy(make_batch(), history_mode="generated")
    assert sum(calls) == 2 and policy.last_loss_components["self_past_rows"] == 2.0
    policy.eval()
    policy(make_batch(), history_mode="expert")
    assert policy.last_loss_components["self_past_rows"] == 0.0  # no stale count in expert mode


def test_stateful_rollout_protocol():
    policy = build().eval()
    policy.reset()
    obs = make_batch(batch=2, episode_step=(0, 0))["obs"]
    out = policy.predict_action(obs)
    assert out["action"].shape == (2, 8, 7) and out["action_pred"].shape == (2, 16, 7)
    assert torch.isfinite(out["action_pred"]).all()
    policy.record_executed_actions(out["action"])
    assert policy._past_valid_buffer.all() and torch.equal(policy._past_buffer, out["action"][:, 1:])
    second = policy.predict_action(obs)
    assert second["action"].shape == (2, 8, 7)


def test_attention_probe_averages_every_action_row():
    policy = build().eval()
    batch = make_batch()
    t = torch.full((3,), 0.4)
    with torch.no_grad():
        prefix_out, layout, _ = policy.vlm_pass(policy.build_prefix(batch["obs"], train_aug=False), None)
        kv, valid = policy._expert_kv(prefix_out, layout, batch["past_action"], batch["past_action_valid"])
        cols = (layout.nonki_len, layout.nonki_len + policy.n_cond_tokens)

        def mass(x_t, rows):
            suffix = build_flow_suffix_layout(prefix_out.pos0, valid, x_t.shape[1])
            return policy.joint.expert_forward(policy.flow_heads.action_in_proj(x_t), kv, suffix.bias,
                                               suffix.positions, cond=flow_condition(policy.flow_heads, t),
                                               probe_cols=cols, probe_rows=rows).probe_mass

        x_t = torch.randn(3, 16, 32)
        every_row = mass(x_t, "all")
        assert every_row.shape == (policy.joint.expert_spec.depth,) and not torch.allclose(every_row, mass(x_t, "last"))
        torch.testing.assert_close(mass(x_t[:, -1:], "all"), mass(x_t[:, -1:], "last"), rtol=0, atol=0)
        _, probe = policy._velocity(kv, prefix_out.pos0, valid, x_t, t, probe_cols=cols)
        torch.testing.assert_close(probe, every_row, rtol=0, atol=0)  # hist_attention_mass reads every row
        with pytest.raises(ValueError, match="probe_rows"):
            mass(x_t, "first")


# ------------------------------------------------------------------------------- eval-only diagnostics
def test_history_interventions_change_only_what_the_head_reads():
    policy = build().eval()
    batch = make_batch()
    obs, past, valid = batch["obs"], batch["past_action"], batch["past_action_valid"]

    def predict(history, history_valid):
        torch.manual_seed(5)
        return policy.predict_action(obs, past_actions=history, past_action_valid=history_valid)["action_pred"]

    plain = predict(past, valid)
    expected = {"zero": predict(torch.zeros_like(past), torch.zeros_like(valid)),
                "swap": predict(past.roll(1, 0), valid.roll(1, 0))}
    for mode, reference in expected.items():
        policy.set_history_intervention(mode)
        torch.testing.assert_close(predict(past, valid), reference, rtol=0, atol=0)
    seen = []
    original = policy.history_tokens
    policy.history_tokens = lambda history, history_valid: (seen.append(history), original(history, history_valid))[1]
    policy.set_history_intervention("noise", noise_std=0.5)
    assert not torch.equal(predict(past, valid), plain)
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


def test_history_intervention_settings_are_validated():
    policy = build().eval()
    for args in (("bogus",), ("noise",), ("zero", 0.1), ("noise", -1.0), ("noise", float("nan"))):
        with pytest.raises(ValueError):
            policy.set_history_intervention(*args)
    with pytest.raises(ValueError, match="no-past"):
        build(use_past=False).set_history_intervention("zero")
    policy.set_history_intervention("swap")
    one = make_batch(batch=1, episode_step=(12,))
    with pytest.raises(ValueError, match="at least 2"):
        policy.predict_action(one["obs"], past_actions=one["past_action"], past_action_valid=one["past_action_valid"])


def test_command_log_maps_runner_chunks_to_episodes_and_summarizes_them():
    import scripts.evaluate_p2n_vla as evaluator

    class Runner:  # the runner's calls: one reset per chunk of parallel episodes, then executed prefixes
        def __init__(self):
            self.calls = []

        def reset(self):
            self.calls.append("reset")

        def record_executed_actions(self, actions, executed_lengths=None):
            self.calls.append(list(executed_lengths))

    policy = Runner()
    log = evaluator.CommandLog(policy)
    ramp = np.zeros((2, 8, 7), np.float32)
    ramp[..., :6] = np.arange(8, dtype=np.float32)[None, :, None] * 0.01  # within-chunk step 0.01 per arm dim
    ramp[..., 6] = -1.0
    jump = ramp + np.float32(0.5)
    jump[..., 6] = 1.0  # gripper closes: one toggle
    policy.reset()
    policy.record_executed_actions(ramp, executed_lengths=np.array([8, 8]))
    policy.record_executed_actions(jump, executed_lengths=np.array([8, 1]))  # env 1 finished after 1 step
    policy.reset()
    policy.record_executed_actions(ramp, executed_lengths=np.array([3, 3]))  # env 1 pads the last chunk
    assert policy.calls == ["reset", [8, 8], [8, 1], "reset", [3, 3]]  # the runner's own methods still run
    episodes = log.episodes(3, 2)
    assert [[len(chunk) for chunk in chunks] for chunks in episodes] == [[8, 8], [8, 1], [3]]
    stats = evaluator.command_statistics(episodes[0], normalize=lambda values: np.asarray(values), tail=4)
    within = np.linalg.norm(np.full(6, 0.01))
    boundary = np.linalg.norm(np.full(6, 0.5 - 0.07))
    assert stats["steps"] == 16 and stats["chunks"] == 2 and stats["gripper_toggles"] == 1
    assert stats["boundary_ratio"] == pytest.approx(boundary / within, rel=1e-4)
    assert stats["final_motion"] == pytest.approx(np.linalg.norm(jump[0, 4:, :6], axis=1).mean(), rel=1e-6)
    assert stats["near_padding_fraction"] == 0.0  # the gripper command is never near the zero padding
    records = [{"success": True}, {"success": False}, {"success": False}]
    stats = [evaluator.command_statistics(chunks) for chunks in episodes]
    summary = evaluator.summarize_commands(stats, records)
    assert summary["success"]["episodes"] == 1 and summary["failure"]["episodes"] == 2
    assert summary["failure"]["gripper_toggles"] == pytest.approx(0.5) and summary["near_padding_fraction"] is None


def test_offline_diagnostics_measure_toggles_reliance_and_innovation():
    from scripts.pi05_past_offline_diagnostics import Accumulator, first_toggles
    assert first_toggles(torch.tensor([[-1., -1, 1, 1], [1., 1, 1, 1], [1., -1, -1, 1]])).tolist() == [1, -1, 0]
    past = torch.zeros(2, 7, 7)
    past[..., -1] = -1.0
    valid = torch.ones(2, 7, dtype=torch.bool)
    target = torch.zeros(2, 16, 7)
    target[..., -1] = -1.0
    target[0, 3:, -1] = 1.0  # window 0 closes the gripper at executed step 3, window 1 never toggles
    plain = target.clone()
    plain[0, 3, -1] = -1.0  # predicted one step late: recalled, timing error 1
    plain[1, 6:, -1] = 1.0  # a false toggle
    history_swap, observation_swap = plain.clone(), plain.clone()
    history_swap[:, :8, 0] += 0.1
    observation_swap[:, :8, 0] += 0.2
    accumulator = Accumulator(16, 7)
    accumulator.add(lambda x: x, past, valid, target, plain, history_swap, observation_swap, tolerance=2)
    report = accumulator.summary()
    assert report["gripper_toggles"] == {"toggle_windows": 1, "recall": 1.0, "timing_error": 1.0,
                                         "false_toggle_rate": 1.0}
    assert report["reliance"]["ratio"] == pytest.approx(0.5)
    accumulator = Accumulator(16, 7)
    moving = target + torch.linspace(0, 1, 16)[None, :, None] * torch.tensor([1.0, 2.0])[:, None, None]
    accumulator.add(lambda x: x, past, valid, moving, moving, moving, moving, tolerance=2)
    assert accumulator.summary()["innovation_r2"] == pytest.approx([None] + [1.0] * 15)  # step 0 has no spread


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_trains_and_samples_through_the_fp32_condition_path():
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
        x_t, t = torch.randn(3, 16, 32, device="cuda"), torch.full((3,), 0.4, device="cuda")
        masked = valid.clone()
        masked[:, layout.nonki_len:] = False
        no_history, _ = policy._velocity(kv, prefix_out.pos0, masked, x_t, t)
        plain, _ = policy._velocity(tuple(prefix_out.kv), prefix_out.pos0, layout.nonki_valid, x_t, t)
        torch.testing.assert_close(no_history, plain, rtol=1e-3, atol=1e-3)


# ------------------------------------------------------------------------------------ artifacts / arms
def test_export_restore_round_trip(tmp_path):
    policy = build().eval()
    path = tmp_path / "snapshot.ckpt"
    torch.save({"format": "p2n_vla_checkpoint_v1", "policy_config": policy.export_config(),
                "metadata": policy.artifact_metadata(),
                "state_dicts": {"model": policy.artifact_state_dict()}}, path)
    restored = Pi05PastPolicy.from_checkpoint(str(path), weights="model")
    assert any(k.startswith("siglip.") for k in policy.artifact_state_dict())
    batch = make_batch()
    for model in (policy, restored):
        model.eval()
    torch.manual_seed(3)
    a = policy.predict_action(batch["obs"], past_actions=batch["past_action"],
                              past_action_valid=batch["past_action_valid"])["action_pred"]
    torch.manual_seed(3)
    b = restored.predict_action(batch["obs"], past_actions=batch["past_action"],
                                past_action_valid=batch["past_action_valid"])["action_pred"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert [n for n, _ in restored.trainable_named_parameters()] == [n for n, _ in policy.trainable_named_parameters()]
    assert policy.artifact_metadata()["pi05_past"]["condition_tokens"] == 9


def test_no_past_control_arm():
    policy = build(use_past=False).train()
    assert policy.n_cond_tokens == 0 and not hasattr(policy, "acc_proj") and policy.self_past_p == 0.0
    loss = policy(make_batch(), history_mode="generated")
    loss.backward()
    assert policy.last_loss_components["hist_attention_mass"] is None
    assert not [n for n, p in policy.named_parameters() if p.requires_grad and p.grad is None]


# -------------------------------------------------------------------------------- launcher + workspace
def _compose(*overrides):
    import scripts.train_p2n_vla as launcher
    return launcher.compose_config("pi05_past", "libero",
                                   ["policy.model_size=tiny", "policy.pi05_weights=null", *overrides])


def test_launcher_validates_the_pi05_past_config():
    cfg = _compose()
    assert cfg.policy._target_ == "oat.policy.pi05_past.Pi05PastPolicy" and cfg.policy.use_past is True
    assert cfg.training.max_train_steps == 400 and cfg.training.max_optimizer_steps == 80000
    assert cfg.training.num_epochs == 800 and cfg.policy.self_past_ramp_steps == 0 and cfg.policy.lambda_ki == 0.0
    # LIBERO evaluation every 10k updates (official protocol, 500 episodes); only the best snapshot is kept.
    assert cfg.task.policy.lazy_eval is False and cfg.training.rollout_every == 100
    runner = cfg.task.policy.env_runner
    assert runner.protocol == "official" and runner.n_test == 500 and runner.n_parallel_envs == 10
    assert cfg.training.keep_snapshots == "best" and cfg.training.snapshot_every == 0
    with pytest.raises(ValueError, match="keep_snapshots"):
        _compose("task.policy.lazy_eval=true")
    with pytest.raises(ValueError, match="keep_snapshots"):  # latest.ckpt holds the latest weights in best mode
        _compose("training.keep_resume_checkpoint=false")
    _compose("policy.use_past=false", "training.validate_generated_history=false")
    with pytest.raises(ValueError, match="no generated history"):
        _compose("policy.use_past=false")
    with pytest.raises(ValueError, match="without a loss"):
        _compose("policy.kv_grad_scale=0", "policy.lambda_ki=0")


def make_libero_like_zarr(path, n_episodes=3, episode_length=16, seed=0):
    """A tiny zarr with the LIBERO-10 schema (copied from tests/test_p2n_vla_workspace.py)."""
    from oat.common.replay_buffer import ReplayBuffer
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer.create_from_path(str(path), mode="w")
    for episode in range(n_episodes):
        quat = rng.standard_normal((episode_length, 4)).astype(np.float32)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        buffer.add_episode({
            "action": np.concatenate([rng.uniform(-0.9, 0.9, (episode_length, 3)),
                                      rng.uniform(-0.2, 0.3, (episode_length, 3)),
                                      np.where(rng.random((episode_length, 1)) > 0.5, 1.0, -1.0)], 1).astype(np.float32),
            "agentview_rgb": rng.integers(0, 256, (episode_length, 128, 128, 3), dtype=np.uint8),
            "robot0_eye_in_hand_rgb": rng.integers(0, 256, (episode_length, 128, 128, 3), dtype=np.uint8),
            "robot0_eef_pos": (rng.standard_normal((episode_length, 3)) * 0.1).astype(np.float32),
            "robot0_eef_quat": quat,
            "robot0_gripper_qpos": rng.uniform(-0.04, 0.04, (episode_length, 2)).astype(np.float32),
            "task_uid": np.full((episode_length, 1), 30 + episode % 10, dtype=np.int64),
        })
    return path


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


@pytest.fixture
def cpu_accelerate(monkeypatch):
    from accelerate.state import AcceleratorState
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    AcceleratorState._reset_state(True)
    yield
    AcceleratorState._reset_state(True)


def _records(path, event):
    return [r for r in map(json.loads, Path(path).read_text().splitlines()) if r.get("event") == event]


@pytest.mark.slow
def test_tiny_pi05_past_trains_validates_snapshots_and_reloads(tmp_path, cpu_accelerate, monkeypatch):
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    workspace = TrainP2NVLAWorkspace(tiny_run_config(zarr_path), output_dir=str(tmp_path / "run"))
    workspace.run()
    assert workspace.completed_optimizer_steps == 4
    steps = _records(tmp_path / "run/logs.jsonl", "train_step")
    assert steps and all(r.get("train/loss_flow") is not None for r in steps)
    epochs = _records(tmp_path / "run/logs.jsonl", "epoch")
    assert any(r.get("val_loss_generated_history") is not None for r in epochs)
    snapshots = sorted((tmp_path / "run/snapshots").glob("upd-*_model.ckpt"))
    assert snapshots, "use_ema=false must write _model snapshots"
    restored = Pi05PastPolicy.from_checkpoint(str(snapshots[-1]), weights="model")
    assert restored.VARIANT == "pi05_past" and restored.train_vision
    # The evaluator accepts the snapshot with a history intervention (dry run: no simulators) ...
    import scripts.evaluate_p2n_vla as evaluate
    from scripts.pi05_past_offline_diagnostics import main as offline_diagnostics
    monkeypatch.chdir(REPO)
    for key, default in (("MUJOCO_GL", "egl"), ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
                         ("HF_HOME", "/workspace/.hf_home")):
        monkeypatch.setenv(key, os.environ.get(key, default))
    output = tmp_path / "eval"
    assert evaluate.main(["--snapshot", str(snapshots[-1]), "--weights", "model", "--output-dir", str(output),
                          "--device", "cpu", "--n-test", "10", "--history-intervention", "noise",
                          "--history-noise-std", "0.2", "--threads", str(torch.get_num_threads()), "--dry-run"]) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "ready" and metadata["variant"] == "pi05_past"
    assert metadata["history_intervention"] == {"mode": "noise", "noise_std": 0.2}
    # ... and the offline diagnostics run on its held-out windows.
    report = offline_diagnostics(["--snapshot", str(snapshots[-1]), "--device", "cpu", "--batch-size", "4",
                                  "--max-windows", "8", "--output", str(tmp_path / "offline.json")])
    assert report["windows"] >= 2 and report["reliance"]["history_swap_change"] > 0
    assert len(report["innovation_r2"]) == 16


@pytest.mark.slow
def test_tiny_pi05_past_resume_is_bitwise_identical_to_an_uninterrupted_run(tmp_path, cpu_accelerate):
    import shutil
    from accelerate.state import AcceleratorState
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")

    def run(output, *overrides):
        AcceleratorState._reset_state(True)
        TrainP2NVLAWorkspace(tiny_run_config(zarr_path, *overrides), output_dir=str(output)).run()

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
def test_tiny_pi05_past_rollouts_keep_only_the_best_snapshot(tmp_path, cpu_accelerate):
    from test_p2n_vla_workspace import ScoredRolloutRunner
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    cfg = tiny_run_config(zarr_path, "task.policy.lazy_eval=false", "training.rollout_every=1",
                          "training.keep_snapshots=best", "training.snapshot_every=0")
    OmegaConf.update(cfg, "task.policy.env_runner", {  # the real policy, a scripted simulator
        "_target_": "test_p2n_vla_workspace.ScoredRolloutRunner", "task_name": "libero10", "protocol": "official",
        "n_test": 4, "n_test_vis": 0, "test_start_seed": 3000, "init_state_offset": 0, "n_parallel_envs": 2,
        "max_episode_steps": 16}, merge=False)
    ScoredRolloutRunner.rates[:] = [0.25, 0.75]
    TrainP2NVLAWorkspace(cfg, output_dir=str(tmp_path / "run")).run()
    rollouts = _records(tmp_path / "run/logs.jsonl", "rollout")
    assert [record["rollout/success_rate"] for record in rollouts] == [0.25, 0.75]
    best = json.loads((tmp_path / "run/snapshots/best.json").read_text())
    assert best["optimizer_step"] == rollouts[1]["optimizer_step"] == rollouts[1]["rollout/best_optimizer_step"]
    assert [path.name for path in (tmp_path / "run/snapshots").glob("upd-*.ckpt")] == [Path(best["snapshot"]).name]
    assert (tmp_path / "run/checkpoints/latest.ckpt").is_file()
    assert Pi05PastPolicy.from_checkpoint(best["snapshot"], weights="model").VARIANT == "pi05_past"


@pytest.mark.slow
def test_tiny_pi05_past_two_rank_gloo_ddp(tmp_path):
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    cfg = tiny_run_config(zarr_path)
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
