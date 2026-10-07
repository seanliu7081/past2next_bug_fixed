"""P2N-VLA policy tests (tiny configs on CPU; full-size checks are marked)."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.vla.lora import lora_parameters
from oat.policy.p2n_vla import P2NVLAPolicy
from oat.policy.p2n_vla_state_gate import P2NVLAStateGatePolicy
from oat.policy.pi05_ki_flow import PI05KIFlowPolicy

TOKENIZER = Path("/workspace/hf_upload/tokenizer_oattok_so3aug_ep4960_mse0.001.ckpt")
pytestmark = pytest.mark.skipif(not TOKENIZER.is_file(), reason=f"OAT tokenizer missing at {TOKENIZER}")

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
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def _actions(*shape, generator):
    xyz = (torch.rand(*shape, 3, generator=generator) * 1.8 - 0.9)
    rot = (torch.rand(*shape, 3, generator=generator) * 0.5 - 0.2)
    grip = torch.where(torch.rand(*shape, 1, generator=generator) > 0.5, 1.0, -1.0)
    return torch.cat((xyz, rot, grip), dim=-1)


def _quat(*shape, generator):
    q = torch.randn(*shape, 4, generator=generator)
    return q / q.norm(dim=-1, keepdim=True)


def _normalizer():
    generator = torch.Generator().manual_seed(7)
    normalizer = LinearNormalizer()
    normalizer.fit({
        "action": _actions(512, generator=generator),
        "robot0_eef_pos": torch.randn(512, 3, generator=generator) * 0.1,
        "robot0_gripper_qpos": torch.rand(512, 2, generator=generator) * 0.08 - 0.04,
    }, last_n_dims=1, mode="limits")
    q01, q99 = -torch.ones(8) * 0.5, torch.ones(8) * 0.5
    scale = 2.0 / (q99 - q01 + 1e-6)
    normalizer["prompt_state"] = SingleFieldLinearNormalizer.create_manual(
        scale, -1.0 - q01 * scale, {"q01": q01, "q99": q99})
    return normalizer


def _obs(batch, generator, frames=1):
    return {
        "agentview_rgb": torch.randint(0, 256, (batch, frames, 128, 128, 3), generator=generator, dtype=torch.uint8),
        "robot0_eye_in_hand_rgb": torch.randint(0, 256, (batch, frames, 128, 128, 3), generator=generator,
                                                dtype=torch.uint8),
        "robot0_eef_pos": torch.randn(batch, frames, 3, generator=generator) * 0.1,
        "robot0_eef_quat": _quat(batch, frames, generator=generator),
        "robot0_gripper_qpos": torch.rand(batch, frames, 2, generator=generator) * 0.08 - 0.04,
        "task_uid": torch.randint(30, 40, (batch, frames, 1), generator=generator),
    }


def _history(obs, steps_valid, generator):
    batch = steps_valid.shape[0]
    valid = torch.arange(8)[None] >= (8 - steps_valid[:, None])
    obs = dict(obs)
    obs["state_history__robot0_eef_pos"] = torch.randn(batch, 8, 3, generator=generator) * 0.1
    obs["state_history__robot0_eef_quat"] = _quat(batch, 8, generator=generator)
    obs["state_history__robot0_gripper_qpos"] = torch.rand(batch, 8, 2, generator=generator) * 0.08 - 0.04
    for key in STATE_KEYS:
        obs[f"state_history__{key}"] = torch.where(valid[..., None], obs[f"state_history__{key}"],
                                                   torch.zeros_like(obs[f"state_history__{key}"]))
    obs["state_history_valid"] = valid
    return obs, valid


def make_batch(batch=2, episode_step=12, gate=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    steps = torch.full((batch,), episode_step) if isinstance(episode_step, int) else torch.tensor(episode_step)
    past_count = steps.clamp(max=7)
    prev_count = (steps - 8).clamp(min=0, max=7)
    past_valid = torch.arange(7)[None] >= (7 - past_count[:, None])
    prev_valid = torch.arange(7)[None] >= (7 - prev_count[:, None])
    out = {
        "obs": _obs(batch, generator), "prev_obs": _obs(batch, generator),
        "action": _actions(batch, 16, generator=generator),
        "past_action": _actions(batch, 7, generator=generator) * past_valid[..., None],
        "past_action_valid": past_valid,
        "prev_past_action": _actions(batch, 7, generator=generator) * prev_valid[..., None],
        "prev_past_action_valid": prev_valid,
        "prev_window_valid": steps >= 8, "episode_step": steps,
    }
    if gate:
        out["obs"], _ = _history(out["obs"], past_count + 1, generator)
        out["prev_obs"], _ = _history(out["prev_obs"], prev_count + 1, generator)
    return out


def build(cls=P2NVLAPolicy, **kwargs):
    torch.manual_seed(kwargs.pop("seed", 0))
    policy = cls(shape_meta=SHAPE_META, model_size="tiny", tokenizer_checkpoint=str(TOKENIZER),
                 activation_checkpointing=False, self_past_warmup_steps=0, self_past_ramp_steps=0, **kwargs)
    policy.set_normalizer(_normalizer())
    return policy


GATE_MODES = ("learned", "open", "closed")


# ---------------------------------------------------------------- construction
def test_construction_and_ports():
    policy = build()
    assert policy.get_observation_ports() == ["agentview_rgb", "robot0_eye_in_hand_rgb", *STATE_KEYS, "task_uid"]
    assert policy.dtype == torch.float32 and policy.max_seq_len == 8
    frozen = set(policy.frozen_base_keys())
    parameters = dict(policy.named_parameters())
    assert frozen and all(not parameters[k].requires_grad for k in frozen)
    assert any(k.endswith(".base.weight") for k in frozen)
    groups = policy.clip_groups()
    trainable = {id(p) for _, p in policy.trainable_named_parameters()}
    assert {id(p) for p in groups["ar"]} | {id(p) for p in groups["ki"]} == trainable
    assert not ({id(p) for p in groups["ar"]} & {id(p) for p in groups["ki"]})
    optimizer = policy.get_optimizer()
    assert {g["name"] for g in optimizer.param_groups} == {"pretrained", "new"}
    gate = build(P2NVLAStateGatePolicy)
    assert gate.get_observation_ports()[-1] == "state_history_valid"
    flow = build(PI05KIFlowPolicy)
    assert not hasattr(flow, "tok_emb") and not hasattr(flow, "raw_proj")


def test_requires_normalizer():
    policy = P2NVLAPolicy(shape_meta=SHAPE_META, model_size="tiny", tokenizer_checkpoint=str(TOKENIZER))
    with pytest.raises(RuntimeError, match="set_normalizer"):
        policy(make_batch())


# ---------------------------------------------------------------- loss and gradient coverage (DDP contract)
@pytest.mark.parametrize("cls,mode", [(P2NVLAPolicy, None), (PI05KIFlowPolicy, None)]
                         + [(P2NVLAStateGatePolicy, m) for m in GATE_MODES])
@pytest.mark.parametrize("episode_step", [0, 12])
@pytest.mark.parametrize("lambda_ki", [0.0, 1.0])
def test_gradient_coverage(cls, mode, episode_step, lambda_ki):
    kwargs = {"lambda_ki": lambda_ki}
    if mode is not None:
        kwargs["history_gate_mode"] = mode
    policy = build(cls, **kwargs).train()
    batch = make_batch(episode_step=episode_step, gate=cls is P2NVLAStateGatePolicy)
    loss = policy(batch)
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, f"frozen {name} received a gradient"
    components = policy.last_loss_components
    assert math.isfinite(components["loss"])
    if lambda_ki == 0:
        assert components["loss_ki"] is None and not any(p.requires_grad for p in lora_parameters(policy.joint))


def test_stop_gradient_both_directions():
    policy = build(P2NVLAStateGatePolicy).train()
    batch = make_batch(gate=True)
    targets = policy.encode_targets(batch["action"])
    prefix = policy.build_prefix(batch["obs"], train_aug=False)
    prefix_out, layout, ki_logits = policy.vlm_pass(prefix, targets)
    cond = policy.build_conditions(batch["obs"], batch["past_action"], batch["past_action_valid"])
    log_gate, closed = policy.compute_log_gate(batch["obs"], prefix, prefix_out, layout, cond)
    logits, _ = policy.teacher_forcing_logits(prefix_out, layout, cond, targets, log_gate, closed)
    loss_ar = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss_ki = torch.nn.functional.cross_entropy(ki_logits.reshape(-1, 5000), targets.reshape(-1))
    vlm_side = lora_parameters(policy.joint) + list(policy.ki_table.parameters())
    ar_side = [p for n, p in policy.trainable_named_parameters() if not policy._is_ki_clip(n)]
    grads = torch.autograd.grad(loss_ar, vlm_side, allow_unused=True)
    assert all(g is None or g.abs().max() == 0 for g in grads), "L_AR must not reach LoRA / KI rows"
    grads = torch.autograd.grad(loss_ki, ar_side, allow_unused=True)
    assert all(g is None or g.abs().max() == 0 for g in grads), "L_KI must not reach the expert side"


# ---------------------------------------------------------------- positions, KI causality, decoding consistency
def test_ki_prefix_does_not_change_expert_logits():
    policy = build(P2NVLAStateGatePolicy).eval()
    batch = make_batch(gate=True)
    targets = policy.encode_targets(batch["action"])
    prefix = policy.build_prefix(batch["obs"], train_aug=False)
    cond = policy.build_conditions(batch["obs"], batch["past_action"], batch["past_action_valid"])
    results = []
    for ki in (targets, None):
        prefix_out, layout, _ = policy.vlm_pass(prefix, ki)
        log_gate, closed = policy.compute_log_gate(batch["obs"], prefix, prefix_out, layout, cond)
        results.append(policy.teacher_forcing_logits(prefix_out, layout, cond, targets, log_gate, closed)[0])
    assert torch.allclose(results[0], results[1], atol=1e-5)


def test_ki_causality():
    policy = build().eval()
    batch = make_batch()
    targets = policy.encode_targets(batch["action"])
    prefix = policy.build_prefix(batch["obs"], train_aug=False)
    _, _, logits = policy.vlm_pass(prefix, targets)
    perturbed = targets.clone()
    perturbed[:, 4:] = (perturbed[:, 4:] + 17) % 5000
    _, _, logits_perturbed = policy.vlm_pass(prefix, perturbed)
    assert torch.allclose(logits[:, :5], logits_perturbed[:, :5], atol=1e-5)  # rows 0..4 read z<4 only
    assert not torch.allclose(logits[:, 5:], logits_perturbed[:, 5:], atol=1e-5)


def test_greedy_decode_is_consistent_with_teacher_forcing():
    policy = build(P2NVLAStateGatePolicy).eval()
    batch = make_batch(gate=True)
    with torch.no_grad():
        prefix = policy.build_prefix(batch["obs"], train_aug=False)
        prefix_out, layout, _ = policy.vlm_pass(prefix, None)
        cond = policy.build_conditions(batch["obs"], batch["past_action"], batch["past_action_valid"])
        log_gate, closed = policy.compute_log_gate(batch["obs"], prefix, prefix_out, layout, cond)
        tokens = policy.generate_tokens(prefix_out, layout, cond, log_gate, closed, 8, 0.0, None)
        logits, _ = policy.teacher_forcing_logits(prefix_out, layout, cond, tokens, log_gate, closed)
    assert (tokens < 5000).all() and (tokens >= 0).all()
    masked = logits.clone()
    masked[..., 5000] = float("-inf")
    assert torch.equal(masked.argmax(-1), tokens)


# ---------------------------------------------------------------- gate semantics at policy level
def test_closed_gate_ignores_history_content_and_learned_gate_init():
    closed = build(P2NVLAStateGatePolicy, history_gate_mode="closed").eval()
    batch = make_batch(gate=True, seed=1)
    other = make_batch(gate=True, seed=1)
    for key in STATE_KEYS:
        other["obs"][f"state_history__{key}"] = other["obs"][f"state_history__{key}"] * 3 + 0.1
    other["obs"]["state_history__robot0_eef_quat"] = torch.nn.functional.normalize(
        other["obs"]["state_history__robot0_eef_quat"], dim=-1)
    with torch.no_grad():
        a = closed(batch, history_mode="expert")
        b = closed(other, history_mode="expert")
    assert torch.equal(a, b)
    learned = build(P2NVLAStateGatePolicy).eval()
    with torch.no_grad():
        learned(batch, history_mode="expert")
    metrics = learned.get_history_gate_metrics()
    assert abs(metrics["history_gate/mean"] - 0.9) < 1e-5


# ---------------------------------------------------------------- self-past
def test_self_past_selects_valid_rows_and_restores_modes():
    policy = build(P2NVLAStateGatePolicy).train()
    batch = make_batch(batch=3, episode_step=[3, 12, 20], gate=True)
    past = policy._maybe_self_past(batch, batch["past_action"], 1.0)
    assert policy.training and not policy.siglip.training and not policy.action_tokenizer.training
    assert torch.equal(past[0], batch["past_action"][0])  # no previous window at step 3
    assert not torch.equal(past[1:], batch["past_action"][1:])
    assert policy._last_self_past_rows == 2
    loss = policy(batch, history_mode="generated")
    loss.backward()
    before = policy.self_past_step
    policy.on_optimizer_step()
    assert policy.self_past_step == before + 1
    policy.eval()
    policy.on_optimizer_step()
    assert policy.self_past_step == before + 1


# ---------------------------------------------------------------- rollout API
@pytest.mark.parametrize("cls", [P2NVLAPolicy, P2NVLAStateGatePolicy, PI05KIFlowPolicy])
def test_predict_action_through_runner_adapter(cls):
    from oat.env_runner.p2n_new_runner import ByteObservationPolicy

    policy = build(cls).eval()
    batch = make_batch(gate=cls is P2NVLAStateGatePolicy)
    obs = {k: (v.float() if v.dtype != torch.bool else v.float()) for k, v in batch["obs"].items()}
    if cls is P2NVLAStateGatePolicy:  # the runner sends a growing history; start at one valid state
        obs["state_history_valid"] = torch.zeros(2, 8)
        obs["state_history_valid"][:, -1] = 1
    wrapped = ByteObservationPolicy(policy)
    policy.reset()
    result = wrapped.predict_action(obs)
    assert result["action"].shape == (2, 8, 7) and result["action_pred"].shape == (2, 16, 7)
    assert torch.isfinite(result["action_pred"]).all()
    with pytest.raises(RuntimeError):
        wrapped.predict_action(obs)
    wrapped.record_executed_actions(result["action"], executed_lengths=torch.tensor([8, 3]))
    if cls is not P2NVLAStateGatePolicy:
        wrapped.predict_action(obs)


# ---------------------------------------------------------------- artifacts
def test_artifact_round_trip(tmp_path):
    policy = build(P2NVLAStateGatePolicy).eval()
    policy.set_self_past_step(123)
    batch = make_batch(gate=True)
    with torch.no_grad():
        expected = policy(batch, history_mode="expert")
    payload = {"policy_config": policy.export_config(), "metadata": policy.artifact_metadata(),
               "state_dicts": {"model": policy.artifact_state_dict()}}
    path = tmp_path / "artifact.ckpt"
    torch.save(payload, path)
    restored = P2NVLAStateGatePolicy.from_checkpoint(path, weights="model")
    assert restored.self_past_step == 123 and bool(restored._normalizer_fitted)
    assert torch.equal(restored.action_normalizer["action"].params_dict["scale"],
                       policy.action_normalizer["action"].params_dict["scale"])
    with torch.no_grad():
        actual = restored(batch, history_mode="expert")
    assert torch.allclose(actual, expected, atol=1e-6)
    with pytest.raises(KeyError):
        restored.load_artifact_state({k: v for k, v in payload["state_dicts"]["model"].items()
                                      if not k.startswith("action_normalizer.")})


# ---------------------------------------------------------------- full size (real weights)
PI05 = Path("/workspace/.hf_home/hub/models--lerobot--pi05_base/snapshots/"
            "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors")


@pytest.mark.requires_pi05
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not PI05.is_file() or not torch.cuda.is_available(), reason="needs pi05 weights and CUDA")
def test_full_size_step0_loss_and_memory(capsys):
    torch.manual_seed(0)
    policy = P2NVLAStateGatePolicy(shape_meta=SHAPE_META, model_size="full", tokenizer_checkpoint=str(TOKENIZER),
                                   self_past_warmup_steps=0, self_past_ramp_steps=0)
    policy.set_normalizer(_normalizer())
    device = torch.device("cuda:0")
    policy.to(device).train()
    torch.cuda.reset_peak_memory_stats(device)
    batch = make_batch(batch=2, gate=True)
    batch = {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device))
             for k, v in batch.items()}
    loss = policy(batch, history_mode="expert")
    loss.backward()
    components = policy.last_loss_components
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    with capsys.disabled():
        print(f"\n[full-size] step-0 L_AR {components['loss_ar']:.3f} L_KI {components['loss_ki']:.3f} "
              f"peak {peak:.2f} GiB")
    assert math.log(5001) - 0.5 <= components["loss_ar"] <= math.log(5001) + 1.0
    policy.cpu()
    torch.cuda.empty_cache()
