"""Integration checks for the additive-only history-gate policy experiment."""

import contextlib
import copy
from datetime import timedelta
import time

import dill
import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import distributed as dist, multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from oat.common.hydra_util import register_new_resolvers
from oat.model.common.normalizer import LinearNormalizer
from oat.model.diffusion.ema_model import EMAModel
from oat.policy.base_policy import BasePolicy
from oat.policy.past2next_state_history_gate import Past2NextStateHistoryGatePolicy
from oat.workspace.train_policy import TrainPolicyWorkspace
from test_state_history_policy import (
    ROOT, SHAPES, make_batch, make_obs, make_policy, policy_config,
)


def gate_config(history_steps=8, mode="learned", **overrides):
    cfg = policy_config(history_steps)
    cfg.update({
        "_target_": "oat.policy.past2next_state_history_gate.Past2NextStateHistoryGatePolicy",
        "history_gate_mode": mode,
        "history_gate_hidden_dim": 16,
        "history_gate_init": 0.9,
    })
    cfg.update(overrides)
    return cfg


def make_gate(history_steps=8, mode="learned", **overrides):
    policy = hydra.utils.instantiate(gate_config(history_steps, mode, **overrides))
    normalizer = LinearNormalizer()
    normalizer.fit({key: torch.tensor([-1., 1.])[:, None].expand(2, shape[0])
                    for key, shape in {**SHAPES, "action": [7]}.items()})
    policy.set_normalizer(normalizer)
    return policy


@pytest.mark.parametrize("mode", ["learned", "open", "closed"])
@pytest.mark.parametrize("history_steps", [8, 16])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_self_past_backward_optimizer_and_ema(mode, history_steps, mixed_precision):
    torch.manual_seed(43)
    policy = make_gate(history_steps, mode).train()
    optimizer = policy.get_optimizer(1e-3, 1e-3, 1e-4, (0.9, 0.95))
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in policy.parameters() if p.requires_grad}
    shadow = copy.deepcopy(policy)
    ema = EMAModel(shadow)
    batch = make_batch(history_steps)
    before = policy.history_gate[-1].weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        amp = torch.autocast("cpu", dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()
        with amp:
            loss = policy(batch)
        assert torch.isfinite(loss)
        loss.backward()
        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, name
                assert torch.isfinite(param.grad).all(), name
        optimizer.step()
        policy.on_optimizer_step()
        ema.step(policy)
    assert policy.self_past_step == 2
    assert policy.history_gate.training and policy.history_encoder.training and policy.model.training
    assert not policy.action_tokenizer.training
    assert all(p.grad is None for p in policy.action_tokenizer.parameters())
    if mode == "learned":
        assert not torch.equal(before, policy.history_gate[-1].weight)
        assert torch.count_nonzero(policy.history_gate[1].weight.grad) > 0
    else:
        assert all(not p.requires_grad for p in policy.history_gate.parameters())
    assert torch.isfinite(shadow.history_gate[-1].weight).all()


def test_initial_gate_features_padding_and_live_gradient():
    policy = make_gate().eval()
    assert policy.get_history_gate_metrics() == {}
    obs = make_obs(valid_steps=3)
    past = torch.randn(2, 7, 7)
    _, log_gate = policy._condition_and_gate(obs, past)
    torch.testing.assert_close(log_gate.exp(), torch.full((2, 1), .9))
    # Once the zero-initialized output layer has learned, all gate inputs can
    # affect the decision. Verify real input features and padding sanitization.
    torch.nn.init.normal_(policy.history_gate[-1].weight, std=.2)
    inputs = []
    handle = policy.history_gate.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone()))
    cond, expected = policy._condition_and_gate(obs, past)
    handle.remove()
    torch.testing.assert_close(inputs[0][:, :16], cond[:, :2].mean(1))
    torch.testing.assert_close(inputs[0][:, 16:32], cond[:, -4:].mean(1))
    torch.testing.assert_close(inputs[0][:, -1], torch.full((2,), 3 / 8))
    poisoned = copy.deepcopy(obs)
    for key in SHAPES:
        poisoned["state_history__" + key][:, :-3] = float("nan")
    # Original raw past conditions are not padding-aware; test the state's
    # invalid prefix only, as required by the inherited original interface.
    actual_cond, actual = policy._condition_and_gate(poisoned, past)
    torch.testing.assert_close(cond, actual_cond)
    torch.testing.assert_close(expected, actual)
    assert not policy._last_history_gate.requires_grad
    expected.sum().backward()
    assert policy.history_gate[1].weight.grad.abs().sum() > 0


def test_open_gate_migrates_c_and_matches_forward_and_generation():
    base = make_policy().eval()
    gated = make_gate(mode="open").eval()
    with pytest.warns(UserWarning, match="state-history C checkpoint"):
        gated.load_state_dict(base.state_dict())
    for name, value in base.state_dict().items():
        torch.testing.assert_close(gated.state_dict()[name], value)
    batch = make_batch()
    with torch.no_grad():
        torch.testing.assert_close(gated(batch, history_mode="expert"),
                                   base(batch, history_mode="expert"), rtol=0, atol=0)
        cond, log_gate = gated._condition_and_gate(batch["obs"], batch["past_action"])
        assert log_gate is None
        prefix = torch.full((2, 1), base.bos_id, dtype=torch.long)
        actual = gated.model.generate(prefix, cond, 2, temperature=0, bos_id=base.bos_id)
        expected = base.model.generate(prefix, cond, 2, temperature=0, bos_id=base.bos_id)
    torch.testing.assert_close(actual, expected)


def test_closed_gate_removes_summary_influence_only():
    policy = make_gate(mode="closed").eval()
    obs = make_obs()
    past = torch.randn(2, 7, 7)
    altered = copy.deepcopy(obs)
    altered["state_history__robot0_eef_pos"][:, :-1] += 20
    tokens = torch.tensor([[policy.bos_id, 0], [policy.bos_id, 1]])
    with torch.no_grad():
        cond, gate = policy._condition_and_gate(obs, past)
        new_cond, new_gate = policy._condition_and_gate(altered, past)
        assert not torch.equal(cond[:, -4:], new_cond[:, -4:])
        torch.testing.assert_close(policy.model(tokens, cond, gate),
                                   policy.model(tokens, new_cond, new_gate))
        changed_actions, _ = policy._condition_and_gate(obs, past + .5)
        assert not torch.allclose(policy.model(tokens, cond, gate),
                                  policy.model(tokens, changed_actions, gate))
    assert policy.get_history_gate_metrics()["history_gate/mean"] == 0


def test_self_past_uses_previous_gate_and_restores_training_mode():
    policy = make_gate().train()
    batch = make_batch()
    seen = []
    handle = policy.history_gate.register_forward_pre_hook(
        lambda module, args: seen.append((module.training, torch.is_inference_mode_enabled())))
    loss = policy(batch, history_mode="generated")
    handle.remove()
    assert seen == [(False, True), (True, False)]
    loss.backward()
    assert policy.history_gate.training
    assert policy.history_gate[-1].weight.grad.abs().sum() > 0
    assert policy._pending_execution_steps is None and policy._past_buffer is None


def test_prediction_and_partial_execution_feedback_are_preserved():
    policy = make_gate().eval()
    obs = make_obs()
    with torch.inference_mode():
        result = policy.predict_action(obs, use_k_tokens=1)
    assert result["action"].shape == (2, 8, 7)
    assert result["action_pred"].shape == (2, 16, 7)
    with pytest.raises(RuntimeError, match="pending"):
        policy.predict_action(obs)
    commands = torch.full((2, 8, 7), float("nan"))
    commands[0, :3] = .25
    policy.record_executed_actions(commands, [3, 0])
    torch.testing.assert_close(policy._past_buffer[0, -3:], commands[0, :3])
    assert policy._past_buffer[1].count_nonzero() == 0
    policy.reset()
    assert policy._past_buffer is None and policy._pending_execution_steps is None


def test_gate_checkpoint_roundtrip_and_missing_parameters_fail(tmp_path):
    policy = make_gate().eval()
    torch.nn.init.normal_(policy.history_gate[-1].weight, std=.1)
    policy.set_self_past_step(75)
    cfg = OmegaConf.create({"policy": gate_config(), "training": {"use_ema": True}})
    path = tmp_path / "gate.ckpt"
    torch.save({"cfg": cfg, "state_dicts": {"ema_model": policy.state_dict()}},
               path, pickle_module=dill)
    loaded = BasePolicy.from_checkpoint(str(path))
    assert isinstance(loaded, Past2NextStateHistoryGatePolicy)
    assert loaded.self_past_step == 75
    batch = make_batch()
    with torch.no_grad():
        torch.testing.assert_close(policy(batch, history_mode="expert"),
                                   loaded(batch, history_mode="expert"))
    state = policy.state_dict()
    del state["history_gate.3.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        loaded.load_state_dict(state)
    state = {k: v for k, v in policy.state_dict().items() if not k.startswith("history_gate.")}
    with pytest.raises(RuntimeError, match="Missing key"):
        loaded.load_state_dict(state)


def test_workspace_initialization_accepts_c_ema_weights(tmp_path):
    base = make_policy().eval()
    base.set_self_past_step(37)
    path = tmp_path / "c.ckpt"
    torch.save({"state_dicts": {"ema_model": base.state_dict()}}, path, pickle_module=dill)
    workspace = object.__new__(TrainPolicyWorkspace)
    workspace.model = make_gate()
    workspace.ema_model = copy.deepcopy(workspace.model)
    with pytest.warns(UserWarning, match="state-history C checkpoint"):
        workspace._initialize_policy_weights(str(path))
    assert workspace.model.self_past_step == workspace.ema_model.self_past_step == 0
    for model in (workspace.model, workspace.ema_model):
        torch.testing.assert_close(model.history_encoder.summary_queries,
                                   base.history_encoder.summary_queries)


@pytest.mark.parametrize("history_steps", [8, 16])
def test_inherited_config_targets_new_policy_only(history_steps):
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name="experimental/train_past2next_state_history_gate", overrides=[
            "policy.action_tokenizer.checkpoint=/tmp/example.ckpt",
            f"state_history_steps={history_steps}",
        ])
    assert cfg.policy._target_.endswith("Past2NextStateHistoryGatePolicy")
    assert cfg.policy.history_gate_mode == "learned" and cfg.policy.history_gate_init == .9
    assert cfg.policy.past_n == cfg.task.policy.dataset.past_n == history_steps - 1
    assert cfg.policy.state_history_steps == cfg.task.policy.env_runner.state_history_steps == history_steps
    assert cfg.task.policy.dataset.val_ratio == .1
    assert cfg.task.policy.env_runner.protocol == "corrected"
    assert cfg.training.use_ema and cfg.training.validate_generated_history


def _gate_ddp_rank(rank, rendezvous, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(100 + rank)
        policy = make_gate(mode=mode, self_past_ramp_steps=0).train()
        ddp = DistributedDataParallel(policy, find_unused_parameters=False)
        optimizer = policy.get_optimizer(1e-3, 1e-3, 0, (.9, .95))
        batch = make_batch()
        for history_mode in ("expert", "configured"):
            optimizer.zero_grad(set_to_none=True)
            ddp(batch, history_mode=history_mode).backward()
            for name, p in policy.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all(), name
            optimizer.step()
            policy.on_optimizer_step()
        for parameter in (policy.history_gate[-1].weight,
                          policy.history_encoder.summary_queries):
            gathered = [torch.empty_like(parameter) for _ in range(2)]
            dist.all_gather(gathered, parameter.detach())
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo unavailable")
@pytest.mark.parametrize("mode", ["learned", "closed"])
def test_two_rank_ddp_has_no_unused_gate_or_history_parameters(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    context = mp.spawn(_gate_ddp_rank, args=((tmp_path / "rendezvous").as_uri(), mode),
                       nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("History gate DDP check timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_mixed_precision_sdpa_gate_gradient():
    policy = make_gate(self_past_ramp_steps=0).cuda().train()
    batch = make_batch()
    def to_cuda(value):
        return {k: to_cuda(v) for k, v in value.items()} if isinstance(value, dict) else value.cuda()
    batch = to_cuda(batch)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=dtype):
        loss = policy(batch, history_mode="generated")
    loss.backward()
    assert torch.isfinite(loss)
    grad = policy.history_gate[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
