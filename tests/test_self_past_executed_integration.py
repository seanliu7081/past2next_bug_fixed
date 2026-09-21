"""Integrate original offline self-past training with acknowledged rollout history."""

from pathlib import Path

import dill
from hydra import compose, initialize_config_dir
import pytest
import torch
import torch.nn.functional as F

from oat.env_runner.executed_action_runner import LiberoExecutedPastRunner, RoboCasaExecutedPastRunner
from oat.model.common.normalizer import LinearNormalizer
from oat.model.diffusion.ema_model import EMAModel
from oat.policy.base_policy import BasePolicy
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy
from oat.policy.past2next_self_past_executed import Past2NextSelfPastExecutedPolicy
from oat.workspace.train_policy import TrainPolicyWorkspace
from test_executed_past_integration import SevenDimensionalVectorEnv
from test_executed_past_policy import observe_condition
from test_self_past_executed_policy import make_policy, make_self_past_batch


def tiny_config(config_name):
    config_dir = Path(__file__).resolve().parents[1] / "oat/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    cfg.shape_meta = {
        "action": {"shape": [7]},
        "obs": {"state": {"shape": [7], "type": "state"}},
    }
    cfg.policy.obs_encoder = {"_target_": "test_history_training.TinyObservationEncoder"}
    cfg.policy.action_tokenizer = {"_target_": "test_history_training.TinyTokenizer"}
    cfg.policy.embed_dim = 8
    cfg.policy.n_layers = 1
    cfg.policy.n_heads = 2
    cfg.policy.dropout = 0
    cfg.policy.self_past_warmup_steps = 2
    cfg.policy.self_past_ramp_steps = 4
    return cfg


@pytest.mark.parametrize("runner_type", [LiberoExecutedPastRunner, RoboCasaExecutedPastRunner])
def test_self_past_policy_uses_execution_feedback_in_original_runner_loops(monkeypatch, runner_type):
    runner = runner_type.__new__(runner_type)
    env = SevenDimensionalVectorEnv(terminal_steps=(2, 10))
    runner.env = env
    runner.env_fns = [None, None]
    runner.env_init_fn_dills = [b"first", b"second"]
    runner.env_seeds = [1000, 1001]
    runner.env_task_names = ["test_task", "test_task"]
    runner.task_name = "test_task"
    runner.max_episode_steps = 16
    runner.n_action_steps = 8
    runner.tqdm_interval_sec = 1000
    runner.protocol = "corrected"
    runner.episode_schedule = [{"episode_index": 0}, {"episode_index": 1}]
    runner.episode_records_path = None
    policy = make_policy()
    policy.set_self_past_step(4)
    histories, _ = observe_condition(monkeypatch, policy)

    log = runner.run(policy, temperature=0)
    assert log["mean_success_rate"] == 1.0
    assert runner.env is env
    assert len(histories) == 2
    torch.testing.assert_close(histories[0], torch.zeros(2, 7, 7))
    expected_second = torch.tensor([[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 4, 5, 6, 7]]).float()
    torch.testing.assert_close(histories[1], expected_second[:, :, None].expand(-1, -1, 7))
    expected_final = torch.tensor([[0, 0, 0, 0, 0, 0, 1], [3, 4, 5, 6, 7, 0, 1]]).float()
    torch.testing.assert_close(policy._past_buffer, expected_final[:, :, None].expand(-1, -1, 7))
    assert policy._pending_execution_steps is None
    assert policy.self_past_step == 4
    assert policy.self_past_probability() == 0.25

    # The existing run loop resets the policy before the next evaluation batch.
    runner.run(policy, temperature=0)
    torch.testing.assert_close(histories[2], torch.zeros(2, 7, 7))
    assert policy.self_past_step == 4
    assert runner.env is env


@pytest.mark.parametrize("weights", ["model", "ema"])
def test_self_past_recipe_backward_ema_and_checkpoint_preserve_both_contracts(tmp_path, monkeypatch, weights):
    cfg = tiny_config("experimental/train_past2next_self_past_executed")
    cfg.policy.self_past_p = 1.0
    workspace = TrainPolicyWorkspace(cfg, output_dir=str(tmp_path), lazy_instantiation=False)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.tensor([-1.0, 1.0])[:, None].expand(2, 7)})
    for policy in (workspace.model, workspace.ema_model):
        policy.set_normalizer(normalizer)
    model = workspace.model.train()
    model.set_self_past_step(6)
    workspace.completed_optimizer_steps = 6
    ema = EMAModel(workspace.ema_model)
    histories, _ = observe_condition(monkeypatch, model)
    batch = make_self_past_batch()
    targets = model.action_tokenizer.tokenize(batch["action"])
    training_calls = []

    def capture_teacher_forcing(module, args, kwargs, output):
        # Inner generation is inference-only; capture the actual supervised pass.
        if torch.is_grad_enabled():
            training_calls.append((args[0].detach().clone(), output.detach().clone()))

    hook = model.model.register_forward_hook(capture_teacher_forcing, with_kwargs=True)

    def forbid_online_execution(*args, **kwargs):
        raise AssertionError("Offline self-past training must not need online execution")

    before = next(model.raw_proj.parameters()).detach().clone()
    with monkeypatch.context() as offline:
        offline.setattr(model, "predict_action", forbid_online_execution)
        offline.setattr(model, "record_executed_actions", forbid_online_execution)
        loss = model(batch)
        assert torch.isfinite(loss)
        loss.backward()
    hook.remove()

    # Real AR generation decodes 0..15; its executed-length slice supplies 1..7.
    assert len(histories) == 2
    torch.testing.assert_close(histories[0], batch["prev_past_action"])
    generated = torch.arange(1, 8).float()[None, :, None].expand(2, 7, 7)
    torch.testing.assert_close(histories[1], generated)
    assert len(training_calls) == 1
    inputs, logits = training_calls[0]
    torch.testing.assert_close(inputs[:, 0], torch.full((2,), model.bos_id))
    torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])
    expected_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    torch.testing.assert_close(loss.detach(), expected_loss)
    assert any(parameter.grad is not None for parameter in model.raw_proj.parameters())
    assert all(parameter.grad is None and not parameter.requires_grad
               for parameter in model.action_tokenizer.parameters())
    assert model.self_past_step == 6
    assert model._past_buffer is None
    assert model._pending_execution_steps is None

    # Match TrainPolicyWorkspace's successful-update and EMA progress hooks.
    workspace.optimizer.step()
    workspace.optimizer.zero_grad(set_to_none=True)
    workspace.completed_optimizer_steps += 1
    model.on_optimizer_step()
    ema.step(model)
    workspace.ema_model.set_self_past_step(model.self_past_step)
    assert not torch.equal(next(model.raw_proj.parameters()).detach(), before)
    assert model.self_past_step == workspace.ema_model.self_past_step == 7

    # Save both weights with acknowledged history AND an outstanding prediction.
    for policy in (model, workspace.ema_model):
        policy.eval()
        result = policy.predict_action(batch["obs"])
        policy.record_executed_actions(result["action"], [3, 8])
        policy.predict_action(batch["obs"])
        assert torch.count_nonzero(policy._past_buffer) > 0
        assert policy._pending_execution_steps == 8
    checkpoint = workspace.save_checkpoint(tag="self_past_executed", use_thread=False)
    restored, saved_cfg = BasePolicy.from_checkpoint(
        checkpoint, weights=weights, return_configuration=True,
    )
    assert isinstance(restored, Past2NextSelfPastExecutedPolicy)
    assert saved_cfg.task.policy.dataset._target_.endswith("ZarrDatasetWithPrevWindow")
    assert saved_cfg.task.policy.env_runner._target_.endswith("LiberoExecutedPastRunner")
    assert restored.self_past_step == 7
    assert restored.self_past_probability() == 1.0
    assert restored._past_buffer is None
    assert restored._pending_execution_steps is None

    # Both held-out loss modes and explicit-history predictions remain stateless.
    with torch.no_grad():
        for mode in ("expert", "generated"):
            assert torch.isfinite(restored(batch, history_mode=mode))
    restored.on_optimizer_step()  # Evaluation must not advance the curriculum.
    explicit = TrainPolicyWorkspace._predict_validation_action(restored, batch)
    assert explicit["action_pred"].shape == (2, 16, 7)
    assert restored.self_past_step == 7
    assert restored._past_buffer is None
    assert restored._pending_execution_steps is None
    result = restored.predict_action(batch["obs"])
    with pytest.raises(RuntimeError, match="feedback"):
        restored.predict_action(batch["obs"])
    restored.record_executed_actions(result["action"], [0, 3])
    restored.predict_action(batch["obs"])



def test_old_checkpoint_explicitly_opts_into_policy_without_migrating_runner(tmp_path):
    cfg = tiny_config("train_past2next_scratch")
    original = make_policy(policy_type=Past2NextSelfPastPolicy)
    original.set_self_past_step(4)
    path = tmp_path / "original_self_past.ckpt"
    torch.save({
        "cfg": cfg,
        "state_dicts": {"model": original.state_dict(), "ema_model": original.state_dict()},
    }, path, pickle_module=dill)
    new_target = "oat.policy.past2next_self_past_executed.Past2NextSelfPastExecutedPolicy"
    restored, saved_cfg = BasePolicy.from_checkpoint(
        str(path), policy_overrides={"_target_": new_target}, return_configuration=True,
    )
    assert isinstance(restored, Past2NextSelfPastExecutedPolicy)
    assert restored.self_past_step == 4
    assert restored.self_past_probability() == original.self_past_probability() == 0.25
    assert restored.state_dict().keys() == original.state_dict().keys()
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    assert saved_cfg.policy._target_ == new_target
    # Policy-only opt-in deliberately leaves runner selection to the caller.
    assert saved_cfg.task.policy.env_runner._target_ == cfg.task.policy.env_runner._target_
    assert saved_cfg.task.policy.env_runner._target_ == "oat.env_runner.libero_runner.LiberoRunner"
    assert restored._past_buffer is None
    assert restored._pending_execution_steps is None

    batch = make_self_past_batch()
    result = restored.predict_action(batch["obs"])
    torch.testing.assert_close(restored._past_buffer, torch.zeros(2, 7, 7))
    with pytest.raises(RuntimeError, match="feedback"):
        restored.predict_action(batch["obs"])
    restored.record_executed_actions(result["action"], [2, 8])
    expected = torch.tensor([[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 4, 5, 6, 7]]).float()
    torch.testing.assert_close(restored._past_buffer, expected[:, :, None].expand(-1, -1, 7))
    restored.predict_action(batch["obs"])
