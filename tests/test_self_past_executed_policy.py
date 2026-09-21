"""Preserve offline self-past while conditioning rollout on executed commands."""

import pytest
import torch

from oat.model.common.normalizer import LinearNormalizer
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy
from oat.policy.past2next_self_past_executed import Past2NextSelfPastExecutedPolicy
from oat.workspace.train_policy import TrainPolicyWorkspace
from test_executed_past_policy import observe_condition
from test_history_training import TinyObservationEncoder, TinyTokenizer, make_batch


def make_policy(policy_type=Past2NextSelfPastExecutedPolicy, **overrides):
    kwargs = dict(
        shape_meta={"action": {"shape": [7]},
                    "obs": {"state": {"shape": [7], "type": "state"}}},
        obs_encoder=TinyObservationEncoder(), action_tokenizer=TinyTokenizer(),
        n_action_steps=8, n_obs_steps=2, past_n=7,
        embed_dim=8, n_layers=1, n_heads=2, dropout=0,
        temperature=0, self_past_p=0.5, self_past_warmup_steps=2,
        self_past_ramp_steps=4, self_past_schedule="optimizer_step",
    )
    kwargs.update(overrides)
    policy = policy_type(**kwargs)
    low = torch.arange(7).float() / 2 - 2
    high = low + torch.arange(7).float() + 2
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.stack([low, high])})
    policy.set_normalizer(normalizer)
    return policy.eval()


def make_self_past_batch(batch_size=2):
    batch = make_batch(batch_size)
    batch["past_action"] = torch.arange(batch_size * 49).reshape(batch_size, 7, 7).float() / 50
    batch.update(
        prev_obs={"state": torch.full((batch_size, 2, 7), 0.25)},
        prev_past_action=batch["past_action"] + 0.1,
        prev_window_valid=torch.ones(batch_size, dtype=torch.bool),
        past_action_valid=torch.ones(batch_size, 7, dtype=torch.bool),
    )
    return batch


@pytest.mark.parametrize("history_mode,step", [
    ("configured", 0), ("configured", 4), ("configured", 6),
    ("generated", 0), ("expert", 6),
])
def test_offline_loss_histories_and_gradients_match_original_self_past(monkeypatch, history_mode, step):
    policy = make_policy().train()
    original = make_policy(policy_type=Past2NextSelfPastPolicy).train()
    policy.set_self_past_step(step)
    original.load_state_dict(policy.state_dict(), strict=True)
    observed, _ = observe_condition(monkeypatch, policy)
    expected, _ = observe_condition(monkeypatch, original)
    batch = make_self_past_batch()
    batch["prev_window_valid"][0] = False

    def forbid_rollout(*args, **kwargs):
        raise AssertionError("Offline self-past must not require environment execution")

    monkeypatch.setattr(policy, "predict_action", forbid_rollout)
    monkeypatch.setattr(policy, "record_executed_actions", forbid_rollout)
    losses = []
    for candidate in (original, policy):
        torch.manual_seed(17)
        loss = candidate(batch, history_mode=history_mode)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    assert len(observed) == len(expected)
    for actual_history, expected_history in zip(observed, expected):
        torch.testing.assert_close(actual_history, expected_history, rtol=0, atol=0)
    for name, parameter in policy.named_parameters():
        other = dict(original.named_parameters())[name]
        if other.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    assert policy.self_past_step == step
    assert policy._past_buffer is None
    assert policy._pending_execution_steps is None
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in policy.action_tokenizer.parameters())
    assert not policy.action_tokenizer.training


def test_generated_training_history_is_used_without_touching_pending_rollout(monkeypatch):
    policy = make_policy()
    batch = make_self_past_batch()
    result = policy.predict_action(batch["obs"])
    policy.record_executed_actions(torch.full_like(result["action"], -13))
    policy.predict_action(batch["obs"])
    saved = policy._past_buffer.clone()
    histories, _ = observe_condition(monkeypatch, policy)
    policy.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = policy(batch, history_mode="generated")
    loss.backward()
    # The tiny tokenizer decodes 0..15; offline self-past still uses 1..7.
    generated = torch.arange(1, 8).float()[None, :, None].expand(2, 7, 7)
    torch.testing.assert_close(histories[-1], generated)
    torch.testing.assert_close(policy._past_buffer, saved)
    assert policy._pending_execution_steps == 8
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in policy.raw_proj.parameters())


def test_partial_execution_drives_raw_and_dynamic_conditions(monkeypatch):
    policy = make_policy()
    histories, projected = observe_condition(monkeypatch, policy)
    obs = make_batch()["obs"]
    prediction = policy.predict_action(obs)
    assert prediction["action_pred"].shape == (2, 16, 7)
    torch.testing.assert_close(policy._past_buffer, torch.zeros(2, 7, 7))
    with pytest.raises(RuntimeError, match="feedback"):
        policy.predict_action(obs)

    # Acknowledged commands may differ from the plan; the unexecuted tail is ignored.
    commands = (torch.arange(8).float().square()[None, :, None]
                + torch.arange(7).float()[None, None, :] / 10).expand(2, -1, -1).clone()
    expected = torch.zeros(2, 7, 7)
    expected[0, -3:] = commands[0, :3]
    expected[1] = commands[1, 1:8]
    commands[0, 3:] = float("nan")
    policy.record_executed_actions(commands, [3, 8])
    commands.fill_(999)
    policy.predict_action(obs)
    torch.testing.assert_close(histories[-1], expected)
    normalized = policy.action_normalizer["action"].normalize(expected)
    torch.testing.assert_close(projected["raw"][-1], normalized)
    torch.testing.assert_close(projected["acc"][-1], normalized[:, -1] - normalized[:, -2])
    torch.testing.assert_close(projected["jerk"][-1],
                               normalized[:, -1] - 2 * normalized[:, -2] + normalized[:, -3])
    # No new executions means no change to either environment's history.
    policy.record_executed_actions(torch.zeros(2, 8, 7), [0, 0])
    policy.predict_action(obs)
    torch.testing.assert_close(histories[-1], expected)
    policy.reset()
    policy.predict_action(make_batch(3)["obs"])
    torch.testing.assert_close(histories[-1], torch.zeros(3, 7, 7))


def test_offline_validation_is_stateless_and_preserves_execution_obligation(monkeypatch):
    policy = make_policy()
    online = make_batch()
    result = policy.predict_action(online["obs"])
    saved = policy._past_buffer.clone()
    histories, _ = observe_condition(monkeypatch, policy)
    batch = make_self_past_batch(3)
    TrainPolicyWorkspace._predict_validation_action(policy, batch)
    torch.testing.assert_close(histories[-1], batch["past_action"])
    policy(batch, history_mode="expert")
    policy(batch, history_mode="generated")
    assert policy.self_past_step == 0
    torch.testing.assert_close(policy._past_buffer, saved)
    with pytest.raises(RuntimeError, match="feedback"):
        policy.predict_action(online["obs"])
    policy.record_executed_actions(result["action"])
    policy.predict_action(online["obs"])
    torch.testing.assert_close(histories[-1], result["action"][:, 1:8])


def test_original_weights_load_strictly_and_schedule_survives_new_checkpoint():
    original = make_policy(policy_type=Past2NextSelfPastPolicy)
    original.set_self_past_step(4)
    policy = make_policy()
    policy.load_state_dict(original.state_dict(), strict=True)
    assert policy.self_past_step == 4
    assert policy.self_past_probability() == 0.25
    policy.train()
    policy.on_optimizer_step()
    assert policy.self_past_step == 5
    policy.eval()
    policy.on_optimizer_step()
    assert policy.self_past_step == 5
    result = policy.predict_action(make_batch()["obs"])
    policy.record_executed_actions(result["action"], [3, 8])
    policy.predict_action(make_batch()["obs"])
    restored = make_policy()
    restored.load_state_dict(policy.state_dict(), strict=True)
    assert restored.self_past_step == 5
    assert restored.self_past_probability() == 0.375
    assert restored._past_buffer is None
    assert restored._pending_execution_steps is None
    # The inference change does not introduce any extra saved model state.
    assert restored.state_dict().keys() == original.state_dict().keys()


@pytest.mark.parametrize("stride", [4, 8])
def test_full_execution_preserves_original_rollout_conditions(monkeypatch, stride):
    original = make_policy(policy_type=Past2NextSelfPastPolicy, n_action_steps=stride)
    policy = make_policy(n_action_steps=stride)
    policy.load_state_dict(original.state_dict(), strict=True)
    expected_histories, _ = observe_condition(monkeypatch, original)
    histories, _ = observe_condition(monkeypatch, policy)
    obs = make_batch()["obs"]
    for _ in range(3):
        expected = original.predict_action(obs)
        actual = policy.predict_action(obs)
        torch.testing.assert_close(histories[-1], expected_histories[-1])
        torch.testing.assert_close(actual["action_pred"], expected["action_pred"])
        policy.record_executed_actions(actual["action"])
        torch.testing.assert_close(policy._past_buffer, original._past_buffer)
