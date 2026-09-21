"""Executed commands alone supply online past-action and dynamic features."""

import numpy as np
import pytest
import torch

from oat.model.common.normalizer import LinearNormalizer
from oat.policy.past2next_executed_past import Past2NextExecutedPastPolicy
from test_history_training import TinyObservationEncoder, TinyTokenizer, make_batch


def make_policy():
    policy = Past2NextExecutedPastPolicy(
        shape_meta={"action": {"shape": [7]},
                    "obs": {"state": {"shape": [7], "type": "state"}}},
        obs_encoder=TinyObservationEncoder(), action_tokenizer=TinyTokenizer(),
        n_action_steps=8, n_obs_steps=2, past_n=7,
        embed_dim=8, n_layers=1, n_heads=2, dropout=0,
        temperature=0,
    )
    low = torch.arange(7).float() / 2 - 2
    high = low + torch.arange(7).float() + 2
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.stack([low, high])})
    policy.set_normalizer(normalizer)
    return policy.eval()


def observe_condition(monkeypatch, policy):
    histories = []
    projected = {"raw": [], "acc": [], "jerk": []}
    original = policy._build_condition

    def build(features, past_actions):
        histories.append(past_actions.detach().clone())
        return original(features, past_actions)

    monkeypatch.setattr(policy, "_build_condition", build)
    for name in projected:
        module = getattr(policy, name + "_proj")

        def record(module, args, key=name):
            projected[key].append(args[0].detach().clone())

        module.register_forward_pre_hook(record)
    return histories, projected


def buffer_copy(policy):
    return None if policy._past_buffer is None else policy._past_buffer.detach().clone()


def assert_buffer_unchanged(policy, previous):
    if previous is None:
        assert policy._past_buffer is None
    else:
        torch.testing.assert_close(policy._past_buffer, previous)


def test_prediction_requires_acknowledgment_and_never_advances_history(monkeypatch):
    policy = make_policy()
    histories, _ = observe_condition(monkeypatch, policy)
    obs = make_batch()["obs"]

    result = policy.predict_action(obs)
    torch.testing.assert_close(histories[-1], torch.zeros(2, 7, 7))
    before = buffer_copy(policy)
    with pytest.raises(RuntimeError):
        policy.predict_action(obs)
    assert_buffer_unchanged(policy, before)

    policy.record_executed_actions(result["action"], executed_lengths=[0, 0])
    policy.predict_action(obs)
    torch.testing.assert_close(histories[-1], torch.zeros(2, 7, 7))


def test_adjusted_executed_commands_supply_both_history_and_dynamics(monkeypatch):
    policy = make_policy()
    histories, projected = observe_condition(monkeypatch, policy)
    obs = make_batch()["obs"]
    result = policy.predict_action(obs)
    assert result["action_pred"].shape == (2, 16, 7)
    assert result["action"].shape == (2, 8, 7)

    # Controller-adjusted commands deliberately differ from every planned action.
    steps = torch.arange(8).float().square()[None, :, None]
    executed = (steps + torch.arange(7)[None, None, :] / 10 - 30).expand(2, -1, -1).clone()
    expected_history = executed[:, 1:8].clone()
    policy.record_executed_actions(executed)
    executed.fill_(999)  # Acknowledged history must not alias the caller's tensor.
    policy.predict_action(obs)

    torch.testing.assert_close(histories[-1], expected_history)
    normalized = policy.action_normalizer["action"].normalize(expected_history)
    torch.testing.assert_close(projected["raw"][-1], normalized)
    torch.testing.assert_close(projected["acc"][-1], normalized[:, -1] - normalized[:, -2])
    torch.testing.assert_close(
        projected["jerk"][-1], normalized[:, -1] - 2 * normalized[:, -2] + normalized[:, -3],
    )


def test_per_environment_partial_execution_appends_only_acknowledged_prefixes(monkeypatch):
    policy = make_policy()
    histories, _ = observe_condition(monkeypatch, policy)
    obs = make_batch(3)["obs"]
    policy.predict_action(obs)
    first = torch.arange(3 * 8 * 7).reshape(3, 8, 7).float()
    policy.record_executed_actions(first)
    policy.predict_action(obs)

    second = (1000 + np.arange(3 * 8 * 7)).reshape(3, 8, 7).astype(np.float64)
    # Values outside the acknowledged prefixes are not executed and must be ignored.
    second[0] = np.nan
    second[1, 3:] = np.nan
    policy.record_executed_actions(second, executed_lengths=np.array([0, 3, 8]))
    policy.predict_action(obs)

    expected = first[:, 1:].clone()
    expected[1] = torch.cat([first[1, 4:], torch.as_tensor(second[1, :3]).float()])
    expected[2] = torch.as_tensor(second[2, 1:]).float()
    torch.testing.assert_close(histories[-1], expected)
    assert torch.isfinite(histories[-1]).all()


def test_explicit_offline_history_bypasses_pending_ack_without_changing_it(monkeypatch):
    policy = make_policy()
    histories, _ = observe_condition(monkeypatch, policy)
    obs = make_batch()["obs"]
    result = policy.predict_action(obs)
    before = buffer_copy(policy)
    offline = make_batch(3)
    offline["past_action"].fill_(-11)

    policy.predict_action(offline["obs"], past_actions=offline["past_action"])
    torch.testing.assert_close(histories[-1], offline["past_action"])
    assert_buffer_unchanged(policy, before)
    with pytest.raises(RuntimeError):
        policy.predict_action(obs)

    policy.record_executed_actions(result["action"])
    policy.predict_action(obs)
    torch.testing.assert_close(histories[-1], result["action"][:, 1:])


def test_offline_prediction_does_not_create_an_execution_to_acknowledge():
    policy = make_policy()
    batch = make_batch()
    policy.predict_action(batch["obs"], past_actions=batch["past_action"])
    with pytest.raises(RuntimeError):
        policy.record_executed_actions(torch.zeros(2, 8, 7))


@pytest.mark.parametrize("case", [
    "negative", "fractional", "over_action_length", "over_submitted_length",
    "wrong_count_shape", "wrong_batch", "wrong_action_dim", "wrong_rank",
    "full_unexecuted_prediction", "nonfinite_executed",
])
def test_invalid_acknowledgment_is_atomic_and_can_be_retried(case):
    policy = make_policy()
    obs = make_batch()["obs"]
    initial = policy.predict_action(obs)
    policy.record_executed_actions(initial["action"])
    pending = policy.predict_action(obs)
    before = buffer_copy(policy)
    actions = torch.zeros(2, 8, 7)
    lengths = [8, 8]
    if case == "negative":
        lengths = [-1, 8]
    elif case == "fractional":
        lengths = [1.5, 8]
    elif case == "over_action_length":
        lengths = [9, 8]
    elif case == "over_submitted_length":
        actions, lengths = torch.zeros(2, 3, 7), [4, 3]
    elif case == "wrong_count_shape":
        lengths = [8]
    elif case == "wrong_batch":
        actions = torch.zeros(1, 8, 7)
    elif case == "wrong_action_dim":
        actions = torch.zeros(2, 8, 6)
    elif case == "wrong_rank":
        actions = torch.zeros(2, 7)
    elif case == "full_unexecuted_prediction":
        actions = pending["action_pred"]
    elif case == "nonfinite_executed":
        actions[1, 0, 0] = torch.inf

    with pytest.raises((ValueError, TypeError)):
        policy.record_executed_actions(actions, executed_lengths=lengths)
    assert_buffer_unchanged(policy, before)
    with pytest.raises(RuntimeError):
        policy.predict_action(obs)
    policy.record_executed_actions(pending["action"])
    policy.predict_action(obs)


def test_acknowledgment_requires_prediction_and_cannot_be_applied_twice():
    policy = make_policy()
    actions = torch.zeros(2, 8, 7)
    with pytest.raises(RuntimeError):
        policy.record_executed_actions(actions)
    policy.predict_action(make_batch()["obs"])
    policy.record_executed_actions(actions)
    before = buffer_copy(policy)
    with pytest.raises(RuntimeError):
        policy.record_executed_actions(actions)
    assert_buffer_unchanged(policy, before)


def test_returned_action_length_bounds_acknowledgment(monkeypatch):
    policy = make_policy()
    monkeypatch.setattr(policy.action_tokenizer, "detokenize",
                        lambda tokens: torch.zeros(tokens.shape[0], 4, 7))
    result = policy.predict_action(make_batch()["obs"])
    assert result["action"].shape[1] == 4
    with pytest.raises(ValueError):
        policy.record_executed_actions(torch.zeros(2, 5, 7), executed_lengths=[4, 4])
    policy.record_executed_actions(result["action"])


def test_reset_clears_history_pending_ack_and_allows_a_new_batch(monkeypatch):
    policy = make_policy()
    histories, _ = observe_condition(monkeypatch, policy)
    obs = make_batch()["obs"]
    policy.predict_action(obs)
    policy.record_executed_actions(torch.full((2, 8, 7), 13.0))
    with pytest.raises((RuntimeError, ValueError)):
        policy.predict_action(make_batch(3)["obs"])
    policy.reset()
    policy.predict_action(make_batch(3)["obs"])
    torch.testing.assert_close(histories[-1], torch.zeros(3, 7, 7))
    policy.reset()  # Reset must also discard the unacknowledged prediction.
    policy.predict_action(obs)
    torch.testing.assert_close(histories[-1], torch.zeros(2, 7, 7))


def test_training_uses_demonstrated_history_without_generating_self_past(monkeypatch):
    policy = make_policy().train()
    histories, projected = observe_condition(monkeypatch, policy)

    def forbid_generation(*args, **kwargs):
        raise AssertionError("Demonstration-history training must not generate its own past")

    monkeypatch.setattr(policy.model, "generate", forbid_generation)
    batch = make_batch()
    batch["past_action"] = torch.arange(98).reshape(2, 7, 7).float() / 20
    loss = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    torch.testing.assert_close(histories[-1], batch["past_action"])
    normalized = policy.action_normalizer["action"].normalize(batch["past_action"])
    torch.testing.assert_close(projected["raw"][-1], normalized)
    torch.testing.assert_close(projected["acc"][-1], normalized[:, -1] - normalized[:, -2])
    torch.testing.assert_close(
        projected["jerk"][-1], normalized[:, -1] - 2 * normalized[:, -2] + normalized[:, -3],
    )
    assert all(not module.training for module in policy.action_tokenizer.modules())
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in policy.action_tokenizer.parameters())
    assert any(parameter.grad is not None for parameter in policy.raw_proj.parameters())


def test_acknowledged_history_is_detached_and_not_checkpoint_state(monkeypatch):
    policy = make_policy()
    initial_state = {name: value.clone() for name, value in policy.state_dict().items()}
    policy.predict_action(make_batch()["obs"])
    actions = torch.full((2, 8, 7), 17.0, requires_grad=True)
    policy.record_executed_actions(actions)
    assert not policy._past_buffer.requires_grad
    assert policy._past_buffer.grad_fn is None
    state = policy.state_dict()
    assert state.keys() == initial_state.keys()
    for name in state:
        torch.testing.assert_close(state[name], initial_state[name], rtol=0, atol=0)

    restored = make_policy()
    restored.load_state_dict(state)
    with pytest.raises(RuntimeError):
        restored.record_executed_actions(actions.detach())
    histories, _ = observe_condition(monkeypatch, restored)
    restored.predict_action(make_batch()["obs"])
    torch.testing.assert_close(histories[-1], torch.zeros(2, 7, 7))
