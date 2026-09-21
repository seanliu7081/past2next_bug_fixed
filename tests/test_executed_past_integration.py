"""Exercise executed history through real policy, runner, workspace, and reload APIs."""

from hydra import compose, initialize_config_dir
from pathlib import Path

import pytest
import torch

from oat.env_runner.executed_action_runner import LiberoExecutedPastRunner, RoboCasaExecutedPastRunner
from oat.model.common.normalizer import LinearNormalizer
from oat.model.diffusion.ema_model import EMAModel
from oat.policy.base_policy import BasePolicy
from oat.policy.past2next_executed_past import Past2NextExecutedPastPolicy
from oat.workspace.train_policy import TrainPolicyWorkspace
from test_executed_action_runner import FakeVectorEnv
from test_executed_past_policy import make_policy, observe_condition
from test_history_training import make_batch


class SevenDimensionalVectorEnv(FakeVectorEnv):
    def observation(self):
        obs = super().observation()
        obs["state"] = obs["state"].repeat(7, axis=-1)
        return obs


@pytest.mark.parametrize("runner_type", [LiberoExecutedPastRunner, RoboCasaExecutedPastRunner])
def test_real_policy_consumes_only_executed_prefixes_in_existing_runner_loop(monkeypatch, runner_type):
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

    # A second evaluation must reset histories before any new command executes.
    runner.run(policy, temperature=0)
    torch.testing.assert_close(histories[2], torch.zeros(2, 7, 7))


@pytest.mark.parametrize("weights", ["model", "ema"])
def test_new_recipe_trains_saves_and_reloads_with_execution_contract(tmp_path, weights):
    config_dir = Path(__file__).resolve().parents[1] / "oat/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="experimental/train_past2next_executed_past")
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
    workspace = TrainPolicyWorkspace(cfg, output_dir=str(tmp_path), lazy_instantiation=False)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.tensor([-1.0, 1.0])[:, None].expand(2, 7)})
    workspace.model.set_normalizer(normalizer)
    workspace.ema_model.set_normalizer(normalizer)
    batch = make_batch()
    batch["past_action"] = torch.randn(2, 7, 7)
    workspace.model.train()
    loss = workspace.model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    workspace.optimizer.step()
    workspace.optimizer.zero_grad(set_to_none=True)
    EMAModel(workspace.ema_model).step(workspace.model)

    # Episode history is runtime-only, even when saving a policy after rollout.
    workspace.model.eval()
    action = workspace.model.predict_action(batch["obs"])["action"]
    workspace.model.record_executed_actions(action, [3, 8])
    checkpoint = workspace.save_checkpoint(tag="executed", use_thread=False)
    restored, saved_cfg = BasePolicy.from_checkpoint(
        checkpoint, weights=weights, return_configuration=True,
    )
    assert isinstance(restored, Past2NextExecutedPastPolicy)
    assert saved_cfg.task.policy.env_runner._target_.endswith("LiberoExecutedPastRunner")
    assert restored._past_buffer is None
    explicit = TrainPolicyWorkspace._predict_validation_action(restored, batch)
    assert explicit["action_pred"].shape == (2, 16, 7)
    assert restored._past_buffer is None
    result = restored.predict_action(batch["obs"])
    with pytest.raises(RuntimeError, match="feedback"):
        restored.predict_action(batch["obs"])
    restored.record_executed_actions(result["action"], [0, 3])
    restored.predict_action(batch["obs"])
