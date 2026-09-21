"""A deterministic plant checks action/state alignment through the real runner."""

from functools import partial

import dill
import gymnasium
import numpy as np
import torch

from oat.env_runner.state_history_runner import (
    LiberoStateHistoryRunner, StateHistoryInitializer,
)
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import VideoRecordingWrapper
from test_state_history_policy import SHAPES, make_policy
from test_state_history_runner import InactiveRecorder


HISTORY_STEPS = 16
OFFSETS = (0, 100, 200)
TERMINAL_STEPS = (2, 10, 21)


class IntegratingPlant(gymnasium.Env):
    """Each executed command changes position by its first three coordinates."""

    metadata = {}
    render_mode = None

    def __init__(self, terminal_step, offset):
        self.terminal_step = terminal_step
        self.offset = offset
        self.cur_step = 0
        self.commands = []
        self.position = np.full(3, offset, dtype=np.float32)
        self.gripper = np.zeros(2, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Dict({
            key: gymnasium.spaces.Box(-np.inf, np.inf, tuple(shape), np.float32)
            for key, shape in SHAPES.items()
        } | {"rgb": gymnasium.spaces.Box(0, 255, (2, 2, 3), np.uint8)})
        self.action_space = gymnasium.spaces.Box(-100, 100, (7,), np.float32)

    def observe(self):
        return {
            "robot0_eef_pos": self.position.copy(),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
            "robot0_gripper_qpos": self.gripper.copy(),
            "rgb": np.full((2, 2, 3), self.cur_step, dtype=np.uint8),
        }

    def configure_episode(self, seed, init_state_id):
        self.configured = (seed, init_state_id)

    def reset(self, *, seed=None, options=None):
        self.cur_step = 0
        self.commands = []
        self.position[:] = self.offset
        self.gripper[:] = 0
        return self.observe(), {}

    def step(self, action):
        command = np.asarray(action).copy()
        self.commands.append(command)
        self.position += command[:3]
        self.gripper += command[-2:]
        self.cur_step += 1
        done = self.cur_step >= self.terminal_step
        return self.observe(), float(done), done, False, {}


def plant_environment(terminal_step, offset):
    return MultiStepWrapper(
        VideoRecordingWrapper(IntegratingPlant(terminal_step, offset), InactiveRecorder()),
        n_obs_steps=2, n_action_steps=8, max_episode_steps=32,
    )


def configure_plant(outer):
    outer.env.env.configure_episode(42, None)


def state_snapshot(outer):
    return outer.env.env.snapshot()


def assert_state_action_alignment(obs, past, counts):
    """Independent oracle: the tiny tokenizer proposes commands 0..7 per chunk."""
    states = torch.as_tensor(obs["state_history__robot0_eef_pos"])
    validity = torch.as_tensor(obs["state_history_valid"]).bool()
    past = torch.as_tensor(past)
    for row, (count, offset) in enumerate(zip(counts, OFFSETS)):
        times = np.arange(count - HISTORY_STEPS + 1, count + 1)
        expected_valid = torch.from_numpy(times >= 0)
        torch.testing.assert_close(validity[row], expected_valid)
        expected_state = torch.tensor([
            offset + int(np.sum(np.arange(step) % 8)) if step >= 0 else 0
            for step in times
        ], dtype=states.dtype)
        torch.testing.assert_close(states[row], expected_state[:, None].expand(-1, 3))
        command_times = times[:-1]
        expected_past = torch.tensor([
            step % 8 if step >= 0 else 0 for step in command_times
        ], dtype=past.dtype)
        torch.testing.assert_close(past[row], expected_past[:, None].expand(-1, 7))
        transitions = expected_valid[:-1] & expected_valid[1:]
        observed_displacement = states[row, 1:] - states[row, :-1]
        torch.testing.assert_close(observed_displacement[transitions], past[row, transitions, :3])
        assert torch.count_nonzero(past[row, ~transitions]) == 0


def test_h16_real_policy_runner_keeps_commands_aligned_with_executed_transitions(monkeypatch):
    vector = AsyncVectorEnv(
        [partial(plant_environment, terminal, offset)
         for terminal, offset in zip(TERMINAL_STEPS, OFFSETS)],
        shared_memory=False, context="fork", autoreset=False,
    )
    runner = LiberoStateHistoryRunner.__new__(LiberoStateHistoryRunner)
    runner.env = vector
    runner.env_fns = [None] * 3
    initializer = dill.dumps(StateHistoryInitializer(dill.dumps(configure_plant), HISTORY_STEPS))
    runner.env_init_fn_dills = [initializer] * 3
    runner.env_seeds = [1000, 1001, 1002]
    runner.env_task_names = ["integrating_plant"] * 3
    runner.task_name = "integrating_plant"
    runner.max_episode_steps = 32
    runner.n_action_steps = 8
    runner.tqdm_interval_sec = 1000
    runner.protocol = "corrected"
    runner.episode_schedule = [{"episode_index": index} for index in range(3)]
    runner.episode_records_path = None
    policy = make_policy(history_steps=HISTORY_STEPS).eval()
    policy.set_self_past_step(77)
    predictions = []
    acknowledgments = []
    encoder_masks = []
    original_predict = policy.predict_action
    original_acknowledge = policy.record_executed_actions

    def observe_prediction(obs, **kwargs):
        # This is the original LiberoRunner loop's maybe_to_torch conversion.
        mask = obs["state_history_valid"]
        assert mask.dtype == policy.dtype and mask.is_floating_point()
        assert ((mask == 0) | (mask == 1)).all()
        assert obs["robot0_eef_pos"].shape[1] == 2
        output = original_predict(obs, **kwargs)
        predictions.append((
            {key: value.detach().clone() for key, value in obs.items()},
            policy._past_buffer.detach().clone(),
        ))
        return output

    def observe_acknowledgment(actions, executed_lengths=None):
        before = policy._past_buffer.detach().clone()
        assert policy._pending_execution_steps == 8
        original_acknowledge(actions, executed_lengths=executed_lengths)
        assert policy._pending_execution_steps is None
        after = policy._past_buffer.detach().clone()
        commands = torch.as_tensor(actions)
        for row, count in enumerate(executed_lengths):
            expected = torch.cat((before[row], commands[row, :count]), dim=0)[-15:]
            torch.testing.assert_close(after[row], expected)
            if count == 0:
                torch.testing.assert_close(after[row], before[row], rtol=0, atol=0)
        acknowledgments.append(np.asarray(executed_lengths).copy())

    def observe_encoder(module, args):
        # The new policy restores strict bool metadata before temporal attention.
        assert args[1].dtype == torch.bool
        encoder_masks.append(args[1].detach().clone())

    monkeypatch.setattr(policy, "predict_action", observe_prediction)
    monkeypatch.setattr(policy, "record_executed_actions", observe_acknowledgment)
    handle = policy.history_encoder.register_forward_pre_hook(observe_encoder)
    try:
        assert vector.single_observation_space["rgb"].shape == (2, 2, 2, 3)
        assert "state_history_valid" not in vector.single_observation_space.spaces
        # A second run exercises actual episode reset after full/partial histories.
        for episode in range(2):
            log = runner.run(policy, temperature=0)
            assert log["mean_success_rate"] == 1
            assert runner.env is vector
            assert policy.self_past_step == 77
            assert policy._pending_execution_steps is None
            current = predictions[episode * 3:(episode + 1) * 3]
            assert len(current) == 3
            for (obs, past), counts in zip(current, ((0, 0, 0), (2, 8, 8), (2, 10, 16))):
                assert_state_action_alignment(obs, past, counts)
            # Completed rows remain untouched while the last worker executes.
            torch.testing.assert_close(current[1][0]["state_history__robot0_eef_pos"][0],
                                       current[2][0]["state_history__robot0_eef_pos"][0])
            np.testing.assert_array_equal(acknowledgments[episode * 3:(episode + 1) * 3],
                                          [[2, 8, 8], [0, 2, 8], [0, 0, 5]])
            snapshots = vector.call("run_dill_function", dill.dumps(state_snapshot))
            final_obs = {key: np.stack([snapshot[key] for snapshot in snapshots])
                         for key in snapshots[0]}
            assert_state_action_alignment(final_obs, policy._past_buffer, TERMINAL_STEPS)
            final_commands = vector.get_attr("commands")
            for commands, count in zip(final_commands, TERMINAL_STEPS):
                expected = (np.arange(count) % 8).astype(np.float32)
                np.testing.assert_array_equal(np.asarray(commands), np.repeat(expected[:, None], 7, axis=1))
            assert [record["policy_steps"] for record in runner.last_episode_records] == list(TERMINAL_STEPS)
        assert len(encoder_masks) == 6
        for mask, (obs, _) in zip(encoder_masks, predictions):
            torch.testing.assert_close(mask, obs["state_history_valid"].bool())
    finally:
        handle.remove()
        vector.close()
