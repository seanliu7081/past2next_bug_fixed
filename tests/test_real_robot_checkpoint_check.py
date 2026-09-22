"""CPU checks for checkpoint validation across predicted and executed histories."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from oat.common.seq_sampler import get_val_mask
from oat.policy.base_policy import BasePolicy
from oat.policy.past2next_state_history_gate_real_robot import Past2NextRealRobotStateHistoryGatePolicy
from oat.tokenizer.oat.tokenizer import OATTok
from scripts.check_real_robot_checkpoint import check_checkpoint, _episode_window
from test_history_training import make_policy as make_baseline
from test_state_history_gate_real_robot import (
    META, SHAPES, TinyRealObservationEncoder, TinyRealTokenizer, make_normalizer,
)


def make_gate():
    policy = Past2NextRealRobotStateHistoryGatePolicy(
        shape_meta=META, obs_encoder=TinyRealObservationEncoder(),
        action_tokenizer=TinyRealTokenizer(), n_obs_steps=2, n_action_steps=8,
        past_n=7, state_history_steps=8, embed_dim=16, n_layers=1, n_heads=2,
        dropout=0, history_embed_dim=16, history_n_heads=2, history_n_layers=1,
        history_summary_tokens=2, history_dropout=0, history_gate_hidden_dim=16,
        history_gate_mode="learned", rotation_6d_layout="rows", temperature=0,
    )
    policy.set_normalizer(make_normalizer())
    return policy


def dataset_config(tmp_path, length, gate):
    # Put the selected holdout between two very different episodes to expose
    # accidental negative slices or reads beyond a short episode's boundary.
    seed = next(seed for seed in range(100) if get_val_mask(3, .34, seed)[1])
    lengths = [20, length, 20]
    total = sum(lengths)
    values = np.concatenate([1000 * (i + 1) + np.arange(n)
                             for i, n in enumerate(lengths)]).astype(np.float32)
    path = tmp_path / "tiny.zarr"
    root = zarr.open(str(path), mode="w")
    root.create_dataset("meta/episode_ends", data=np.cumsum(lengths))
    for key, width in {"state": 7, "robot0_eef_pos": 3, "robot0_gripper_qpos": 1,
                       "action": 7}.items():
        root.create_dataset("data/" + key, data=np.repeat(values[:, None], width, axis=1))
    root.create_dataset("data/robot0_eef_rot6d", data=np.tile(
        np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (total, 1)))
    cfg = OmegaConf.create({"policy": {"action_tokenizer": {"checkpoint": "source.ckpt"}},
                           "task": {"policy": {"dataset": {
                               "zarr_path": str(path), "obs_keys": list(SHAPES) if gate else ["state"],
                               "action_key": "action", "val_ratio": .34, "seed": seed,
                           }}}})
    return cfg


@pytest.mark.parametrize("length", [1, 3, 25])
@pytest.mark.parametrize("gate", [False, True])
def test_checker_replays_episode_aligned_history(tmp_path, monkeypatch, gate, length):
    policy = make_gate() if gate else make_baseline()
    cfg = dataset_config(tmp_path, length, gate)
    source = copy.deepcopy(policy.action_tokenizer)
    monkeypatch.setattr(BasePolicy, "from_checkpoint", lambda *args, **kwargs: (policy, cfg))
    monkeypatch.setattr(OATTok, "from_checkpoint", lambda *args, **kwargs: source)
    stateful = []
    predict = policy.predict_action

    def observe(obs, **kwargs):
        if kwargs.get("past_actions") is None:
            stateful.append({key: value.clone() for key, value in obs.items()})
        return predict(obs, **kwargs)

    monkeypatch.setattr(policy, "predict_action", observe)
    report = check_checkpoint(tmp_path / "policy.ckpt", "cpu")
    assert report["validation_episode"] == 1
    assert report["tokenizer_frozen"] and report["tokenizer_unchanged"]
    assert report["explicit_and_stateful_inference_finite"]
    assert report["state_history_checked"] == gate
    stride = min(8, length - 1)
    assert report["stateful_observation_stride"] == stride
    state_key = "robot0_eef_pos" if gate else "state"
    assert stateful[0][state_key][0, :, 0].tolist() == [2000, 2000]
    assert stateful[1][state_key][0, -1, 0].item() == 2000 + stride
    if gate:
        assert report["history_source"] == "acknowledged_dataset_commands"
        assert stateful[0]["state_history_valid"].tolist() == [[False] * 7 + [True]]
        for obs, anchor in zip(stateful, [0, stride]):
            valid = [step >= 0 for step in range(anchor - 7, anchor + 1)]
            assert obs["state_history_valid"].tolist() == [valid]
            expected = [2000 + step if step >= 0 else 0 for step in range(anchor - 7, anchor + 1)]
            assert obs["state_history__robot0_eef_pos"][0, :, 0].tolist() == expected
        count = stride + min(8, length - stride)
        expected = [2000 + step if step >= 0 else 0 for step in range(count - 7, count)]
        assert policy._past_buffer[0, :, 0].tolist() == expected
        assert policy._pending_execution_steps is None
    else:
        assert report["history_source"] == "predicted_actions"
        assert policy._past_buffer[0, :, 0].tolist() == list(range(1, 8))


def test_window_only_reads_the_selected_episode():
    class GuardedArray:
        dtype = np.dtype("float64")
        shape = (30, 1)

        def __getitem__(self, item):
            assert 20 <= item.start <= item.stop <= 23
            return np.arange(item.start, item.stop, dtype=self.dtype)[:, None]

    array = GuardedArray()
    window = _episode_window(array, 20, 23, 14, 22)
    assert window.dtype == np.float32
    assert window[:, 0].tolist() == [0] * 6 + [20, 21]
    edge = _episode_window(array, 20, 23, 19, 21, edge=True)
    assert edge[:, 0].tolist() == [20, 20]
    assert _episode_window(array, 20, 23, 20, 20).shape == (0, 1)


@pytest.mark.parametrize("failure", ["changed", "unfrozen"])
def test_checker_still_rejects_invalid_tokenizer(tmp_path, monkeypatch, failure):
    policy = make_gate()
    cfg = dataset_config(tmp_path, 3, True)
    source = copy.deepcopy(policy.action_tokenizer)
    if failure == "changed":
        with torch.no_grad():
            source.placeholder.add_(1)
    else:
        policy.action_tokenizer.placeholder.requires_grad_(True)
    monkeypatch.setattr(BasePolicy, "from_checkpoint", lambda *args, **kwargs: (policy, cfg))
    monkeypatch.setattr(OATTok, "from_checkpoint", lambda *args, **kwargs: source)
    with pytest.raises(AssertionError):
        check_checkpoint(tmp_path / "policy.ckpt", "cpu")
