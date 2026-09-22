"""Reload a trusted local real-robot checkpoint and verify dataset inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _episode_window(array, episode_start, episode_end, start, stop, *, edge=False):
    """Read a bounded window without fetching rows from neighboring episodes."""
    import numpy as np

    dtype = np.float32 if array.dtype.kind == "f" else array.dtype
    window = np.zeros((stop - start, *array.shape[1:]), dtype=dtype)
    first, last = max(start, episode_start), min(stop, episode_end)
    if first < last:
        values = np.asarray(array[first:last], dtype=dtype)
        offset = first - start
        window[offset:offset + len(values)] = values
        if edge:
            window[:offset] = values[0]
            window[offset + len(values):] = values[-1]
    return window


def _dataset_observation(root, dataset, policy, start, end, anchor, device):
    import numpy as np
    import torch

    def tensor(values):
        return torch.from_numpy(values).unsqueeze(0).to(device)

    obs = {key: tensor(_episode_window(
        root[f"data/{key}"], start, end,
        anchor - policy.n_obs_steps + 1, anchor + 1, edge=True,
    )) for key in dataset.obs_keys}
    if hasattr(policy, "state_history_keys"):
        first = anchor - policy.state_history_steps + 1
        for key in policy.state_history_keys:
            obs["state_history__" + key] = tensor(_episode_window(
                root[f"data/{key}"], start, end, first, anchor + 1,
            ).astype(np.float32))
        obs["state_history_valid"] = tensor(
            np.arange(first, anchor + 1) >= start)
    return obs


def _buffer_snapshot(policy):
    return {name: value.detach().clone() if hasattr(value, "detach") else value
            for name in ("_past_buffer", "_pending_execution_steps")
            if hasattr(policy, name) for value in (getattr(policy, name),)}


def _assert_buffers_unchanged(policy, snapshot):
    import torch

    for name, previous in snapshot.items():
        current = getattr(policy, name)
        if isinstance(previous, torch.Tensor):
            torch.testing.assert_close(current, previous, rtol=0, atol=0, msg=name)
        elif current != previous:
            raise AssertionError(f"Explicit history mutated {name}")


def _check_dataset_inference(policy, cfg, root, device):
    """Replay recorded commands with aligned states; this is not a robot rollout."""
    import numpy as np
    import torch
    from oat.common.seq_sampler import get_val_mask

    dataset = cfg.task.policy.dataset
    ends = root["meta/episode_ends"][:]
    validation = get_val_mask(len(ends), dataset.val_ratio, dataset.seed)
    starts = np.r_[0, ends[:-1]]
    candidates = np.flatnonzero(validation & (ends > starts))
    if not len(candidates):
        raise ValueError("Checkpoint validation requires a nonempty held-out episode")
    episode = int(candidates[0])
    start, end = int(starts[episode]), int(ends[episode])
    actions = root[f"data/{dataset.get('action_key', 'action')}"]
    acknowledged = callable(getattr(policy, "record_executed_actions", None))
    outputs = []

    def observation(anchor):
        return _dataset_observation(root, dataset, policy, start, end, anchor, device)

    def explicit_prediction(anchor):
        past = torch.from_numpy(_episode_window(
            actions, start, end, anchor - policy.past_n, anchor,
        )).unsqueeze(0).to(device)
        before = _buffer_snapshot(policy)
        outputs.append(policy.predict_action(observation(anchor), past_actions=past, temperature=0))
        _assert_buffers_unchanged(policy, before)

    def acknowledge(anchor, count):
        commands = torch.from_numpy(_episode_window(
            actions, start, end, anchor, anchor + count,
        )).unsqueeze(0).to(device)
        before = policy._past_buffer.detach().clone()
        expected = torch.cat((before, commands.to(before)), dim=1)[:, -policy.past_n:]
        policy.record_executed_actions(commands)
        torch.testing.assert_close(policy._past_buffer, expected, rtol=0, atol=0)
        if policy._pending_execution_steps is not None:
            raise AssertionError("Execution acknowledgement left pending feedback")

    policy.reset()
    with torch.inference_mode():
        explicit_prediction(min(start + 16, end - 1))
        first = policy.predict_action(observation(start), temperature=0)
        outputs.append(first)
        if acknowledged:
            torch.testing.assert_close(policy._past_buffer, torch.zeros_like(policy._past_buffer))
            if policy._pending_execution_steps != first["action"].shape[1]:
                raise AssertionError("Stateful prediction did not request execution feedback")
        # Also test explicit inference with an existing buffer/pending feedback.
        explicit_prediction(start)
        stride = min(policy.n_action_steps, end - start - 1)
        if acknowledged:
            acknowledge(start, stride)
        before = policy._past_buffer.detach().clone()
        repeated = policy.predict_action(observation(start + stride), temperature=0)
        outputs.append(repeated)
        if acknowledged:
            torch.testing.assert_close(policy._past_buffer, before, rtol=0, atol=0)
            acknowledge(start + stride, min(policy.n_action_steps, end - start - stride))
        else:
            expected = torch.cat((before, repeated["action"]), dim=1)[:, -policy.past_n:]
            torch.testing.assert_close(policy._past_buffer, expected, rtol=0, atol=0)
    for result in outputs:
        for key, shape in (("action", (1, 8, 7)), ("action_pred", (1, 16, 7))):
            if tuple(result[key].shape) != shape or not torch.isfinite(result[key]).all():
                raise AssertionError(f"Invalid inference output for {key}")
    return {"validation_episode": episode,
            "explicit_and_stateful_inference_finite": True,
            "history_source": "acknowledged_dataset_commands" if acknowledged else "predicted_actions",
            "state_history_checked": hasattr(policy, "state_history_keys"),
            "stateful_observation_stride": stride,
            "action_shape": [1, 8, 7], "prediction_shape": [1, 16, 7]}


def check_checkpoint(checkpoint, device):
    import torch
    import zarr
    from oat.policy.base_policy import BasePolicy
    from oat.tokenizer.oat.tokenizer import OATTok

    torch.set_num_threads(4)
    policy, cfg = BasePolicy.from_checkpoint(
        str(checkpoint), weights="ema", return_configuration=True)
    source_tokenizer = OATTok.from_checkpoint(cfg.policy.action_tokenizer.checkpoint)
    expected = source_tokenizer.state_dict()
    actual = policy.action_tokenizer.state_dict()
    if expected.keys() != actual.keys():
        raise AssertionError("Tokenizer state keys changed during policy training")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
    if any(p.requires_grad for p in policy.action_tokenizer.parameters()):
        raise AssertionError("Policy tokenizer is not frozen")
    if policy.action_dim != 7 or policy.n_action_steps != 8 or policy.past_n != 7:
        raise AssertionError("Unexpected action/history dimensions")
    policy.to(device).eval()

    root = zarr.open(str(cfg.task.policy.dataset.zarr_path), mode="r")
    inference = _check_dataset_inference(policy, cfg, root, device)
    return {"checkpoint": str(checkpoint.resolve()), "dataset": str(cfg.task.policy.dataset.zarr_path),
            "tokenizer_unchanged": True, "tokenizer_frozen": True,
            **inference, "device": device}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_checkpoint(args.checkpoint, args.device)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
