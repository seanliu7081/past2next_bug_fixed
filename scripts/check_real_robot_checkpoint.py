"""Reload a trusted local real-robot checkpoint and verify dataset inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_checkpoint(checkpoint, device):
    import numpy as np
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
    from oat.common.seq_sampler import get_val_mask
    ends = root["meta/episode_ends"][:]
    validation = get_val_mask(len(ends), cfg.task.policy.dataset.val_ratio,
                              cfg.task.policy.dataset.seed)
    episode = int(np.flatnonzero(validation)[0])
    start = 0 if episode == 0 else int(ends[episode - 1])
    anchor = start + 16
    obs = {}
    for key in cfg.task.policy.dataset.obs_keys:
        values = root[f"data/{key}"][anchor - 1:anchor + 1]
        if values.dtype.kind == "f":
            values = values.astype(np.float32)
        obs[key] = torch.from_numpy(values.copy()).unsqueeze(0).to(device)
    past = torch.from_numpy(root["data/action"][anchor - 7:anchor].copy()).unsqueeze(0).to(device)
    policy.reset()
    with torch.inference_mode():
        explicit = policy.predict_action(obs, past_actions=past, temperature=0)
        if policy._past_buffer is not None:
            raise AssertionError("Explicit history mutated the rollout buffer")
        stateful = policy.predict_action(obs, temperature=0)
        repeated = policy.predict_action(obs, temperature=0)
    for result in (explicit, stateful, repeated):
        for key, shape in (("action", (1, 8, 7)), ("action_pred", (1, 16, 7))):
            if tuple(result[key].shape) != shape or not torch.isfinite(result[key]).all():
                raise AssertionError(f"Invalid inference output for {key}")
    torch.testing.assert_close(policy._past_buffer, repeated["action_pred"][:, 1:8])
    return {"checkpoint": str(checkpoint.resolve()), "dataset": str(cfg.task.policy.dataset.zarr_path),
            "validation_episode": episode, "tokenizer_unchanged": True,
            "tokenizer_frozen": True, "explicit_and_stateful_inference_finite": True,
            "action_shape": [1, 8, 7], "prediction_shape": [1, 16, 7],
            "device": device}


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
