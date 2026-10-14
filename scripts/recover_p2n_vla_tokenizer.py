#!/usr/bin/env python3
"""Recover the embedded frozen OAT from a trusted P2N-VLA checkpoint into a new file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dill
import hydra
from omegaconf import OmegaConf
import torch

from oat.model.vla.pi05_checkpoint import file_sha256
from oat.tokenizer.oat.latent_adapter import FrozenOATLatentAdapter


def recover_tokenizer(checkpoint, output, *, weights="ema"):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing file: {output}")
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    config = payload["policy_config"]["tokenizer_config"]
    original = dict(payload["policy_config"].get("tokenizer_metadata") or {})
    state_key = {"ema": "ema_model", "model": "model"}[weights]
    prefix = "latent_adapter.tokenizer."
    state = {key[len(prefix):]: value.detach().clone()
             for key, value in payload["state_dicts"][state_key].items() if key.startswith(prefix)}
    if not state:
        raise ValueError(f"No embedded OAT tensors in {state_key}")
    tokenizer = hydra.utils.instantiate(config)
    tokenizer.load_state_dict(state, strict=True)
    FrozenOATLatentAdapter(tokenizer)
    provenance = {
        "recovery": "embedded_frozen_oat_from_p2n_vla",
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_weights": state_key, "original_tokenizer": original,
        "tensor_count": len(state), "parameter_and_buffer_elements": sum(t.numel() for t in state.values()),
    }
    result = {"cfg": OmegaConf.create({"tokenizer": config, "task": original.get("training_task", {})}),
              "state_dicts": {"ema_model": state}, "metadata": provenance}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(result, stream, pickle_module=dill)
    return {"output": str(output), "sha256": file_sha256(output), **provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted P2N-VLA artifact containing frozen OAT")
    parser.add_argument("--output", required=True, help="New tokenizer checkpoint path; never overwritten")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    args = parser.parse_args()
    print(json.dumps(recover_tokenizer(args.checkpoint, args.output, weights=args.weights), indent=2))


if __name__ == "__main__":
    main()
