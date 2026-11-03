"""Auditable, reversible weight edits for PI0.5's final flow-velocity projection.

These utilities perform no fitting, backward pass, or optimization. A patch is an
explicit rank-one update chosen by an experimenter from diagnostic evidence:
``delta_weight[rows] = coefficients[:, None] * direction[None, :]``.
The coefficients act on *velocity*, not final actions: PI0.5 integrates with
negative time steps. A negative z-velocity correction initially raises the sampled
z action; later denoising feedback means the final displacement is not exact.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


TARGET = "flow_heads.action_out_proj"
SCHEMA_VERSION = 1


def tensor_sha256(value: Tensor) -> str:
    """Fingerprint exact tensor dtype, shape, and bytes, including bfloat16."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_patch_spec(path: str | Path) -> dict[str, Any]:
    """Read a JSON patch, resolving its optional NPZ/NPY direction relative to it."""
    path = Path(path).expanduser().resolve()
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError("A patch specification must be a JSON object")
    if "direction_file" in spec:
        direction = Path(spec["direction_file"]).expanduser()
        spec["direction_file"] = str((path.parent / direction).resolve())
    spec["spec_path"] = str(path)
    spec["spec_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return spec


def _head(policy) -> nn.Linear:
    head = policy.flow_heads.action_out_proj
    if not isinstance(head, nn.Linear) or head.bias is None:
        raise TypeError(f"{TARGET} must be an nn.Linear with bias")
    return head


def _direction(spec: dict[str, Any], width: int) -> Tensor:
    if ("direction" in spec) == ("direction_file" in spec):
        raise ValueError("Specify exactly one of direction or direction_file")
    if "direction" in spec:
        values = np.asarray(spec["direction"], dtype=np.float32)
    else:
        path = Path(spec["direction_file"])
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                values = np.asarray(loaded[spec.get("direction_key", "direction")], dtype=np.float32)
        else:
            values = np.asarray(loaded, dtype=np.float32)
    direction = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    if tuple(direction.shape) != (width,) or not torch.isfinite(direction).all():
        raise ValueError(f"direction must contain {width} finite numbers")
    norm = float(direction.double().norm())
    if norm == 0:
        raise ValueError("direction must be nonzero")
    normalize = spec.get("normalize_direction", True)
    if not isinstance(normalize, bool):
        raise TypeError("normalize_direction must be a bool")
    return direction / norm if normalize else direction


@dataclass
class AppliedPatch:
    """An in-memory edit and its exact original values; never writes a checkpoint."""

    head: nn.Linear
    original_weight: Tensor
    original_bias: Tensor
    provenance: dict[str, Any]
    restored: bool = False

    @torch.no_grad()
    def restore(self) -> None:
        if self.restored:
            return
        expected = self.provenance["after_sha256"]
        if tensor_sha256(self.head.weight) != expected["weight"] or \
                tensor_sha256(self.head.bias) != expected["bias"]:
            raise RuntimeError("The head changed after this patch; refusing to overwrite another edit")
        self.head.weight.copy_(self.original_weight)
        self.head.bias.copy_(self.original_bias)
        self.restored = True


@torch.no_grad()
def apply_output_head_patch(policy, spec: dict[str, Any]) -> AppliedPatch:
    """Apply an explicit sparse-row rank-one patch and return an audited undo handle.

    Required fields: schema_version=1, kind='action_output_rank_one', rows,
    coefficients, and direction or direction_file. Optional bias_delta aligns with
    rows. Rows must be distinct *real* action dimensions (padding is protected).
    Optional expected_weight_sha256/expected_bias_sha256 pin the source tensors.
    """
    if spec.get("schema_version") != SCHEMA_VERSION or spec.get("kind") != "action_output_rank_one":
        raise ValueError("Expected schema_version=1 and kind=action_output_rank_one")
    if spec.get("target", TARGET) != TARGET:
        raise ValueError(f"This patch type only supports {TARGET}")
    head = _head(policy)
    if not torch.isfinite(head.weight).all() or not torch.isfinite(head.bias).all():
        raise ValueError("Cannot patch nonfinite source weights")
    rows = spec.get("rows")
    action_dim = int(getattr(policy, "action_dim", head.out_features))
    if not isinstance(rows, list) or not rows or any(
            isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < action_dim for row in rows):
        raise ValueError(f"rows must list integer action indices in [0, {action_dim})")
    if len(set(rows)) != len(rows):
        raise ValueError("rows must be unique")
    coefficients = torch.as_tensor(spec.get("coefficients", []), dtype=torch.float32)
    bias_delta = torch.as_tensor(spec.get("bias_delta", [0.0] * len(rows)), dtype=torch.float32)
    for name, value in (("coefficients", coefficients), ("bias_delta", bias_delta)):
        if tuple(value.shape) != (len(rows),) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must have one finite value per selected row")
    direction = _direction(spec, head.in_features)
    before = {"weight": tensor_sha256(head.weight), "bias": tensor_sha256(head.bias)}
    for name, value in before.items():
        expected = spec.get(f"expected_{name}_sha256")
        if expected is not None and expected != value:
            raise ValueError(f"Source {name} fingerprint does not match the patch")
    original_weight = head.weight.detach().clone()
    original_bias = head.bias.detach().clone()
    proposed_weight = original_weight.float().clone()
    proposed_bias = original_bias.float().clone()
    row_index = torch.tensor(rows, dtype=torch.long, device=head.weight.device)
    proposed_weight[row_index] += (coefficients[:, None] * direction[None, :]).to(head.weight.device)
    proposed_bias[row_index] += bias_delta.to(head.bias.device)
    proposed_weight = proposed_weight.to(head.weight.dtype)
    proposed_bias = proposed_bias.to(head.bias.dtype)
    if not torch.isfinite(proposed_weight).all() or not torch.isfinite(proposed_bias).all():
        raise ValueError("Patch would produce nonfinite parameters")
    # All validation finishes before either parameter is changed.
    head.weight.copy_(proposed_weight)
    head.bias.copy_(proposed_bias)
    delta_weight = head.weight.float() - original_weight.float()
    delta_bias = head.bias.float() - original_bias.float()
    changed_weight = int(torch.count_nonzero(head.weight != original_weight))
    changed_bias = int(torch.count_nonzero(head.bias != original_bias))
    provenance = {
        "schema_version": SCHEMA_VERSION, "kind": "action_output_rank_one", "target": TARGET,
        "description": str(spec.get("description", "")), "rows": rows,
        "coefficients": coefficients.tolist(), "bias_delta": bias_delta.tolist(),
        "normalize_direction": spec.get("normalize_direction", True),
        "direction_sha256": tensor_sha256(direction), "direction_norm": float(direction.double().norm()),
        "direction": direction.tolist(), "before_sha256": before,
        "after_sha256": {"weight": tensor_sha256(head.weight), "bias": tensor_sha256(head.bias)},
        "changed_weight_values": changed_weight, "changed_bias_values": changed_bias,
        "changed_values": changed_weight + changed_bias,
        "weight_delta_l2": float(delta_weight.double().norm()),
        "bias_delta_l2": float(delta_bias.double().norm()),
        "weight_delta_max_abs": float(delta_weight.abs().max()),
        "bias_delta_max_abs": float(delta_bias.abs().max()),
        "weight_shape": list(head.weight.shape), "weight_dtype": str(head.weight.dtype),
        "interpretation": "Static velocity-projection edit; final action effect depends on Euler feedback",
        "gradient_updates": 0,
    }
    for key in ("spec_path", "spec_sha256", "direction_file", "direction_key"):
        if key in spec:
            provenance[key] = spec[key]
    return AppliedPatch(head, original_weight, original_bias, provenance)


class OutputHeadCapture:
    """Capture projection input/output at each denoising step without changing them.

    ``pop()`` returns CPU numpy ``hidden[S,B,H,D]`` and ``velocity[S,B,H,A]``.
    Clear before a prediction, pop after it, and close when finished. Captures stay
    on CPU so a full rollout does not retain GPU activations or autograd graphs.
    """

    def __init__(self, policy, expected_steps: int | None = None):
        self.expected_steps = expected_steps if expected_steps is not None else getattr(policy, "flow_num_steps", None)
        self.records: list[tuple[Tensor, Tensor]] = []
        self.handle = _head(policy).register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        hidden = inputs[0]
        if hidden.ndim != 3 or output.ndim != 3 or hidden.shape[:2] != output.shape[:2]:
            raise ValueError("Expected projection inputs and outputs with shape [batch, horizon, width]")
        self.records.append((hidden.detach().float().cpu().clone(), output.detach().float().cpu().clone()))

    def clear(self) -> None:
        self.records.clear()

    def pop(self) -> dict[str, Any]:
        count = len(self.records)
        if not count:
            raise RuntimeError("No output-head calls captured")
        if self.expected_steps is not None and count != self.expected_steps:
            raise RuntimeError(f"Expected {self.expected_steps} denoising calls, captured {count}")
        result = {
            "hidden": torch.stack([record[0] for record in self.records]).numpy(),
            "velocity": torch.stack([record[1] for record in self.records]).numpy(),
            "flow_steps": count,
        }
        self.clear()
        return result

    def close(self) -> None:
        self.handle.remove()
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@torch.no_grad()
def compare_action_chunks(original: Tensor, edited: Tensor, normalizer) -> dict[str, Any]:
    """Compare paired predictions using the same observation, history, and noise.

    ``original`` and ``edited`` are raw action chunks. ``normalizer`` is the
    policy's action_normalizer['action']. This function performs no policy call.
    """
    if original.shape != edited.shape or original.ndim != 3:
        raise ValueError("Action chunks must have matching [batch, horizon, action_dim] shapes")
    if not torch.isfinite(original).all() or not torch.isfinite(edited).all():
        raise ValueError("Action chunks must be finite")
    raw_delta = edited.float() - original.float()
    normalized_delta = normalizer.normalize(edited.float()) - normalizer.normalize(original.float())
    return {
        "raw_delta": raw_delta.detach().cpu().numpy(),
        "normalized_delta": normalized_delta.detach().cpu().numpy(),
        "raw_l2": float(raw_delta.double().norm()),
        "normalized_l2": float(normalized_delta.double().norm()),
        "raw_max_abs": float(raw_delta.abs().max()),
        "normalized_max_abs": float(normalized_delta.abs().max()),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_output_head(payload: dict[str, Any], weights: str = "model"):
    """Build only the output projection, in the dtype used by restored PI05PastPolicy.

    Snapshot masters can be bf16, but the policy's trainable projection is restored
    into fp32 nn.Linear parameters. This small CPU object reproduces that conversion
    without loading the backbone and is suitable for source-hash checks/export.
    """
    if payload.get("format") != "p2n_vla_checkpoint_v1":
        raise ValueError("Expected a p2n_vla_checkpoint_v1 artifact")
    target = payload.get("policy_config", {}).get("_target_")
    if target != "oat.policy.pi05_past.Pi05PastPolicy":
        raise ValueError("Export currently supports Pi05PastPolicy checkpoints only")
    key = {"model": "model", "ema": "ema_model"}.get(weights)
    if key is None or key not in payload["state_dicts"]:
        raise ValueError(f"Checkpoint does not contain requested {weights!r} weights")
    state = payload["state_dicts"][key]
    weight, bias = state[f"{TARGET}.weight"], state[f"{TARGET}.bias"]
    head = nn.Linear(weight.shape[1], weight.shape[0], dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(weight.float())
        head.bias.copy_(bias.float())
    dim = int(payload["policy_config"]["shape_meta"]["action"]["shape"][0])
    return SimpleNamespace(flow_heads=SimpleNamespace(action_out_proj=head), action_dim=dim), key


def export_edited_snapshot(source: str | Path, destination: str | Path,
                           spec: dict[str, Any], weights: str = "model") -> dict[str, Any]:
    """Materialize a compatible edited snapshot without changing any source bytes.

    Untouched tensors/configuration are preserved. The two edited head tensors are
    saved in fp32 so reload reproduces the runtime patch without bf16 rounding.
    A same-directory temporary file is linked atomically to a never-overwritten
    destination. No backbone, simulator, GPU, gradients, or optimizer is used.
    """
    import dill

    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if source == destination or destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    source_hash = file_sha256(source)
    payload = torch.load(source, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    policy, key = checkpoint_output_head(payload, weights)
    base_state = payload["state_dicts"][key]
    native_hashes = {name: tensor_sha256(base_state[f"{TARGET}.{name}"]) for name in ("weight", "bias")}
    edit = apply_output_head_patch(policy, spec)
    provenance = copy.deepcopy(edit.provenance)
    provenance.update({
        "source_snapshot": str(source), "source_snapshot_sha256": source_hash,
        "source_weights": weights, "source_stored_head_sha256": native_hashes,
        "source_stored_head_dtype": str(base_state[f"{TARGET}.weight"].dtype),
        "exported_head_dtype": "torch.float32",
        "unchanged_tensor_count": len(base_state) - 2,
        "calibration": copy.deepcopy(spec.get("calibration")),
        "note": "Only output projection weight/bias replaced; original artifact unchanged",
    })
    result = dict(payload)
    result["state_dicts"] = dict(payload["state_dicts"])
    updated = dict(base_state)
    updated[f"{TARGET}.weight"] = edit.head.weight.detach().cpu().clone()
    updated[f"{TARGET}.bias"] = edit.head.bias.detach().cpu().clone()
    result["state_dicts"][key] = updated
    result["metadata"] = copy.deepcopy(payload.get("metadata", {}))
    result["metadata"]["weight_space_edit"] = provenance
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(result, stream, pickle_module=dill)
            stream.flush()
            os.fsync(stream.fileno())
        if file_sha256(source) != source_hash:
            raise RuntimeError("Source snapshot changed during export; refusing publication")
        os.link(temporary, destination)  # Exclusive: a racing writer cannot be overwritten.
    finally:
        temporary.unlink(missing_ok=True)
    return {"destination": str(destination), "destination_sha256": file_sha256(destination),
            "provenance": provenance}
