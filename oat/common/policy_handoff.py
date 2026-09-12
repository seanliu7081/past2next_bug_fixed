"""Hold retired policy launches before model or CUDA initialization.

An external supervisor worker performs the handoff. This gate never releases
an old policy, starts a replacement, or changes a tokenizer's execution.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import time

from omegaconf import OmegaConf


def _absolute_path(request, key):
    value = request.get(key)
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"Policy handoff {key} must be an absolute path")
    return Path(value).resolve()


def maybe_wait_for_policy_handoff(cfg, output_dir):
    """Ignore absent requests; valid requests block; invalid ones fail closed."""
    output = Path(output_dir).resolve()
    # Tokenizer and other stage directories are outside this gate's scope.
    if output.name != "policy":
        return
    request_path = output.parent / "policy_handoff.json"
    if not request_path.exists():
        return
    raw_request = request_path.read_bytes()
    try:
        request = json.loads(raw_request)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid policy handoff request: {request_path}") from exc
    if not isinstance(request, dict):
        raise ValueError("Policy handoff request must be an object")
    if type(request.get("schema_version")) is not int or request["schema_version"] != 1:
        raise ValueError("Unsupported policy handoff schema_version")
    if request.get("state") != "armed":
        raise ValueError("Existing policy handoff request must be armed; old policy cannot start")
    if _absolute_path(request, "legacy_policy_dir") != output:
        raise ValueError("Policy handoff legacy_policy_dir does not match this policy")
    if (request.get("task") != "nut_washer"
            or OmegaConf.select(cfg, "_target_") != "oat.workspace.train_policy.TrainPolicyWorkspace"
            or OmegaConf.select(cfg, "task.policy.task_name") != "nut_washer"
            or OmegaConf.select(cfg, "task.policy.name") != "real_robot_nut_washer"):
        raise ValueError("Policy handoff only supports the declared nut_washer policy")
    variant = request.get("variant")
    if variant not in ("current", "left_noise") or variant != output.parent.name:
        raise ValueError("Policy handoff variant does not match its legacy output directory")
    if (type(request.get("expected_world_size")) is not int
            or request["expected_world_size"] != 2):
        raise ValueError("Policy handoff requires expected_world_size=2")
    destination_service = request.get("destination_service")
    if (not isinstance(destination_service, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", destination_service)):
        raise ValueError("Policy handoff destination_service must be a supervisor program name")
    destination = _absolute_path(request, "destination_output_dir")
    if destination in (output, output.parent):
        raise ValueError("Policy handoff destination must preserve the legacy output directory")
    marker_dir = _absolute_path(request, "marker_dir")
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Policy handoff requires an explicit torchrun rank environment") from exc
    if world_size != 2 or rank not in (0, 1) or local_rank != rank:
        raise ValueError("Policy handoff expected two local torchrun ranks")
    marker = {
        "schema_version": 1, "state": "waiting_for_supervisor_handoff",
        "pid": os.getpid(), "ppid": os.getppid(),
        "rank": rank, "local_rank": local_rank, "world_size": world_size,
        "legacy_policy_dir": str(output), "request_path": str(request_path),
        "request_sha256": hashlib.sha256(raw_request).hexdigest(),
        "task": request["task"], "variant": variant,
        "destination_service": destination_service,
        "destination_output_dir": str(destination),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    marker_dir.mkdir(parents=True, exist_ok=True)
    temporary = marker_dir / f".rank-{rank}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(marker, indent=2) + "\n")
    temporary.replace(marker_dir / f"rank-{rank}.json")
    print(f"Policy rank {rank} waiting for supervisor handoff to {destination_service}", flush=True)
    while True:
        time.sleep(5)
