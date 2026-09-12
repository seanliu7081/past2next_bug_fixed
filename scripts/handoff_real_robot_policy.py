"""Hand completed, gated nut/washer tokenizers to prepared single-GPU policy jobs."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_real_robot import best_tokenizer_checkpoint
from train_real_robot_policy import _command_config, sha256_file, verify_tokenizer


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def supervisor(service, action="status"):
    result = subprocess.run(["supervisorctl", action, service], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    if action != "status":
        if result.returncode:
            raise RuntimeError(f"Supervisor {action} {service} failed: {result.stdout.strip()}")
        return result.stdout.strip()
    fields = result.stdout.strip().split()
    if len(fields) < 2 or fields[0] != service or fields[1] not in {
            "RUNNING", "STARTING", "STOPPED", "STOPPING", "EXITED", "FATAL", "BACKOFF"}:
        raise RuntimeError(f"Cannot read Supervisor state for {service}: {result.stdout.strip()}")
    match = re.search(r"\bpid (\d+)", result.stdout)
    return {"state": fields[1], "pid": int(match.group(1)) if match else None}


def process_parent(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
        raise ValueError(f"Gate PID {pid} is not live")
    return int(fields[1])


def descends_from(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = process_parent(pid)
    return False


def validate_plan(plan):
    if plan.get("schema_version") != 1 or not plan.get("jobs"):
        raise ValueError("Expected a schema-version-1 handoff plan with jobs")
    if plan.get("current_jobs_registry") and not Path(plan["current_jobs_registry"]).is_absolute():
        raise ValueError("Current-job registry path must be absolute")
    seen_services, seen_destinations = set(), set()
    for job in plan["jobs"]:
        if job.get("task") != "nut_washer" or job.get("variant") not in ("current", "left_noise"):
            raise ValueError("This handoff worker only accepts the two nut/washer variants")
        if job.get("expected_world_size") != 2 or type(job.get("gpu")) is not int or job["gpu"] < 0:
            raise ValueError("Expected two legacy ranks and one nonnegative destination GPU")
        if not re.fullmatch(r"[a-f0-9]{64}", job.get("request_sha256", "")):
            raise ValueError("Invalid handoff request SHA-256")
        for key in ("legacy_service", "destination_service"):
            if not re.fullmatch(r"real-robot-[A-Za-z0-9_.-]+", job.get(key, "")):
                raise ValueError(f"Invalid experiment service name: {job.get(key)}")
            if job[key] in seen_services:
                raise ValueError("Handoff plan service names must be unique")
            seen_services.add(job[key])
        for key in ("legacy_output_dir", "legacy_policy_dir", "marker_dir", "destination_output_dir"):
            if not Path(job[key]).is_absolute():
                raise ValueError(f"Handoff {key} must be absolute")
        if Path(job["legacy_policy_dir"]) != Path(job["legacy_output_dir"]) / "policy":
            raise ValueError("Legacy policy directory must belong to the legacy output directory")
        if job["destination_output_dir"] in seen_destinations:
            raise ValueError("Destination directories must be unique")
        seen_destinations.add(job["destination_output_dir"])
        if Path(job["destination_output_dir"]) == Path(job["legacy_output_dir"]):
            raise ValueError("Destination output must preserve the legacy output directory")


def validate_request(job):
    path = Path(job["legacy_output_dir"]) / "policy_handoff.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != job["request_sha256"]:
        raise ValueError("Handoff request changed after the plan was prepared")
    request = json.loads(raw)
    expected = {"schema_version": 1, "state": "armed", **{key: job[key] for key in (
        "legacy_policy_dir", "task", "variant", "expected_world_size",
        "destination_service", "destination_output_dir", "marker_dir")}}
    for key, value in expected.items():
        if request.get(key) != value:
            raise ValueError(f"Handoff request mismatch: {key}")
    return path


def verify_markers(job, legacy_pid):
    request_path = validate_request(job)
    markers = []
    for rank in range(job["expected_world_size"]):
        path = Path(job["marker_dir"]) / f"rank-{rank}.json"
        if not path.is_file():
            return None
        marker = read_json(path)
        expected = {"schema_version": 1, "state": "waiting_for_supervisor_handoff",
                    "rank": rank, "local_rank": rank, "world_size": 2,
                    "request_path": str(request_path), **{key: job[key] for key in (
                        "legacy_policy_dir", "request_sha256", "task", "variant",
                        "destination_service", "destination_output_dir")}}
        for key, value in expected.items():
            if marker.get(key) != value:
                raise ValueError(f"Gate rank {rank} mismatch: {key}")
        pid, ppid = marker.get("pid"), marker.get("ppid")
        if type(pid) is not int or type(ppid) is not int or pid <= 1 or ppid <= 1:
            raise ValueError("Gate marker has invalid process IDs")
        if process_parent(pid) != ppid or not descends_from(pid, legacy_pid):
            raise ValueError(f"Gate rank {rank} is not a current child of the legacy service")
        markers.append(marker)
    if len({marker["pid"] for marker in markers}) != 2:
        raise ValueError("Both gate markers must identify distinct live rank processes")
    return markers


def verify_finished_tokenizer(job, status):
    legacy = Path(job["legacy_output_dir"])
    frozen = legacy / "frozen_tokenizer.ckpt"
    source = Path(status["tokenizer_source"])
    if Path(status["frozen_tokenizer"]) != frozen or source.parent != legacy / "tokenizer/checkpoints":
        raise ValueError("Legacy tokenizer source/frozen paths do not match this job")
    finished = False
    with (legacy / "tokenizer/logs.json").open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                continue
            record = json.loads(line)
            metric = record.get("test_reconst_mse")
            if record.get("epoch") == 5000 and isinstance(metric, (int, float)) and math.isfinite(metric):
                finished = True
    if not finished:
        raise ValueError("Tokenizer did not record its completed final epoch 5000")
    best, metric = best_tokenizer_checkpoint(legacy / "tokenizer")
    if best.resolve() != source.resolve():
        raise ValueError("Legacy tokenizer selection is not the best retained full-precision checkpoint")
    digest = sha256_file(source)
    if sha256_file(frozen) != digest:
        raise ValueError("Frozen tokenizer differs from the selected trained checkpoint")
    metadata = verify_tokenizer(frozen, job["task"], job["variant"])
    return {"source": str(source), "frozen": str(frozen), "sha256": digest,
            "reconstruction_mse": metric, "completed_final_epoch": 5000, "metadata": metadata}


def validate_destination(job, tokenizer):
    output = Path(job["destination_output_dir"])
    path = output / "manifest.json"
    if not path.is_file():
        return False
    manifest = read_json(path)
    for key, value in (("task", job["task"]), ("variant", job["variant"]), ("gpu", job["gpu"]),
                       ("output_dir", str(output)), ("policy_config", "train_past2next_scratch_all500"),
                       ("policy_epochs", 251), ("smoke", False), ("global_batch_size", 64)):
        if manifest.get(key) != value:
            raise ValueError(f"Existing destination manifest mismatch: {key}")
    if manifest.get("wandb", {}).get("mode") != "online":
        raise ValueError("Destination is not the prepared online policy run")
    for key in ("entity", "project"):
        if key in job.get("wandb", {}) and manifest["wandb"].get(key) != job["wandb"][key]:
            raise ValueError(f"Destination W&B {key} differs from the prepared destination")
    command = manifest.get("commands", {}).get("policy", [])
    if "--nproc_per_node=1" not in command or "--config-name=train_past2next_scratch_all500" not in command:
        raise ValueError("Destination command is not a single-GPU all500 policy")
    cfg = _command_config(command)
    if (cfg.training.checkpoint_every != 20 or cfg.training.snapshot_every != 0
            or cfg.checkpoint.topk.k != 0 or cfg.checkpoint.get("save_all") is not True
            or cfg.checkpoint.topk.monitor_key != "test_reconst_mse"
            or cfg.checkpoint.topk.mode != "min"
            or cfg.checkpoint.topk.format_str != "ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt"):
        raise ValueError("Destination command does not preserve every-20-epoch MSE checkpoint settings")
    if cfg.dataloader.batch_size != 64 or cfg.val_dataloader.batch_size != 64:
        raise ValueError("Destination command must use one GPU with batch size 64")
    copied = manifest["tokenizer"]
    if (copied.get("source") != tokenizer["frozen"] or copied.get("source_sha256") != tokenizer["sha256"]
            or copied.get("copied_sha256") != tokenizer["sha256"]
            or Path(copied.get("copied_path", "")) != output / "frozen_tokenizer.ckpt"
            or sha256_file(output / "frozen_tokenizer.ckpt") != tokenizer["sha256"]):
        raise ValueError("Destination does not use this exact completed tokenizer")
    return True


class HandoffWorker:
    def __init__(self, plan_path):
        self.plan_path = plan_path.resolve()
        self.plan = read_json(self.plan_path)
        validate_plan(self.plan)
        self.status_path = self.plan_path.parent / "handoff_status.json"
        digest = sha256_file(self.plan_path)
        self.status = read_json(self.status_path) if self.status_path.exists() else {
            "schema_version": 1, "plan_sha256": digest, "jobs": {}}
        if self.status.get("plan_sha256") != digest:
            raise ValueError("Existing handoff status belongs to a different plan")

    def save(self):
        self.status["updated_at"] = now()
        write_json(self.status_path, self.status)

    def advance(self, job):
        entry = self.status["jobs"].setdefault(job["legacy_service"], {"state": "waiting_tokenizer"})
        if entry["state"] == "handed_off":
            return True
        validate_request(job)
        legacy_path = Path(job["legacy_output_dir"]) / "status.json"
        legacy_status = read_json(legacy_path)
        old = supervisor(job["legacy_service"])
        if "verified_tokenizer" not in entry:
            if legacy_status.get("state") == "training_tokenizer":
                entry["state"] = "waiting_tokenizer"
                return False
            if legacy_status.get("state") != "training_policy":
                raise ValueError(f"Unexpected legacy stage: {legacy_status.get('state')}")
            if not all(legacy_status.get(key) and Path(legacy_status[key]).is_file()
                       for key in ("tokenizer_source", "frozen_tokenizer")):
                entry["state"] = "waiting_tokenizer_handoff_files"
                return False
            if old["state"] != "RUNNING" or old["pid"] is None:
                raise ValueError("Legacy policy must be live and rank-gated before handoff")
            markers = verify_markers(job, old["pid"])
            if markers is None:
                entry["state"] = "waiting_policy_gate"
                return False
            tokenizer = verify_finished_tokenizer(job, legacy_status)
            entry.update(state="validated", verified_tokenizer=tokenizer,
                         legacy_pid=old["pid"], gate_markers=markers, request_sha256=job["request_sha256"])
            self.save()
        tokenizer = entry["verified_tokenizer"]
        if entry.get("request_sha256") != job["request_sha256"]:
            raise ValueError("Saved handoff validation does not match this request")
        if any(sha256_file(Path(tokenizer[key])) != tokenizer["sha256"] for key in ("source", "frozen")):
            raise ValueError("Completed tokenizer changed after handoff validation")
        old = supervisor(job["legacy_service"])
        if old["state"] == "RUNNING":
            if old["pid"] != entry["legacy_pid"] or verify_markers(job, old["pid"]) is None:
                raise ValueError("Legacy service or gate changed after handoff validation")
            entry.update(state="stop_requested", stop_requested_at=now())
            self.save()  # Durable intent permits recovery if the worker dies after stop.
            supervisor(job["legacy_service"], "stop")
            old = supervisor(job["legacy_service"])
        if old["state"] not in ("STOPPED", "EXITED"):
            raise ValueError(f"Legacy service did not stop: {old['state']}")
        if entry["state"] not in ("stop_requested", "legacy_stopped", "starting_destination", "waiting_destination"):
            raise ValueError("Legacy stopped without this worker's recorded stop intent")
        entry["state"] = "legacy_stopped"
        self.save()
        legacy_status = read_json(legacy_path)
        legacy_status.update(state="redirected_to_single_gpu_policy",
                             destination_service=job["destination_service"],
                             destination_output_dir=job["destination_output_dir"], destination_gpu=job["gpu"],
                             handoff_request_sha256=job["request_sha256"], updated_at=now())
        write_json(legacy_path, legacy_status)
        destination = supervisor(job["destination_service"])
        valid_manifest = validate_destination(job, tokenizer)
        destination_status = Path(job["destination_output_dir"]) / "status.json"
        completed = (valid_manifest and destination_status.is_file()
                     and read_json(destination_status).get("state") == "completed")
        if (destination["state"] == "RUNNING" and valid_manifest) or completed:
            self.update_registry(job, tokenizer)
            entry.update(state="handed_off", handed_off_at=now(), destination_service=job["destination_service"],
                         destination_output_dir=job["destination_output_dir"], destination_state=destination["state"])
            self.save()
            return True
        if destination["state"] in ("RUNNING", "STARTING"):
            entry["state"] = "waiting_destination"
            return False
        if Path(job["destination_output_dir"]).exists():
            raise ValueError("Destination output already exists but is not running/completed; manual resume required")
        if destination["state"] != "STOPPED":
            raise ValueError(f"Prepared destination service is not stopped: {destination['state']}")
        entry.update(state="starting_destination", destination_start_requested_at=now())
        self.save()
        supervisor(job["destination_service"], "start")
        entry["state"] = "waiting_destination"
        self.save()
        return False

    def update_registry(self, job, tokenizer):
        registry = self.plan.get("current_jobs_registry")
        if not registry:
            return
        path = Path(registry)
        rows = read_json(path)
        matches = [index for index, row in enumerate(rows)
                   if row.get("name") in (job["legacy_service"], job["destination_service"])]
        if len(matches) != 1:
            raise ValueError("Current-job registry must contain exactly one row for this handoff")
        index = matches[0]
        row = dict(rows[index])
        row.update({key: job[key] for key in ("task", "variant", "command", "wrapper", "supervisor_config")
                    if key in job})
        row.update(name=job["destination_service"], output_dir=job["destination_output_dir"],
                   state="training_policy", checkpoint_metric="test_reconst_mse",
                   checkpoint_format="ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt",
                   gpu=str(job["gpu"]), gpus=str(job["gpu"]),
                   previous_service=job["legacy_service"], previous_output_dir=job["legacy_output_dir"],
                   tokenizer_output_dir=job["legacy_output_dir"], source_tokenizer=tokenizer["frozen"],
                   source_tokenizer_sha256=tokenizer["sha256"])
        row["wandb"] = read_json(Path(job["destination_output_dir"]) / "manifest.json")["wandb"]
        for key in ("checkpoint_every", "checkpoint_retention"):
            if key in job:
                row[key] = job[key]
        rows[index] = row
        write_json(path, rows)

    def run_once(self):
        results, errors = [], {}
        for job in self.plan["jobs"]:
            entry = self.status["jobs"].setdefault(job["legacy_service"], {"state": "waiting_tokenizer"})
            try:
                results.append(self.advance(job))
                entry.pop("error", None)
                entry.pop("error_at", None)
            except Exception as exc:
                # Preserve durable stop/start intent so this job remains recoverable.
                entry.update(error=str(exc), error_at=now())
                errors[job["legacy_service"]] = str(exc)
                results.append(False)
            self.save()
        completed = all(results)
        self.status.update(state="completed" if completed else "failed" if errors else "waiting",
                           errors=errors)
        self.status.pop("error", None)
        self.save()
        return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Process one handoff pass, then exit")
    args = parser.parse_args()
    with args.plan.resolve().with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = HandoffWorker(args.plan)
        while True:
            completed = worker.run_once()
            print(json.dumps({"state": worker.status["state"], "jobs": {
                key: entry["state"] for key, entry in worker.status["jobs"].items()}}), flush=True)
            if args.once and worker.status["state"] == "failed":
                raise SystemExit(1)
            if completed or args.once:
                return
            time.sleep(15)


if __name__ == "__main__":
    main()
