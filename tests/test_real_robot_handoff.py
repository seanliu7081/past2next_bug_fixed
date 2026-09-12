"""Service actions are faked: exercise only the managed handoff decisions."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("handoff_real_robot_policy", ROOT / "scripts/handoff_real_robot_policy.py")
worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker_module)
from train_real_robot_policy import policy_command


@pytest.fixture
def setup(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy/current"
    legacy.mkdir(parents=True)
    markers = tmp_path / "markers"
    markers.mkdir()
    destination = tmp_path / "destination/current"
    job = {"legacy_service": "real-robot-nut-washer-current", "legacy_output_dir": str(legacy),
           "legacy_policy_dir": str(legacy / "policy"), "task": "nut_washer", "variant": "current",
           "expected_world_size": 2, "marker_dir": str(markers),
           "destination_service": "real-robot-nut-washer-current-1gpu",
           "destination_output_dir": str(destination), "gpu": 4,
           "command": ["python", "train_real_robot_policy.py"], "wrapper": "prepared.sh",
           "supervisor_config": "prepared.conf",
           "wandb": {"entity": "test-entity", "project": "real_robot", "mode": "online"}}
    request = {"schema_version": 1, "state": "armed", **{key: job[key] for key in (
        "legacy_policy_dir", "task", "variant", "expected_world_size",
        "destination_service", "destination_output_dir", "marker_dir")}}
    request_path = legacy / "policy_handoff.json"
    request_path.write_text(json.dumps(request))
    job["request_sha256"] = worker_module.sha256_file(request_path)
    for rank, pid in enumerate((110, 111)):
        marker = {"schema_version": 1, "state": "waiting_for_supervisor_handoff", "pid": pid,
                  "ppid": 105, "rank": rank, "local_rank": rank, "world_size": 2,
                  "request_path": str(request_path), **{key: job[key] for key in (
                      "legacy_policy_dir", "request_sha256", "task", "variant",
                      "destination_service", "destination_output_dir")}}
        (markers / f"rank-{rank}.json").write_text(json.dumps(marker))
    parents = {110: 105, 111: 105, 105: 100, 100: 1}
    monkeypatch.setattr(worker_module, "process_parent", lambda pid: parents[pid])
    checkpoint_dir = legacy / "tokenizer/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    source = checkpoint_dir / "ep-4990_mse-0.000.ckpt"
    source.write_bytes(b"completed trained tokenizer")
    frozen = legacy / "frozen_tokenizer.ckpt"
    frozen.write_bytes(source.read_bytes())
    (legacy / "tokenizer/logs.json").write_text(
        '{"epoch": 4990, "test_reconst_mse": 0.0001}\n'
        '{"epoch": 5000, "test_reconst_mse": 0.0002}\n')
    status = {"state": "training_policy", "tokenizer_source": str(source),
              "frozen_tokenizer": str(frozen), "original_field": "preserved"}
    (legacy / "status.json").write_text(json.dumps(status))
    monkeypatch.setattr(worker_module, "verify_tokenizer", lambda *args: {"weights_finite": True})
    registry = tmp_path / "current_jobs.json"
    registry.write_text(json.dumps([{"name": "fruit-service", "task": "fruits"},
                                    {"name": job["legacy_service"], "task": "nut_washer"}]))
    plan_path = tmp_path / "handoff_plan.json"
    plan_path.write_text(json.dumps({"schema_version": 1, "jobs": [job],
                                    "current_jobs_registry": str(registry)}))
    services = {job["legacy_service"]: {"state": "RUNNING", "pid": 100},
                job["destination_service"]: {"state": "STOPPED", "pid": None}}
    actions = []

    def create_destination():
        destination.mkdir(parents=True, exist_ok=True)
        copied = destination / "frozen_tokenizer.ckpt"
        copied.write_bytes(frozen.read_bytes())
        digest = worker_module.sha256_file(frozen)
        manifest = {"task": job["task"], "variant": job["variant"], "gpu": job["gpu"],
                    "output_dir": str(destination), "policy_config": "train_past2next_scratch_all500",
                    "policy_epochs": 251, "smoke": False, "global_batch_size": 64,
                    "wandb": {"mode": "online", "run_id": "fresh-nut-policy",
                              "entity": "test-entity", "project": "real_robot"},
                    "commands": {"policy": policy_command(
                        "nut_washer", "current", destination, copied, "test-entity", "real_robot",
                        "fresh-nut-policy", "prepared-policy")},
                    "tokenizer": {"source": str(frozen), "source_sha256": digest,
                                  "copied_path": str(copied), "copied_sha256": digest}}
        (destination / "manifest.json").write_text(json.dumps(manifest))

    def fake_supervisor(service, action="status"):
        if action == "status":
            return dict(services[service])
        actions.append((action, service))
        if action == "stop":
            services[service] = {"state": "STOPPED", "pid": None}
        elif action == "start":
            create_destination()
            services[service] = {"state": "RUNNING", "pid": 200}

    monkeypatch.setattr(worker_module, "supervisor", fake_supervisor)
    return {"worker": worker_module.HandoffWorker(plan_path), "plan_path": plan_path,
            "job": job, "legacy": legacy, "status": status, "markers": markers,
            "source": source, "frozen": frozen, "actions": actions, "services": services,
            "supervisor": fake_supervisor, "create_destination": create_destination,
            "registry": registry, "parents": parents}


def test_tokenizer_in_flight_is_never_stopped(setup):
    (setup["legacy"] / "status.json").write_text('{"state": "training_tokenizer"}')
    assert setup["worker"].run_once() is False
    assert setup["actions"] == []


def test_complete_handoff_preserves_artifacts_updates_registry_and_is_idempotent(setup):
    assert setup["worker"].run_once() is False  # Starts destination; verifies it next pass.
    assert setup["worker"].run_once() is True
    job = setup["job"]
    assert setup["actions"] == [("stop", job["legacy_service"]), ("start", job["destination_service"])]
    legacy_status = json.loads((setup["legacy"] / "status.json").read_text())
    assert legacy_status["state"] == "redirected_to_single_gpu_policy"
    assert legacy_status["original_field"] == "preserved"
    assert setup["source"].read_bytes() == setup["frozen"].read_bytes()
    rows = json.loads(setup["registry"].read_text())
    assert rows[0] == {"name": "fruit-service", "task": "fruits"}
    assert rows[1]["name"] == job["destination_service"]
    assert rows[1]["state"] == "training_policy"
    assert rows[1]["tokenizer_output_dir"] == job["legacy_output_dir"]
    assert worker_module.HandoffWorker(setup["plan_path"]).run_once() is True
    assert len(setup["actions"]) == 2


@pytest.mark.parametrize("fault", ["wrong_digest", "stale_pid", "missing_rank"])
def test_unready_or_mismatched_rank_markers_never_stop_legacy(setup, fault):
    marker_path = setup["markers"] / "rank-1.json"
    if fault == "missing_rank":
        marker_path.unlink()
        assert setup["worker"].run_once() is False
    else:
        if fault == "wrong_digest":
            marker = json.loads(marker_path.read_text())
            marker["request_sha256"] = "0" * 64
            marker_path.write_text(json.dumps(marker))
        else:
            setup["parents"][105] = 999
            setup["parents"][999] = 1
        assert setup["worker"].run_once() is False
        assert setup["worker"].status["errors"]
    assert setup["actions"] == []


def test_unfinished_tokenizer_and_changed_weights_fail_closed(setup):
    log = setup["legacy"] / "tokenizer/logs.json"
    original = log.read_text()
    log.write_text('{"epoch": 4990, "test_reconst_mse": 0.0001}\n')
    assert setup["worker"].run_once() is False
    assert "final epoch" in setup["worker"].status["errors"][setup["job"]["legacy_service"]]
    log.write_text(original)
    setup["frozen"].write_bytes(b"wrong tokenizer")
    assert setup["worker"].run_once() is False
    assert "differs" in setup["worker"].status["errors"][setup["job"]["legacy_service"]]
    assert setup["actions"] == []


def test_crash_after_legacy_stop_recovers_without_repeating_stop(setup, monkeypatch):
    def crash_on_start(service, action="status"):
        if action == "start":
            raise RuntimeError("simulated worker interruption")
        return setup["supervisor"](service, action)

    monkeypatch.setattr(worker_module, "supervisor", crash_on_start)
    assert setup["worker"].run_once() is False
    assert "interruption" in setup["worker"].status["errors"][setup["job"]["legacy_service"]]
    assert setup["actions"] == [("stop", setup["job"]["legacy_service"])]
    monkeypatch.setattr(worker_module, "supervisor", setup["supervisor"])
    recovered = worker_module.HandoffWorker(setup["plan_path"])
    assert recovered.run_once() is False
    assert recovered.run_once() is True
    assert setup["actions"] == [("stop", setup["job"]["legacy_service"]),
                                 ("start", setup["job"]["destination_service"])]


def test_existing_destination_is_never_restarted(setup):
    setup["create_destination"]()
    setup["services"][setup["job"]["destination_service"]] = {"state": "RUNNING", "pid": 200}
    assert setup["worker"].run_once() is True
    assert setup["actions"] == [("stop", setup["job"]["legacy_service"])]


def test_failure_of_one_job_does_not_starve_another(setup, monkeypatch):
    worker = setup["worker"]
    other = {**setup["job"], "legacy_service": "real-robot-nut-washer-left-noise"}
    worker.plan["jobs"].append(other)
    visited = []

    def advance(job):
        visited.append(job["legacy_service"])
        if job["legacy_service"] == setup["job"]["legacy_service"]:
            raise ValueError("first job is not safe")
        return True

    monkeypatch.setattr(worker, "advance", advance)
    assert worker.run_once() is False
    assert visited == [setup["job"]["legacy_service"], other["legacy_service"]]
    assert worker.status["jobs"][setup["job"]["legacy_service"]]["state"] == "waiting_tokenizer"
    assert worker.status["state"] == "failed"


def test_destination_with_old_checkpoint_settings_is_rejected(setup):
    setup["create_destination"]()
    path = Path(setup["job"]["destination_output_dir"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["commands"]["policy"] = [
        "training.checkpoint_every=1" if arg == "training.checkpoint_every=20" else arg
        for arg in manifest["commands"]["policy"]]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every-20-epoch"):
        worker_module.validate_destination(setup["job"], {"frozen": str(setup["frozen"]),
                                            "sha256": worker_module.sha256_file(setup["frozen"])})
