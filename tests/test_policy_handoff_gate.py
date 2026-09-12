"""The legacy policy cannot initialize before a supervised stage handoff."""
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from oat.common import policy_handoff as gate


class WaitingAtGate(BaseException):
    pass


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    output = tmp_path / "legacy" / "current" / "policy"
    output.mkdir(parents=True)
    request_path = output.parent / "policy_handoff.json"
    request = {
        "schema_version": 1, "state": "armed", "legacy_policy_dir": str(output),
        "task": "nut_washer", "variant": "current", "expected_world_size": 2,
        "destination_service": "real-robot-nut-washer-current-1gpu",
        "destination_output_dir": str(tmp_path / "replacement" / "current"),
        "marker_dir": str(tmp_path / "markers"),
    }
    cfg = OmegaConf.create({
        "_target_": "oat.workspace.train_policy.TrainPolicyWorkspace",
        "task": {"policy": {"task_name": "nut_washer", "name": "real_robot_nut_washer"}},
    })
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    return output, request_path, request, cfg


def test_absent_request_is_noop(handoff, monkeypatch):
    output, _, request, cfg = handoff
    monkeypatch.delenv("RANK")
    assert gate.maybe_wait_for_policy_handoff(cfg, output) is None
    assert not Path(request["marker_dir"]).exists()


def test_tokenizer_ignores_even_malformed_policy_request(handoff):
    output, request_path, _, _ = handoff
    request_path.write_text("{invalid")
    cfg = OmegaConf.create({"_target_": "oat.workspace.train_oattok.TrainOATTokWorkspace"})
    assert gate.maybe_wait_for_policy_handoff(cfg, output.parent / "tokenizer") is None


@pytest.mark.parametrize("rank", [0, 1])
def test_valid_request_publishes_live_rank_and_waits(handoff, monkeypatch, rank):
    output, request_path, request, cfg = handoff
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    request_path.write_text(json.dumps(request))

    def stop_wait(seconds):
        assert seconds == 5
        raise WaitingAtGate

    monkeypatch.setattr(gate.time, "sleep", stop_wait)
    with pytest.raises(WaitingAtGate):
        gate.maybe_wait_for_policy_handoff(cfg, output)
    marker_dir = Path(request["marker_dir"])
    marker = json.loads((marker_dir / f"rank-{rank}.json").read_text())
    assert marker["pid"] == os.getpid()
    assert marker["ppid"] == os.getppid()
    assert marker["rank"] == marker["local_rank"] == rank
    assert marker["world_size"] == 2
    assert marker["legacy_policy_dir"] == str(output)
    assert marker["request_sha256"] == hashlib.sha256(request_path.read_bytes()).hexdigest()
    assert marker["destination_service"] == request["destination_service"]
    assert marker["state"] == "waiting_for_supervisor_handoff"
    assert list(marker_dir.iterdir()) == [marker_dir / f"rank-{rank}.json"]


@pytest.mark.parametrize("bad_value", ["{bad", "[]", "null"])
def test_malformed_request_fails_closed(handoff, bad_value):
    output, request_path, request, cfg = handoff
    request_path.write_text(bad_value)
    with pytest.raises(ValueError):
        gate.maybe_wait_for_policy_handoff(cfg, output)
    assert not Path(request["marker_dir"]).exists()


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("state", "completed"),
    ("state", "disabled"), ("legacy_policy_dir", "/different/policy"),
    ("legacy_policy_dir", "relative/policy"), ("task", "fruits"),
    ("variant", "left_noise"), ("expected_world_size", 1),
    ("destination_output_dir", "relative"), ("marker_dir", "relative"),
    ("destination_service", "service with spaces"),
])
def test_mismatched_request_fails_closed(handoff, field, value):
    output, request_path, request, cfg = handoff
    request[field] = value
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError):
        gate.maybe_wait_for_policy_handoff(cfg, output)


@pytest.mark.parametrize("field,value", [
    ("_target_", "oat.workspace.train_oattok.TrainOATTokWorkspace"),
    ("task.policy.task_name", "fruits"), ("task.policy.name", "real_robot_fruits"),
])
def test_wrong_workspace_or_task_cannot_publish_readiness(handoff, field, value):
    output, request_path, request, cfg = handoff
    OmegaConf.update(cfg, field, value)
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError):
        gate.maybe_wait_for_policy_handoff(cfg, output)
    assert not Path(request["marker_dir"]).exists()


@pytest.mark.parametrize("name,value", [("WORLD_SIZE", "1"), ("RANK", "2"),
                                          ("LOCAL_RANK", "1"), ("RANK", "invalid")])
def test_wrong_rank_environment_cannot_publish_readiness(handoff, monkeypatch, name, value):
    output, request_path, request, cfg = handoff
    monkeypatch.setenv(name, value)
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError):
        gate.maybe_wait_for_policy_handoff(cfg, output)
    assert not Path(request["marker_dir"]).exists()
