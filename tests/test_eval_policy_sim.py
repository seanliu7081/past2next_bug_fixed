"""CPU-only CLI regressions; policies and simulator runners are always mocked."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from click.testing import CliRunner
from omegaconf import OmegaConf
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval_policy_sim.py"
LIBERO_RUNNER = "oat.env_runner.libero_runner.LiberoRunner"


@pytest.fixture
def evaluator(monkeypatch):
    streams = sys.stdout, sys.stderr
    spec = importlib.util.spec_from_file_location("eval_policy_sim_cli_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (sys.stdout, sys.stderr) == streams

    state = SimpleNamespace(
        module=module, runner_config={"_target_": LIBERO_RUNNER},
        loads=[], instances=[], runs=[], closed=[],
    )

    class FakePolicy:
        def to(self, device):
            assert str(device) == "cpu"
            return self

        def eval(self):
            return self

    def load(checkpoint, **kwargs):
        state.loads.append((checkpoint, kwargs))
        cfg = OmegaConf.create({"task": {"policy": {"env_runner": state.runner_config}}})
        return FakePolicy(), cfg

    def instantiate(config, **kwargs):
        state.instances.append((OmegaConf.to_container(config), kwargs))
        instance = SimpleNamespace(protocol=kwargs.get("protocol", config.get("protocol")))
        call_count = 0

        def run(policy, **inference):
            nonlocal call_count
            call_count += 1
            state.runs.append(inference)
            return {"mean_success_rate": 0.25 if call_count == 1 else 0.75}

        instance.run = run
        instance.close = lambda: state.closed.append(instance)
        return instance

    monkeypatch.setattr(module.BasePolicy, "from_checkpoint", staticmethod(load))
    monkeypatch.setattr(module.hydra.utils, "instantiate", instantiate)
    return state


def invoke(evaluator, output, *extra, checkpoint="unused.ckpt", input=None):
    return CliRunner().invoke(
        evaluator.module.eval_policy_sim,
        ["--checkpoint", str(checkpoint), "--output_dir", str(output), "--device", "cpu", *extra],
        input=input,
    )


@pytest.mark.parametrize("saved_protocol", [None, "legacy"])
def test_libero_defaults_to_corrected_for_old_checkpoints(evaluator, tmp_path, saved_protocol):
    if saved_protocol is not None:
        evaluator.runner_config["protocol"] = saved_protocol
    output = tmp_path / "evaluation"
    result = invoke(evaluator, output)
    assert result.exit_code == 0, result.output
    assert evaluator.instances[0][1]["protocol"] == "corrected"
    assert "LIBERO evaluation protocol: corrected" in result.output
    assert json.loads((output / "eval_log.json").read_text())["protocol"] == "corrected"
    assert len(evaluator.closed) == 1


@pytest.mark.parametrize("protocol", ["corrected", "official", "legacy"])
def test_explicit_libero_protocol_is_applied_and_recorded(evaluator, tmp_path, protocol):
    evaluator.runner_config["protocol"] = "legacy"
    output = tmp_path / "evaluation"
    result = invoke(evaluator, output, "--protocol", protocol)
    assert result.exit_code == 0, result.output
    assert evaluator.instances[0][1]["protocol"] == protocol
    assert json.loads((output / "eval_log.json").read_text())["protocol"] == protocol


def test_unrelated_runner_does_not_receive_libero_protocol(evaluator, tmp_path):
    evaluator.runner_config = {"_target_": "oat.env_runner.robocasa_multitask_runner.RoboCasaMultiTaskRunner"}
    output = tmp_path / "evaluation"
    result = invoke(evaluator, output)
    assert result.exit_code == 0, result.output
    assert "protocol" not in evaluator.instances[0][1]
    assert "protocol" not in json.loads((output / "eval_log.json").read_text())


def test_explicit_protocol_rejects_unrelated_runner(evaluator, tmp_path):
    evaluator.runner_config = {"_target_": "other.Runner"}
    result = invoke(evaluator, tmp_path / "evaluation", "--protocol", "legacy")
    assert result.exit_code == 2
    assert "only supported for the LIBERO runner" in result.output
    assert not evaluator.instances


def test_checkpoint_directory_repeats_crops_and_inference_options(evaluator, tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for name in ("first.ckpt", "second.ckpt", "latest.ckpt", "notes.txt"):
        (checkpoints / name).write_text("mock checkpoint")
    output = tmp_path / "evaluation"
    result = invoke(
        evaluator, output, "--num_exp", "2", "--temperature", "0", "--topk", "3",
        "--use_k_tokens", "4", checkpoint=checkpoints,
    )
    assert result.exit_code == 0, result.output
    assert {Path(path).name for path, _ in evaluator.loads} == {"first.ckpt", "second.ckpt"}
    assert all(kwargs["policy_overrides"] == {
        "obs_encoder": {"vision_encoder": {"eval_fixed_crop": True}}
    } for _, kwargs in evaluator.loads)
    assert evaluator.runs == [{"temperature": 0.0, "topk": 3, "use_k_tokens": 4}] * 4
    assert len(evaluator.closed) == 2
    for name in ("first", "second"):
        report = json.loads((output / name / "eval_log.json").read_text())
        assert report["num_exp"] == 2
        assert report["mean_success_rate_mean"] == 0.5
        assert report["protocol"] == "corrected"


def test_confirmed_cleanup_treats_spaces_and_shell_metacharacters_literally(evaluator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "results ; touch injected"
    output.mkdir()
    (output / "old.txt").write_text("replace me")
    neighbor = tmp_path / "results"
    neighbor.mkdir()
    (neighbor / "keep.txt").write_text("keep me")
    # Recursive cleanup must not follow symlinks within the removed directory.
    (output / "linked_neighbor").symlink_to(neighbor, target_is_directory=True)
    result = invoke(evaluator, output, input="y\n")
    assert result.exit_code == 0, result.output
    assert not (output / "old.txt").exists()
    assert (output / "eval_log.json").is_file()
    assert (neighbor / "keep.txt").read_text() == "keep me"
    assert not (tmp_path / "injected").exists()


@pytest.mark.parametrize("target_exists", [True, False])
def test_confirmed_output_symlink_is_replaced_without_touching_target(evaluator, tmp_path, target_exists):
    target = tmp_path / "target"
    if target_exists:
        target.mkdir()
        (target / "keep.txt").write_text("keep me")
    output = tmp_path / "linked output"
    output.symlink_to(target, target_is_directory=True)
    result = invoke(evaluator, output, input="y\n")
    assert result.exit_code == 0, result.output
    assert "Overwrite?" in result.output
    assert output.is_dir() and not output.is_symlink()
    assert (output / "eval_log.json").is_file()
    if target_exists:
        assert list(target.iterdir()) == [target / "keep.txt"]
        assert (target / "keep.txt").read_text() == "keep me"
    else:
        assert not target.exists()


def test_declined_cleanup_preserves_existing_output_and_does_not_load(evaluator, tmp_path):
    output = tmp_path / "results ; untouched"
    output.mkdir()
    (output / "keep.txt").write_text("keep me")
    result = invoke(evaluator, output, input="n\n")
    assert result.exit_code == 1
    assert (output / "keep.txt").read_text() == "keep me"
    assert not evaluator.loads
    assert not evaluator.instances
