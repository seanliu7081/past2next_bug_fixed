"""LIBERO configuration and device-selection regressions, without simulator jobs."""
import json
from types import SimpleNamespace

import pytest

from scripts import train_p2n_new_convnext_libero10 as launcher


@pytest.mark.parametrize("variant", ["p2n_new", "p2n_state_gate_new"])
def test_libero_contract_and_evaluation_defaults(variant):
    cfg = launcher.compose_config(variant)
    original = launcher.nano.legacy.compose_config(variant, "libero")
    assert cfg.task.policy.lazy_eval is False
    assert cfg.training.rollout_every == 100
    assert cfg.training.num_demo == 500
    assert cfg.training.num_epochs == 251
    assert cfg.task.policy.dataset == original.task.policy.dataset
    runner = dict(cfg.task.policy.env_runner)
    original_runner = dict(original.task.policy.env_runner)
    assert runner.pop("_target_").endswith(original_runner.pop("_target_").split(".")[-1])
    assert runner == original_runner
    assert cfg.policy.task == "libero"
    assert cfg.policy.convnext_frozen is True
    assert cfg.policy.expected_action_tokens == 8
    assert cfg.shape_meta.obs.robot0_eef_quat.shape == [4]
    assert cfg.shape_meta.obs.robot0_gripper_qpos.shape == [2]
    assert "libero10-oat-so3aug" in cfg.policy.tokenizer_checkpoint
    assert "nut_washer" not in cfg.policy.tokenizer_checkpoint


@pytest.mark.parametrize("override,match", [
    ("training.rollout_every=0", "rollout_every"),
    ("task.policy.lazy_eval=invalid", "lazy_eval"),
    ("task.policy.env_runner.n_test=0", "n_test"),
    ("task.policy.env_runner.n_parallel_envs=0", "n_parallel_envs"),
    ("policy.convnext_frozen=false", "convnext_frozen"),
    ("task.policy.env_runner._target_=oat.env_runner.p2n_new_runner.P2NStateGateNewLiberoRunner", "runner"),
])
def test_invalid_libero_contract_fails_before_loading(override, match):
    with pytest.raises(ValueError, match=match):
        launcher.compose_config("p2n_new", [override])


def test_dry_run_never_touches_model_simulator_or_devices(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("Configuration dry-run attempted runtime work")
    monkeypatch.setattr(launcher.nano, "preflight", unexpected)
    monkeypatch.setattr(launcher.nano, "check_gpu_idle", unexpected)
    monkeypatch.setattr(launcher, "inspect_simulator", unexpected)
    monkeypatch.setattr(launcher, "configure_training_devices", unexpected)
    launcher.main(["--variant", "p2n_state_gate_new", "--dry-run", "--num-processes", "4",
                   "--", "dataloader.batch_size=64", "training.gradient_accumulate_every=1",
                   "logging.mode=online", "training.rollout_every=100"])
    output = capsys.readouterr().out
    assert '"effective_batch": 256' in output
    assert '"lazy_eval": false' in output
    assert '"eval_every": 100' in output
    assert "mode: online" in output


def test_gpu_uuid_selection_maps_renderer_independently(monkeypatch):
    for name, value in (("CUDA_VISIBLE_DEVICES", "GPU-bb,GPU-aa"),
                        ("CUDA_DEVICE_ORDER", "FASTEST_FIRST"),
                        ("MUJOCO_EGL_DEVICE_ID", "6"),
                        ("P2N_LIBERO_EGL_DEVICE_ID", "6"), ("MUJOCO_GL", "egl")):
        monkeypatch.setenv(name, value)
    def run(command, **kwargs):
        assert "CUDA_VISIBLE_DEVICES" not in kwargs["env"]
        assert kwargs["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        return SimpleNamespace(stdout=json.dumps([
            dict(uuid="aa", cuda_ordinal=0, egl_device_id=3, name="A"),
            dict(uuid="bb", cuda_ordinal=2, egl_device_id=7, name="B")]))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    selected = [dict(index="2", uuid="GPU-bb", name="B"),
                dict(index="0", uuid="GPU-aa", name="A")]
    result = launcher.configure_training_devices(selected)
    assert result["renderer"]["egl_device_id"] == 7
    assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-bb,GPU-aa"
    assert "MUJOCO_EGL_DEVICE_ID" not in launcher.os.environ
    assert launcher.os.environ["P2N_LIBERO_EGL_DEVICE_ID"] == "7"


def test_selected_gpu_without_renderer_fails_without_rewriting_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout='[]'))
    with pytest.raises(RuntimeError, match="EGL renderer"):
        launcher.configure_training_devices([dict(index="7", uuid="GPU-bb", name="GPU")])
    assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_lazy_training_uses_uuid_mask_without_egl_probe(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    monkeypatch.setenv("P2N_LIBERO_EGL_DEVICE_ID", "1")
    def unexpected(*args, **kwargs):
        pytest.fail("Lazy evaluation attempted EGL enumeration")
    monkeypatch.setattr(launcher.subprocess, "run", unexpected)
    result = launcher.configure_training_devices([dict(index="7", uuid="GPU-bb", name="GPU")], render=False)
    assert result["renderer"] is None
    assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-bb"
    assert "P2N_LIBERO_EGL_DEVICE_ID" not in launcher.os.environ


def test_lazy_evaluation_skips_simulator_dependency_checks(monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", "/missing/libero/config")
    cfg = launcher.compose_config("p2n_new", ["task.policy.lazy_eval=true"])
    assert launcher.inspect_simulator(cfg) == {"enabled": False}


@pytest.mark.parametrize("fail", [False, True])
def test_renderer_environment_restores_training_mask_on_all_exits(monkeypatch, fail):
    from oat.env_runner.p2n_new_convnext_libero10_runner import renderer_env
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aa,GPU-bb")
    monkeypatch.setenv("P2N_LIBERO_EGL_DEVICE_ID", "7")
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    try:
        with renderer_env():
            assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "7"
            assert launcher.os.environ["MUJOCO_EGL_DEVICE_ID"] == "7"
            if fail:
                raise RuntimeError("construction failure")
    except RuntimeError:
        assert fail
    assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-aa,GPU-bb"
    assert "MUJOCO_EGL_DEVICE_ID" not in launcher.os.environ


@pytest.mark.parametrize("name", ["P2NNewLiberoRunner", "P2NStateGateNewLiberoRunner"])
def test_runner_preserves_policy_and_history_protocol(monkeypatch, name):
    from oat.env_runner import p2n_new_convnext_libero10_runner as scoped
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aa")
    monkeypatch.setenv("P2N_LIBERO_EGL_DEVICE_ID", "3")
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    calls = []
    class ExistingRunner:
        def __init__(self, **kwargs):
            assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "3"
            self.protocol = kwargs["protocol"]
        def run(self, policy, **kwargs):
            calls.append((policy, kwargs))
            return {"score": 1}
        def close(self):
            calls.append("closed")
    monkeypatch.setattr(scoped.importlib, "import_module", lambda module: SimpleNamespace(**{name: ExistingRunner}))
    runner = getattr(scoped, name)(protocol="corrected")
    assert runner.protocol == "corrected"
    assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-aa"
    policy = object()
    assert runner.run(policy, example=True) == {"score": 1}
    runner.close()
    assert calls == [(policy, {"example": True}), "closed"]
