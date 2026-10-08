"""P2N-VLA M2: dataset mixin (prompt-state q01/q99), LIBERO task configs, and the real LIBERO-10 zarr."""
import inspect
import json
import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from oat.common.replay_buffer import ReplayBuffer
from oat.common.seq_sampler import get_val_mask
from oat.dataset.real_robot_dataset import RealRobotZarrDatasetWithPrevWindow
from oat.dataset.real_robot_state_history import RealRobotZarrDatasetWithStateHistory
from oat.dataset.vla_dataset import (PROMPT_STATE_FIELD, PromptStateStatsMixin, VLAZarrDatasetWithPrevWindow,
                                     VLAZarrDatasetWithStateHistory)
from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow
from oat.model.vla.image_preprocess import ImagePreprocessor
from oat.model.vla.paligemma_prompt import (DEFAULT_SPM_PATH, LIBERO10_INSTRUCTIONS, LIBERO_PROMPT_STATE,
                                            PaliGemmaTokenizer, PromptBuilder, PromptStateSpec, discretize_state,
                                            uid_from_obs)

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "oat/config"
LIBERO_ZARR = Path("/workspace/past_action/data/libero/libero10_N500.zarr")
# Raw LIBERO-10 demonstrations (robosuite renders as recorded); the zarr was converted from these.
LIBERO10_HDF5_DIR = Path("/workspace/past_action/third_party/LIBERO/libero/datasets/libero_10")
CAMERAS = (("agentview_rgb", "agentview_rgb", "agentview"),
           ("robot0_eye_in_hand_rgb", "eye_in_hand_rgb", "robot0_eye_in_hand"))  # (zarr, HDF5, env camera)
TASKS = {"libero10_vla": VLAZarrDatasetWithPrevWindow, "libero10_vla_state_history": VLAZarrDatasetWithStateHistory}
STUB = dict(n_obs_steps=1, horizon=16, n_action_steps=8, past_n=7, num_demo=500)
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
RGB_KEYS = ("agentview_rgb", "robot0_eye_in_hand_rgb")


def compose_task(name, **overrides):
    values = {**STUB, **overrides}
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(overrides=[
            f"+task/policy=libero/{name}", f"+n_obs_steps={values['n_obs_steps']}", f"+horizon={values['horizon']}",
            f"+n_action_steps={values['n_action_steps']}", f"+past_n={values['past_n']}",
            f"+training.num_demo={values['num_demo']}"])


def robosuite_prompt_states(pos, quat, grip):
    """Independent reference: openpi LIBERO state with robosuite's own quat2axisangle (float64)."""
    robosuite_t = pytest.importorskip("robosuite.utils.transform_utils")
    axis_angle = np.stack([robosuite_t.quat2axisangle(q.astype(np.float64)) for q in quat])
    return np.concatenate([pos.astype(np.float64), axis_angle, grip.astype(np.float64)], axis=1)


def normalizer_quantiles(field):
    scale = field.params_dict["scale"].double()
    offset = field.params_dict["offset"].double()
    q01 = (-1.0 - offset) / scale
    q99 = q01 + 2.0 / scale - 1e-6
    return q01.numpy(), q99.numpy()


def assert_frozen(normalizer):
    params = list(normalizer.parameters())
    assert params and all(not p.requires_grad for p in params)


# ---------------------------------------------------------------------------------------------- task configs
@pytest.mark.parametrize("name", sorted(TASKS))
def test_task_configs_compose_and_resolve(name):
    cfg = OmegaConf.to_container(compose_task(name), resolve=True)
    task = cfg["task"]["policy"]
    # Plain OmegaConf with stub parent values resolves to the same task block as Hydra composition.
    loaded = OmegaConf.load(CONFIG_DIR / f"task/policy/libero/{name}.yaml")
    stub = OmegaConf.create({"n_obs_steps": 1, "horizon": 16, "n_action_steps": 8, "past_n": 7,
                             "training": {"num_demo": 500}, "task": {"policy": loaded}})
    assert OmegaConf.to_container(stub.task.policy, resolve=True) == task
    assert task["name"] == task["task_name"] == "libero10" and task["lazy_eval"] is True
    assert task["task_uids"] == list(range(30, 40))
    obs = task["shape_meta"]["obs"]
    assert {k: (v["type"], v["shape"]) for k, v in obs.items()} == {
        "agentview_rgb": ("rgb", [128, 128, 3]), "robot0_eye_in_hand_rgb": ("rgb", [128, 128, 3]),
        "robot0_eef_pos": ("state", [3]), "robot0_eef_quat": ("state", [4]),
        "robot0_gripper_qpos": ("state", [2]), "task_uid": ("state", [1])}
    assert task["shape_meta"]["action"]["shape"] == [7]
    assert task["rgb_ports"] == list(RGB_KEYS) and task["wrist_ports"] == ["robot0_eye_in_hand_rgb"]
    assert task["prompt_state"] == LIBERO_PROMPT_STATE
    assert task["prompt"] == {"instruction_source": "libero10", "state_keys": LIBERO_PROMPT_STATE["keys"],
                              "state_transforms": LIBERO_PROMPT_STATE["transforms"], "max_len": 96,
                              "dummy_uid": 30}
    assert PromptStateSpec(task["prompt"]["state_keys"], task["prompt"]["state_transforms"]).output_dim(
        {k: v["shape"] for k, v in obs.items()}) == 8
    dataset = dict(task["dataset"])
    assert get_class(dataset.pop("_target_")) is TASKS[name]
    expected = dict(zarr_path=str(LIBERO_ZARR), obs_keys=[*RGB_KEYS, *STATE_KEYS, "task_uid"], action_key="action",
                    n_obs_steps=1, n_action_steps=16, seed=42, val_ratio=0.1, max_train_episodes=None, past_n=7,
                    n_exec_steps=8, history_padding="zero", return_history_validity=True,
                    prompt_state=LIBERO_PROMPT_STATE)
    runner = dict(task["env_runner"])
    # In-training rollouts use LIBERO's official protocol: 50 episodes per task (each fixed initial state once).
    common_runner = dict(task_name="libero10", protocol="official", n_test=500, n_test_vis=0,
                         test_start_seed=3000, init_state_offset=0, n_obs_steps=1,
                         n_action_steps=8, fps=20, n_parallel_envs=10, image_size=128,
                         camera_names=["agentview", "robot0_eye_in_hand"], state_ports=list(STATE_KEYS),
                         max_episode_steps=550)
    if name == "libero10_vla_state_history":
        assert task["state_history_steps"] == STUB["past_n"] + 1 == 8
        assert task["state_history_keys"] == list(STATE_KEYS)
        expected.update(state_history_steps=8, state_history_keys=list(STATE_KEYS))
        common_runner.update(state_history_steps=8, state_history_keys=list(STATE_KEYS))
        assert runner.pop("_target_") == "oat.env_runner.p2n_new_runner.P2NStateGateNewLiberoRunner"
    else:
        assert runner.pop("_target_") == "oat.env_runner.p2n_new_runner.P2NNewLiberoRunner"
    assert dataset == expected
    assert runner == common_runner
    # Every sampler keyword is accepted by the explicit ZarrDatasetWithPrevWindow signature.
    inspect.signature(ZarrDatasetWithPrevWindow.__init__).bind(None, **{
        k: v for k, v in expected.items() if k not in ("prompt_state", "state_history_steps", "state_history_keys")})


def test_task_config_variants_share_every_common_block():
    base = OmegaConf.to_container(compose_task("libero10_vla"), resolve=True)["task"]["policy"]
    history = OmegaConf.to_container(compose_task("libero10_vla_state_history"), resolve=True)["task"]["policy"]
    for key in ("name", "task_name", "lazy_eval", "task_uids", "shape_meta", "rgb_ports", "wrist_ports",
                "prompt_state", "prompt"):
        assert base[key] == history[key], key
    for block in ("dataset", "env_runner"):
        for key, value in base[block].items():
            if key != "_target_":
                assert history[block][key] == value, (block, key)


def test_task_config_follows_run_level_values():
    cfg = OmegaConf.to_container(compose_task("libero10_vla", n_obs_steps=2, num_demo=50), resolve=True)
    task = cfg["task"]["policy"]
    assert task["dataset"]["n_obs_steps"] == task["env_runner"]["n_obs_steps"] == 2
    assert task["dataset"]["zarr_path"] == "/workspace/past_action/data/libero/libero10_N50.zarr"
    assert task["dataset"]["seed"] == 42  # pinned to the OAT tokenizer split, not the run seed


def test_task_config_composes_into_existing_train_config():
    """The p2n_new training config accepts the VLA task as its task/policy option."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="train_p2n_new", overrides=["task/policy=libero/libero10_vla"])
    task = OmegaConf.to_container(cfg.task.policy, resolve=True)
    assert task["dataset"]["_target_"] == "oat.dataset.vla_dataset.VLAZarrDatasetWithPrevWindow"
    assert task["dataset"]["prompt_state"] == LIBERO_PROMPT_STATE
    assert task["env_runner"]["_target_"] == "oat.env_runner.p2n_new_runner.P2NNewLiberoRunner"


SKIP_EXIT_CODE = 77


def run_isolated(code, *args, env=None, timeout=600):
    """Run ``code`` in a fresh interpreter at the repo root.

    Importing the simulator stack (libero/robosuite/mujoco/cv2) rewrites os.environ (MUJOCO_GL,
    PYOPENGL_PLATFORM, LD_LIBRARY_PATH, Qt paths); doing it in-process would leak into every later test
    and into the subprocesses they spawn (e.g. the gloo DDP smokes). Exit code 77 means "skip".
    """
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code), *args], cwd=str(REPO),
                            env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=timeout)
    if result.returncode == SKIP_EXIT_CODE:
        pytest.skip(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "skipped in subprocess")
    assert result.returncode == 0, f"isolated check failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr[-4000:]}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_runner_targets_accept_the_resolved_kwargs():
    cases = []
    for name in sorted(TASKS):
        kwargs = OmegaConf.to_container(compose_task(name).task.policy.env_runner, resolve=True)
        cases.append([kwargs.pop("_target_"), kwargs, name == "libero10_vla_state_history"])
    before = dict(os.environ)
    checked = run_isolated("""
        import inspect, json, sys
        try:
            from hydra.utils import get_class
            from oat.env_runner.libero_runner import LiberoRunner
            classes = {target: get_class(target) for target, _, _ in json.loads(sys.argv[1])}
        except ImportError as exc:
            print(f"LIBERO runner stack is not importable here: {exc!r}")
            sys.exit(77)
        checked = []
        for target, kwargs, history in json.loads(sys.argv[1]):
            if history:
                bound = inspect.signature(classes[target].__init__).bind(None, output_dir="out", **kwargs)
                assert bound.arguments["state_history_steps"] == 8, bound.arguments
                kwargs = {k: v for k, v in kwargs.items() if not k.startswith("state_history_")}
            inspect.signature(LiberoRunner.__init__).bind(None, output_dir="out", **kwargs)
            checked.append(target)
        print(json.dumps(checked))
        """, json.dumps(cases))
    assert checked == [target for target, _, _ in cases]
    assert dict(os.environ) == before  # the simulator import stayed out of this process


# ---------------------------------------------------------------------------------------------- synthetic zarr
LENGTHS = (23, 31, 19, 27, 25, 21)
VAL_RATIO = 0.34


@pytest.fixture
def synthetic_zarr(tmp_path):
    """Six LIBERO-shaped episodes; validation episodes are shifted so leaking them would change q01/q99."""
    rng = np.random.default_rng(0)
    holdout = get_val_mask(len(LENGTHS), VAL_RATIO, 42)
    assert holdout.sum() == 2
    replay = ReplayBuffer.create_empty_numpy()
    for index, length in enumerate(LENGTHS):
        shift = 5.0 if holdout[index] else 0.0
        axis = rng.normal(size=(length, 3))
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        angle = rng.uniform(0.5, 3.0, size=(length, 1))
        quat = np.concatenate([axis * np.sin(angle / 2), np.cos(angle / 2)], axis=-1)
        quat[::3] *= -1  # raw sign is kept; robosuite's map then gives angles above pi
        replay.add_episode({
            "action": rng.normal(size=(length, 7)).astype(np.float32),
            "agentview_rgb": rng.integers(0, 256, size=(length, 8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_rgb": rng.integers(0, 256, size=(length, 8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": (rng.normal(size=(length, 3)) + shift).astype(np.float32),
            "robot0_eef_quat": quat.astype(np.float32),
            "robot0_gripper_qpos": (rng.uniform(-0.04, 0.04, size=(length, 2)) + shift).astype(np.float32),
            "task_uid": np.full((length, 1), 30 + index, dtype=np.int64),
        })
    path = tmp_path / "libero_like.zarr"
    replay.save_to_path(str(path))
    return str(path), holdout


def synthetic_kwargs(path, **overrides):
    kwargs = dict(zarr_path=path, obs_keys=[*RGB_KEYS, *STATE_KEYS, "task_uid"], action_key="action",
                  n_obs_steps=1, n_action_steps=16, seed=42, val_ratio=VAL_RATIO, past_n=7, n_exec_steps=8,
                  history_padding="zero", return_history_validity=True, prompt_state=LIBERO_PROMPT_STATE)
    kwargs.update(overrides)
    return kwargs


def training_reference(path, holdout):
    replay = ReplayBuffer.copy_from_path(path, keys=list(STATE_KEYS))
    frames = np.repeat(~holdout, np.diff(np.concatenate(([0], replay.episode_ends))))
    return robosuite_prompt_states(*(replay[key][frames] for key in STATE_KEYS)), frames


@pytest.mark.parametrize("dataset_cls", [VLAZarrDatasetWithPrevWindow, VLAZarrDatasetWithStateHistory])
def test_prompt_state_normalizer_uses_training_frames_only(synthetic_zarr, dataset_cls):
    path, holdout = synthetic_zarr
    dataset = dataset_cls(**synthetic_kwargs(path))
    assert dataset.prompt_state_dim == 8 and dataset.prompt_state_spec.to_dict() == LIBERO_PROMPT_STATE
    normalizer = dataset.get_normalizer()
    assert set(normalizer.params_dict) == {"action", *RGB_KEYS, *STATE_KEYS, "task_uid", PROMPT_STATE_FIELD}
    assert_frozen(normalizer)
    field = normalizer[PROMPT_STATE_FIELD]
    assert field.params_dict["scale"].shape == field.params_dict["offset"].shape == (8,)
    assert field.params_dict["scale"].dtype == torch.float32
    assert set(field.params_dict["input_stats"]) == {"min", "max", "mean", "std"}
    reference, frames = training_reference(path, holdout)
    q01, q99 = np.quantile(reference, [0.01, 0.99], axis=0)
    got_q01, got_q99 = normalizer_quantiles(field)
    np.testing.assert_allclose(got_q01, q01, rtol=0, atol=2e-5)
    np.testing.assert_allclose(got_q99, q99, rtol=0, atol=2e-5)
    stats = field.params_dict["input_stats"]
    np.testing.assert_allclose(stats["min"].numpy(), reference.min(0), atol=2e-5)
    np.testing.assert_allclose(stats["max"].numpy(), reference.max(0), atol=2e-5)
    # openpi map: q01 -> -1 and q99 -> +1.
    ends = field.normalize(torch.from_numpy(np.stack([q01, q99])).float())
    torch.testing.assert_close(ends, torch.tensor([[-1.0] * 8, [1.0] * 8]), rtol=0, atol=1e-4)
    all_frames = robosuite_prompt_states(*(ReplayBuffer.copy_from_path(path, keys=list(STATE_KEYS))[key][:]
                                           for key in STATE_KEYS))
    assert np.abs(np.quantile(all_frames, 0.99, axis=0) - q99).max() > 1.0  # the holdout would have leaked
    # A validation view never refits on validation frames.
    val = dataset.get_validation_dataset()
    assert np.array_equal(val.train_mask, holdout) and len(val) == frames.size - frames.sum()
    val_state = val.get_normalizer().state_dict()
    for key, value in normalizer.state_dict().items():
        torch.testing.assert_close(val_state[key], value, rtol=0, atol=0)
    provenance = dataset.prompt_state_provenance()
    assert provenance["train_episodes"] == 4 and provenance["train_frames"] == int(frames.sum())
    assert provenance["dim"] == 8 and provenance["quantiles"] == [0.01, 0.99]


@pytest.mark.parametrize("dataset_cls,parent_cls", [
    (VLAZarrDatasetWithPrevWindow, RealRobotZarrDatasetWithPrevWindow),
    (VLAZarrDatasetWithStateHistory, RealRobotZarrDatasetWithStateHistory)])
def test_samples_and_inherited_fields_are_unchanged(synthetic_zarr, dataset_cls, parent_cls):
    path, _ = synthetic_zarr
    kwargs = synthetic_kwargs(path)
    dataset = dataset_cls(**kwargs)
    kwargs.pop("prompt_state")
    parent = parent_cls(**kwargs)
    assert len(dataset) == len(parent)
    for index in (0, 3, 8, 9, len(dataset) // 2, len(dataset) - 1):
        ours, theirs = dataset[index], parent[index]
        assert ours.keys() == theirs.keys()
        for key in ours:
            if isinstance(ours[key], dict):
                assert ours[key].keys() == theirs[key].keys()
                for name in ours[key]:
                    assert torch.equal(ours[key][name], theirs[key][name]), (key, name)
            else:
                assert torch.equal(ours[key], theirs[key]), key
    ours, theirs = dataset.get_normalizer(), parent.get_normalizer()
    for key in theirs.params_dict:
        for name, value in theirs[key].params_dict.state_dict().items():
            torch.testing.assert_close(ours[key].params_dict.state_dict()[name], value, rtol=0, atol=0)


def test_mixin_validates_eagerly(synthetic_zarr):
    path, _ = synthetic_zarr
    with pytest.raises(ValueError, match="obs_keys"):
        VLAZarrDatasetWithPrevWindow(**synthetic_kwargs(path, obs_keys=[*RGB_KEYS, "robot0_eef_pos", "task_uid"]))
    with pytest.raises(ValueError, match="reserved"):
        VLAZarrDatasetWithPrevWindow(**synthetic_kwargs(path, obs_keys=[*RGB_KEYS, *STATE_KEYS, "prompt_state"]))
    with pytest.raises(ValueError, match="transform"):
        VLAZarrDatasetWithPrevWindow(**synthetic_kwargs(
            path, prompt_state={"keys": list(STATE_KEYS), "transforms": {"robot0_eef_quat": "euler"}}))
    with pytest.raises(ValueError, match="needs 4 values"):
        VLAZarrDatasetWithPrevWindow(**synthetic_kwargs(
            path, prompt_state={"keys": ["robot0_eef_pos"], "transforms": {"robot0_eef_pos": "quat_to_axis_angle"}}))
    with pytest.raises(TypeError, match="keyword"):
        VLAZarrDatasetWithPrevWindow(7, **synthetic_kwargs(path))
    with pytest.raises(TypeError):
        VLAZarrDatasetWithPrevWindow(**{k: v for k, v in synthetic_kwargs(path).items() if k != "prompt_state"})
    assert issubclass(VLAZarrDatasetWithPrevWindow, PromptStateStatsMixin)
    assert VLAZarrDatasetWithStateHistory.__mro__[1] is PromptStateStatsMixin


@pytest.mark.parametrize("name", sorted(TASKS))
def test_synthetic_dataset_from_config_kwargs(synthetic_zarr, name):
    """The resolved task-config kwargs (only the path and split ratio swapped) construct the dataset."""
    path, _ = synthetic_zarr
    dataset = instantiate(compose_task(name).task.policy.dataset, zarr_path=path, val_ratio=VAL_RATIO)
    assert type(dataset) is TASKS[name]
    assert dataset.prompt_state_spec.to_dict() == LIBERO_PROMPT_STATE
    sample = dataset[10]
    assert sample["obs"]["task_uid"].dtype == torch.int64 and sample["obs"]["robot0_eef_quat"].shape == (1, 4)
    if name == "libero10_vla_state_history":
        assert sample["obs"]["state_history__robot0_eef_quat"].shape == (8, 4)
    assert PROMPT_STATE_FIELD in dataset.get_normalizer().params_dict


# ---------------------------------------------------------------------------------------------- real LIBERO-10
@contextmanager
def shared_replay_buffer(buffer, path):
    original = ReplayBuffer.copy_from_path

    def cached(zarr_path, *args, keys=None, **kwargs):
        if Path(zarr_path).resolve() == Path(path).resolve() and keys is not None and \
                set(keys) <= set(buffer.keys()) and not args and not kwargs:
            return buffer
        return original(zarr_path, *args, keys=keys, **kwargs)

    with mock.patch.object(ReplayBuffer, "copy_from_path", new=cached):
        yield


@pytest.fixture(scope="module")
def libero_datasets():
    """Both VLA datasets from the resolved task configs (n_obs_steps=1), loading the ~12.6 GB RGB once."""
    if not LIBERO_ZARR.exists():
        pytest.skip(f"LIBERO-10 zarr not found at {LIBERO_ZARR}")
    prev = instantiate(compose_task("libero10_vla").task.policy.dataset)
    with shared_replay_buffer(prev.replay_buffer, LIBERO_ZARR):
        history = instantiate(compose_task("libero10_vla_state_history").task.policy.dataset)
    assert history.replay_buffer is prev.replay_buffer
    return prev, history


def anchor(dataset, index, sample):
    buffer_start, _, sample_start, _ = dataset.seq_sampler.indices[index]
    frame = int(buffer_start + dataset.pad_before - sample_start)
    step = int(sample["episode_step"])
    episode = int(np.searchsorted(dataset.replay_buffer.episode_ends, frame, side="right"))
    end = int(dataset.replay_buffer.episode_ends[episode])
    return frame, frame - step, end


def sample_indices(dataset):
    rng = np.random.default_rng(0)
    fixed = [0, 1, 7, 8, 9, 15, 16, 100, len(dataset) // 2, len(dataset) - 1]
    return fixed + rng.choice(len(dataset), size=20, replace=False).tolist()


@pytest.mark.requires_data
@pytest.mark.slow
def test_libero_prev_window_samples_align(libero_datasets):
    dataset, _ = libero_datasets
    replay = dataset.replay_buffer
    assert type(dataset) is VLAZarrDatasetWithPrevWindow
    assert (dataset.n_obs_steps, dataset.n_action_steps, dataset.past_n, dataset.n_exec_steps) == (1, 16, 7, 8)
    assert dataset.normalization_train_mask.sum() == 450 and len(dataset.train_mask) == 500
    for index in sample_indices(dataset):
        sample = dataset[index]
        frame, start, end = anchor(dataset, index, sample)
        step = frame - start
        assert set(sample) == {"obs", "action", "past_action", "past_action_valid", "prev_obs", "prev_past_action",
                               "prev_past_action_valid", "prev_window_valid", "episode_step"}
        for name, at in (("obs", frame), ("prev_obs", max(frame - 8, start))):
            obs = sample[name]
            assert set(obs) == {*RGB_KEYS, *STATE_KEYS, "task_uid"}
            for key in RGB_KEYS:
                assert obs[key].shape == (1, 128, 128, 3) and obs[key].dtype == torch.uint8
                assert np.array_equal(obs[key][0].numpy(), replay[key][at])
            for key, width in zip(STATE_KEYS, (3, 4, 2)):
                assert obs[key].shape == (1, width) and obs[key].dtype == torch.float32
                assert np.array_equal(obs[key][0].numpy(), replay[key][at])
            assert obs["task_uid"].shape == (1, 1) and obs["task_uid"].dtype == torch.int64
            assert 30 <= int(obs["task_uid"]) <= 39 and int(obs["task_uid"]) == int(replay["task_uid"][at, 0])
        assert sample["action"].shape == (16, 7) and sample["action"].dtype == torch.float32
        expected_action = replay["action"][np.minimum(frame + np.arange(16), end - 1)]
        assert np.array_equal(sample["action"].numpy(), expected_action)
        for name, valid_name, last in (("past_action", "past_action_valid", frame),
                                       ("prev_past_action", "prev_past_action_valid", frame - 8)):
            positions = last - 7 + np.arange(7)
            valid = positions >= start
            assert sample[name].shape == (7, 7) and sample[valid_name].dtype == torch.bool
            assert np.array_equal(sample[valid_name].numpy(), valid)
            assert np.array_equal(sample[name].numpy()[valid], replay["action"][positions[valid]])
            assert (sample[name].numpy()[~valid] == 0).all()
        assert sample["episode_step"].dtype == torch.int64 and int(sample["episode_step"]) == step
        assert bool(sample["prev_window_valid"]) == (step >= 8)


@pytest.mark.requires_data
@pytest.mark.slow
def test_libero_state_history_samples_align(libero_datasets):
    _, dataset = libero_datasets
    replay = dataset.replay_buffer
    assert type(dataset) is VLAZarrDatasetWithStateHistory
    assert dataset.state_history_steps == 8 and dataset.state_history_keys == STATE_KEYS
    for index in sample_indices(dataset):
        sample = dataset[index]
        frame, start, _ = anchor(dataset, index, sample)
        for name, last, past_valid in (("obs", frame, "past_action_valid"),
                                       ("prev_obs", frame - 8, "prev_past_action_valid")):
            obs = sample[name]
            valid = obs["state_history_valid"]
            positions = last - 7 + np.arange(8)
            assert valid.shape == (8,) and valid.dtype == torch.bool
            assert np.array_equal(valid.numpy(), positions >= start)
            assert torch.equal(valid, torch.sort(valid.int()).values.bool())  # contiguous valid suffix
            assert torch.equal(sample[past_valid], valid[:-1] & valid[1:])
            for key, width in zip(STATE_KEYS, (3, 4, 2)):
                history = obs["state_history__" + key]
                assert history.shape == (8, width) and history.dtype == torch.float32
                ok = valid.numpy()
                assert np.array_equal(history.numpy()[ok], replay[key][positions[ok]])
                assert (history.numpy()[~ok] == 0).all()
            if name == "obs":
                assert bool(valid[-1])  # the current state is always valid


@pytest.mark.requires_data
@pytest.mark.slow
def test_libero_normalizer_fields_and_prompt_state_quantiles(libero_datasets):
    for dataset in libero_datasets:
        normalizer = dataset.get_normalizer()
        assert set(normalizer.params_dict) == {"action", *RGB_KEYS, *STATE_KEYS, "task_uid", PROMPT_STATE_FIELD}
        assert_frozen(normalizer)
        for key in RGB_KEYS:  # fixed byte range, never fitted
            torch.testing.assert_close(normalizer[key].params_dict["scale"], torch.full((3,), 2 / 255))
        frames = dataset.training_frame_mask()
        assert frames.sum() == dataset.prompt_state_provenance()["train_frames"]
        replay = dataset.replay_buffer
        reference = robosuite_prompt_states(*(replay[key][frames] for key in STATE_KEYS))
        q01, q99 = np.quantile(reference, [0.01, 0.99], axis=0)
        got_q01, got_q99 = normalizer_quantiles(normalizer[PROMPT_STATE_FIELD])
        np.testing.assert_allclose(got_q01, q01, rtol=0, atol=2e-5)
        np.testing.assert_allclose(got_q99, q99, rtol=0, atol=2e-5)
        normalized = normalizer[PROMPT_STATE_FIELD].normalize(torch.from_numpy(reference).float())
        inside = ((normalized >= -1 - 1e-4) & (normalized <= 1 + 1e-4)).float().mean(0)
        assert (inside > 0.975).all(), inside
        val_state = dataset.get_validation_dataset().get_normalizer().state_dict()
        for key, value in normalizer.state_dict().items():
            torch.testing.assert_close(val_state[key], value, rtol=0, atol=0)
    first, second = (dataset.get_normalizer().state_dict() for dataset in libero_datasets)
    for key, value in first.items():
        torch.testing.assert_close(second[key], value, rtol=0, atol=0)


@pytest.mark.requires_data
@pytest.mark.slow
def test_libero_batch_through_the_prompt_and_image_pipeline(libero_datasets):
    """The policy's prefix inputs from a real batch: images, prompt state bins and <= 96-token prompts."""
    if not DEFAULT_SPM_PATH.is_file():
        pytest.skip(f"PaliGemma SentencePiece model missing at {DEFAULT_SPM_PATH}")
    dataset, _ = libero_datasets
    normalizer = dataset.get_normalizer()
    batch = torch.utils.data.default_collate([dataset[i] for i in np.linspace(0, len(dataset) - 1, 32).astype(int)])
    obs = batch["obs"]
    spec = PromptStateSpec.from_config(LIBERO_PROMPT_STATE)
    state = normalizer[PROMPT_STATE_FIELD].normalize(spec.extract(obs))
    bins = discretize_state(state)
    uids = uid_from_obs(obs)
    assert uids.tolist() == obs["task_uid"][:, -1, 0].tolist()
    ids, valid = PromptBuilder(PaliGemmaTokenizer(str(DEFAULT_SPM_PATH)), "libero10").build(uids, bins)
    assert ids.shape == (32, 96) and valid.sum(1).max() <= 96 and valid[:, 0].all()
    # Rollout delivers task_uid as float and RGB as float bytes; the same prompts and pixels result.
    rollout_obs = {k: v.float() for k, v in obs.items()}
    torch.testing.assert_close(PromptBuilder(PaliGemmaTokenizer(str(DEFAULT_SPM_PATH)), "libero10").build(
        uid_from_obs(rollout_obs), discretize_state(normalizer[PROMPT_STATE_FIELD].normalize(
            spec.extract(rollout_obs))))[0], ids)
    pre = ImagePreprocessor(list(RGB_KEYS), ["robot0_eye_in_hand_rgb"], 224)
    pixels = pre(obs, train_aug=False)
    assert pixels.shape == (32, 2, 3, 224, 224) and -1 <= pixels.min() and pixels.max() <= 1
    torch.testing.assert_close(pre(rollout_obs, train_aug=False), pixels, rtol=0, atol=0)
    augmented = pre(obs, train_aug=True, generator=torch.Generator().manual_seed(0))
    assert augmented.shape == pixels.shape and -1 <= augmented.min() and augmented.max() <= 1


@pytest.mark.requires_data
def test_zarr_prompt_field_is_a_truncated_prefix_of_the_instruction_table():
    """Confirms the uid -> instruction table against the data, and why the zarr 'prompt' must not be used."""
    if not LIBERO_ZARR.exists():
        pytest.skip(f"LIBERO-10 zarr not found at {LIBERO_ZARR}")
    import zarr
    data = zarr.open(str(LIBERO_ZARR), "r")["data"]
    uids, prompts = data["task_uid"][:, 0], data["prompt"][:]
    assert data["prompt"].dtype == np.dtype("<U31")
    pairs = set(zip(uids.tolist(), prompts.tolist()))
    assert {uid for uid, _ in pairs} == set(range(30, 40)) and len(pairs) == 10
    for uid, prompt in pairs:
        assert LIBERO10_INSTRUCTIONS[uid].startswith(prompt) and len(prompt) == min(31, len(LIBERO10_INSTRUCTIONS[uid]))
    assert dict(pairs)[30] == dict(pairs)[37]


@pytest.mark.requires_data
@pytest.mark.slow
def test_libero_prompt_bins_match_openpi_quantile_tokenization(libero_datasets):
    """Every training frame's state bins equal openpi's float64 pipeline (robosuite axis-angle -> quantile map
    -> digitize), apart from rare fp32 ties at a bin edge; below-q01 values are clipped to bin 0 (plan), where
    openpi's unclipped digitize would emit "-1"."""
    dataset, _ = libero_datasets
    frames = dataset.training_frame_mask()
    replay = dataset.replay_buffer
    reference = robosuite_prompt_states(*(replay[key][frames] for key in STATE_KEYS))
    q01, q99 = np.quantile(reference, [0.01, 0.99], axis=0)
    normalized = (reference - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    edges = np.linspace(-1, 1, 257)[:-1]
    expected = np.digitize(np.clip(normalized, -1, 1), edges) - 1
    spec = PromptStateSpec.from_config(LIBERO_PROMPT_STATE)
    obs = {key: torch.from_numpy(np.asarray(replay[key][frames], dtype=np.float32)) for key in STATE_KEYS}
    got = discretize_state(dataset.get_normalizer()[PROMPT_STATE_FIELD].normalize(spec.extract(obs))).numpy()
    difference = np.abs(got - expected)
    assert difference.max() <= 1 and (difference > 0).mean() < 1e-4, (difference > 0).sum()
    assert got.min() == 0 and got.max() == 255
    assert ((np.digitize(normalized, edges) - 1) < 0).mean() > 0.005  # openpi without the plan's clip


# ---------------------------------------------------------------------------------------------- frame orientation
def locate_libero_demo(episode=0):
    """The raw LIBERO-10 HDF5 demo a zarr episode was converted from (matched by language and actions)."""
    if not LIBERO_ZARR.exists() or not LIBERO10_HDF5_DIR.is_dir():
        pytest.skip(f"LIBERO-10 zarr ({LIBERO_ZARR}) or raw HDF5 demos ({LIBERO10_HDF5_DIR}) missing")
    h5py = pytest.importorskip("h5py")
    import zarr
    root = zarr.open(str(LIBERO_ZARR), "r")
    ends = root["meta"]["episode_ends"][:]
    start, end = (0 if episode == 0 else int(ends[episode - 1])), int(ends[episode])
    uid, actions = int(root["data"]["task_uid"][start, 0]), root["data"]["action"][start:end]
    for path in sorted(LIBERO10_HDF5_DIR.glob("*.hdf5")):
        with h5py.File(path, "r") as stream:
            data = stream["data"]
            if json.loads(data.attrs["problem_info"])["language_instruction"] != LIBERO10_INSTRUCTIONS[uid]:
                continue
            for name in data:
                if data[name]["actions"].shape[0] == end - start and \
                        np.array_equal(data[name]["actions"][:].astype(np.float32), actions):
                    task_name = data.attrs["bddl_file_name"].split("/")[-1][:-5]
                    return dict(path=str(path), demo=name, task_name=task_name, start=start, end=end, uid=uid)
    pytest.fail(f"No raw LIBERO-10 demo reproduces zarr episode {episode} (uid {uid})")


def test_libero_env_flips_live_renders_vertically_only():
    """LiberoEnv turns each raw robosuite render into an observation by a vertical flip only (no mirror)."""
    flipped = run_isolated("""
        import json, sys
        import numpy as np
        try:
            from oat.env.libero.env import LiberoEnv
        except Exception as exc:
            print(f"LIBERO env is not importable here: {exc!r}")
            sys.exit(77)
        env = object.__new__(LiberoEnv)
        env.state_ports, env.task_prompt, env.task_uid = ["robot0_eef_pos"], "x", 38
        env.camera_names = ["agentview", "robot0_eye_in_hand"]
        rng = np.random.default_rng(0)
        raw = {"robot0_eef_pos": np.zeros(3)}
        raw.update({f"{cam}_image": rng.integers(0, 256, (6, 5, 3), dtype=np.uint8) for cam in env.camera_names})
        obs = env._extract_obs(raw)
        print(json.dumps({cam: bool(np.array_equal(obs[f"{cam}_rgb"], raw[f"{cam}_image"][::-1]))
                          for cam in env.camera_names}))
        """)
    assert flipped == {"agentview": True, "robot0_eye_in_hand": True}


@pytest.mark.requires_data
def test_zarr_frames_are_vertically_flipped_raw_libero_renders():
    """Training frames are the raw recorded renders flipped vertically only, the transform LiberoEnv applies to
    live renders: train and rollout frames share one orientation (upright, never mirrored or rotated 180
    degrees as in openpi's LIBERO data)."""
    h5py = pytest.importorskip("h5py")
    import zarr
    demo = locate_libero_demo(0)
    data = zarr.open(str(LIBERO_ZARR), "r")["data"]
    with h5py.File(demo["path"], "r") as stream:
        recorded = stream["data"][demo["demo"]]["obs"]
        for zarr_key, hdf5_key, _ in CAMERAS:
            raw = recorded[hdf5_key][:6]
            stored = data[zarr_key][demo["start"]:demo["start"] + 6]
            assert stored.dtype == np.uint8 and np.array_equal(stored, raw[:, ::-1]), zarr_key
            for wrong in (raw, raw[:, :, ::-1], raw[:, ::-1, ::-1]):
                assert not np.array_equal(stored, wrong), zarr_key


@pytest.mark.requires_data
@pytest.mark.gpu
@pytest.mark.slow
def test_live_libero_render_matches_training_frame_orientation_and_quaternion_sign():
    """Render the simulator at a demo's initial state through LiberoEnv: both cameras match the zarr training
    frame upright (not flipped or mirrored), and the raw eef quaternion has the zarr's sign, so the robosuite
    axis-angle in the prompt state means the same thing in training and rollout."""
    demo = locate_libero_demo(0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    device = visible.split(",")[0].strip() if visible else "0"
    result = run_isolated("""
        import json, sys
        import h5py, numpy as np, zarr
        demo, zarr_path, cameras = json.loads(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
        # LiberoEnv seeds the simulator only after its construction-time reset, whose fixture placement
        # lives in the model rather than in the MuJoCo state set_init_state restores; seed it explicitly.
        np.random.seed(0)
        try:
            from oat.env.libero.env import LiberoEnv
            env = LiberoEnv(demo["task_name"], image_size=128, protocol="corrected")
        except Exception as exc:
            print(f"LIBERO rendering is unavailable here: {exc!r}")
            sys.exit(77)
        with h5py.File(demo["path"], "r") as stream:
            state0 = stream["data"][demo["demo"]]["states"][0]
        obs = env._extract_obs(env.env.set_init_state(state0))
        data = zarr.open(zarr_path, "r")["data"]
        out = {}
        for zarr_key, _, _ in cameras:
            live, stored = obs[zarr_key].astype(np.float64), data[zarr_key][demo["start"]].astype(np.float64)
            out[zarr_key] = {name: float(np.abs(live - view).mean()) for name, view in (
                ("upright", stored), ("vflip", stored[::-1]), ("hflip", stored[:, ::-1]),
                ("rot180", stored[::-1, ::-1]))}
        out["quat_dot"] = float(np.dot(obs["robot0_eef_quat"], data["robot0_eef_quat"][demo["start"]]))
        env.env.close()
        print(json.dumps(out))
        """, json.dumps(demo), str(LIBERO_ZARR), json.dumps(CAMERAS),
        env={"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "MUJOCO_EGL_DEVICE_ID": device})
    # Unseeded placements gave upright/best-flip ratios of 0.07-0.15 (agentview) and 0.15-0.26 (wrist: the
    # gripper view is nearly mirror-symmetric); a flipped frame would give the inverse, about 4-14.
    margins = {"agentview_rgb": 0.25, "robot0_eye_in_hand_rgb": 0.35}
    for zarr_key, _, _ in CAMERAS:
        errors = result[zarr_key]
        assert errors["upright"] < 20.0, (zarr_key, errors)
        best_flip = min(errors["vflip"], errors["hflip"], errors["rot180"])
        assert errors["upright"] < margins[zarr_key] * best_flip, (zarr_key, errors)
    assert result["quat_dot"] > 0.99, result
