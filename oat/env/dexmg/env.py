"""Gymnasium wrapper around a DexMimicGen (robosuite 1.5) environment.

The environment is constructed from the ``env_args`` stored by
``scripts/convert_dexmg_to_zarr.py`` (the dataset's own controller, robots and cameras), with
render/horizon overrides. Observations use the zarr key names and conventions:
``<camera>_image`` uint8 HWC, ``<arm>_eef_rot6d`` from ``<arm>_eef_quat_site`` (rows layout) and
a constant ``task_uid``. Rewards are 1.0 iff ``_check_success()``; ``done`` is sticky.

Seeding: robosuite 1.5.2 builds every placement sampler with an unseeded ``default_rng()`` and
rebuilds them on each hard reset, so a pre-hook on ``_reset_internal`` reseeds all samplers from
the scheduled episode seed; ``np.random`` is seeded too (CanSort draws its can colour from it in
``_load_model``) and so is ``env.rng`` (robot initial-joint noise).

Requires robosuite 1.5.2 (the local checkout whose Inspire-hand mounts match the dataset) and
dexmimicgen on PYTHONPATH; both are imported lazily so this module is importable without them.
"""
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium
import numpy as np
import zarr

from oat.env.dexmg.factory import ARM_PREFIXES, get_task
from oat.env.dexmg.geometry import quat_xyzw_to_rot6d_rows

DROP_ENV_KWARGS = ("env_lang",)


def load_env_args(zarr_path):
    root = zarr.open(str(zarr_path), mode="r")
    return json.loads(json.dumps(dict(root.attrs)["env_args"]))


def _iter_samplers(sampler):
    yield sampler
    for child in getattr(sampler, "samplers", {}).values():
        yield from _iter_samplers(child)


class DexMGEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_name="box_cleanup",
        env_args=None,
        zarr_path=None,
        state_ports=None,
        camera_names=None,
        image_size=84,
        # Stored HDF5 frames are upright; robosuite's default "opengl" frames are their vertical
        # mirror (gate V3: flipped renders equal the stored frames exactly).
        image_flip=True,
        max_episode_steps=None,
        enable_render=True,
        render_gpu_device_id=-1,
        video_camera=None,
        video_resolution=256,
        hard_reset=True,
        seed=None,
    ):
        super().__init__()
        import robosuite
        import dexmimicgen  # noqa: F401 -- registers the TwoArm* environments

        self.task = get_task(task_name)
        if env_args is None:
            if zarr_path is None:
                raise ValueError("DexMGEnv needs env_args or the converted zarr_path")
            env_args = load_env_args(zarr_path)
        kwargs = dict(env_args["env_kwargs"])
        for key in DROP_ENV_KWARGS:
            kwargs.pop(key, None)
        self.camera_names = list(camera_names or self.task["cameras"])
        self.video_camera = video_camera or self.camera_names[0]
        render_cameras = list(dict.fromkeys(self.camera_names + [self.video_camera]))
        self.max_episode_steps = int(max_episode_steps or self.task["horizon"])
        kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=enable_render,
            use_camera_obs=enable_render,
            camera_names=render_cameras,
            camera_heights=image_size,
            camera_widths=image_size,
            camera_depths=False,
            ignore_done=True,
            horizon=self.max_episode_steps,
            hard_reset=hard_reset,
            render_gpu_device_id=render_gpu_device_id,
        )
        self.env = robosuite.make(self.task["env_name"], **kwargs)
        self.image_flip = bool(image_flip)
        self.enable_render = enable_render
        self.video_resolution = video_resolution
        self.arm_prefixes = ARM_PREFIXES[self.task["embodiment"]]
        self.state_ports = list(state_ports or self.default_state_ports())
        self.done = False
        self.cur_step = 0
        self._episode_seed = seed
        self._sampler_seed = None
        self._zarr_path = zarr_path
        self._dataset_init = None
        self._model_files = None
        original_reset_internal = self.env._reset_internal

        def reset_internal_with_seeded_samplers():
            if self._sampler_seed is not None and getattr(self.env, "placement_initializer", None) is not None:
                for index, sampler in enumerate(_iter_samplers(self.env.placement_initializer)):
                    sampler.rng = np.random.default_rng([self._sampler_seed, index])
            original_reset_internal()

        self.env._reset_internal = reset_internal_with_seeded_samplers

        obs = self._extract_obs(self.env._get_observations(force_update=True))
        spaces = {}
        for key, value in obs.items():
            if key.endswith("_image"):
                spaces[key] = gymnasium.spaces.Box(0, 255, shape=value.shape, dtype=np.uint8)
            elif key == "task_uid":
                spaces[key] = gymnasium.spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
            else:
                spaces[key] = gymnasium.spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float32)
        # Camera spaces are declared even without rendering, so a render-free probe env (used by
        # the vector env for space checks) matches the rendering workers.
        for camera in self.camera_names:
            spaces[f"{camera}_image"] = gymnasium.spaces.Box(0, 255, shape=(image_size, image_size, 3),
                                                             dtype=np.uint8)
        self.observation_space = gymnasium.spaces.Dict(spaces)
        low, high = self.env.action_spec
        self.action_space = gymnasium.spaces.Box(low.astype(np.float32), high.astype(np.float32),
                                                 dtype=np.float32)

    def default_state_ports(self):
        ports = []
        for prefix in self.arm_prefixes:
            ports += [f"{prefix}_eef_pos", f"{prefix}_eef_rot6d", f"{prefix}_gripper_qpos"]
        return ports

    def _extract_obs(self, raw):
        obs = {}
        if self.enable_render:
            for camera in self.camera_names:
                image = raw[f"{camera}_image"]
                obs[f"{camera}_image"] = np.ascontiguousarray(image[::-1] if self.image_flip else image,
                                                              dtype=np.uint8)
        for port in self.state_ports:
            if port.endswith("_eef_rot6d"):
                quat = raw[port[: -len("_eef_rot6d")] + "_eef_quat_site"]
                obs[port] = quat_xyzw_to_rot6d_rows(quat).astype(np.float32)
            elif port == "task_uid":
                continue
            else:
                obs[port] = np.asarray(raw[port], dtype=np.float32)
        obs["task_uid"] = np.array([self.task["uid"]], dtype=np.uint8)
        return obs

    def raw_observation(self):
        return self.env._get_observations(force_update=True)

    def step(self, action):
        _, _, _, info = self.env.step(np.asarray(action, dtype=np.float64))
        # step()'s own observations are sampled on the observables' internal clock and lag the
        # final substep (up to ~2e-2 on Box); the dataset rows equal force-updated observations
        # taken right after the last substep (verified to <=1e-7 over live replays).
        raw = self.raw_observation()
        self.cur_step += 1
        reward = 1.0 if self.env._check_success() else 0.0
        self.done = self.done or reward >= 1 or self.cur_step >= self.max_episode_steps
        return self._extract_obs(raw), reward, self.done, False, info

    def configure_dataset_init(self, episode):
        """Make the next reset restore recorded episode ``episode`` (scene XML + initial state)."""
        if self._zarr_path is None:
            raise ValueError("Dataset initialisation needs the zarr_path")
        self._dataset_init = int(episode)

    def _dataset_initial_state(self, episode):
        import gzip
        from pathlib import Path
        root = zarr.open(str(self._zarr_path), mode="r")
        if self._model_files is None:
            sidecar = Path(self._zarr_path).with_name(root.attrs["model_file_sidecar"])
            with gzip.open(sidecar, "rt") as stream:
                self._model_files = [json.loads(line)["model_file"] for line in stream]
        ends = root["meta/episode_ends"][:]
        start = int(ends[episode - 1]) if episode else 0
        return self._model_files[episode], root["data/sim_state"][start]

    def configure_episode(self, seed):
        """Apply the scheduled seed to the next reset (object placement, colours, joint noise)."""
        self._episode_seed = int(seed)

    def reset(self, seed=None, options=None):
        if self._dataset_init is not None and not options:
            model_xml, state = self._dataset_initial_state(self._dataset_init)
            options = {"model_xml": model_xml, "state": state}
            self._dataset_init = None
        if options and "state" in options:
            # Start from a recorded dataset state (diagnostics / dataset-init evaluation). Note that
            # task attributes sampled in _load_model (CanSort's is_red) are not restored by this.
            super().reset(seed=None)
            self._episode_seed = None
            return self.reset_to(options["model_xml"], options["state"]), {}
        episode_seed = self._episode_seed if seed is None else int(seed)
        super().reset(seed=episode_seed)
        self._sampler_seed = episode_seed
        if episode_seed is not None:
            np.random.seed(episode_seed)
            self.env.rng = np.random.default_rng(episode_seed)
            self._episode_seed = None
        self.env.reset()
        raw = self.raw_observation()
        self._sampler_seed = None
        self.done = False
        self.cur_step = 0
        return self._extract_obs(raw), {}

    def reset_to(self, model_xml, state):
        """Restore a recorded scene + MuJoCo state (DexMimicGen playback_datasets.reset_to)."""
        self.env.reset()
        self.env.reset_from_xml_string(self.env.edit_model_xml(model_xml))
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(np.asarray(state))
        self.env.sim.forward()
        if hasattr(self.env, "update_state"):
            self.env.update_state()
        self.done = False
        self.cur_step = 0
        return self._extract_obs(self.raw_observation())

    def render(self, mode="rgb_array"):
        frame = self.env.sim.render(height=self.video_resolution, width=self.video_resolution,
                                    camera_name=self.video_camera)
        return np.ascontiguousarray(frame[::-1] if self.image_flip else frame, dtype=np.uint8)

    def close(self):
        self.env.close()
