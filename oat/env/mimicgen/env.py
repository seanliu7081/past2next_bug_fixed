"""Seeded rollouts of the six official MimicGen tasks used by the PRT dataset."""
from copy import deepcopy
import os
import random

import cv2
import gymnasium as gym
import numpy as np


# In training task_uid order. Horizons follow MimicGen's core dataset registry.
_TASKS = {
    "stack_three_d1": ("StackThree_D1", 400),
    "hammer_cleanup_d1": ("HammerCleanup_D1", 500),
    "mug_cleanup_d1": ("MugCleanup_D1", 500),
    "coffee_d2": ("Coffee_D2", 400),
    "square_d2": ("Square_D2", 400),
    "threading_d2": ("Threading_D2", 400),
}
TASK_UIDS = {name: uid for uid, name in enumerate(_TASKS)}
TASK_HORIZONS = {name: spec[1] for name, spec in _TASKS.items()}
_CAMERAS = ("agentview", "robot0_eye_in_hand")
_STATE_DIMS = {"robot0_joint_pos": 7, "robot0_eef_pos": 3, "robot0_eef_quat": 4,
               "robot0_gripper_qpos": 2}

# Identical env_kwargs in all six core HDF5 data.attrs["env_args"], revision
# 33016f8a62c02334f929f2913af8fdd2a8a129e1 of amandlek/mimicgen_datasets.
_ENV_KWARGS = {
    "robots": ["Panda"], "control_freq": 20, "reward_shaping": False, "ignore_done": True,
    "has_renderer": False, "has_offscreen_renderer": True, "use_object_obs": True,
    "use_camera_obs": True, "camera_depths": False, "camera_heights": 84, "camera_widths": 84,
    "camera_names": list(_CAMERAS), "render_gpu_device_id": 0,
    "controller_configs": {
        "type": "OSC_POSE", "input_max": 1, "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150, "damping": 1, "impedance_mode": "fixed", "kp_limits": [0, 300],
        "damping_limits": [0, 10], "position_limits": None, "orientation_limits": None,
        "uncouple_pos_ori": True, "control_delta": True, "interpolation": None, "ramp_ratio": 0.2,
    },
}


class MimicGenEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, task_name, task_uid, image_size=128, render_gpu_device_id=None,
                 max_episode_steps=None):
        super().__init__()
        if task_name not in TASK_UIDS or task_uid != TASK_UIDS[task_name]:
            raise ValueError(f"Unknown MimicGen task/UID pair: {task_name} / {task_uid}")
        self.task_name = task_name
        self.task_uid = task_uid
        self.image_size = image_size
        self.max_episode_steps = TASK_HORIZONS[task_name] if max_episode_steps is None else max_episode_steps
        if self.image_size <= 0 or self.max_episode_steps <= 0:
            raise ValueError("image_size and max_episode_steps must be positive")
        self.render_gpu_device_id = render_gpu_device_id
        self.env = None
        self.last_obs = None
        self.steps = 0
        spaces = {key: gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)
                  for key, dim in _STATE_DIMS.items()}
        for camera in _CAMERAS:
            spaces[f"{camera}_rgb"] = gym.spaces.Box(0, 255, (image_size, image_size, 3), np.uint8)
        spaces["task_uid"] = gym.spaces.Box(0, len(TASK_UIDS) - 1, (1,), np.int64)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Box(-1., 1., (7,), np.float32)

    def _build(self):
        # The parent vector-env space probe never initializes MuJoCo or EGL.
        os.environ.setdefault("MUJOCO_GL", "egl")
        if self.render_gpu_device_id is not None:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.render_gpu_device_id)
        import mimicgen  # registers the official tasks with robosuite
        import robosuite
        from robosuite import macros

        self._flip_rgb = macros.IMAGE_CONVENTION == "opengl"
        kwargs = deepcopy(_ENV_KWARGS)
        if self.render_gpu_device_id is not None:
            kwargs["render_gpu_device_id"] = self.render_gpu_device_id
        self.env = robosuite.make(_TASKS[self.task_name][0], **kwargs)
        self.env.modify_observable("robot0_joint_pos", "active", True)

    def _observation(self, raw):
        obs = {key: np.asarray(raw[key], dtype=np.float32) for key in _STATE_DIMS}
        for camera in _CAMERAS:
            frame = raw[f"{camera}_image"]
            if self._flip_rgb:
                frame = frame[::-1]
            obs[f"{camera}_rgb"] = cv2.resize(frame, (self.image_size, self.image_size),
                                              interpolation=cv2.INTER_LINEAR)
        obs["task_uid"] = np.array([self.task_uid], dtype=np.int64)
        self.last_obs = obs
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.env is None:
            self._build()
        # robosuite 1.4.1 and MimicGen sample the robot and scene using these
        # two legacy RNGs. Seed after construction so repeated resets agree.
        episode_seed = int(self.np_random.integers(2**32)) if seed is None else int(seed)
        random.seed(episode_seed)
        np.random.seed(episode_seed % 2**32)
        self.steps = 0
        return self._observation(self.env.reset()), {"success": False, "steps": 0}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected seven finite delta-action and gripper commands")
        raw, _, _, _ = self.env.step(action)
        self.steps += 1
        success = bool(self.env._check_success())
        return (self._observation(raw), float(success), success,
                self.steps >= self.max_episode_steps and not success,
                {"success": success, "steps": self.steps})

    def render(self):
        return self.last_obs["agentview_rgb"]

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.last_obs = None
