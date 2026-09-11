import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium

from typing import List, Optional, Dict



class RoboCasaEnv(gymnasium.Env):
    def __init__(
        self,
        env_name: str = "TurnOnSinkFaucet",
        robots: str = "PandaOmron",
        image_size: int = 128,
        seed: int = 42,
        camera_names: List[str] = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        state_ports: List[str] = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_base_pos",
            "robot0_base_quat",
        ],
        video_camera: str = "robot0_agentview_left",
        video_resolution: int = 512,
        max_episode_steps: int = 500,
        enable_render: bool = True,
        layout_ids: int = -1,
        style_ids: int = -1,
    ):
        super().__init__()

        # Simulator dependencies are needed only when evaluation is enabled.
        import robosuite
        from robosuite.controllers import load_composite_controller_config
        import robocasa  # noqa: F401 -- registers the kitchen environments

        controller_config = load_composite_controller_config(robot=robots)

        env = robosuite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=enable_render,
            use_camera_obs=enable_render,
            camera_names=list(set(list(camera_names) + [video_camera])),
            camera_heights=image_size,
            camera_widths=image_size,
            camera_depths=False,
            control_freq=20,
            horizon=max_episode_steps,
            ignore_done=True,
            seed=seed,
            layout_ids=layout_ids,
            style_ids=style_ids,
        )

        self.env = env
        self.env_name = env_name
        self.state_ports = state_ports
        self.camera_names = camera_names
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.done = False
        self.cur_step = 0
        self._episode_seed = seed

        # Setup gymnasium spaces
        obs_dict = env._get_observations()
        observation_space = gymnasium.spaces.Dict({})
        for port in state_ports:
            observation_space.spaces[port] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=obs_dict[port].shape, dtype=np.float32,
            )
        for cam_name in camera_names:
            observation_space.spaces[f"{cam_name}_image"] = gymnasium.spaces.Box(
                low=0, high=255,
                shape=(image_size, image_size, 3), dtype=np.uint8,
            )
        self.observation_space = observation_space
        self.action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(env.action_dim,), dtype=np.float32,
        )

    def _extract_obs(
        self, raw_obs: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        if raw_obs is None:
            raw_obs = self.env._get_observations()

        obs_dict = {}

        # Robot state
        for port in self.state_ports:
            obs_dict[port] = raw_obs[port].astype(np.float32)

        # RGB images — robosuite returns "{cam_name}_image" with origin at
        # bottom-left, so flip vertically to get standard top-left origin
        for cam_name in self.camera_names:
            obs_dict[f"{cam_name}_image"] = np.flip(
                raw_obs[f"{cam_name}_image"], axis=0
            ).astype(np.uint8)

        return obs_dict

    def step(self, action: np.ndarray):
        obs, reward, terminated, info = self.env.step(action)
        self.cur_step += 1
        if self.env._check_success():
            reward = 1.0
        else:
            reward = 0.0
        self.done = (
            self.done or terminated or (reward >= 1)
            or (self.cur_step >= self.max_episode_steps)
        )
        return self._extract_obs(obs), reward, self.done, False, info

    def configure_episode(self, seed):
        """Apply the scheduled seed to the next reset, including kitchen layout."""
        self._episode_seed = int(seed)

    def reset(self, seed=None, options=None):
        episode_seed = self._episode_seed if seed is None else int(seed)
        super().reset(seed=episode_seed)
        if episode_seed is not None:
            # RoboCasa uses this Generator when hard reset rebuilds the scene.
            self.env.seed = episode_seed
            self.env.rng = np.random.default_rng(episode_seed)
            np.random.seed(episode_seed)
            self._episode_seed = None
        obs = self.env.reset()
        obs_dict = self._extract_obs(obs)
        self.done = False
        self.cur_step = 0
        return obs_dict, {}

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        frame = np.flip(
            self.env.sim.render(
                height=self.video_resolution,
                width=self.video_resolution,
                camera_name=self.video_camera,
            ),
            axis=0,
        ).astype(np.uint8)
        return frame

    def close(self):
        self.env.close()
