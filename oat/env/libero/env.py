import os
import string
import random
import numpy as np
import gymnasium
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

from typing import List, Optional, Dict


task_name_to_suite_and_ids = {
    # task_name: (suite_name, task_idx_in_suite, global_task_uid)
}
global_task_id = 0
for suite_name, suite_task_names in libero_task_map.items():
    for local_id, task_name in enumerate(suite_task_names):
        task_name_to_suite_and_ids[task_name] = (suite_name, local_id, global_task_id)
        global_task_id += 1
num_libero_tasks = global_task_id


class LiberoEnv(gymnasium.Env):
    def __init__(self,
        task_name: str,
        image_size: int = 128,
        seed: int = 42,
        camera_names: List = [
            'agentview',
            'robot0_eye_in_hand',
        ],
        state_ports: List = [
            'robot0_joint_pos',
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ],
        video_camera: str = 'agentview',
        video_resolution: int = 512,
        max_episode_steps: int = 550,
        enable_render: bool = True,
        protocol: str = "legacy",
    ):
        super().__init__()
        if protocol not in {"legacy", "corrected", "official"}:
            raise ValueError(f"Unknown evaluation protocol: {protocol}")
        self.protocol = protocol
        self.episode_seed = seed
        self.init_state_id = None
        self.init_states = None
        self.init_states_path = None

        libero_suite, task_suite_id, task_uid = task_name_to_suite_and_ids[task_name]
        task = benchmark.get_benchmark_dict()[libero_suite]().get_task(task_suite_id)
        if protocol == "official":
            # Same state file as LIBERO's evaluate_one_task_success. Explicit
            # CPU loading supports current torch defaults and trusted old files.
            import torch
            self.init_states_path = os.path.join(
                get_libero_path("init_states"), task.problem_folder,
                task.init_states_file,
            )
            self.init_states = torch.load(
                self.init_states_path, map_location="cpu", weights_only=False,
            )
        env = ControlEnv(
            bddl_file_name=os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file
            ),
            camera_names=list(set(list(camera_names) + [video_camera])),
            camera_heights=image_size,
            camera_widths=image_size,
            has_renderer=False,
            use_camera_obs=enable_render,
            has_offscreen_renderer=enable_render,
        )
        # env.env.hard_reset = False  # TODO: check if it's safe to set to False
        env.seed(seed)

        self.env = env
        self.task_name = task.name
        self.task_prompt = task.language
        self.task_uid = task_uid
        self.state_ports = state_ports
        self.camera_names = camera_names
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.done = False
        self.cur_step = 0

        # setup gym spaces
        obs_dict = env.env._get_observations()
        observation_space = gymnasium.spaces.Dict({})
        for port in state_ports:
            observation_space.spaces[port] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=obs_dict[port].shape, dtype=np.float32
            )
        for cam_name in camera_names:
            observation_space.spaces[f"{cam_name}_rgb"] = gymnasium.spaces.Box(
                low=0, high=255, 
                shape=(image_size, image_size, 3), dtype=np.uint8
            )
        observation_space.spaces['prompt'] = gymnasium.spaces.Text(
            min_length=0, max_length=512,
            charset=string.printable
        )
        observation_space.spaces['task_uid'] = gymnasium.spaces.Box(
            low=0, high=num_libero_tasks-1,
            shape=(1,), dtype=np.uint8
        )
        self.observation_space = observation_space
        self.action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(env.env.action_dim,), dtype=np.float32
        )
        self._let_objects_fall()

    def _let_objects_fall(self):
        # libero env needs a few steps to let objects fall to the table/ground
        dummy_action = [0.] * 6 + [-1.]
        raw_obs = None
        for _ in range(10):
            raw_obs, _, _, _ = self.env.step(dummy_action)
        return raw_obs

    def configure_episode(self, seed: int, init_state_id: Optional[int] = None):
        """Select the next reset; the runner performs that reset exactly once."""
        if self.protocol == "legacy":
            raise ValueError("Legacy protocol intentionally retains its original reset behavior")
        if self.protocol == "official":
            if init_state_id is None or not 0 <= init_state_id < len(self.init_states):
                raise ValueError(
                    f"Initial-state index {init_state_id} outside [0, {len(self.init_states)})"
                )
        elif init_state_id is not None:
            raise ValueError("Initial states require the official protocol")
        self.episode_seed = int(seed)
        self.init_state_id = init_state_id

    def _extract_obs(self, 
        raw_obs: Optional[Dict[str, np.ndarray]]=None
    ) -> Dict[str, np.ndarray]:
        if raw_obs is None:
            raw_obs = self.env.env._get_observations()

        obs_dict = {}

        # robot state
        for port in self.state_ports:
            obs_dict[port] = raw_obs[port].astype(np.float32)

        # rgb
        for cam_name in self.camera_names:
            obs_dict[f"{cam_name}_rgb"] = np.flip(
                raw_obs[f"{cam_name}_image"], axis=0).astype(np.uint8)
            
        # prompt & task uid
        obs_dict['prompt'] = self.task_prompt
        obs_dict['task_uid'] = np.array([self.task_uid,], dtype=np.uint8)

        return obs_dict
    
    def step(self, action: np.ndarray):
        obs, reward, terminated, info = self.env.step(action)
        self.cur_step += 1
        if self.env.check_success():
            reward = 1.0
        else:
            reward = 0.0
        self.done = self.done or terminated or (reward >= 1) \
            or (self.cur_step >= self.max_episode_steps)
        return self._extract_obs(obs), reward, self.done, False, info
    
    def reset(self, seed=None, options=None):
        if self.protocol == "legacy":
            # Preserve the historical stale observation and RNG behavior for
            # explicitly named reproductions of old reported results.
            obs = self.env.reset()
            obs_dict = self._extract_obs(obs)
            self.done = False
            self.cur_step = 0
            self._let_objects_fall()
            return obs_dict, {'prompt': self.task_prompt}

        episode_seed = self.episode_seed if seed is None else int(seed)
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        self.env.seed(episode_seed)
        raw_obs = self.env.reset()
        if self.protocol == "official":
            if self.init_state_id is None:
                raise ValueError("configure_episode must set an official initial-state index")
            raw_obs = self.env.set_init_state(self.init_states[self.init_state_id])
            # Exact convention in installed LIBERO/libero/lifelong/metric.py:
            # five all-zero actions after set_init_state, including gripper.
            for _ in range(5):
                raw_obs, _, _, _ = self.env.step(np.zeros(7))
        else:
            raw_obs = self._let_objects_fall()
        self.done = False
        self.cur_step = 0
        return self._extract_obs(raw_obs), {
            'prompt': self.task_prompt,
            'episode_seed': episode_seed,
            'init_state_id': self.init_state_id,
            'protocol': self.protocol,
        }
    
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        frame = np.flip(self.env.sim.render(
            height=self.video_resolution, width=self.video_resolution, 
            camera_name=self.video_camera
        ), axis=0).astype(np.uint8)
        return frame

    def close(self):
        self.env.close()
