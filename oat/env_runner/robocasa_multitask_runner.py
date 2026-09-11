"""Seeded RoboCasa evaluation for the Sink3 suite.

Each worker injects the canonical dataset task ID. Episode order and seeds are
independent of worker count, and completed episodes remain stopped until the
next scheduled reset.
"""

import math
import pathlib

import dill
import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wandb_video

from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.env.robocasa.factory import get_subtasks
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from oat.common.pytorch_util import dict_apply

from typing import Optional, List


def maybe_to_torch(x, device, dtype):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    else:
        return x


class RoboCasaMultiTaskRunner(BaseRunner):

    def __init__(
        self,
        output_dir,
        task_name: str = "sink3",
        robots: str = "PandaOmron",
        n_test: int = 150,
        n_test_vis: int = 9,
        test_start_seed: int = 1000,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 20,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_parallel_envs: Optional[int] = None,
        image_size: int = 128,
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
        max_episode_steps: int = 500,
        layout_ids: int = -1,
        style_ids: int = -1,
        protocol: str = "corrected",
    ):
        super().__init__(output_dir)

        if n_parallel_envs is None:
            n_parallel_envs = n_test
        n_parallel_envs = min(n_parallel_envs, n_test)

        if n_test < 1 or n_parallel_envs < 1:
            raise ValueError("Positive episode and worker counts are required")
        if not 0 <= n_test_vis <= n_test:
            raise ValueError("n_test_vis must be between zero and n_test")
        if protocol != "corrected":
            raise ValueError("RoboCasa supports the corrected seeded protocol")

        # Detect GPUs for distributing EGL rendering across devices.
        n_gpus = max(torch.cuda.device_count(), 1)

        # Round-robin ordering keeps task coverage independent of worker count.
        subtask_names = get_subtasks(task_name)
        env_task_names = [subtask_names[i % len(subtask_names)] for i in range(n_test)]

        def make_env(this_task_name, seed, enable_render, gpu_id):
            from oat.env.robocasa.multitask_env import MultiTaskRoboCasaEnv
            import os
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    MultiTaskRoboCasaEnv(
                        env_name=this_task_name,
                        robots=robots,
                        image_size=image_size,
                        seed=seed,
                        camera_names=camera_names,
                        state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                        enable_render=enable_render,
                        layout_ids=layout_ids,
                        style_ids=style_ids,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=1,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                reward_agg_method="max",
            )

        # Setup envs
        env_seeds = []
        env_fns = []
        env_init_fn_dills = []

        for i in range(n_test):
            this_task_name = env_task_names[i]
            this_seed = test_start_seed + i
            env_seeds.append(this_seed)
            enable_render = i < n_test_vis

            if i < n_parallel_envs:
                def env_fn(task_name=this_task_name, seed=this_seed, gpu_id=i % n_gpus):
                    return make_env(task_name, seed, enable_render=True, gpu_id=gpu_id)
                env_fns.append(env_fn)

            def init_fn(env, task_name=this_task_name, seed=this_seed,
                        enable_render=enable_render, gpu_id=(i % n_parallel_envs) % n_gpus):
                from oat.env.robocasa.multitask_env import MultiTaskRoboCasaEnv
                # A slot may change tasks when the worker count is not divisible
                # by the suite size. Every episode still gets its explicit seed.
                if env.env.env.env_name != task_name:
                    import os
                    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
                    env.env.env.close()
                    env.env.env = MultiTaskRoboCasaEnv(
                        env_name=task_name,
                        robots=robots,
                        image_size=image_size,
                        seed=seed,
                        camera_names=camera_names,
                        state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                        layout_ids=layout_ids,
                        style_ids=style_ids,
                    )
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        f"media/{task_name}",
                        wandb_video.util.generate_id() + ".mp4",
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                env.env.env.configure_episode(seed)

            env_init_fn_dills.append(dill.dumps(init_fn))

        assert len(env_fns) == n_parallel_envs
        assert len(env_init_fn_dills) == n_test

        # Render-free env for space initialization (no OpenGL context).
        def dummy_env_fn():
            return make_env(
                env_task_names[0], seed=test_start_seed,
                enable_render=False, gpu_id=0,
            )

        env = AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            dummy_env_fn=dummy_env_fn,
            context="forkserver",
            autoreset=False,
        )

        self.env = env
        self.task_name = task_name
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_task_names = env_task_names
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    @torch.inference_mode()
    def run(self, policy: BasePolicy, **kwargs):
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_success = [False] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # Init envs
            self.env.call_each(
                "run_dill_function",
                args_list=[(x,) for x in this_init_fns],
            )

            # Start rollout
            obs, _ = self.env.reset()
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Eval {policy_name} in RoboCasa::{self.task_name} "
                     f"{chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            while not done and pbar.n < pbar.total:
                obs_dict = dict_apply(
                    obs,
                    lambda x: maybe_to_torch(x, device=device, dtype=dtype),
                )

                with torch.inference_mode():
                    action = policy.predict_action(
                        {
                            port: obs_dict[port]
                            for port in policy.get_observation_ports()
                        },
                        **kwargs,
                    )["action"].detach().cpu().numpy()

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN or Inf action")

                obs, reward, env_done, _, _ = self.env.step(action)

                done = np.logical_or(
                    env_done[this_local_slice],
                    all_success[this_global_slice][this_local_slice],
                )
                done = np.all(done[this_local_slice])

                all_success[this_global_slice] = np.logical_or(
                    all_success[this_global_slice],
                    [r >= 1 for r in reward[this_local_slice]],
                )

                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = self.env.render()[this_local_slice]

        # Log
        log_data = dict()

        for task_name in set(self.env_task_names):
            task_success = [
                all_success[i]
                for i in range(n_inits)
                if self.env_task_names[i] == task_name
            ]
            log_data[f"{task_name}/mean_success_rate"] = np.mean(task_success)

        for i in range(n_inits):
            seed = self.env_seeds[i]
            task_name = self.env_task_names[i]
            video_path = all_video_paths[i]
            if video_path is not None:
                video = wandb.Video(video_path, format="mp4")
                log_data[f"{task_name}/video_{seed}"] = video

        log_data["mean_success_rate"] = np.mean(all_success)

        return log_data

    def close(self):
        self.env.close()
