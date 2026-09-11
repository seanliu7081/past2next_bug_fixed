import wandb
import numpy as np
import torch
import tqdm
import math
import pathlib
import dill
import json
import os
import tempfile
import time
import wandb.sdk.data_types.video as wandb_video

from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.env.libero.env import LiberoEnv
from oat.env.libero.factory import get_subtasks
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from oat.common.pytorch_util import dict_apply

from typing import Optional, List

def atomic_write_records(path, records):
    """Replace a complete JSONL snapshot after each finished rollout batch."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_episode_schedule(task_names, n_test, n_parallel_envs,
                           test_start_seed=1000, protocol="legacy",
                           init_state_offset=0):
    """Explicit schedules independent of worker count in corrected protocols."""
    if protocol not in {"legacy", "corrected", "official"}:
        raise ValueError(f"Unknown protocol {protocol}")
    if n_test < 1 or n_parallel_envs < 1 or not task_names:
        raise ValueError("Positive trials/workers and at least one task are required")
    if len(set(task_names)) != len(task_names):
        raise ValueError("Task filters must not contain duplicates")
    if init_state_offset < 0:
        raise ValueError("init_state_offset must be nonnegative")
    if protocol == "legacy":
        names = []
        for start in range(0, len(task_names), n_parallel_envs):
            batch = task_names[start:start + n_parallel_envs]
            for _ in range(math.ceil(n_test / len(task_names))):
                names.extend(batch)
        names = names[:n_test]
    else:
        names = [task_names[i % len(task_names)] for i in range(n_test)]
    task_counts = dict.fromkeys(task_names, 0)
    schedule = []
    for i, name in enumerate(names):
        task_trial = task_counts[name]
        schedule.append({
            "episode_index": i, "task_name": name, "task_trial": task_trial,
            "episode_seed": int(test_start_seed + i),
            "episode_seed_applied": protocol != "legacy",
            "init_state_id": int(init_state_offset + task_trial) if protocol == "official" else None,
            "protocol": protocol,
        })
        task_counts[name] += 1
    return schedule


def maybe_to_torch(x, device, dtype):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    else:
        return x
    
class LiberoRunner(BaseRunner):
    # import multiprocessing as mp
    # mp.set_start_method('spawn', force=True)

    def __init__(self,
        output_dir,
        task_name: str,
        n_test: int,
        n_test_vis: int,
        test_start_seed: int = 1000,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 20,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_parallel_envs: Optional[int] = None,
        # other libero env args
        image_size: int = 128,
        camera_names: List[str] = [
            "agentview",
            "robot0_eye_in_hand",
        ],
        state_ports: List[str] = [
            'robot0_joint_pos',
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ],
        max_episode_steps: int = 550,
        protocol: str = "legacy",
        task_names: Optional[List[str]] = None,
        init_state_offset: int = 0,
        episode_records_path: Optional[str] = None,
    ):
        super().__init__(output_dir)

        if n_parallel_envs is None:
            n_parallel_envs = n_test
        n_parallel_envs = min(n_parallel_envs, n_test)

        assert n_parallel_envs > 0, "n_parallel_envs must be positive"
        assert n_test_vis <= n_test, "n_test_vis must be less than or equal to n_test"

        # Preserve legacy ordering; corrected schedules do not depend on workers.
        all_subtask_names = get_subtasks(task_name)
        subtask_names = all_subtask_names if task_names is None else list(task_names)
        unknown = set(subtask_names) - set(all_subtask_names)
        if unknown:
            raise ValueError(f"Unknown tasks for {task_name}: {sorted(unknown)}")
        schedule = build_episode_schedule(
            subtask_names, n_test, n_parallel_envs, test_start_seed,
            protocol, init_state_offset,
        )
        env_task_names = [episode["task_name"] for episode in schedule]

        # setup env
        env_seeds = []
        env_fns = []
        env_init_fn_dills = []
        for i in range(n_test):
            this_task_name = env_task_names[i]
            this_seed = schedule[i]["episode_seed"]
            this_init_state_id = schedule[i]["init_state_id"]
            env_seeds.append(this_seed)
            enable_render = i < n_test_vis

            if i < n_parallel_envs:
                def env_fn(task_name=this_task_name, seed=this_seed):
                    return MultiStepWrapper(
                        VideoRecordingWrapper(
                            LiberoEnv(
                                task_name=task_name,
                                image_size=image_size,
                                seed=seed,
                                camera_names=camera_names,
                                state_ports=state_ports,
                                max_episode_steps=max_episode_steps,
                                protocol=protocol,
                            ),
                            video_recoder=VideoRecorder.create_h264(
                                fps=fps,
                                codec='h264',
                                input_pix_fmt='rgb24',
                                crf=crf,
                                thread_type='FRAME',
                                thread_count=1
                            ),
                            file_path=None,
                            steps_per_render=1
                        ),
                        n_obs_steps=n_obs_steps,
                        n_action_steps=n_action_steps,
                        max_episode_steps=max_episode_steps,
                        reward_agg_method='max'
                    )
                env_fns.append(env_fn)
        
            def init_fn(env, task_name=this_task_name, seed=this_seed,
                        enable_render=enable_render, init_state_id=this_init_state_id):
                if env.env.env.task_name != task_name:
                    env.env.env.close()
                    env.env.env = LiberoEnv(
                        task_name=task_name,
                        image_size=image_size,
                        seed=seed,
                        camera_names=camera_names,
                        state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                        protocol=protocol,
                    )
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        f'media/{task_name}', wandb_video.util.generate_id() + '.mp4'
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                if protocol == "legacy":
                    env.reset()
                else:
                    env.env.env.configure_episode(seed, init_state_id)
            env_init_fn_dills.append(dill.dumps(init_fn))

        assert len(env_fns) == n_parallel_envs
        assert len(env_init_fn_dills) == n_test

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    LiberoEnv(
                        task_name=env_task_names[0],
                        image_size=image_size,
                        # seed=seed,
                        camera_names=camera_names,
                        state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                        enable_render=False,
                        protocol=protocol,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=1
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                reward_agg_method='max'
            )

        env = AsyncVectorEnv(env_fns, shared_memory=False,
            dummy_env_fn=dummy_env_fn,
            # MultiStepWrapper already retains a completed episode. Automatic
            # resets otherwise discard its history and can skip the first action
            # after an explicit reset when shared_memory=False.
            autoreset=protocol == "legacy",
            # context='spawn',
        )  # NOTE: turn off shared_memory to use Text space

        # attr assignment
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
        self.protocol = protocol
        self.episode_schedule = schedule
        self.episode_records_path = episode_records_path
        self.last_episode_records = []

    @torch.inference_mode()
    def run(self, 
        policy: BasePolicy,
        # policy inference args
        **kwargs,
    ):
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        self.last_episode_records = []
        run_started = time.monotonic()

        # allocate data
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

            chunk_started = time.monotonic()
            # init envs
            self.env.call_each(
                'run_dill_function', 
                args_list=[(x,) for x in this_init_fns]
            )

            # start rollout
            obs, _ = self.env.reset()
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Eval {policy_name} in Libero::{self.task_name} {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec
            )

            done = False
            while not done and pbar.n < pbar.total:
                # create obs dict
                obs_dict = dict_apply(
                    obs,
                    lambda x: maybe_to_torch(x, device=device, dtype=dtype)
                )

                # run policy
                with torch.inference_mode():
                    action = policy.predict_action({
                        port: obs_dict[port] 
                        for port in policy.get_observation_ports()
                    }, **kwargs)['action'].detach().cpu().numpy()

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN of Inf action")

                # step env
                obs, reward, done, _, _ = self.env.step(action) # NOTE: reward=1 if success
                done = np.logical_or(
                    done[this_local_slice],
                    all_success[this_global_slice][this_local_slice]
                )
                done = np.all(done[this_local_slice])

                all_success[this_global_slice] = np.logical_or(
                    all_success[this_global_slice],
                    [r >= 1 for r in reward[this_local_slice]]
                )

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = self.env.render()[this_local_slice]
            episode_rewards = self.env.call("get_rewards")
            chunk_seconds = time.monotonic() - chunk_started
            for local_i, global_i in enumerate(range(start, end)):
                record = dict(self.episode_schedule[global_i])
                steps = len(episode_rewards[local_i])
                record.update({
                    "success": bool(all_success[global_i]),
                    "policy_steps": steps,
                    "max_episode_steps": self.max_episode_steps,
                    "n_action_steps": self.n_action_steps,
                    "video_path": all_video_paths[global_i],
                    "rollout_batch_seconds": chunk_seconds,
                    "elapsed_seconds": time.monotonic() - run_started,
                    "termination": "success" if all_success[global_i] else
                                   "timeout" if steps >= self.max_episode_steps else "terminated",
                })
                self.last_episode_records.append(record)
            if self.episode_records_path is not None:
                atomic_write_records(self.episode_records_path, self.last_episode_records)

        # clear out video buffer
        if self.protocol == "legacy":
            _ = self.env.reset()

        # log
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

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                video = wandb.Video(video_path, format='mp4')
                log_data[f"{task_name}/video_{seed}"] = video
            
        # log aggregate metrics
        log_data['mean_success_rate'] = np.mean(all_success)
        
        return log_data

    def close(self):
        self.env.close()
