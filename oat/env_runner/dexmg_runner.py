"""Seeded single-task DexMimicGen evaluation (robosuite 1.5; LIBERO is never imported).

Episode i of a run uses seed ``test_start_seed + i``; which episodes this process evaluates is
``range(shard_index, n_test, num_shards)``, so the episode set and its seeds are independent of the
worker count and of how evaluation is sharded across processes/GPUs. Workers never autoreset and a
finished episode stays finished until its slot's next scheduled reset.

EGL: each worker selects its render device by setting ``MUJOCO_EGL_DEVICE_ID`` to an entry of
``egl_device_ids`` and dropping ``CUDA_VISIBLE_DEVICES`` *inside the worker, before robosuite is
imported* (forkserver workers do not see later parent environment changes; robosuite asserts the
two agree at import time and otherwise reads CUDA_VISIBLE_DEVICES as an EGL index, which is the
wrong GPU on hosts whose /dev/nvidia minors are reversed). EGL indices are not CUDA ordinals:
probe them with ``oat/common/libero_egl_devices.py``.
"""

from contextlib import contextmanager
import json
import math
import os
import pathlib
import uuid

import dill
import numpy as np
import torch
import tqdm
import wandb

from oat.common.p2n_new_capabilities import policy_capability
from oat.common.pytorch_util import dict_apply
from oat.env.dexmg.factory import get_task
from oat.env_runner.base_runner import BaseRunner
from oat.env_runner.byte_observation_policy import ByteObservationPolicy
from oat.env_runner.executed_action_common import _ExecutedPastRunnerMixin
from oat.env_runner.state_history_common import _StateHistoryRunnerMixin
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper


def _to_torch(value, device, dtype):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device=device, dtype=dtype)
    return value


@contextmanager
def _without_parent_egl_selection():
    """The policy process must not carry MUJOCO_EGL_DEVICE_ID into robosuite's import-time assert."""
    previous = os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = previous


def _make_env(task_name, zarr_path, egl_device_id, image_size, camera_names, state_ports,
              max_episode_steps, n_obs_steps, n_action_steps, fps, crf, enable_render):
    if enable_render and egl_device_id is not None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(egl_device_id))
    from oat.env.dexmg.env import DexMGEnv
    env = DexMGEnv(
        task_name=task_name, zarr_path=zarr_path, state_ports=state_ports,
        camera_names=camera_names, image_size=image_size, max_episode_steps=max_episode_steps,
        enable_render=enable_render,
        render_gpu_device_id=int(egl_device_id) if enable_render and egl_device_id is not None else -1,
    )
    return MultiStepWrapper(
        VideoRecordingWrapper(
            env,
            video_recoder=VideoRecorder.create_h264(
                fps=fps, codec="h264", input_pix_fmt="rgb24", crf=crf,
                thread_type="FRAME", thread_count=1),
            file_path=None,
            steps_per_render=1,
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
        reward_agg_method="max",
    )


class DexMGRunner(BaseRunner):
    def __init__(
        self,
        output_dir,
        task_name="box_cleanup",
        zarr_path=None,
        n_test=200,
        n_test_vis=4,
        test_start_seed=100000,
        shard_index=0,
        num_shards=1,
        n_obs_steps=2,
        n_action_steps=8,
        fps=20,
        crf=22,
        tqdm_interval_sec=5.0,
        n_parallel_envs=25,
        image_size=84,
        camera_names=None,
        state_ports=None,
        max_episode_steps=None,
        egl_device_ids=None,
        protocol="corrected",
        dataset_init_episodes=None,
    ):
        super().__init__(output_dir)
        if protocol != "corrected":
            raise ValueError("DexMimicGen evaluation supports the corrected seeded protocol only")
        if zarr_path is None:
            raise ValueError("zarr_path (the converted dataset holding env_args) is required")
        if not 0 <= shard_index < num_shards:
            raise ValueError("shard_index must be in [0, num_shards)")
        if not 0 <= n_test_vis <= n_test:
            raise ValueError("n_test_vis must be between zero and n_test")
        if not egl_device_ids or any(int(x) < 0 for x in egl_device_ids):
            raise ValueError("egl_device_ids must list probed EGL indices (not CUDA ordinals)")
        task = get_task(task_name)
        camera_names = list(camera_names or task["cameras"])
        max_episode_steps = int(max_episode_steps or task["horizon"])
        # dataset_init_episodes: restore those recorded episodes' initial states instead of seeded
        # resets (diagnostic: separates policy quality from initial-state distribution shift).
        if dataset_init_episodes is not None:
            dataset_init_episodes = [int(x) for x in dataset_init_episodes]
            n_test = len(dataset_init_episodes)
            n_test_vis = min(n_test_vis, n_test)
        episodes = list(range(shard_index, n_test, num_shards))
        if not episodes:
            raise ValueError("This shard has no episodes")
        n_parallel_envs = max(1, min(int(n_parallel_envs), len(episodes)))
        egl_device_ids = [int(x) for x in egl_device_ids]

        make = dict(task_name=task_name, zarr_path=str(zarr_path), image_size=image_size,
                    camera_names=camera_names, state_ports=state_ports,
                    max_episode_steps=max_episode_steps, n_obs_steps=n_obs_steps,
                    n_action_steps=n_action_steps, fps=fps, crf=crf)
        env_fns = []
        for slot in range(n_parallel_envs):
            def env_fn(egl=egl_device_ids[slot % len(egl_device_ids)]):
                return _make_env(egl_device_id=egl, enable_render=True, **make)
            env_fns.append(env_fn)

        media = pathlib.Path(output_dir) / "media" / task_name
        init_fn_dills, seeds = [], []
        for episode in episodes:
            dataset_episode = None if dataset_init_episodes is None else dataset_init_episodes[episode]
            seed = int(test_start_seed + episode) if dataset_episode is None else -1 - dataset_episode
            seeds.append(seed)
            record = episode < n_test_vis

            def init_fn(env, seed=seed, record=record, dataset_episode=dataset_episode):
                env.env.video_recoder.stop()
                env.env.file_path = None
                if record:
                    media.mkdir(parents=True, exist_ok=True)
                    tag = f"seed{seed}" if dataset_episode is None else f"demo{dataset_episode}"
                    env.env.file_path = str(media / f"{tag}_{uuid.uuid4().hex[:8]}.mp4")
                if dataset_episode is None:
                    env.env.env.configure_episode(seed)
                else:
                    env.env.env.configure_dataset_init(dataset_episode)

            init_fn_dills.append(dill.dumps(init_fn))

        def dummy_env_fn():
            return _make_env(egl_device_id=None, enable_render=False, **make)

        with _without_parent_egl_selection():
            self.env = AsyncVectorEnv(env_fns, shared_memory=False, dummy_env_fn=dummy_env_fn,
                                      context="forkserver", autoreset=False)
        self.task_name = task_name
        self.env_fns = env_fns
        self.env_seeds = seeds
        self.episodes = episodes
        self.env_init_fn_dills = init_fn_dills
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.episode_log = pathlib.Path(output_dir) / f"dexmg_eval_episodes_shard{shard_index}of{num_shards}.jsonl"

    @torch.inference_mode()
    def run(self, policy, **kwargs):
        device, dtype = policy.device, policy.dtype
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        success = [False] * n_inits
        first_success_step = [None] * n_inits
        video_paths = [None] * n_inits
        for chunk in range(math.ceil(n_inits / n_envs)):
            start, end = chunk * n_envs, min(n_inits, (chunk + 1) * n_envs)
            active = end - start
            init_fns = list(self.env_init_fn_dills[start:end])
            init_fns += [self.env_init_fn_dills[0]] * (n_envs - active)
            self.env.call_each("run_dill_function", args_list=[(fn,) for fn in init_fns])
            obs, _ = self.env.reset()
            policy.reset()
            pbar = tqdm.tqdm(total=self.max_episode_steps, leave=False, mininterval=self.tqdm_interval_sec,
                             desc=f"Eval {policy.get_policy_name()} DexMG::{self.task_name} {chunk + 1}")
            steps = 0
            while steps < self.max_episode_steps:
                obs_t = dict_apply(obs, lambda x: _to_torch(x, device=device, dtype=dtype))
                action = policy.predict_action(
                    {port: obs_t[port] for port in policy.get_observation_ports()}, **kwargs,
                )["action"].detach().cpu().numpy()
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN or Inf action")
                obs, reward, env_done, _, _ = self.env.step(action)
                steps += action.shape[1]
                for local in range(active):
                    index = start + local
                    if reward[local] >= 1 and not success[index]:
                        success[index], first_success_step[index] = True, steps
                pbar.update(action.shape[1])
                if all(env_done[:active]) or all(success[start:end]):
                    break
            pbar.close()
            video_paths[start:end] = self.env.render()[:active]

        self.episode_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.episode_log, "a") as stream:
            for index in range(n_inits):
                stream.write(json.dumps({
                    "task": self.task_name, "episode": self.episodes[index], "seed": self.env_seeds[index],
                    "success": bool(success[index]), "first_success_step": first_success_step[index],
                }) + "\n")
        log = {"mean_success_rate": float(np.mean(success)), f"{self.task_name}/mean_success_rate": float(np.mean(success)),
               "n_episodes": n_inits}
        for index, path in enumerate(video_paths):
            if path is not None:
                log[f"{self.task_name}/video_{self.env_seeds[index]}"] = wandb.Video(path, format="mp4")
        return log

    def close(self):
        self.env.close()


class DexMGExecutedPastRunner(_ExecutedPastRunnerMixin, DexMGRunner):
    """Acknowledge executed command prefixes (R1g / R2 / R3 policies)."""


class DexMGStateHistoryRunner(_StateHistoryRunnerMixin, DexMGExecutedPastRunner):
    """Executed commands plus per-control-step measured-state history (R1g)."""


class P2NNewDexMGRunner(DexMGExecutedPastRunner):
    def run(self, policy, **kwargs):
        if not policy_capability(policy, "requires_execution_acknowledgement"):
            raise ValueError("This runner requires execution acknowledgement")
        if policy_capability(policy, "requires_state_history"):
            raise ValueError("State-history policy requires P2NStateGateNewDexMGRunner")
        return super().run(ByteObservationPolicy(policy), **kwargs)


class P2NStateGateNewDexMGRunner(DexMGStateHistoryRunner):
    def run(self, policy, **kwargs):
        if not policy_capability(policy, "requires_execution_acknowledgement") or not policy_capability(
                policy, "requires_state_history"):
            raise ValueError("Gate runner requires measured state history and execution acknowledgement")
        return super().run(ByteObservationPolicy(policy), **kwargs)
