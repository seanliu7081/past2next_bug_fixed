"""Seeded MimicGen rollouts with feedback for each executed action chunk."""
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch

from oat.env_runner.base_runner import BaseRunner
from oat.env_runner.p2n_new_convnext_libero10_runner import renderer_env
from oat.env_runner.p2n_vla_rollout import close_runner, resolve_renderer
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper


class _EpisodeChunks(MultiStepWrapper):
    """Keep finished vector slots stationary and return actual primitive step counts."""

    def reset(self, **kwargs):
        self.terminated = self.truncated = self.success = False
        return super().reset(**kwargs)

    def step(self, actions):
        executed = 0
        for action in actions:
            if self.terminated or self.truncated:
                break
            obs, _, self.terminated, self.truncated, info = self.env.step(action)
            self.obs.append(obs)
            executed += 1
            self.success = bool(info.get("success", self.terminated))
            self.terminated = self.terminated or self.success
        return (self._get_obs(self.n_obs_steps), float(self.success), self.terminated,
                self.truncated, {"executed": executed, "success": self.success})


def _make_env(task_name, uid, image_size, egl_device, max_episode_steps, n_obs_steps, n_action_steps):
    from oat.env.mimicgen.env import MimicGenEnv
    env = MimicGenEnv(task_name, uid, image_size=image_size, render_gpu_device_id=egl_device,
                      max_episode_steps=max_episode_steps)
    return _EpisodeChunks(env, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


class MimicGenRunner(BaseRunner):
    def __init__(self, output_dir, task_names, n_per_task=10, n_parallel_envs=4, n_obs_steps=2,
                 n_action_steps=8, image_size=128, seed=42, max_episode_steps=None):
        super().__init__(output_dir)
        from oat.env.mimicgen.env import TASK_UIDS
        if not task_names or len(task_names) != len(set(task_names)):
            raise ValueError("task_names must contain distinct MimicGen tasks")
        self.tasks = [(name, TASK_UIDS[name]) for name in task_names]
        if min(n_per_task, n_parallel_envs, n_obs_steps, n_action_steps, image_size) < 1 or seed < 0:
            raise ValueError("MimicGen rollout counts must be positive and seed must be nonnegative")
        self.n_per_task = n_per_task
        self.n_parallel_envs = n_parallel_envs
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.image_size = image_size
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.env = None

    def _open_pool(self, name, uid, width, renderer):
        egl_device = renderer["egl_device_id"] if renderer is not None else None
        factory = partial(_make_env, name, uid, self.image_size, egl_device,
                          self.max_episode_steps, self.n_obs_steps, self.n_action_steps)
        # Spawn a clean process for each simulator, with the physical renderer selected
        # before robosuite imports EGL. The parent retains its training CUDA mask.
        scope = renderer_env(egl_device) if renderer is not None else nullcontext()
        with scope:
            return AsyncVectorEnv([factory] * width, context="spawn", autoreset=False)

    def _episodes(self, policy, seeds, predict_kwargs):
        obs, _ = self.env.reset(seed=seeds)
        policy.reset()
        ports = policy.get_observation_ports()
        while True:
            inputs = {key: torch.as_tensor(obs[key], device=policy.device, dtype=policy.dtype)
                      for key in ports}
            prediction = policy.predict_action(inputs, **predict_kwargs)["action"]
            actions = prediction.detach().to(device="cpu", dtype=torch.float32).numpy()
            if actions.shape != (len(seeds), self.n_action_steps, 7) or not np.isfinite(actions).all():
                raise ValueError("Expected finite [batch, n_action_steps, 7] MimicGen actions")
            obs, _, terminated, truncated, info = self.env.step(actions)
            policy.record_executed_actions(actions, executed_lengths=info["executed"])
            if np.all(terminated | truncated):
                return int(np.count_nonzero(info["success"]))

    @torch.inference_mode()
    def run(self, policy, **kwargs):
        renderer = resolve_renderer(policy.device)
        metrics = {}
        total_successes = 0
        for name, uid in self.tasks:
            successes = 0
            start = 0
            while start < self.n_per_task:
                width = min(self.n_parallel_envs, self.n_per_task - start)
                self.env = self._open_pool(name, uid, width, renderer)
                failed = True
                try:
                    while self.n_per_task - start >= width:
                        seeds = [int(np.random.SeedSequence([self.seed, uid, ep]).generate_state(1)[0])
                                 for ep in range(start, start + width)]
                        successes += self._episodes(policy, seeds, kwargs)
                        start += width
                    failed = False
                finally:
                    close_runner(self, force=failed)
                    self.env = None
            metrics[f"{name}/mean_success_rate"] = successes / self.n_per_task
            total_successes += successes
        metrics["n_episodes"] = self.n_per_task * len(self.tasks)
        metrics["mean_success_rate"] = total_successes / metrics["n_episodes"]
        return metrics

    def close(self):
        if self.env is not None:
            self.env.close()
