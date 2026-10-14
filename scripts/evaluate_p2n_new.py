#!/usr/bin/env python3
"""Evaluate modern P2N LIBERO checkpoints with their embedded policy and runner."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_candidate import atomic_json, git_text, sha256, summarize_records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path, help='Must not already exist')
    p.add_argument('--gpu', default='0', help='Physical GPU index or UUID from nvidia-smi')
    p.add_argument('--n-test', type=int, default=500, help='Total episodes across the suite')
    p.add_argument('--n-parallel-envs', type=int, default=10)
    p.add_argument('--protocol', choices=('corrected', 'official'), default='corrected')
    p.add_argument('--weights', choices=('ema', 'model'), default='ema')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--episode-start-seed', type=int, default=1000)
    p.add_argument('--max-episode-steps', type=int, default=550)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    p.add_argument('--wandb-project', help='Defaults to the checkpoint training project')
    p.add_argument('--wandb-name')
    p.add_argument('--dry-run', action='store_true', help='Restore on CPU and save provenance/schedule; no GPU, simulator, or W&B run')
    return p


def validate_args(args):
    for name in ('n_test', 'n_parallel_envs', 'max_episode_steps', 'threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if not 0 <= args.seed < 2**32:
        raise ValueError('--seed must fit an unsigned 32-bit integer')
    if args.episode_start_seed < 0 or args.episode_start_seed + args.n_test > 2**32:
        raise ValueError('Episode seeds must fit unsigned 32-bit integers')
    if not args.gpu or ',' in args.gpu:
        raise ValueError('--gpu selects exactly one physical GPU')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f'Output directory already exists: {args.output_dir}')


def load_policy(args, payload):
    """Dispatch to the modern class loader, which restores embedded assets."""
    from hydra.utils import get_class
    from oat.policy.p2n_new_common import P2NNewCommonPolicy

    config = payload.get('policy_config', payload['cfg']['policy'])
    cls = get_class(config['_target_'])
    if not issubclass(cls, P2NNewCommonPolicy):
        raise ValueError('This evaluator requires a modern P2N policy checkpoint')
    return cls.from_checkpoint(
        str(args.checkpoint), return_configuration=True, weights=args.weights,
    )


def runner_config_for(args, cfg, policy):
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(cfg.task.policy.env_runner, resolve=True)
    expected_name = ('P2NStateGateNewLiberoRunner' if getattr(policy, 'variant', None) == 'p2n_state_gate_new'
                     else 'P2NNewLiberoRunner')
    if str(config.get('_target_', '')).rsplit('.', 1)[-1] != expected_name:
        raise ValueError('Checkpoint runner does not match the modern LIBERO policy')
    # The wrapper preserves the same history/byte adapters while scoping EGL.
    config['_target_'] = 'oat.env_runner.p2n_new_convnext_libero10_runner.' + expected_name
    config.update(
        output_dir=str(args.output_dir), n_test=args.n_test, n_test_vis=0,
        n_parallel_envs=min(args.n_parallel_envs, args.n_test),
        test_start_seed=args.episode_start_seed, protocol=args.protocol,
        max_episode_steps=args.max_episode_steps, init_state_offset=0,
        n_action_steps=policy.n_action_steps, n_obs_steps=policy.n_obs_steps,
        episode_records_path=str(args.output_dir / 'episodes.jsonl'),
    )
    return config


def seed_everything(seed, *, cuda=False):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


class EpisodeProgress:
    """Log complete atomic episode snapshots while the runner is busy."""
    def __init__(self, path, run):
        self.path, self.run = path, run
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.last_count = 0
        self.error = None

    def _watch(self):
        while not self.stop_event.wait(2):
            try:
                if not self.path.exists():
                    continue
                records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
                if len(records) <= self.last_count:
                    continue
                successes = sum(int(record['success']) for record in records)
                self.run.log({'eval/episodes': len(records), 'eval/successes': successes,
                              'eval/success_rate': successes / len(records)})
                self.last_count = len(records)
            except Exception as error:
                self.error = f'{type(error).__name__}: {error}'
                return

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)


def main(argv=None):
    args = parser().parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().absolute()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT_DIR)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(args.threads))
    started = time.monotonic()
    metadata = {
        'schema_version': 1, 'status': 'initializing',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint),
        'repository': str(ROOT_DIR), 'python': sys.executable,
        'git_head': git_text('rev-parse', 'HEAD'), 'git_status': git_text('status', '--porcelain'),
        'weights': args.weights,
    }
    atomic_json(args.output_dir / 'metadata.json', metadata)
    runner = run = progress = None
    completed = False
    try:
        devices = None
        if args.dry_run:
            # Avoid device initialization even if the invoking shell has a mask.
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ.pop('MUJOCO_EGL_DEVICE_ID', None)
        else:
            from scripts.train_p2n_new_convnext import check_gpu_idle
            from scripts.train_p2n_new_convnext_libero10 import configure_training_devices
            selected = check_gpu_idle(args.gpu, required_devices=1)
            devices = configure_training_devices(selected)

        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf
        from oat.env.libero.factory import get_subtasks
        # No contexts are created by these imports. During a real run, scope
        # robosuite's import-time renderer check to the selected EGL device.
        from contextlib import nullcontext
        from oat.env_runner.p2n_new_convnext_libero10_runner import renderer_env
        with nullcontext() if args.dry_run else renderer_env():
            from oat.env_runner.libero_runner import build_episode_schedule

        torch.set_num_threads(args.threads)
        seed_everything(args.seed, cuda=not args.dry_run)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        payload = torch.load(str(args.checkpoint), map_location='cpu', pickle_module=dill,
                             weights_only=False, mmap=True)
        policy, cfg = load_policy(args, payload)
        checkpoint_metadata = payload.get('metadata', {})
        del payload
        policy.to('cpu' if args.dry_run else 'cuda:0').eval()
        runner_config = runner_config_for(args, cfg, policy)
        all_tasks = get_subtasks(runner_config['task_name'])
        tasks = runner_config.get('task_names') or all_tasks
        schedule = build_episode_schedule(tasks, args.n_test, runner_config['n_parallel_envs'],
                                          args.episode_start_seed, args.protocol, 0)
        counts = Counter(record['task_name'] for record in schedule)
        atomic_json(args.output_dir / 'schedule.json', schedule)
        atomic_json(args.output_dir / 'runner_config.json', runner_config)
        atomic_json(args.output_dir / 'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
        inference = {'temperature': policy.temperature, 'topk': policy.topk,
                     'use_k_tokens': policy.max_seq_len, 'n_action_steps': policy.n_action_steps}
        metadata.update(
            status='ready' if args.dry_run else 'running', devices=devices,
            policy_class=f'{type(policy).__module__}.{type(policy).__name__}',
            policy_metadata=checkpoint_metadata, effective_inference=inference,
            task_episode_counts=dict(counts), all_tasks_covered=set(counts) == set(all_tasks),
            balanced_schedule=len(set(counts.values())) == 1,
            settling_steps=5 if args.protocol == 'official' else 10,
            load_seconds=time.monotonic() - started,
        )
        atomic_json(args.output_dir / 'metadata.json', metadata)
        if args.dry_run:
            print(json.dumps({'status': 'ready', 'output_dir': str(args.output_dir),
                              'task_episode_counts': dict(counts)}, sort_keys=True))
            return 0

        runner = hydra.utils.instantiate(runner_config)
        import wandb
        project = args.wandb_project or OmegaConf.select(cfg, 'logging.project') or 'p2n_new_libero10'
        run = wandb.init(project=project, mode=args.wandb_mode, job_type='evaluation',
                         name=args.wandb_name or args.output_dir.name, dir=str(args.output_dir),
                         config={**metadata['arguments'], 'policy_class': metadata['policy_class'],
                                 'checkpoint_sha256': metadata['checkpoint_sha256'],
                                 'effective_inference': inference})
        metadata['wandb_url'] = run.url
        atomic_json(args.output_dir / 'metadata.json', metadata)
        if run.url:
            print(f'W&B: {run.url}', flush=True)
        progress = EpisodeProgress(args.output_dir / 'episodes.jsonl', run)
        progress.start()
        seed_everything(args.seed, cuda=True)
        rollout_started = time.monotonic()
        runner.run(policy)
        rollout_seconds = time.monotonic() - rollout_started
        progress.stop()
        records = runner.last_episode_records
        if len(records) != args.n_test:
            raise RuntimeError(f'Expected {args.n_test} episodes, received {len(records)}')
        summary = summarize_records(records)
        summary.update(checkpoint=str(args.checkpoint), weights=args.weights, protocol=args.protocol,
                       seed=args.seed, max_episode_steps=args.max_episode_steps,
                       rollout_seconds=rollout_seconds, effective_inference=inference,
                       all_tasks_covered=metadata['all_tasks_covered'],
                       balanced_schedule=metadata['balanced_schedule'])
        atomic_json(args.output_dir / 'summary.json', summary)
        metrics = {'eval/episodes': summary['trials'], 'eval/successes': summary['successes'],
                   'eval/success_rate': summary['success_rate'],
                   'eval/macro_task_success_rate': summary['macro_task_success_rate'],
                   'eval/rollout_seconds': rollout_seconds}
        metrics.update({f'eval/task/{name}/success_rate': value['success_rate']
                        for name, value in summary['per_task'].items()})
        run.log(metrics)
        run.summary.update(metrics)
        if progress.error:
            metadata['wandb_progress_error'] = progress.error
        completed = True
        metadata.update(status='complete', total_seconds=time.monotonic() - started)
        atomic_json(args.output_dir / 'metadata.json', metadata)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}',
                        traceback=traceback.format_exc(), total_seconds=time.monotonic() - started)
        atomic_json(args.output_dir / 'metadata.json', metadata)
        raise
    finally:
        if progress is not None:
            progress.stop()
        if runner is not None:
            runner.close()
        if run is not None:
            run.finish(exit_code=0 if completed else 1)


if __name__ == '__main__':
    raise SystemExit(main())
