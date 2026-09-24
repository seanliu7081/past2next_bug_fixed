#!/usr/bin/env python3
"""Bounded real-weight acceptance probe; no downloads and no training loop.

Examples (run from the repository with its Python environment):
  python scripts/smoke_p2n_new.py --variant p2n_new --task real_robot \
    --dino /local/snapshots/COMMIT --device cuda --output /tmp/p2n-smoke.json
  torchrun --standalone --nproc_per_node=2 scripts/smoke_p2n_new.py \
    --variant p2n_state_gate_new --task real_robot --dino /local/snapshots/COMMIT \
    --distributed --device cuda --output /tmp/p2n-gate-ddp.json
  python scripts/smoke_p2n_new.py --variant p2n_new --task real_robot \
    --count-only --output /tmp/p2n-counts.json

--count-only creates the complete production architecture and uses the selected
OAT EMA/normalizers, but does not execute uninitialized DINO weights or count as
real-weight acceptance. Hydra overrides after -- are passed to the task config.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate

from oat.common.hydra_util import register_new_resolvers
from oat.policy.p2n_new_common import file_sha256
from scripts.train_p2n_new import inspect_dataset, validate_config, validate_tokenizer_source


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', required=True, choices=('p2n_new', 'p2n_state_gate_new'))
    parser.add_argument('--task', required=True, choices=('libero', 'real_robot'))
    parser.add_argument('--dino', help='Approved local DINOv3-S/16 snapshot; never downloaded by this script.')
    parser.add_argument('--dino-revision', help='Exact 40-character commit; inferred from HF snapshot folder if omitted.')
    parser.add_argument('--tokenizer', help='Frozen OAT checkpoint with EMA weights; real-robot config supplies the selected checkpoint.')
    parser.add_argument('--dataset', help='Override the selected task zarr_path.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--val-batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=2, help='Full predict_action warmup calls per batch size.')
    parser.add_argument('--iterations', type=int, default=5, help='Timed predict_action calls per batch size.')
    parser.add_argument('--max-scan-samples', type=int, default=4096)
    parser.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--distributed', action='store_true', help='Use an existing torchrun process group (NCCL on CUDA, Gloo on CPU).')
    parser.add_argument('--count-only', action='store_true')
    parser.add_argument('--output', type=Path, help='JSON report, written only by rank zero.')
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.val_batch_size, args.iterations, args.max_scan_samples) < 1 or args.warmup < 0:
        parser.error('Batch sizes, iterations and scan limit must be positive; warmup must be nonnegative.')
    if args.count_only and args.distributed:
        parser.error('--count-only is a single-process CPU architecture inspection.')
    if not args.count_only and not args.dino:
        parser.error('Real-weight smoke requires --dino with an approved local snapshot.')
    return args


def compose_config(args):
    name = f'train_{args.variant}'
    if args.task == 'real_robot':
        name = f'experimental/{name}_real_robot'
    overrides = args.overrides[1:] if args.overrides[:1] == ['--'] else args.overrides
    register_new_resolvers()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / 'oat/config')):
        cfg = hydra.compose(config_name=name, overrides=overrides)
    if cfg.variant != args.variant or cfg.task_type != args.task:
        raise ValueError('Hydra overrides may not change the explicitly selected variant/task.')
    if args.tokenizer:
        cfg.policy.tokenizer_checkpoint = args.tokenizer
    if args.dino:
        cfg.policy.dino_path = args.dino
    if args.dino_revision:
        cfg.policy.dino_revision = args.dino_revision
    if args.dataset:
        cfg.task.policy.dataset.zarr_path = args.dataset
    if cfg.training.resume:
        raise ValueError('This bounded smoke probe uses fresh frozen weights; resume acceptance belongs to the training checkpoint path.')
    cfg.policy.construction_mode = 'fresh'
    return cfg


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move(item, device) for key, item in value.items()}
    return value


def take(value, count):
    if isinstance(value, torch.Tensor):
        return value[:count]
    if isinstance(value, Mapping):
        return {key: take(item, count) for key, item in value.items()}
    return value


def valid_batch(dataset, size, max_scan, rank):
    """Use actual windows; do not manufacture previous-window validity."""
    selected, indices = [], []
    start = rank * size
    for offset in range(min(len(dataset), max_scan)):
        index = (start + offset) % len(dataset)
        sample = dataset[index]
        if 'prev_window_valid' not in sample:
            raise KeyError('Dataset must emit prev_window_valid and independent action-validity masks.')
        if bool(sample['prev_window_valid']) and bool(sample['past_action_valid'].all()):
            selected.append(sample)
            indices.append(index)
            if len(selected) == size:
                return default_collate(selected), indices
    raise ValueError(f'Could not find {size} real windows with valid previous windows in {min(len(dataset), max_scan)} samples.')


def memory_stats(device):
    if device.type != 'cuda':
        return {'max_memory_allocated_bytes': None, 'max_memory_reserved_bytes': None}
    return {'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'max_memory_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(function, device):
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    value = function()
    synchronize(device)
    return value, {'seconds': time.perf_counter() - start, **memory_stats(device)}


def count_policy(cfg):
    """Actual modules, full dimensions, no forward with uninitialized DINO."""
    from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessorFast

    source = Path(cfg.policy.tokenizer_checkpoint)
    payload = torch.load(source, map_location='cpu', pickle_module=dill, weights_only=False)
    tokenizer_config = OmegaConf.to_container(payload['cfg'].tokenizer, resolve=True)
    tokenizer = hydra.utils.instantiate(tokenizer_config)
    tokenizer.load_state_dict(payload['state_dicts']['ema_model'], strict=True)
    del payload
    return hydra.utils.instantiate(
        cfg.policy, _recursive_=False, construction_mode='restore', action_tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        tokenizer_metadata={'source': str(source), 'sha256': file_sha256(source), 'weights': 'ema'},
        dino_config=DINOv3ViTConfig(num_register_tokens=4).to_dict(),
        processor_config=DINOv3ViTImageProcessorFast().to_dict(),
    )


def main(args=None):
    args = parse_args(args)
    cfg = compose_config(args)
    validate_config(cfg, args.variant, args.task)
    validate_tokenizer_source(cfg)
    inspected_split = inspect_dataset(cfg)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    rank, world_size = 0, 1
    device = torch.device('cpu' if args.count_only else args.device)
    if args.distributed:
        if 'RANK' not in os.environ or int(os.environ.get('WORLD_SIZE', 1)) < 2:
            raise ValueError('--distributed requires torchrun with at least two ranks.')
        if device.type == 'cuda':
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
            torch.cuda.set_device(device)
        distributed.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
    try:
        # Fail on weight/config issues before loading a potentially large dataset.
        policy = count_policy(cfg) if args.count_only else hydra.utils.instantiate(cfg.policy)
        dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
        validation = dataset.get_validation_dataset()
        policy.set_normalizer(dataset.get_normalizer())
        report = {'status': 'architecture_count_only' if args.count_only else 'smoke_passed',
                  'real_dino_weights_validated': not args.count_only,
                  'variant': args.variant, 'task': args.task, 'rank': rank, 'world_size': world_size,
                  'device': str(device), 'precision': 'not_executed' if args.count_only else args.precision,
                  'validation_weights': 'raw', 'training_timing_includes_gradient_checks': True,
                  'metadata': policy.artifact_metadata(),
                  'dataset': {'path': str(cfg.task.policy.dataset.zarr_path),
                              'train_windows': len(dataset), 'validation_windows': len(validation),
                              'train_episode_ids': np.flatnonzero(dataset.train_mask).tolist(),
                              'validation_episode_ids': np.flatnonzero(validation.train_mask).tolist(),
                              'normalizer_source': 'actual_training_episode_mask',
                              'preflight': inspected_split},
                  'component_parameters': {
                      name: {'total': sum(p.numel() for p in module.parameters()),
                             'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)}
                      for name, module in policy.named_children()},
                  'module_tree': [name for name, _ in policy.named_children()]}
        if args.count_only:
            report['limitation'] = 'DINO architecture only: no approved pretrained DINO weights loaded, no forward/training/GPU acceptance performed.'
            return write_report(report, args.output, rank)
        if device.type == 'cuda' and args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Requested BF16 is unsupported on this CUDA device; choose an explicit supported precision.')
        report['hardware'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
        train_batch, train_indices = valid_batch(dataset, args.batch_size, args.max_scan_samples, rank)
        val_batch, val_indices = valid_batch(validation, args.val_batch_size, args.max_scan_samples, rank)
        report['dataset'].update(train_sample_indices=train_indices, validation_sample_indices=val_indices)
        train_batch, val_batch = move(train_batch, device), move(val_batch, device)
        policy.to(device)
        # Match training's EMA residency and update after every successful step.
        ema = copy.deepcopy(policy).requires_grad_(False).eval() if cfg.training.use_ema else None
        ema_helper = hydra.utils.instantiate(cfg.ema, model=ema) if ema is not None else None
        report['ema_resident'] = ema is not None
        optimizer = policy.get_optimizer(**cfg.optimizer)
        model = DistributedDataParallel(policy, device_ids=[device.index] if device.type == 'cuda' else None,
                                        find_unused_parameters=False, broadcast_buffers=False) if args.distributed else policy
        autocast = lambda: (torch.autocast(device.type, dtype=torch.bfloat16) if args.precision == 'bf16'
                            else contextlib.nullcontext())

        def train_step(history_mode):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = model(train_batch, history_mode=history_mode)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite {history_mode} training loss')
            loss.backward()
            for name, parameter in policy.named_parameters():
                if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f'Missing or nonfinite gradient for {name}')
                if not parameter.requires_grad and parameter.grad is not None:
                    raise RuntimeError(f'Frozen parameter received a gradient: {name}')
            gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.max_grad_norm)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError('Nonfinite global gradient norm')
            optimizer.step()
            policy.on_optimizer_step()
            if ema is not None:
                ema_helper.step(policy)
                ema.set_self_past_step(policy.self_past_step)
            return float(loss.detach())

        # Establish Adam moments before measuring normal training/self-past peaks.
        train_step('expert')
        report['adam_parameter_states'] = len(optimizer.state)
        report['measurements'] = {}
        for history_mode in ('expert', 'generated'):
            loss, stats = measure(lambda: train_step(history_mode), device)
            report['measurements'][f'train_{history_mode}'] = {'loss': loss, 'batch_size': args.batch_size, **stats}
        policy.eval()
        for history_mode in ('expert', 'generated'):
            def validate():
                with torch.no_grad(), autocast():
                    value = policy(val_batch, history_mode=history_mode)
                if not torch.isfinite(value):
                    raise RuntimeError(f'Nonfinite {history_mode} validation loss')
                return float(value)
            loss, stats = measure(validate, device)
            report['measurements'][f'validation_{history_mode}'] = {'loss': loss, 'batch_size': args.val_batch_size, **stats}
        report['measurements']['predict_action'] = []
        for batch_size in sorted({1, args.val_batch_size}):
            batch = take(val_batch, batch_size)
            def predict():
                with torch.no_grad(), autocast():
                    prediction = policy.predict_action(batch['obs'], past_actions=batch['past_action'],
                                                       past_action_valid=batch['past_action_valid'])
                if any(not torch.isfinite(value).all() for value in prediction.values()):
                    raise RuntimeError('Nonfinite full predict_action output')
                return prediction
            for _ in range(args.warmup):
                predict()
            samples, allocated, reserved = [], [], []
            for _ in range(args.iterations):
                _, stats = measure(predict, device)
                samples.append(stats['seconds'] * 1000)
                allocated.append(stats['max_memory_allocated_bytes'])
                reserved.append(stats['max_memory_reserved_bytes'])
            report['measurements']['predict_action'].append({
                'batch_size': batch_size, 'warmup': args.warmup, 'iterations': args.iterations,
                'p50_ms': float(np.percentile(samples, 50)), 'p95_ms': float(np.percentile(samples, 95)),
                'max_memory_allocated_bytes': max(allocated) if device.type == 'cuda' else None,
                'max_memory_reserved_bytes': max(reserved) if device.type == 'cuda' else None})
        report['self_past_chunk_size'] = policy.self_past_chunk_size
        report['successful_optimizer_updates'] = policy.self_past_step
        report['ema_optimizer_updates'] = ema_helper.optimization_step if ema_helper is not None else None
        if args.distributed:
            rank_reports = [None] * world_size
            distributed.all_gather_object(rank_reports, report)
            report = {'status': 'distributed_smoke_passed', 'variant': args.variant,
                      'world_size': world_size, 'ranks': rank_reports}
        return write_report(report, args.output, rank)
    finally:
        if args.distributed and distributed.is_initialized():
            distributed.destroy_process_group()


def write_report(report, output, rank):
    if rank == 0:
        text = json.dumps(report, indent=2, allow_nan=False)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + '\n')
        print(text)
    return report


if __name__ == '__main__':
    main()
