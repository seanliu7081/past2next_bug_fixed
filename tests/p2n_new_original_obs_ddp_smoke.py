"""Bounded, download-free two-rank original-fused acceptance.

    /venv/oat/bin/python tests/p2n_new_original_obs_ddp_smoke.py
    CUDA_VISIBLE_DEVICES=6,7 /venv/oat/bin/python tests/p2n_new_original_obs_ddp_smoke.py --device cuda --bf16

Real trainable ResNet18/GN and real tiny OAT are used. The default AR is small;
--production-ar selects the 16x768/12-head/2048 modern AR (OAT stays tiny).
This synthetic contract check is not a dataset/robot quality or speed benchmark.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import hydra
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]

from test_p2n_new_original_obs_policy import make_policy, batch_for
from oat.model.diffusion.ema_model import EMAModel
from oat.workspace.base_workspace import _copy_to_cpu


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--production-ar', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.accumulation) < 1 or args.updates < 2:
        parser.error('batch-size and accumulation must be positive; updates must be at least two')
    if args.device == 'cuda' and torch.cuda.device_count() < 2:
        parser.error('CUDA smoke requires two visible GPUs')
    return args


def move(value, device):
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value.to(device)


def synchronized_time(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def rng_state(device):
    return {'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def update(model, optimizer, scheduler, ema, args, device):
    policy = model.module
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = synchronized_time(device)
    times = dict(data_wait_seconds=0., forward_seconds=0., backward_seconds=0.)
    losses = []
    for micro in range(args.accumulation):
        started = synchronized_time(device)
        batch = move(batch_for(policy, args.batch_size, previous=True), device)
        times['data_wait_seconds'] += synchronized_time(device) - started
        synchronize = micro == args.accumulation - 1
        with (nullcontext() if synchronize else model.no_sync()):
            started = synchronized_time(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                loss = model(batch) / args.accumulation
            times['forward_seconds'] += synchronized_time(device) - started
            assert torch.isfinite(loss), (dist.get_rank(), policy.variant)
            started = synchronized_time(device)
            loss.backward()
            times['backward_seconds'] += synchronized_time(device) - started
        losses.append(float(loss.detach()))
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (dist.get_rank(), policy.variant, name)
            assert torch.isfinite(parameter.grad).all(), (dist.get_rank(), policy.variant, name)
        else:
            assert parameter.grad is None, name
    started = synchronized_time(device)
    optimizer.step()
    scheduler.step()
    policy.on_optimizer_step()
    ema.step(policy)
    ema.averaged_model.set_self_past_step(policy.self_past_step)
    times['optimizer_and_ema_seconds'] = synchronized_time(device) - started
    optimizer.zero_grad(set_to_none=True)
    times['step_seconds'] = synchronized_time(device) - start
    times['loss'] = sum(losses)
    return times


def worker(rank, args, store, report_path):
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo',
                            rank=rank, world_size=2, init_method='file://' + store)
    reports = []
    try:
        for gate in (False, True):
            for probability in (0., .5):
                options = dict(self_past_p=probability)
                if args.production_ar:
                    options.update(embed_dim=768, n_layers=16, n_heads=12, ffn_dim=2048, dropout=.1)
                    if gate:
                        options.update(history_embed_dim=128, history_n_heads=4,
                                       history_n_layers=2, history_gate_hidden_dim=128)
                policy = make_policy(gate, **options).to(device).train()
                model = DistributedDataParallel(policy, device_ids=[rank] if device.type == 'cuda' else None,
                                                 find_unused_parameters=False)
                optimizer = policy.get_optimizer()
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                ema = EMAModel(copy.deepcopy(policy))
                torch.manual_seed(987 + rank)
                measurements = []
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                for step in range(args.updates):
                    measurements.append(update(model, optimizer, scheduler, ema, args, device))
                assert policy.self_past_step == ema.optimization_step == args.updates
                snapshot = dict(config=policy.export_config(), model=_copy_to_cpu(policy.state_dict()),
                                optimizer=_copy_to_cpu(optimizer.state_dict()),
                                scheduler=copy.deepcopy(scheduler.state_dict()),
                                ema=_copy_to_cpu(ema.averaged_model.state_dict()),
                                ema_step=ema.optimization_step, ema_decay=ema.decay,
                                rng=rng_state(device))
                update(model, optimizer, scheduler, ema, args, device)
                expected = _copy_to_cpu(policy.state_dict())
                expected_ema = _copy_to_cpu(ema.averaged_model.state_dict())
                expected_scheduler = copy.deepcopy(scheduler.state_dict())
                expected_draw = torch.rand(4, device=device)
                peak = ({'allocated_bytes': torch.cuda.max_memory_allocated(device),
                         'reserved_bytes': torch.cuda.max_memory_reserved(device)}
                        if device.type == 'cuda' else None)
                del model, optimizer, scheduler, ema, policy
                restored = hydra.utils.instantiate(OmegaConf.create(snapshot['config']))
                restored.load_state_dict(snapshot['model'], strict=True)
                restored.to(device)
                optimizer = restored.get_optimizer()
                optimizer.load_state_dict(snapshot['optimizer'])
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                scheduler.load_state_dict(snapshot['scheduler'])
                restored_ema = copy.deepcopy(restored)
                restored_ema.load_state_dict(snapshot['ema'], strict=True)
                restored_ema.to(device)
                ema = EMAModel(restored_ema)
                ema.optimization_step, ema.decay = snapshot['ema_step'], snapshot['ema_decay']
                model = DistributedDataParallel(restored, device_ids=[rank] if device.type == 'cuda' else None,
                                                 find_unused_parameters=False)
                restore_rng(snapshot['rng'], device)
                update(model, optimizer, scheduler, ema, args, device)
                tolerance = 1e-6 if device.type == 'cuda' else 0.
                for actual, expected_state in ((restored.state_dict(), expected),
                                               (ema.averaged_model.state_dict(), expected_ema)):
                    for key, value in actual.items():
                        torch.testing.assert_close(value.cpu(), expected_state[key], rtol=tolerance, atol=tolerance)
                assert scheduler.state_dict() == expected_scheduler
                assert restored.self_past_step == ema.optimization_step == args.updates + 1
                torch.testing.assert_close(torch.rand(4, device=device), expected_draw, rtol=0, atol=0)
                warmed = [item['step_seconds'] for item in measurements[1:]]
                entry = dict(variant=restored.variant, rank=rank, self_past_probability=probability,
                             context_tokens=15 if gate else 11, observation_tokens=2,
                             crop_shape=[112, 112], cnn_trainable=True,
                             batch_per_rank=args.batch_size, accumulation=args.accumulation,
                             ar='production_16x768' if args.production_ar else 'tiny_2x16',
                             oat='tiny_real_fsq', precision='bf16' if args.bf16 else 'fp32',
                             successful_optimizer_updates=restored.self_past_step,
                             all_trainable_gradients_finite=True, exact_rng_continuation=True,
                             strict_resume_matches_uninterrupted=True, measurements=measurements,
                             step_p50_seconds=statistics.median(warmed),
                             throughput_local_examples_per_second=args.batch_size * args.accumulation / statistics.mean(warmed),
                             cuda_peak=peak)
                reports.append(entry)
                dist.barrier()
                if rank == 0:
                    print(f'{restored.variant} p={probability}: two ranks, finite full gradients, '
                          f'accumulation={args.accumulation}, EMA and strict resume passed', flush=True)
                del model, optimizer, scheduler, ema, restored, restored_ema, snapshot, expected, expected_ema
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        gathered = [None, None]
        dist.all_gather_object(gathered, reports)
        if rank == 0:
            Path(report_path).write_text(json.dumps({
                'status': 'passed', 'scope': 'synthetic two-rank correctness smoke; tiny OAT',
                'world_size': 2, 'device': args.device,
                'measurements_note': 'Instrumented timings; first Adam update excluded from aggregate throughput. No production speed claim.',
                'runs': [entry for rank_reports in gathered for entry in rank_reports],
            }, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def main():
    args = arguments()
    with tempfile.TemporaryDirectory(prefix='original_obs_ddp_') as temporary:
        report = args.output or Path(temporary) / 'report.json'
        report.parent.mkdir(parents=True, exist_ok=True)
        mp.spawn(worker, args=(args, str(Path(temporary) / 'store'), str(report)), nprocs=2, join=True)
        print(report.read_text())


if __name__ == '__main__':
    main()
