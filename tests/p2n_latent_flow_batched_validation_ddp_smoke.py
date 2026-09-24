#!/usr/bin/env python
"""Opt-in two-rank CPU validation: uneven batches and an empty rank.

CUDA_VISIBLE_DEVICES= /venv/real_robot/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/p2n_latent_flow_batched_validation_ddp_smoke.py

Runs actual workspace validation and tiny gated DiTX/OAT with a local mock DINO;
no optimizer, training, GPU allocation, network access, or W&B run is created.
"""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import torch
import transformers
from omegaconf import OmegaConf

from test_p2n_new_policy import MockDINOBackbone
from test_p2n_latent_flow_batched_validation import (
    CPUValidationAccelerator, ValidationDataset, _make_case, _select,
    _workspace_validation_reference,
)
from oat.workspace.train_p2n_latent_flow import (
    NonPaddingDistributedSampler, TrainP2NLatentFlowWorkspace,
)


def main():
    torch.set_num_threads(1)
    transformers.DINOv3ViTModel = MockDINOBackbone
    torch.distributed.init_process_group('gloo', timeout=timedelta(seconds=60))
    try:
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        if world != 2:
            raise ValueError('This CPU smoke expects exactly two ranks')
        policy, full_batch, _ = _make_case('dinov3', True)
        accelerator = CPUValidationAccelerator(rank, world)
        reports = []
        with tempfile.TemporaryDirectory(prefix=f'flow-validation-rank{rank}-') as directory:
            for size in (5, 1):
                batch = _select(full_batch, list(range(size)))
                dataset = ValidationDataset(batch)
                sampler = NonPaddingDistributedSampler(dataset, rank, world)
                loader = torch.utils.data.DataLoader(dataset, batch_size=2,
                    sampler=sampler, drop_last=False, num_workers=0)
                cfg = OmegaConf.create({'training': {'seed': 327, 'tqdm_interval_sec': 0},
                    'policy': {'flow': {'inference_steps': 8}}})
                workspace = TrainP2NLatentFlowWorkspace(cfg,
                    output_dir=str(Path(directory) / str(size)))
                workspace.ema_model = policy
                expected = _workspace_validation_reference(policy, batch, dataset, 327, True)
                result = workspace._validate(accelerator, loader, dataset, generated=True, decode=True)
                assert result['validation_samples'] == size
                for name, value in expected.items():
                    torch.testing.assert_close(torch.tensor(result[name], dtype=torch.float64),
                        torch.tensor(value, dtype=torch.float64), rtol=2e-5, atol=2e-6,
                        msg=lambda text: f'{name}: {text}')
                progress = Path(workspace.output_dir) / 'progress.jsonl'
                assert progress.exists() == (rank == 0)
                if rank == 0:
                    events = [json.loads(line) for line in progress.read_text().splitlines()]
                    assert events[-1]['completed'] == len(sampler)
                reports.append({'dataset_size': size, 'local_samples': len(sampler),
                                'local_batches': len(loader), 'validation_samples': result['validation_samples']})
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, {'rank': rank, 'cases': reports})
        if rank == 0:
            print(json.dumps({'status': 'passed', 'mode': 'cpu_actual_workspace_validation',
                              'ranks': gathered}, sort_keys=True), flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
