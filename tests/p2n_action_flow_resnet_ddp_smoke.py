#!/usr/bin/env python
"""Opt-in two-rank CPU DDP smoke for direct action flow with ResNet adapters.

  /venv/real_robot/bin/python -m torch.distributed.run --standalone \
    --nproc_per_node=2 tests/p2n_action_flow_resnet_ddp_smoke.py \
    --variant p2n_action_flow

Repeat with --variant p2n_state_gate_action_flow. Only the per-camera feature
networks are replaced with tiny convolutional modules. Production observation
adapters, normalized random crops, history, flow, packed FM/CT, Adam, EMA, and
strict training continuation all execute. This is not a full-model GPU test;
actual ResNet18 gradients are covered in test_p2n_action_flow_resnet_policy.py.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from oat.model.diffusion.ema_model import EMAModel
from oat.workspace.train_p2n_action_flow import (
    MetricSums, NonPaddingDistributedSampler, _module_digest,
    assert_teacher_synchronized, make_fresh_ema, successful_update,
    validate_optimizer_ownership,
)
from test_p2n_action_flow_resnet_policy import (
    first_rgb_conv, make_resnet_action_batch, make_resnet_action_flow,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('p2n_action_flow', 'p2n_state_gate_action_flow'),
                        default='p2n_action_flow')
    parser.add_argument('--updates', type=int, default=3)
    args = parser.parse_args()
    if args.updates < 3:
        parser.error('--updates must be at least three to test beyond zero initialization')
    rank = int(os.environ.get('RANK', 0))
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo')
    try:
        if torch.distributed.get_world_size() != 2:
            raise ValueError('Smoke expects exactly two CPU ranks')
        # Distinct initial weights exercise creation of EMA after DDP sync.
        torch.manual_seed(231 + rank)
        student = make_resnet_action_flow(args.variant == 'p2n_state_gate_action_flow',
                                         self_past_p=1., tiny_backbones=True)
        optimizer = student.get_optimizer(policy_lr=.005, obs_enc_lr=.002)
        batch = make_resnet_action_batch(student, 4)
        wrapped = DDP(student, find_unused_parameters=False)
        teacher = make_fresh_ema(student)
        assert _module_digest(student) == _module_digest(teacher)
        assert_teacher_synchronized(teacher)
        validate_optimizer_ownership(student, teacher, optimizer)
        ema = EMAModel(teacher, power=.75)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        generator = torch.Generator().manual_seed(998 + rank)
        past_generator = torch.Generator().manual_seed(1898 + rank)
        original_conv = first_rgb_conv(student).weight.detach().clone()
        def prepare():
            return student.prepare_training_batch(batch, teacher=teacher,
                generator=generator, self_past_generator=past_generator)
        start, losses = time.monotonic(), []
        for update in range(args.updates):
            wrapped.train()
            optimizer.zero_grad(set_to_none=True)
            group_length = 1 if update == args.updates - 1 else 2
            for microbatch in range(group_length):
                prepared = prepare()
                assert not prepared.prepared_visual.requires_grad
                with wrapped.no_sync() if microbatch < group_length - 1 else nullcontext():
                    loss = wrapped(prepared)
                    assert torch.isfinite(loss)
                    (loss / group_length).backward()
                losses.append(float(loss.detach()))
            missing = [name for name, p in student.named_parameters()
                       if p.requires_grad and p.grad is None]
            assert not missing, f'Unused trainable parameters: {missing}'
            assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.requires_grad)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
            successful_update(student, ema, scheduler)
        assert not torch.equal(original_conv, first_rgb_conv(student).weight)
        assert ema.optimization_step == student.self_past_step == args.updates
        assert_teacher_synchronized(teacher)
        assert all(p.grad is None for p in teacher.parameters())
        report = dict(mode='cpu_tiny_camera_contract_only', obs_encoder_type='resnet18',
            variant=args.variant, rank=rank, successful_updates=ema.optimization_step,
            accumulation=2, tail_group_size=1, seconds=time.monotonic() - start, losses=losses)
        # The continuation must reproduce crops, generated past, teacher targets,
        # DDP gradients, Adam state, and the next full EMA update exactly.
        stream = io.BytesIO()
        torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            generator=generator.get_state(), past_generator=past_generator.get_state(),
            torch_rng=torch.get_rng_state(), ema_step=ema.optimization_step,
            ema_decay=ema.decay), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        teacher_copy = make_fresh_ema(student)
        teacher_copy.load_state_dict(saved['teacher'], strict=True)
        assert _module_digest(teacher_copy) == _module_digest(teacher)
        def continuation_update():
            wrapped.train()
            optimizer.zero_grad(set_to_none=True)
            next_loss = wrapped(prepare())
            next_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
            successful_update(student, ema, scheduler)
            return float(next_loss.detach()), _module_digest(student), _module_digest(teacher)
        generator.set_state(saved['generator'])
        past_generator.set_state(saved['past_generator'])
        torch.set_rng_state(saved['torch_rng'])
        expected = continuation_update()
        student.load_state_dict(saved['student'], strict=True)
        teacher.load_state_dict(saved['teacher'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['generator'])
        past_generator.set_state(saved['past_generator'])
        ema.optimization_step, ema.decay = saved['ema_step'], saved['ema_decay']
        torch.set_rng_state(saved['torch_rng'])
        actual = continuation_update()
        assert expected == actual, (expected, actual)
        assert ema.optimization_step == args.updates + 1
        report['resume_next_update_exact'] = True
        # A rank with no validation rows must still participate in metric reduction.
        sampler = NonPaddingDistributedSampler(range(1), rank, 2)
        sums = MetricSums(['masked_mse'], torch.device('cpu'))
        for _ in sampler:
            sums.add({'masked_mse': {'sum': torch.tensor(21.), 'count': 7}})
        assert sums.reduce()['masked_mse'] == 3.
        with torch.no_grad():
            sample = {key: value[:1] for key, value in batch['obs'].items()}
            for steps in (8, 2):
                result = teacher.predict_action(sample, past_actions=batch['past_action'][:1],
                    past_action_valid=batch['past_action_valid'][:1], num_flow_steps=steps,
                    generator=torch.Generator().manual_seed(718 + rank))
                assert result['action_pred'].shape == (1, 16, 7)
                assert torch.isfinite(result['action_pred']).all()
        print(json.dumps(report), flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
