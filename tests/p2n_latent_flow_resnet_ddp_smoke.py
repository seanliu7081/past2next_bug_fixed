#!/usr/bin/env python
"""Two-rank CPU smoke using actual original ResNet18 camera backbones.

  /venv/real_robot/bin/python -m torch.distributed.run --standalone \
    --nproc_per_node=2 tests/p2n_latent_flow_resnet_ddp_smoke.py \
    --variant p2n_latent_flow

Repeat with --variant p2n_state_gate_latent_flow. The existing distributed
harness checks accumulated/tail updates, complete gradients, synchronized EMA,
strict state/RNG roundtrip and uneven validation. Only the small CPU policy
factory changes; no external DINO weights are involved. This does not start
production training or allocate a GPU.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import p2n_latent_flow_ddp_smoke as harness
from test_p2n_latent_flow_resnet_policy import make_resnet_flow, make_resnet_batch
from oat.policy.p2n_latent_flow_common import prepare_flow_training_batch


def resnet_cpu_mode(args, rank, device):
    student = make_resnet_flow(args.variant == 'p2n_state_gate_latent_flow',
                               self_past_p=1.).to(device)
    optimizer = student.get_optimizer(policy_lr=.005, obs_enc_lr=.0005)
    batch = make_resnet_batch(student, 4)
    def prepare(student, teacher, generator, self_past_generator):
        return prepare_flow_training_batch(batch, student=student, teacher=teacher,
            generator=generator, self_past_generator=self_past_generator)
    return student, optimizer, prepare, batch


if __name__ == '__main__':
    if '--real' in sys.argv:
        raise SystemExit('This ResNet distributed smoke supports CPU mode only; omit --real.')
    harness.toy_mode = resnet_cpu_mode
    harness.main()
