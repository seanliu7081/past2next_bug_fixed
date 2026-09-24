"""Reproducible two-rank CPU/gloo smoke for both new policy variants.

Run from the repository root::

    /venv/real_robot/bin/python tests/p2n_new_ddp_smoke.py

Uses actual tiny OAT/FSQ, AR/resampler, and an explicitly mocked DINO backbone.
No GPU or external weights are used; this is not production pretrained-model
acceptance. Importing the contract-test helpers only defines functions and
classes; it neither invokes pytest nor runs fixture setup.
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)


def worker(rank, world_size, store):
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('p2n_new_policy_contracts',
                                                 str(Path(REPO) / 'tests/test_p2n_new_policy.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import transformers
    transformers.DINOv3ViTModel = module.MockDINOBackbone
    dist.init_process_group('gloo', rank=rank, world_size=world_size,
                            init_method='file://' + store)
    try:
        for gate in (False, True):
            policy = module.make_policy(gate, self_past_chunk_size=1).train()
            model = DistributedDataParallel(policy, find_unused_parameters=False)
            optimizer = policy.get_optimizer()
            for step in range(2):
                torch.manual_seed(441 + rank + step)
                batch = module.batch_for(policy, previous=True)
                if step == 0:
                    batch['past_action_valid'].fill_(False)
                    batch['past_action'].fill_(float('nan'))
                    if gate:
                        batch['obs']['state_history_valid'][:, :-1] = False
                        for key in policy.state_history_keys:
                            batch['obs']['state_history__' + key][:, :-1] = float('nan')
                    mode = 'expert'
                else:
                    batch['prev_window_valid'][1] = False
                    batch['prev_past_action'][1] = float('nan')
                    for key in policy.obs_encoder.state_ports:
                        batch['prev_obs'][key][1] = float('nan')
                    if gate:
                        batch['prev_obs']['state_history_valid'][1] = False
                    mode = 'generated'
                loss = model(batch, history_mode=mode)
                assert torch.isfinite(loss), (rank, gate, step)
                loss.backward()
                for name, parameter in policy.named_parameters():
                    if parameter.requires_grad:
                        assert parameter.grad is not None, (rank, gate, step, name)
                        assert torch.isfinite(parameter.grad).all(), (rank, gate, step, name)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                policy.on_optimizer_step()
            assert policy.self_past_step == 2
            dist.barrier()
            if rank == 0:
                print(f'{policy.variant}: 2 ranks x 2 updates passed; shortest-history expert + generated self-past; every trainable gradient finite', flush=True)
            del model, optimizer, policy
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    descriptor, store = tempfile.mkstemp(prefix='p2n_new_gloo_', dir='/tmp')
    os.close(descriptor)
    os.unlink(store)
    try:
        mp.spawn(worker, args=(2, store), nprocs=2, join=True)
    finally:
        if os.path.exists(store):
            os.unlink(store)
