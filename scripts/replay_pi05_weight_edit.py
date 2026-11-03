#!/usr/bin/env python
"""Replay frozen Pi05-past observations/history/noise before and after a weight edit.

This is an offline intervention check, not a robot rollout. It first reproduces
the recorded unedited prediction and repeats it with the same RNG. Only after
both checks pass does it evaluate the static patched weights on the same input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def replay_prediction(policy, archive):
    import numpy as np
    import torch
    obs = {key: torch.from_numpy(np.array(archive['obs__' + key], copy=True)).to(policy.device)
           for key in policy.get_observation_ports()}
    past = torch.from_numpy(np.array(archive['history_actions'], copy=True)).to(policy.device, policy.dtype)
    valid = torch.from_numpy(np.array(archive['history_valid'], copy=True)).to(policy.device, torch.bool)
    policy.reset()
    torch.set_rng_state(torch.from_numpy(np.array(archive['torch_rng_cpu'], copy=True)).to(torch.uint8))
    if torch.device(policy.device).type == 'cuda':
        torch.cuda.set_rng_state(torch.from_numpy(np.array(archive['torch_rng_cuda'], copy=True)).to(torch.uint8),
                                 device=policy.device)
    with torch.inference_mode():
        result = policy.predict_action(obs, past_actions=past, past_action_valid=valid)
    return {key: value.detach().float().cpu().numpy().copy() for key, value in result.items()}


def reproduction_metrics(reference, actual, *, atol, rtol):
    import numpy as np
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        return {'passed': False, 'bitwise_equal': False, 'reason': 'invalid shape or nonfinite output'}
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return {'passed': bool(np.allclose(reference, actual, atol=atol, rtol=rtol)),
            'bitwise_equal': bool(np.array_equal(reference, actual)),
            'max_abs': float(np.max(np.abs(difference))), 'mean_abs': float(np.mean(np.abs(difference))),
            'atol': atol, 'rtol': rtol}


def dimension_metrics(delta):
    import numpy as np
    values = np.asarray(delta, dtype=np.float64).reshape(-1, delta.shape[-1])
    labels = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
    return {label: {'mean': float(values[:, i].mean()), 'mean_abs': float(np.abs(values[:, i]).mean()),
                    'max_abs': float(np.abs(values[:, i]).max()),
                    'min': float(values[:, i].min()), 'max': float(values[:, i].max())}
            for i, label in enumerate(labels[:values.shape[-1]])}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--chunks', default='0', help='Comma-separated chunk IDs, or all')
    p.add_argument('--patch-json', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--weights', choices=['model', 'ema'], default='model')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--atol', type=float, default=1e-6)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--pi05', type=Path)
    p.add_argument('--spm', type=Path)
    return p


def main(args):
    import numpy as np
    import torch
    from scripts.evaluate_p2n_vla import read_payload, load_policy
    from scripts.evaluate_pi05_weight_edit import sha256, write_json
    from oat.model.vla.weight_space_edit import (
        load_patch_spec, apply_output_head_patch, compare_action_chunks,
    )
    if args.atol < 0 or args.rtol < 0:
        raise ValueError('Replay tolerances must be nonnegative')
    if args.chunks == 'all':
        paths = sorted(args.capture_dir.glob('chunk_*.npz'))
    else:
        indices = [int(item.strip()) for item in args.chunks.split(',')]
        if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
            raise ValueError('Chunk IDs must be distinct nonnegative integers')
        paths = [args.capture_dir / f'chunk_{index:04d}.npz' for index in indices]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError('One or more requested chunk captures are missing')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output_dir / 'replay_source.py')
    torch.set_num_threads(args.threads)
    payload = read_payload(args.snapshot)
    policy = load_policy(args, payload)
    spec = None if args.patch_json is None else load_patch_spec(args.patch_json)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'snapshot': str(args.snapshot.resolve()), 'snapshot_sha256': sha256(args.snapshot),
              'capture_dir': str(args.capture_dir.resolve()), 'device': args.device,
              'method': 'exact saved policy observations, executed history, CPU/CUDA Torch RNG; no simulator',
              'patch': None, 'records': [], 'all_baselines_reproduced': True,
              'interpretation': 'Action counterfactuals only; do not establish task success or collision avoidance'}
    for path in paths:
        with np.load(path, allow_pickle=False) as opened:
            archive = {key: opened[key] for key in opened.files}
        baseline = replay_prediction(policy, archive)
        repeat = replay_prediction(policy, archive)
        checks = {
            'stored_action': reproduction_metrics(archive['actions'], baseline['action'], atol=args.atol, rtol=args.rtol),
            'stored_full_prediction': reproduction_metrics(archive['action_pred'], baseline['action_pred'], atol=args.atol, rtol=args.rtol),
            'repeat_action': reproduction_metrics(baseline['action'], repeat['action'], atol=args.atol, rtol=args.rtol),
            'repeat_full_prediction': reproduction_metrics(baseline['action_pred'], repeat['action_pred'], atol=args.atol, rtol=args.rtol),
        }
        record = {'capture': str(path.resolve()), 'capture_sha256': sha256(path),
                  'policy_step': int(archive['policy_step']), 'baseline_checks': checks,
                  'baseline_reproduced': all(value['passed'] for value in checks.values())}
        arrays = {'stored_action': archive['actions'], 'stored_action_pred': archive['action_pred'],
                  'baseline_action': baseline['action'], 'baseline_action_pred': baseline['action_pred'],
                  'repeat_action': repeat['action'], 'repeat_action_pred': repeat['action_pred']}
        if not record['baseline_reproduced']:
            report['all_baselines_reproduced'] = False
            report['records'].append(record)
            np.savez_compressed(args.output_dir / path.name, **arrays)
            write_json(args.output_dir / 'summary.json', report)
            raise RuntimeError(f'Baseline replay failed for {path.name}; no patch applied to this context')
        if spec is not None:
            patch = apply_output_head_patch(policy, spec)
            try:
                report['patch'] = patch.provenance
                edited = replay_prediction(policy, archive)
            finally:
                patch.restore()
            arrays.update(edited_action=edited['action'], edited_action_pred=edited['action_pred'])
            record['changes'] = {}
            for key in ['action', 'action_pred']:
                delta = compare_action_chunks(torch.as_tensor(baseline[key], device=policy.device),
                                              torch.as_tensor(edited[key], device=policy.device),
                                              policy.action_normalizer['action'])
                arrays[key + '_raw_delta'] = delta.pop('raw_delta')
                arrays[key + '_normalized_delta'] = delta.pop('normalized_delta')
                record['changes'][key] = {**delta,
                    'raw_per_dimension': dimension_metrics(arrays[key + '_raw_delta']),
                    'normalized_per_dimension': dimension_metrics(arrays[key + '_normalized_delta'])}
        report['records'].append(record)
        np.savez_compressed(args.output_dir / path.name, **arrays)
        write_json(args.output_dir / 'summary.json', report)
        print(f'{path.name}: baseline reproduced; patch={spec is not None}', flush=True)
    report['complete'] = True
    write_json(args.output_dir / 'summary.json', report)


if __name__ == '__main__':
    main(parser().parse_args())
