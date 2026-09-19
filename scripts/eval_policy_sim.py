"""
Usage:
python scripts/eval_policy_sim.py --checkpoint path/to/ckpt -o path/to/output_dir
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import shutil
import click
import hydra
import torch
import wandb
import json
import numpy as np
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from typing import List, Optional

@click.command()
@click.option('-c', '--checkpoint', required=True, help="either a .ckpt file or a directory containing .ckpt files")
@click.option('-o', '--output_dir', required=True, help="output directory for eval info dump")
@click.option('-n', '--num_exp', default=1, help="num experiments to run")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('--temperature', default=None, type=float, help="temperature for policy inference")
@click.option('--topk', default=None, type=int, help="topk for policy inference")
@click.option('--use_k_tokens', default=None, type=int, help="number of tokens to use for policy inference")
@click.option('--protocol', type=click.Choice(['corrected', 'official', 'legacy']), default=None,
              help="LIBERO evaluation protocol; defaults to corrected regardless of the saved checkpoint. Use legacy explicitly to reproduce historical resets.")
def eval_policy_sim(
    checkpoint: str,
    output_dir: str,
    num_exp: int = 1,
    device: str = 'cuda:0',
    # policy inference args
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    use_k_tokens: Optional[int] = None,
    protocol: Optional[str] = None,
):
    output_path = pathlib.Path(output_dir)
    if output_path.exists() or output_path.is_symlink():
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        # Treat paths literally and remove a symlink itself, never its target.
        if output_path.is_symlink() or output_path.is_file():
            output_path.unlink()
        else:
            shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # grab all checkpoints
    ckpts: List[str]    # file paths to checkpoints to evaluate
    if os.path.isdir(checkpoint):
        ckpts = [
            os.path.join(checkpoint, f) 
            for f in os.listdir(checkpoint) 
            if f.endswith('.ckpt') and f != 'latest.ckpt'
        ]
    else:
        ckpts = [checkpoint,]

    base_output_dir = output_dir
    for ckpt in ckpts:
        # format output dir
        if len(ckpts) > 1:
            ckpt_name = os.path.basename(ckpt).replace('.ckpt', '')
            output_dir = os.path.join(base_output_dir, ckpt_name)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir
        
        # load checkpoint
        policy, cfg = BasePolicy.from_checkpoint(
            ckpt, return_configuration=True,
            policy_overrides={"obs_encoder": {"vision_encoder": {"eval_fixed_crop": True}}},
        )
        
        device = torch.device(device)
        policy.to(device)
        policy.eval()
        
        # run eval
        print(f"Running evaluation on {ckpt}")
        runner_cfg = cfg.task.policy.env_runner
        is_libero = runner_cfg.get('_target_') == 'oat.env_runner.libero_runner.LiberoRunner'
        runner_kwargs = {'output_dir': output_dir}
        if is_libero:
            runner_kwargs['protocol'] = protocol or 'corrected'
        elif protocol is not None:
            raise click.UsageError('--protocol is only supported for the LIBERO runner')
        env_runner: BaseRunner = hydra.utils.instantiate(runner_cfg, **runner_kwargs)
        effective_protocol = env_runner.protocol if is_libero else None
        if is_libero:
            print(f"LIBERO evaluation protocol: {effective_protocol}")
        
        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if topk is not None:
            kwargs['topk'] = topk
        if use_k_tokens is not None:
            kwargs['use_k_tokens'] = use_k_tokens
        runner_log = env_runner.run(
            policy,
            **kwargs
        )
        
        # Store all runs for computing statistics
        all_runs = []
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                runner_log[key] = [value]
        all_runs.append({k: v for k, v in runner_log.items() if not isinstance(v, list)})
        print(f"Exp 1: success rate = {runner_log['mean_success_rate']}")
        
        for i in range(num_exp - 1):
            this_log = env_runner.run(policy, **kwargs)
            print(f"Exp {i + 2}: success rate = {this_log['mean_success_rate']}")
            all_runs.append({k: v for k, v in this_log.items() if not isinstance(v, list)})
            # merge logs
            for key, value in this_log.items():
                assert key in runner_log
                if isinstance(value, wandb.sdk.data_types.video.Video):
                    runner_log[key].append(value)
                else:
                    runner_log[key] += value
        
        # Compute mean and std for all numeric metrics
        numeric_keys = [k for k in all_runs[0].keys()]
        mean_log = {}
        std_log = {}
        
        for key in numeric_keys:
            values = [run[key] for run in all_runs]
            mean_log[key] = np.mean(values)
            if num_exp > 1:
                std_log[key] = np.std(values, ddof=1)  # sample std
        
        env_runner.close()
        
        # dump log to json
        json_log = dict()
        json_log['checkpoint'] = ckpt
        json_log['num_exp'] = num_exp
        if is_libero:
            json_log['protocol'] = effective_protocol
        
        # Add mean values
        for key, value in mean_log.items():
            json_log[f'{key}_mean'] = float(value)
        
        # Add standard deviation & error values if multiple experiments
        if num_exp > 1:
            for key, value in std_log.items():
                json_log[f'{key}_std'] = float(value)
                json_log[f'{key}_stderr'] = float(value / np.sqrt(num_exp))
        
        # Add video paths
        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    assert isinstance(video, wandb.sdk.data_types.video.Video)
                    json_log[f'{key}_{i}'] = video._path
        
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    # Configure CLI logging without replacing streams when imported by callers.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(line_buffering=True)
    eval_policy_sim()
