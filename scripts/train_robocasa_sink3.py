"""Train a Sink3 tokenizer, then a fresh task-LR policy with that tokenizer frozen."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def best_tokenizer_checkpoint(stage_dir):
    """Select a retained checkpoint using unrounded reconstruction metrics."""
    candidates = []
    for line in (stage_dir / "logs.json").read_text().splitlines():
        record = json.loads(line)
        metric = record.get("test_reconst_mse")
        if metric is None or not math.isfinite(metric):
            continue
        path = stage_dir / "checkpoints" / (
            f"ep-{record['epoch']:04d}_mse-{metric:.3f}.ckpt")
        if path.is_file():
            candidates.append((metric, -record["epoch"], path))
    if not candidates:
        raise RuntimeError("Stage 1 produced no checkpoint with a finite validation reconstruction MSE")
    metric, _, path = min(candidates)
    return path, metric


def stage_command(stage, output_dir, num_gpus, tokenizer=None, smoke=False):
    common = ["training.num_demo=600", "training.resume=false", "logging.mode=offline",
              "logging.resume=false", "training.tqdm_interval_sec=30",
              f"hydra.run.dir={output_dir}"]
    if stage == "tokenizer":
        config = "train_oattok_so3aug"
        overrides = ["task/tokenizer=robocasa/sink3",
                     "tokenizer.action_aug.mode=left_noise",
                     "tokenizer.action_aug.augment_position=false",
                     f"dataloader.batch_size={256 // num_gpus}",
                     f"val_dataloader.batch_size={256 // num_gpus}"]
    else:
        config = "train_past2next_scratch_tasklr"
        overrides = ["task/policy=robocasa/sink3_with_prev_window",
                     f"policy.action_tokenizer.checkpoint={tokenizer}",
                     "training.init_checkpoint=null", "task.policy.lazy_eval=true",
                     f"dataloader.batch_size={64 // num_gpus}",
                     f"val_dataloader.batch_size={64 // num_gpus}",
                     "checkpoint.topk.monitor_key=val_loss", "checkpoint.topk.mode=min",
                     "checkpoint.topk.format_str='ep-{epoch:04d}_val-{val_loss:.6f}.ckpt'"]
    if smoke:
        overrides += ["training.num_epochs=1", "training.max_train_steps=2",
                      "training.max_val_steps=1", "training.max_reconst_steps=1",
                      "dataloader.num_workers=0", "dataloader.persistent_workers=false",
                      "val_dataloader.num_workers=0", "val_dataloader.persistent_workers=false"]
        if stage == "policy":
            overrides += ["policy.self_past_warmup_steps=0", "policy.self_past_ramp_steps=0",
                          "policy.self_past_p=1.0"]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={num_gpus}", str(ROOT / "scripts/run_workspace.py"),
            f"--config-name={config}", *common, *overrides]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--smoke", action="store_true", help="Two training batches per stage, one epoch")
    args = parser.parse_args()
    num_gpus = len(args.gpus.split(","))
    if num_gpus not in (1, 2, 4, 8):
        parser.error("GPU count must divide both global batch sizes (256 and 64)")
    output = args.output_dir.resolve()
    # Refuse reuse: neither stage should silently overwrite or resume a prior run.
    output.mkdir(parents=True, exist_ok=False)
    status = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "state": "starting", "gpus": args.gpus, "smoke": args.smoke,
              "tokenizer_epochs": 1 if args.smoke else 5001,
              "policy_epochs": 1 if args.smoke else 251,
              "commands": {}}

    def save_status():
        temporary = output / "status.tmp.json"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(output / "status.json")

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED="1", WANDB_MODE="offline",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               MUJOCO_GL="egl", HYDRA_FULL_ERROR="1")
    tokenizer = None
    try:
        for stage in ("tokenizer", "policy"):
            stage_dir = output / stage
            command = stage_command(stage, stage_dir, num_gpus, tokenizer, args.smoke)
            status.update(state=f"training_{stage}")
            status["commands"][stage] = command
            save_status()
            print(f"Starting {stage}: {json.dumps(command)}", flush=True)
            with (output / f"{stage}.log").open("w") as logfile:
                subprocess.run(command, cwd=ROOT, env=env, stdout=logfile,
                               stderr=subprocess.STDOUT, check=True)
            if stage == "tokenizer":
                selected, metric = best_tokenizer_checkpoint(stage_dir)
                tokenizer = output / "frozen_tokenizer.ckpt"
                shutil.copy2(selected, tokenizer)
                status.update(tokenizer_source=str(selected), tokenizer_mse=metric,
                              frozen_tokenizer=str(tokenizer))
                save_status()
                print(f"Freezing tokenizer checkpoint {selected} (MSE {metric:.8f})", flush=True)
        status["state"] = "completed"
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        status["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_status()


if __name__ == "__main__":
    main()
