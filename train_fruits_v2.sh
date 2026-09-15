#!/usr/bin/env bash
# Train the fruits v2 tokenizer, then its policy, on GPUs 6 and 7.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=6,7
export WANDB_MODE=online
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

TRAIN_PY=/venv/real_robot/bin/python
RUN="$PWD/output/fruits_v2_current_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output"
mkdir "$RUN"
printf 'Training output: %s\n' "$RUN"

COMMON=(
  seed=42
  training.num_demo=49
  training.resume=false
  logging.mode=online
  logging.resume=false
  logging.project=real_robot
  logging.group=fruits_v2_current
  val_dataloader.drop_last=false
)

# Tokenizer: 5001 epochs; retain the best three by validation MSE.
"$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/run_workspace.py --config-name=train_oattok_so3aug \
  "${COMMON[@]}" \
  task/tokenizer=real_robot/fruits_v2 \
  training.num_epochs=5001 \
  dataloader.batch_size=256 \
  val_dataloader.batch_size=128 \
  checkpoint.topk.k=3 \
  checkpoint.topk.monitor_key=test_reconst_mse \
  checkpoint.topk.mode=min \
  "checkpoint.topk.format_str='ep-{epoch:04d}_mse-{test_reconst_mse:.3f}.ckpt'" \
  "hydra.run.dir=$RUN/tokenizer"

# Select using full-precision logged MSE and freeze the best tokenizer.
"$TRAIN_PY" - "$RUN" <<'PY'
import shutil
import sys
from pathlib import Path
from scripts.train_real_robot import best_tokenizer_checkpoint

run = Path(sys.argv[1])
checkpoint, mse = best_tokenizer_checkpoint(run / "tokenizer")
shutil.copy2(checkpoint, run / "frozen_tokenizer.ckpt")
print(f"Selected tokenizer: {checkpoint} (MSE={mse:.8f})", flush=True)
PY

# Policy: 1001 epochs; save indices 0, 50, ..., 1000 with their MSE.
"$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/run_workspace.py --config-name=train_past2next_scratch_all500 \
  "${COMMON[@]}" \
  task/policy=real_robot/fruits_v2_with_prev_window \
  "policy.action_tokenizer.checkpoint=$RUN/frozen_tokenizer.ckpt" \
  training.num_epochs=1001 \
  training.checkpoint_every=50 \
  training.snapshot_every=0 \
  training.init_checkpoint=null \
  training.offline_validation_enabled=true \
  training.offline_validation_reason=held_out_real_robot_episodes \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=32 \
  checkpoint.topk.k=0 \
  checkpoint.topk.monitor_key=test_reconst_mse \
  checkpoint.topk.mode=min \
  ++checkpoint.save_all=true \
  "checkpoint.topk.format_str='ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt'" \
  "hydra.run.dir=$RUN/policy"
