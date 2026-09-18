#!/usr/bin/env bash
# Train a fruit dataset tokenizer, then its policy. Default: V3 N40 on GPU 7.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-7}"
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPU_IDS[@]}
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

TRAIN_PY=/venv/real_robot/bin/python
DATASET_PATH="${DATASET_PATH:-/workspace/ysk/zarr/fruitV3_40.zarr}"
NUM_DEMOS=$("$TRAIN_PY" -c 'import sys,zarr; print(len(zarr.open_group(sys.argv[1],mode="r")["meta/episode_ends"]))' "$DATASET_PATH")
DATASET_NAME=$(basename -- "$DATASET_PATH" .zarr)
RUN_GROUP="${DATASET_NAME}_current"
RUN="$PWD/output/${RUN_GROUP}_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output"
mkdir "$RUN"
printf 'Training output: %s\n' "$RUN"

COMMON=(
  seed=42
  "training.num_demo=$NUM_DEMOS"
  training.resume=false
  "logging.mode=$WANDB_MODE"
  logging.resume=false
  logging.project=real_robot
  "logging.group=$RUN_GROUP"
  val_dataloader.drop_last=false
)

# Tokenizer: 2001 epochs; retain the best three by validation MSE.
"$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  scripts/run_workspace.py --config-name=train_oattok_so3aug \
  "${COMMON[@]}" \
  task/tokenizer=real_robot/fruits_v2 \
  "task.tokenizer.name=real_robot_$DATASET_NAME" \
  "task.tokenizer.task_name=$DATASET_NAME" \
  "task.tokenizer.dataset.zarr_path=$DATASET_PATH" \
  training.num_epochs=3001 \
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

# Policy: 501 epochs; save every 50 epochs with their MSE.
"$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  scripts/run_workspace.py --config-name=train_past2next_scratch_all500 \
  "${COMMON[@]}" \
  task/policy=real_robot/fruits_v2_with_prev_window \
  "task.policy.name=real_robot_$DATASET_NAME" \
  "task.policy.task_name=$DATASET_NAME" \
  "task.policy.dataset.zarr_path=$DATASET_PATH" \
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

# Verify the selected policy reloads with its frozen tokenizer and finite actions.
"$TRAIN_PY" - "$RUN" <<'PY_CHECK'
import json
import sys
from pathlib import Path
from scripts.train_real_robot import best_policy_checkpoint
from scripts.check_real_robot_checkpoint import check_checkpoint

run = Path(sys.argv[1])
checkpoint, mse = best_policy_checkpoint(run / "policy")
report = check_checkpoint(checkpoint, "cuda:0")
report["held_out_action_mse"] = mse
(run / "checkpoint_check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
PY_CHECK
