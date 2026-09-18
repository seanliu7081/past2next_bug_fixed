#!/usr/bin/env bash
# Train a pen_cabinet tokenizer, then Past2Next, with the current fruit settings.
# Foreground launcher; the caller owns the terminal/process manager.
set -euo pipefail
SCRIPT_PATH=$(realpath -- "${BASH_SOURCE[0]}")
cd -- "$(dirname -- "$SCRIPT_PATH")"

DRY_RUN=false
case "${1:-}" in
  --dry-run) DRY_RUN=true; shift ;;
  -h|--help)
    cat <<'HELP'
Usage: bash train_pen_cabinet.sh [--dry-run]

Environment settings:
  DATASET_PATH      Default: /workspace/ysk/zarr/pen_cabinet_N67.zarr
  TRAIN_GPUS        Physical GPU indices, comma-separated (default: 7)
  TRAIN_PY          Default: /venv/real_robot/bin/python
  TOKENIZER_EPOCHS  Default: 3001
  POLICY_EPOCHS     Default: 1001
  WANDB_MODE        online or offline (default: online); project: real_robot
  RUN_DIR           Fresh output directory (default: timestamped under output/)

Matches train_fruits_v2.sh: tokenizer batch 256, policy batch 64 per GPU,
128x128 camera images, 112x112 crops, and a shared 90/10 episode split.
--dry-run validates the dataset and resolves both stage configs without
starting training, creating a run directory, or initializing W&B.
HELP
    exit 0 ;;
  "") ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
(( $# == 0 )) || { printf 'Unexpected extra arguments.\n' >&2; exit 2; }

TRAIN_PY="${TRAIN_PY:-/venv/real_robot/bin/python}"
DATASET_PATH="${DATASET_PATH:-/workspace/ysk/zarr/pen_cabinet_N67.zarr}"
TOKENIZER_EPOCHS="${TOKENIZER_EPOCHS:-3001}"
POLICY_EPOCHS="${POLICY_EPOCHS:-1001}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-7}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  printf 'TRAIN_GPUS must contain comma-separated GPU indices.\n' >&2; exit 2;
}
[[ "$TOKENIZER_EPOCHS" =~ ^[1-9][0-9]*$ && "$POLICY_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'TOKENIZER_EPOCHS and POLICY_EPOCHS must be positive integers.\n' >&2; exit 2;
}
[[ "$WANDB_MODE" == online || "$WANDB_MODE" == offline ]] || {
  printf 'WANDB_MODE must be online or offline.\n' >&2; exit 2;
}
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPU_IDS[@]}

# Check schema and split without loading the full cameras or modifying the Zarr.
NUM_DEMOS=$("$TRAIN_PY" - "$DATASET_PATH" "$CUDA_VISIBLE_DEVICES" <<'PY'
import sys
import numpy as np
import zarr
from oat.common.seq_sampler import get_val_mask

gpu_ids = [int(value) for value in sys.argv[2].split(',')]
if len(set(gpu_ids)) != len(gpu_ids):
    raise ValueError('TRAIN_GPUS must contain distinct GPU indices')
root = zarr.open_group(sys.argv[1], mode='r')
ends = np.asarray(root['meta/episode_ends'][:])
if (ends.ndim != 1 or ends.dtype.kind not in 'iu' or len(ends) < 2
        or np.any(np.diff(np.r_[0, ends]) <= 0)):
    raise ValueError('Expected at least two episodes with increasing integer boundaries')
shapes = {
    'action': (7,), 'agentview_rgb': (128, 128, 3),
    'robot0_eye_in_hand_rgb': (128, 128, 3), 'robot0_eef_pos': (3,),
    'robot0_eef_rot6d': (6,), 'robot0_gripper_qpos': (1,), 'task_uid': (1,),
}
for key, shape in shapes.items():
    array = root[f'data/{key}']
    if array.shape != (int(ends[-1]), *shape):
        raise ValueError(f'{key}: expected {(int(ends[-1]), *shape)}, got {array.shape}')
    if key.endswith('_rgb'):
        if array.dtype != np.uint8:
            raise ValueError(f'{key} must be uint8 RGB')
    elif array.dtype.kind not in 'fiu' or not np.isfinite(array[:]).all():
        raise ValueError(f'{key} must contain finite numeric values')
if np.unique(root['data/task_uid'][:]).tolist() != [0]:
    raise ValueError('This single-task configuration requires task_uid=0')
val_mask = get_val_mask(len(ends), 0.1, 42)
print(f'Dataset: {len(ends)} episodes, {int(ends[-1])} frames; '
      f'{int((~val_mask).sum())} train / {int(val_mask.sum())} validation (seed 42)',
      file=sys.stderr)
print(len(ends))
PY
)

DATASET_NAME=$(basename -- "$DATASET_PATH" .zarr)
RUN_GROUP="${DATASET_NAME}_current"
RUN=$(realpath -m -- "${RUN_DIR:-$PWD/output/${RUN_GROUP}_$(date -u +%Y%m%d_%H%M%S)}")
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

TOKENIZER_ARGS=(
  --config-name=train_oattok_so3aug
  "${COMMON[@]}"
  task/tokenizer=real_robot/pen_cabinet
  "task.tokenizer.name=real_robot_$DATASET_NAME"
  "task.tokenizer.task_name=$DATASET_NAME"
  "task.tokenizer.dataset.zarr_path=$DATASET_PATH"
  "training.num_epochs=$TOKENIZER_EPOCHS"
  dataloader.batch_size=256
  val_dataloader.batch_size=128
  checkpoint.topk.k=3
  checkpoint.topk.monitor_key=test_reconst_mse
  checkpoint.topk.mode=min
  "checkpoint.topk.format_str='ep-{epoch:04d}_mse-{test_reconst_mse:.3f}.ckpt'"
  "hydra.run.dir=$RUN/tokenizer"
)
POLICY_ARGS=(
  --config-name=train_past2next_scratch_all500
  "${COMMON[@]}"
  task/policy=real_robot/pen_cabinet_with_prev_window
  "task.policy.name=real_robot_$DATASET_NAME"
  "task.policy.task_name=$DATASET_NAME"
  "task.policy.dataset.zarr_path=$DATASET_PATH"
  "policy.action_tokenizer.checkpoint=$RUN/frozen_tokenizer.ckpt"
  "training.num_epochs=$POLICY_EPOCHS"
  training.checkpoint_every=50
  training.snapshot_every=0
  training.init_checkpoint=null
  training.offline_validation_enabled=true
  training.offline_validation_reason=held_out_real_robot_episodes
  dataloader.batch_size=64
  val_dataloader.batch_size=32
  checkpoint.topk.k=0
  checkpoint.topk.monitor_key=test_reconst_mse
  checkpoint.topk.mode=min
  ++checkpoint.save_all=true
  "checkpoint.topk.format_str='ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt'"
  "hydra.run.dir=$RUN/policy"
)

if [[ "$DRY_RUN" == true ]]; then
  "$TRAIN_PY" scripts/run_workspace.py "${TOKENIZER_ARGS[@]}" --cfg job --resolve
  "$TRAIN_PY" scripts/run_workspace.py "${POLICY_ARGS[@]}" --cfg job --resolve
  exit 0
fi

mkdir -p -- "$(dirname -- "$RUN")"
mkdir -- "$RUN"
cp -- "$SCRIPT_PATH" "$RUN/launcher.sh"
TOKENIZER_COMMAND=("$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC"
  scripts/run_workspace.py "${TOKENIZER_ARGS[@]}")
POLICY_COMMAND=("$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC"
  scripts/run_workspace.py "${POLICY_ARGS[@]}")
{
  printf 'export CUDA_VISIBLE_DEVICES=%q WANDB_MODE=%q\n' "$CUDA_VISIBLE_DEVICES" "$WANDB_MODE"
  printf '%q ' "${TOKENIZER_COMMAND[@]}"; printf '\n'
  printf '%q ' "${POLICY_COMMAND[@]}"; printf '\n'
} > "$RUN/launch_commands.txt"
exec > >(tee -a "$RUN/console.log") 2>&1
printf 'Dataset: %s (%s episodes)\n' "$DATASET_PATH" "$NUM_DEMOS"
printf 'GPUs: %s; tokenizer/policy epochs: %s/%s; W&B: %s\n' \
  "$CUDA_VISIBLE_DEVICES" "$TOKENIZER_EPOCHS" "$POLICY_EPOCHS" "$WANDB_MODE"
printf 'Images: 128x128; crops: 112x112; output: %s\n' "$RUN"

"${TOKENIZER_COMMAND[@]}"

# Select using full-precision logged MSE and freeze the best tokenizer.
"$TRAIN_PY" - "$RUN" <<'PY'
import shutil
import sys
from pathlib import Path
from scripts.train_real_robot import best_tokenizer_checkpoint

run = Path(sys.argv[1])
checkpoint, mse = best_tokenizer_checkpoint(run / 'tokenizer')
shutil.copy2(checkpoint, run / 'frozen_tokenizer.ckpt')
print(f'Selected tokenizer: {checkpoint} (MSE={mse:.8f})', flush=True)
PY

"${POLICY_COMMAND[@]}"

# Reload the selected policy and verify its frozen tokenizer and finite actions.
"$TRAIN_PY" - "$RUN" <<'PY'
import json
import sys
from pathlib import Path
from scripts.train_real_robot import best_policy_checkpoint
from scripts.check_real_robot_checkpoint import check_checkpoint

run = Path(sys.argv[1])
checkpoint, mse = best_policy_checkpoint(run / 'policy')
report = check_checkpoint(checkpoint, 'cuda:0')
report['held_out_action_mse'] = mse
(run / 'checkpoint_check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
PY
