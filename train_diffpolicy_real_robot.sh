#!/usr/bin/env bash
# Single-stage diffusion training. Foreground launcher; use a terminal or tmux.
set -euo pipefail
SCRIPT_PATH=$(realpath -- "${BASH_SOURCE[0]}")
cd -- "$(dirname -- "$SCRIPT_PATH")"

DRY_RUN=false
case "${1:-}" in
  --dry-run) DRY_RUN=true; shift ;;
  -h|--help)
    cat <<'HELP'
Usage: bash train_diffpolicy_real_robot.sh [--dry-run]

Environment settings:
  DATASET_PATH     Default: /workspace/ysk/zarr/pen_cabinet_lp3_N67.zarr
  TRAIN_GPUS       Physical GPU indices, comma-separated (default: 7)
  TRAIN_PY         Default: /venv/real_robot/bin/python
  POLICY_EPOCHS    Default: 1001
  BATCH_SIZE       Per-GPU training batch (default: 64)
  VAL_BATCH_SIZE   Per-GPU validation batch (default: 32)
  WANDB_MODE       online or offline (default: online); project: real_robot
  RUN_DIR          Fresh output directory (default: timestamped under output/)

Trains directly on actions; no tokenizer stage or checkpoint is needed.
Uses 128x128 images, 112x112 crops, and a 90/10 episode split (seed 42).
--dry-run validates the dataset and resolves the configuration without training,
creating a run directory, or initializing W&B.
HELP
    exit 0 ;;
  "") ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
(( $# == 0 )) || { printf 'Unexpected extra arguments.\n' >&2; exit 2; }

TRAIN_PY="${TRAIN_PY:-/venv/real_robot/bin/python}"
DATASET_PATH="${DATASET_PATH:-/workspace/ysk/zarr/pen_cabinet_lp3_N67.zarr}"
POLICY_EPOCHS="${POLICY_EPOCHS:-1001}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-7}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  printf 'TRAIN_GPUS must contain comma-separated GPU indices.\n' >&2; exit 2;
}
for value in "$POLICY_EPOCHS" "$BATCH_SIZE" "$VAL_BATCH_SIZE"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Epoch and batch counts must be positive integers.\n' >&2; exit 2;
  }
done
[[ "$WANDB_MODE" == online || "$WANDB_MODE" == offline ]] || {
  printf 'WANDB_MODE must be online or offline.\n' >&2; exit 2;
}
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPU_IDS[@]}
DATASET_NAME=$(basename -- "$DATASET_PATH" .zarr)
RUN=$(realpath -m -- "${RUN_DIR:-$PWD/output/${DATASET_NAME}_diffpolicy_$(date -u +%Y%m%d_%H%M%S)}")

# Inspect metadata/states only; do not load the full image arrays or modify data.
# JSON quoting also keeps paths with spaces or commas valid as Hydra overrides.
DATA_INFO=$("$TRAIN_PY" - "$DATASET_PATH" "$CUDA_VISIBLE_DEVICES" "$RUN" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np
import zarr
from oat.common.seq_sampler import get_val_mask

path = Path(sys.argv[1]).resolve()
gpu_ids = [int(value) for value in sys.argv[2].split(',')]
if len(set(gpu_ids)) != len(gpu_ids):
    raise ValueError('TRAIN_GPUS must contain distinct GPU indices')
root = zarr.open_group(str(path), mode='r')
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
mask = get_val_mask(len(ends), 0.1, 42)
print(f'Dataset: {len(ends)} episodes, {int(ends[-1])} frames; '
      f'{int((~mask).sum())} train / {int(mask.sum())} validation (seed 42)',
      file=sys.stderr)
print(len(ends))
print(json.dumps(str(path)))
print(json.dumps(Path(sys.argv[1]).stem))
print(json.dumps(sys.argv[3]))
PY
)
mapfile -t INFO <<< "$DATA_INFO"

POLICY_ARGS=(
  --config-name=train_diffpolicy_real_robot
  "dataset_path=${INFO[1]}"
  "dataset_name=${INFO[2]}"
  "training.num_demo=${INFO[0]}"
  "training.num_epochs=$POLICY_EPOCHS"
  "dataloader.batch_size=$BATCH_SIZE"
  "val_dataloader.batch_size=$VAL_BATCH_SIZE"
  "logging.mode=$WANDB_MODE"
  "hydra.run.dir=${INFO[3]}"
)

if [[ "$DRY_RUN" == true ]]; then
  "$TRAIN_PY" scripts/run_workspace.py "${POLICY_ARGS[@]}" --cfg job --resolve
  exit 0
fi

mkdir -p -- "$(dirname -- "$RUN")"
mkdir -- "$RUN"
cp -- "$SCRIPT_PATH" "$RUN/launcher.sh"
COMMAND=("$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC"
  scripts/run_workspace.py "${POLICY_ARGS[@]}")
{
  printf 'cd -- %q\n' "$PWD"
  printf 'export CUDA_VISIBLE_DEVICES=%q WANDB_MODE=%q\n' "$CUDA_VISIBLE_DEVICES" "$WANDB_MODE"
  printf '%q ' "${COMMAND[@]}"; printf '\n'
} > "$RUN/launch_commands.txt"
exec > >(tee -a "$RUN/console.log") 2>&1
printf 'Diffusion policy: %s; GPUs: %s; epochs: %s; batch per GPU: %s\n' \
  "$DATASET_NAME" "$CUDA_VISIBLE_DEVICES" "$POLICY_EPOCHS" "$BATCH_SIZE"
printf 'Output: %s\n' "$RUN"
"${COMMAND[@]}"
