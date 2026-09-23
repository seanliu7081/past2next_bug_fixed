#!/usr/bin/env bash
# Train a tokenizer, then Past2Next, on a single-task real-robot dataset.
# Foreground launcher; the caller owns the terminal/process manager.
set -euo pipefail
SCRIPT_PATH=$(realpath -- "${BASH_SOURCE[0]}")

usage() {
  cat <<'HELP'
Usage: bash train_pen_cabinet.sh [options]

Options (command-line values override environment settings):
  --tokenizer-config NAME  Hydra tokenizer config (default: train_oattok_so3aug)
  --dataset PATH          Real-robot Zarr dataset
  --policy-config NAME    Hydra Past2Next policy config (default: train_past2next_scratch_all500)
  --gpus IDS              Physical GPU indices, comma-separated
  --output-dir PATH       Fresh output directory, or existing run with --policy-only
  --policy-only           Start policy from a completed run's best tokenizer
  --dry-run               Validate data and print resolved stage configs
  -h, --help              Show this help

Example (tokenizer without action augmentation):
  bash train_pen_cabinet.sh --tokenizer-config oattok \
    --dataset /workspace/ysk/zarr/pen_cabinet_lp3_N67.zarr \
    --policy-config train_past2next_scratch_all500 --gpus 4,5

Environment settings:
  TOKENIZER_CONFIG Default: train_oattok_so3aug
  POLICY_CONFIG    Default: train_past2next_scratch_all500
  DATASET_PATH     Default: /workspace/ysk/zarr/pen_cabinet_N67.zarr
  TRAIN_GPUS       Physical GPU indices, comma-separated (default: 7)
  TRAIN_PY         Default: /venv/real_robot/bin/python
  TOKENIZER_EPOCHS  Default: 3001
  POLICY_EPOCHS    Default: 1001; real-robot gate uses its policy config (2001)
  WANDB_MODE       online or offline (default: online); project: real_robot
  RUN_DIR          Fresh output directory (default: timestamped under output/)

Uses the real_robot/pen_cabinet task schema: 7D actions, two 128x128 RGB
cameras, and task_uid=0. The split is 90/10 by episode, seed 42.
Tokenizer batch is 256 per GPU. Standard Past2Next policies use batch 64
and checkpoints every 50 epochs. The real-robot state-history gate preserves
its config recipe: batch 32, checkpoints every 100 epochs, ranked by val_loss.
Supported policies: root train_past2next*.yaml recipes and
experimental/train_past2next_state_history_gate_real_robot.
Architecture and crops come from the selected configs.
Config names are relative to oat/config; .yaml is optional.
Dataset and output paths are relative to the caller's working directory.
--dry-run validates the dataset and resolves both stage configs without
starting training, creating a run directory, or initializing W&B.
--policy-only requires --output-dir (or RUN_DIR), a completed tokenizer,
and no existing policy output. It inherits the saved tokenizer config and
dataset unless explicitly provided, selects by full-precision held-out MSE,
and preserves the original launcher and command records. Combine it with
--dry-run to validate recovery and resolve only the policy config.
HELP
}

fail() { printf '%s\n' "$*" >&2; exit 2; }

# Bash parses a function in full before executing it. Keep the entire pipeline
# here so editing this file during tokenizer training cannot corrupt the handoff.
main() {
TOKENIZER_CONFIG_EXPLICIT=${TOKENIZER_CONFIG:+true}
DATASET_PATH_EXPLICIT=${DATASET_PATH:+true}
TRAIN_PY="${TRAIN_PY:-/venv/real_robot/bin/python}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-train_oattok_so3aug}"
POLICY_CONFIG="${POLICY_CONFIG:-train_past2next_scratch_all500}"
DATASET_PATH="${DATASET_PATH:-/workspace/ysk/zarr/pen_cabinet_N67.zarr}"
TRAIN_GPUS="${TRAIN_GPUS:-7}"
TOKENIZER_EPOCHS="${TOKENIZER_EPOCHS:-3001}"
POLICY_EPOCHS="${POLICY_EPOCHS:-}"
DRY_RUN=false
POLICY_ONLY=false
while (( $# )); do
  option=${1%%=*}
  case "$option" in
    --tokenizer-config|--dataset|--policy-config|--gpus|--output-dir)
      if [[ "$1" == *=* ]]; then
        value=${1#*=}
        shift
      else
        (( $# >= 2 )) || fail "Missing value for $option"
        value=$2
        shift 2
      fi
      [[ -n "$value" && "$value" != --* ]] || fail "Missing value for $option"
      case "$option" in
        --tokenizer-config) TOKENIZER_CONFIG=$value; TOKENIZER_CONFIG_EXPLICIT=true ;;
        --dataset) DATASET_PATH=$value; DATASET_PATH_EXPLICIT=true ;;
        --policy-config) POLICY_CONFIG=$value ;;
        --gpus) TRAIN_GPUS=$value ;;
        --output-dir) RUN_DIR=$value ;;
      esac ;;
    --dry-run) [[ "$1" == --dry-run ]] || fail "Unexpected value for --dry-run"; DRY_RUN=true; shift ;;
    --policy-only) [[ "$1" == --policy-only ]] || fail "Unexpected value for --policy-only"; POLICY_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1 (see --help)" ;;
  esac
done

DATASET_PATH=$(realpath -m -- "$DATASET_PATH")
if [[ -n "${RUN_DIR:-}" ]]; then
  RUN_DIR=$(realpath -m -- "$RUN_DIR")
fi
cd -- "$(dirname -- "$SCRIPT_PATH")"
TOKENIZER_CONFIG=${TOKENIZER_CONFIG%.yaml}
POLICY_CONFIG=${POLICY_CONFIG%.yaml}
if [[ "$POLICY_ONLY" == true ]]; then
  [[ -n "${RUN_DIR:-}" ]] || fail '--policy-only requires --output-dir or RUN_DIR'
  [[ -d "$RUN_DIR" ]] || fail "Existing run directory not found: $RUN_DIR"
  [[ ! -e "$RUN_DIR/policy" && ! -L "$RUN_DIR/policy" ]] || fail "Refusing existing policy output: $RUN_DIR/policy"
  # Defaults must not silently pair an existing tokenizer with a different
  # dataset or augmentation recipe. Read saved settings before validating data.
  RECOVERY_SETTINGS=$("$TRAIN_PY" - "$RUN_DIR" "$DATASET_PATH" "$TOKENIZER_CONFIG" \
    "${DATASET_PATH_EXPLICIT:-false}" "${TOKENIZER_CONFIG_EXPLICIT:-false}" <<'PY'
import json
import math
import sys
from pathlib import Path
from omegaconf import OmegaConf

run = Path(sys.argv[1])
stage = run / 'tokenizer'
cfg = OmegaConf.load(stage / '.hydra/config.yaml')
hydra = OmegaConf.load(stage / '.hydra/hydra.yaml')
dataset = Path(str(cfg.task.tokenizer.dataset.zarr_path)).resolve()
config_name = str(hydra.hydra.job.config_name).removesuffix('.yaml')
if sys.argv[4] == 'true' and Path(sys.argv[2]).resolve() != dataset:
    raise ValueError(f'Dataset does not match saved tokenizer: {dataset}')
if sys.argv[5] == 'true' and sys.argv[3] != config_name:
    raise ValueError(f'Config does not match saved tokenizer: {config_name}')
last = None
with (stage / 'logs.json').open() as stream:
    for line in stream:
        if line.strip():
            last = json.loads(line)
final_epoch = int(cfg.training.num_epochs) - 1
if last is None or last.get('epoch') != final_epoch:
    raise ValueError(f'Tokenizer is incomplete: expected final epoch {final_epoch}')
# Scheduled validation/reconstruction appear only in the completed epoch record,
# never in intermediate minibatch records from that same epoch.
for cadence, metric in [('val_every', 'val_loss'), ('sample_every', 'test_reconst_mse')]:
    if final_epoch % int(cfg.training[cadence]) == 0:
        if metric not in last or not math.isfinite(last[metric]):
            raise ValueError(f'Tokenizer final epoch is missing finite {metric}')
if not (run / 'tokenizer_completed.json').is_file() and all(
        final_epoch % int(cfg.training[cadence]) != 0
        for cadence in ('val_every', 'sample_every')):
    raise ValueError('Cannot verify tokenizer completion without an epoch-end metric or completion marker')
print(dataset)
print(config_name)
print(cfg.training.num_epochs)
print(f'Completed tokenizer: epoch {final_epoch}; dataset {dataset}', file=sys.stderr)
PY
  )
  mapfile -t RECOVERY_SETTINGS <<< "$RECOVERY_SETTINGS"
  DATASET_PATH=${RECOVERY_SETTINGS[0]}
  TOKENIZER_CONFIG=${RECOVERY_SETTINGS[1]}
  TOKENIZER_EPOCHS=${RECOVERY_SETTINGS[2]}
fi
STATE_HISTORY_GATE=false
if [[ "$POLICY_CONFIG" == experimental/train_past2next_state_history_gate_real_robot ]]; then
  STATE_HISTORY_GATE=true
else
  POLICY_EPOCHS="${POLICY_EPOCHS:-1001}"
fi
[[ -f "oat/config/$TOKENIZER_CONFIG.yaml" ]] || fail "Tokenizer config not found: $TOKENIZER_CONFIG"
[[ -f "oat/config/$POLICY_CONFIG.yaml" ]] || fail "Policy config not found: $POLICY_CONFIG"
[[ -d "$DATASET_PATH" ]] || fail "Dataset directory not found: $DATASET_PATH"

# Quote for Hydra as well as the shell, so paths may contain spaces or commas.
hydra_string() {
  local value="$1"
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '"%s"' "$value"
}

export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  printf 'TRAIN_GPUS must contain comma-separated GPU indices.\n' >&2; exit 2;
}
[[ "$TOKENIZER_EPOCHS" =~ ^[1-9][0-9]*$ && ( -z "$POLICY_EPOCHS" || "$POLICY_EPOCHS" =~ ^[1-9][0-9]*$ ) ]] || {
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
  "logging.group=$(hydra_string "$RUN_GROUP")"
  val_dataloader.drop_last=false
)

TOKENIZER_ARGS=(
  "--config-name=$TOKENIZER_CONFIG"
  "${COMMON[@]}"
  task/tokenizer=real_robot/pen_cabinet
  "task.tokenizer.name=$(hydra_string "real_robot_$DATASET_NAME")"
  "task.tokenizer.task_name=$(hydra_string "$DATASET_NAME")"
  "task.tokenizer.dataset.zarr_path=$(hydra_string "$DATASET_PATH")"
  "training.num_epochs=$TOKENIZER_EPOCHS"
  dataloader.batch_size=256
  val_dataloader.batch_size=128
  checkpoint.topk.k=3
  checkpoint.topk.monitor_key=test_reconst_mse
  checkpoint.topk.mode=min
  "checkpoint.topk.format_str='ep-{epoch:04d}_mse-{test_reconst_mse:.3f}.ckpt'"
  "hydra.run.dir=$(hydra_string "$RUN/tokenizer")"
)
POLICY_ARGS=(
  "--config-name=$POLICY_CONFIG"
  "${COMMON[@]}"
  task/policy=real_robot/pen_cabinet_with_prev_window
  "task.policy.name=$(hydra_string "real_robot_$DATASET_NAME")"
  "task.policy.task_name=$(hydra_string "$DATASET_NAME")"
  "task.policy.dataset.zarr_path=$(hydra_string "$DATASET_PATH")"
  "policy.action_tokenizer.checkpoint=$(hydra_string "$RUN/frozen_tokenizer.ckpt")"
  ++training.snapshot_every=0
  training.init_checkpoint=null
  ++training.offline_validation_enabled=true
  ++training.offline_validation_reason=held_out_real_robot_episodes
  "hydra.run.dir=$(hydra_string "$RUN/policy")"
)
# The gate config owns its batch, checkpoint cadence, and validation-loss metric.
# Keep the existing recipe for the original Past2Next policy choices.
if [[ "$STATE_HISTORY_GATE" == false ]]; then
  POLICY_ARGS+=(
    training.checkpoint_every=50
    dataloader.batch_size=64
    val_dataloader.batch_size=32
    checkpoint.topk.k=0
    checkpoint.topk.monitor_key=test_reconst_mse
    checkpoint.topk.mode=min
    ++checkpoint.save_all=true
    "checkpoint.topk.format_str='ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt'"
  )
fi
if [[ -n "$POLICY_EPOCHS" ]]; then
  POLICY_ARGS+=("training.num_epochs=$POLICY_EPOCHS")
fi

if [[ "$DRY_RUN" == true ]]; then
  if [[ "$POLICY_ONLY" == false ]]; then
    "$TRAIN_PY" scripts/run_workspace.py "${TOKENIZER_ARGS[@]}" --cfg job --resolve
  fi
  "$TRAIN_PY" scripts/run_workspace.py "${POLICY_ARGS[@]}" --cfg job --resolve
  exit 0
fi

# Resolve both stages before starting the expensive tokenizer training.
if [[ "$POLICY_ONLY" == false ]]; then
  "$TRAIN_PY" scripts/run_workspace.py "${TOKENIZER_ARGS[@]}" --cfg job --resolve > /dev/null
fi
"$TRAIN_PY" scripts/run_workspace.py "${POLICY_ARGS[@]}" --cfg job --resolve > /dev/null

if [[ "$POLICY_ONLY" == false ]]; then
  mkdir -p -- "$(dirname -- "$RUN")"
  mkdir -- "$RUN"
fi
exec {RUN_LOCK_FD}> "$RUN/.training.lock"
flock -n "$RUN_LOCK_FD" || fail "Training already holds the run lock: $RUN"
[[ ! -e "$RUN/policy" && ! -L "$RUN/policy" ]] || fail "Refusing existing policy output: $RUN/policy"
LAUNCH_RECORD_DIR=$RUN
if [[ "$POLICY_ONLY" == true ]]; then
  LAUNCH_RECORD_DIR=$(mktemp -d "$RUN/policy_recovery.XXXXXXXX")
fi
cp -- "$SCRIPT_PATH" "$LAUNCH_RECORD_DIR/launcher.sh"
TOKENIZER_COMMAND=("$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC"
  scripts/run_workspace.py "${TOKENIZER_ARGS[@]}")
POLICY_COMMAND=("$TRAIN_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC"
  scripts/run_workspace.py "${POLICY_ARGS[@]}")
{
  printf 'export CUDA_VISIBLE_DEVICES=%q WANDB_MODE=%q\n' "$CUDA_VISIBLE_DEVICES" "$WANDB_MODE"
  if [[ "$POLICY_ONLY" == false ]]; then
    printf '%q ' "${TOKENIZER_COMMAND[@]}"; printf '\n'
  fi
  printf '%q ' "${POLICY_COMMAND[@]}"; printf '\n'
} > "$LAUNCH_RECORD_DIR/launch_commands.txt"
exec > >(tee -a "$RUN/console.log") 2>&1
printf 'Dataset: %s (%s episodes)\n' "$DATASET_PATH" "$NUM_DEMOS"
printf 'Configs: tokenizer=%s; policy=%s\n' "$TOKENIZER_CONFIG" "$POLICY_CONFIG"
printf 'GPUs: %s; tokenizer/policy epochs: %s/%s; W&B: %s\n' \
  "$CUDA_VISIBLE_DEVICES" "$TOKENIZER_EPOCHS" "${POLICY_EPOCHS:-policy config default}" "$WANDB_MODE"
printf 'Images: 128x128; output: %s\n' "$RUN"

if [[ "$POLICY_ONLY" == false ]]; then
  "${TOKENIZER_COMMAND[@]}"
  printf '{"num_epochs": %s}\n' "$TOKENIZER_EPOCHS" > "$RUN/tokenizer_completed.json"
else
  printf 'Policy-only recovery: using the completed tokenizer; records: %s\n' "$LAUNCH_RECORD_DIR"
fi

# Select using full-precision logged MSE and freeze the best tokenizer.
"$TRAIN_PY" - "$RUN" <<'PY'
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from scripts.train_real_robot import best_tokenizer_checkpoint

run = Path(sys.argv[1])
checkpoint, mse = best_tokenizer_checkpoint(run / 'tokenizer')
if checkpoint.stat().st_size == 0:
    raise ValueError(f'Selected tokenizer checkpoint is empty: {checkpoint}')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

frozen = run / 'frozen_tokenizer.ckpt'
source_hash = sha256(checkpoint)
with tempfile.NamedTemporaryFile(dir=run, prefix='.frozen_tokenizer.', delete=False) as stream:
    temporary = Path(stream.name)
try:
    shutil.copy2(checkpoint, temporary)
    if sha256(temporary) != source_hash:
        raise RuntimeError('Frozen tokenizer copy failed its SHA256 check')
    os.replace(temporary, frozen)
finally:
    temporary.unlink(missing_ok=True)
selection = {
    'source': str(checkpoint.resolve()),
    'frozen_checkpoint': str(frozen.resolve()),
    'epoch': int(re.search(r'ep-(\d+)_', checkpoint.name).group(1)),
    'metric_name': 'test_reconst_mse',
    'metric': mse,
    'sha256': source_hash,
}
with tempfile.NamedTemporaryFile(mode='w', dir=run, prefix='.tokenizer_selection.',
                                 delete=False) as stream:
    json.dump(selection, stream, indent=2)
    stream.write('\n')
    temporary = Path(stream.name)
try:
    os.replace(temporary, run / 'tokenizer_selection.json')
finally:
    temporary.unlink(missing_ok=True)
print(f'Selected tokenizer: {checkpoint} (MSE={mse:.17g}, SHA256={source_hash})', flush=True)
PY

"${POLICY_COMMAND[@]}"

# Reload the selected policy and verify its frozen tokenizer and finite actions.
"$TRAIN_PY" - "$RUN" <<'PY'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf
from scripts.train_real_robot import best_policy_checkpoint
from scripts.check_real_robot_checkpoint import check_checkpoint

run = Path(sys.argv[1])
policy_cfg = OmegaConf.load(run / 'policy/.hydra/config.yaml')
metric_name = policy_cfg.checkpoint.topk.monitor_key
checkpoint, metric = best_policy_checkpoint(
    run / 'policy', metric_name=metric_name,
    filename=policy_cfg.checkpoint.topk.format_str,
)
report = check_checkpoint(checkpoint, 'cuda:0')
report['held_out_metric'] = {'name': metric_name, 'value': metric}
if metric_name == 'test_reconst_mse':
    report['held_out_action_mse'] = metric
(run / 'checkpoint_check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
PY
}

# Exit from an already parsed block without reading this file again after main.
{ main "$@"; exit; }
