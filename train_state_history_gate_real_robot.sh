#!/usr/bin/env bash
# Real-robot rot6d gate experiment: fresh policy run, two GPUs, online W&B.
# Reads demonstrations only; does not connect to or control a robot.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash train_state_history_gate_real_robot.sh [--dry-run] [TOKENIZER_CHECKPOINT [OUTPUT_DIR]]

Creates a fresh run: two GPUs, batch 32 per GPU (global 64), H8, 2001 epochs.
W&B is online; held-out demonstration validation runs every epoch.
Saves all checkpoints every 100 epochs, named by validation token loss.
No simulator rollout or live robot execution is performed.
Existing output directories are rejected for real training.

GATE_MODE=learned|open|closed (default learned)
GATE_INIT=0.9                     Initial learned gate, strictly between 0 and 1.
GATE_HIDDEN_DIM=128               Gate MLP hidden width.
STATE_HISTORY_STEPS=8             H state frames and H-1 past commands (H >= 4).
SEED=42                          Model/training seed; dataset split stays seed 42.
NUM_EPOCHS=2001 CHECKPOINT_EVERY=100 VAL_EVERY=1
                                 CHECKPOINT_EVERY must be divisible by VAL_EVERY.
INIT_CHECKPOINT=/path/policy.ckpt Optional compatible real-robot history/gate weights (EMA).
                                 Weight initialization only; optimizer and self-past schedule reset.
TOKENIZER_CHECKPOINT=/path/tok.ckpt is used if no positional tokenizer path is given.
DATASET_PATH=/path/real_robot.zarr overrides the pen_cabinet_lp3_N67 dataset.
TRAIN_PY=/venv/real_robot/bin/python     Python executable, or a command on PATH.
CUDA_VISIBLE_DEVICES=0,1          Select exactly two GPUs.

Paths are relative to the caller's working directory.
Output defaults to a new timestamped directory under output/training/.
Defaults use the supplied pen_cabinet_lp3_N67 frozen tokenizer and dataset.
--dry-run resolves config without training or W&B; input path existence is NOT checked.
A successful dry run does not establish checkpoint compatibility or GPU readiness.
HELP
}

DRY_RUN=false
case "${1:-}" in
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
esac
if (( $# > 2 )); then
    usage >&2
    exit 2
fi

fail() { printf '%s\n' "$*" >&2; exit 2; }
positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer; got: $2"
}

GATE_MODE="${GATE_MODE:-learned}"
ROTATION_6D_LAYOUT="${ROTATION_6D_LAYOUT:-rows}"
GATE_INIT="${GATE_INIT:-0.9}"
GATE_HIDDEN_DIM="${GATE_HIDDEN_DIM:-128}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-8}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-2001}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
VAL_EVERY="${VAL_EVERY:-1}"
case "$GATE_MODE" in learned|open|closed) ;; *) fail "GATE_MODE must be learned, open, or closed." ;; esac
case "$ROTATION_6D_LAYOUT" in rows|columns) ;; *) fail "ROTATION_6D_LAYOUT must be rows or columns." ;; esac
positive_integer GATE_HIDDEN_DIM "$GATE_HIDDEN_DIM"
positive_integer STATE_HISTORY_STEPS "$STATE_HISTORY_STEPS"
(( STATE_HISTORY_STEPS >= 4 )) || fail "STATE_HISTORY_STEPS must be at least 4 (three past commands for command differences)."
positive_integer NUM_EPOCHS "$NUM_EPOCHS"
positive_integer CHECKPOINT_EVERY "$CHECKPOINT_EVERY"
positive_integer VAL_EVERY "$VAL_EVERY"
(( CHECKPOINT_EVERY % VAL_EVERY == 0 )) || fail "CHECKPOINT_EVERY must be divisible by VAL_EVERY so every checkpoint has a validation loss."
[[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]] || fail "SEED must be a nonnegative integer."

REPO_DIR=$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")
DEFAULT_TOKENIZER=/workspace/ysk/past2next_bug_fixed/output/training/pen_cabinet_lp3_N67_current_20260919_213053/frozen_tokenizer.ckpt
TOKENIZER_ARG="${1:-${TOKENIZER_CHECKPOINT:-$DEFAULT_TOKENIZER}}"
TOKENIZER_PATH=$(realpath -m -- "$TOKENIZER_ARG")
DATASET_PATH=$(realpath -m -- "${DATASET_PATH:-/workspace/ysk/zarr/pen_cabinet_lp3_N67.zarr}")
RUN_DIR=$(realpath -m -- "${2:-$REPO_DIR/output/training/pen_cabinet_lp3_N67_gate_${GATE_MODE}_seed${SEED}_$(date -u +%Y%m%d_%H%M%S_%N)}")
INIT_PATH=""
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    INIT_PATH=$(realpath -m -- "$INIT_CHECKPOINT")
fi
TRAIN_PY="${TRAIN_PY:-/venv/real_robot/bin/python}"
TRAIN_PY_PATH=$(command -v -- "$TRAIN_PY") || fail "Python executable not found: $TRAIN_PY"
[[ -x "$TRAIN_PY_PATH" ]] || fail "Python executable is not executable: $TRAIN_PY_PATH"
TRAIN_PY_PATH=$(realpath -ms -- "$TRAIN_PY_PATH")
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a GPU_SELECTION <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#GPU_SELECTION[@]} -eq 2 && -n "${GPU_SELECTION[0]}" && -n "${GPU_SELECTION[1]}" ]] \
    || fail "CUDA_VISIBLE_DEVICES must select exactly two GPUs."
[[ "${GPU_SELECTION[0]}" != "${GPU_SELECTION[1]}" ]] \
    || fail "CUDA_VISIBLE_DEVICES must select two distinct GPUs."

# Check the scalar before Hydra composition; importing Python does not start training.
"$TRAIN_PY_PATH" -c 'import sys; value = float(sys.argv[1]); (0.0 < value < 1.0) or sys.exit("GATE_INIT must be strictly between 0 and 1.")' "$GATE_INIT"
if [[ "$DRY_RUN" == false ]]; then
    [[ -d "$DATASET_PATH" && -r "$DATASET_PATH" ]] \
        || fail "Dataset directory not found or not readable: $DATASET_PATH (set DATASET_PATH)."
    [[ -f "$TOKENIZER_PATH" && -r "$TOKENIZER_PATH" ]] \
        || fail "Tokenizer checkpoint not found or not readable: $TOKENIZER_PATH"
    if [[ -n "$INIT_PATH" ]]; then
        [[ -f "$INIT_PATH" && -r "$INIT_PATH" ]] \
            || fail "Initialization checkpoint not found or not readable: $INIT_PATH"
    fi
    [[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] \
        || fail "Output path already exists; choose a new run directory: $RUN_DIR"
fi

# Shell quoting and Hydra string quoting are separate requirements.
hydra_string() {
    local value="$1"
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '"%s"' "$value"
}
INIT_OVERRIDE=null
if [[ -n "$INIT_PATH" ]]; then
    INIT_OVERRIDE=$(hydra_string "$INIT_PATH")
fi

ARGS=(
    --config-name=experimental/train_past2next_state_history_gate_real_robot
    "policy.action_tokenizer.checkpoint=$(hydra_string "$TOKENIZER_PATH")"
    "policy.history_gate_mode=$GATE_MODE"
    "policy.rotation_6d_layout=$ROTATION_6D_LAYOUT"
    "policy.history_gate_init=$GATE_INIT"
    "policy.history_gate_hidden_dim=$GATE_HIDDEN_DIM"
    "state_history_steps=$STATE_HISTORY_STEPS"
    "seed=$SEED"
    task.policy.dataset.seed=42
    "task.policy.dataset.zarr_path=$(hydra_string "$DATASET_PATH")"
    training.resume=false
    "training.init_checkpoint=$INIT_OVERRIDE"
    training.use_ema=true
    "training.num_epochs=$NUM_EPOCHS"
    "training.val_every=$VAL_EVERY"
    "training.checkpoint_every=$CHECKPOINT_EVERY"
    training.snapshot_every=0
    checkpoint.save_all=true
    checkpoint.topk.k=0
    checkpoint.save_last_ckpt=false
    checkpoint.save_last_snapshot=false
    dataloader.batch_size=32
    val_dataloader.batch_size=32
    logging.mode=online
    logging.resume=false
    task.policy.lazy_eval=true
    "hydra.run.dir=$(hydra_string "$RUN_DIR")"
)
if [[ "$DRY_RUN" == true ]]; then
    ARGS+=(--cfg job --resolve)
fi

cd -- "$REPO_DIR"
printf 'Gate mode: %s; initial value: %s\nTokenizer: %s\nDataset: %s\nOutput directory: %s\n' \
    "$GATE_MODE" "$GATE_INIT" "$TOKENIZER_PATH" "$DATASET_PATH" "$RUN_DIR"
if [[ -n "$INIT_PATH" ]]; then
    printf 'Initialize policy from EMA weights: %s (fresh optimizer and self-past schedule)\n' "$INIT_PATH"
fi
if [[ "$DRY_RUN" == true ]]; then
    printf 'DRY RUN: config only; dataset/checkpoint existence and contents are NOT checked.\n'
    exec "$TRAIN_PY_PATH" scripts/run_workspace.py "${ARGS[@]}"
fi
exec "$TRAIN_PY_PATH" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/run_workspace.py "${ARGS[@]}"
