#!/usr/bin/env bash
# Two-GPU self-past training, live W&B, and execution-aware evaluation.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash train_self_past_executed_live.sh [--dry-run] [TOKENIZER_CHECKPOINT [OUTPUT_DIR]]

Defaults to the existing conjugate tokenizer checkpoint (ep-1300).
Starts a fresh policy on two GPUs with corrected EMA, W&B online, and lazy_eval=false.
Uses batch size 32 per GPU (global batch size 64).
Training remains offline self-past; executed histories are used during rollout.
Relative paths are resolved from the caller's working directory.
OUTPUT_DIR defaults to a fresh timestamped directory under output/training/.
TRAIN_PY overrides the Python executable (default: /venv/oat/bin/python).
MUJOCO_GL defaults to egl for simulator rendering.
CUDA_VISIBLE_DEVICES defaults to 0,1; override it to select another GPU pair.

--dry-run resolves and prints the config without starting training or W&B.
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

DEFAULT_TOKENIZER=/workspace/past2next_clean/output/20260827/070913_train_oattok_so3aug_libero10_N500/checkpoints/ep-1300_mse-0.001.ckpt
TOKENIZER_ARG="${1:-$DEFAULT_TOKENIZER}"
if [[ ! -f "$TOKENIZER_ARG" ]]; then
    printf 'Tokenizer checkpoint not found: %s\n' "$TOKENIZER_ARG" >&2
    exit 2
fi

REPO_DIR=$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")
TOKENIZER_PATH=$(realpath -- "$TOKENIZER_ARG")
RUN_DIR=$(realpath -m -- "${2:-$REPO_DIR/output/training/self_past_executed_live_seed42_$(date -u +%Y%m%d_%H%M%S_%N)}")
TRAIN_PY="${TRAIN_PY:-/venv/oat/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Shell quoting and Hydra string quoting are separate requirements.
hydra_string() {
    local value="$1"
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '"%s"' "$value"
}

ARGS=(
    --config-name=experimental/train_past2next_self_past_executed
    "policy.action_tokenizer.checkpoint=$(hydra_string "$TOKENIZER_PATH")"
    seed=42
    task.policy.dataset.seed=42
    training.resume=false
    training.init_checkpoint=null
    training.use_ema=true
    training.num_epochs=2001
    training.rollout_every=100
    training.val_every=1
    dataloader.batch_size=32
    val_dataloader.batch_size=32
    logging.mode=online
    logging.resume=false
    task.policy.lazy_eval=false
    "hydra.run.dir=$(hydra_string "$RUN_DIR")"
)
if [[ "$DRY_RUN" == true ]]; then
    ARGS+=(--cfg job --resolve)
fi

cd -- "$REPO_DIR"
printf 'Tokenizer: %s\nOutput directory: %s\n' "$TOKENIZER_PATH" "$RUN_DIR"
if [[ "$DRY_RUN" == true ]]; then
    exec "$TRAIN_PY" scripts/run_workspace.py "${ARGS[@]}"
fi
exec "$TRAIN_PY" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/run_workspace.py "${ARGS[@]}"
