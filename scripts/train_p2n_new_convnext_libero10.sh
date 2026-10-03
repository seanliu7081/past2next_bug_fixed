#!/usr/bin/env bash
# Train frozen ConvNeXt Nano P2N variants on LIBERO-10 with simulator evaluation.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: train_p2n_new_convnext_libero10.sh [options]

Trains base P2N, then state-gated P2N by default.

  --variant NAME            both, p2n_new, p2n_state_gate_new (default: both)
  --batch-size N            Training batch per GPU (default: 8)
  --gpu IDS                 Comma-separated GPU indices or UUIDs (default: 0,1)
                            One training process is used per selected GPU.
  --convnext-frozen true    Frozen pretrained backbone (false is unsupported)
  --gradient N             Gradient accumulation steps (default: 4)
  --num-epochs N           Training epochs for each variant (default: 251)
  --lazy-eval BOOL         Skip simulator evaluation if true (default: false)
  --eval-every N           Simulator rollout interval in epochs (default: 100)
                            Existing epoch labels are 0, 100, 200, ...
  --eval-episodes N        Total rollout episodes across 10 tasks (default: 100)
  --eval-envs N            Parallel simulator environments (default: 10)
  --wandb-mode MODE        online, offline, disabled (default: offline)
  --tokenizer PATH         Override the LIBERO-10 tokenizer checkpoint
  --dataset PATH           Override the LIBERO-10 Zarr dataset
  --python PATH            Python executable (default: starvla-heading environment)
  --dry-run                Resolve configurations without training or loading data
  --preflight              Check local weights and data without training
  -h, --help               Show this help

Aliases: --batch_size, --devices, --convnext_frozen,
         --gradient-accumulation, --num_epochs, --lazy_eval,
         --eval_every, --eval_episodes, --eval_envs, --wandb_mode.
USAGE
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

require_value() {
  if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
    die "$1 requires a value. Use --help for available options."
  fi
}

P2N_VARIANT_SELECTION=both
P2N_BATCH_SIZE=8
P2N_DEVICES=0,1
P2N_CONVNEXT_FROZEN=true
P2N_GRADIENT_ACCUMULATION=4
P2N_NUM_EPOCHS=251
P2N_LAZY_EVAL=false
P2N_EVAL_EVERY=100
P2N_EVAL_EPISODES=100
P2N_EVAL_ENVS=10
P2N_WANDB_MODE=offline
P2N_PYTHON=/workspace/venvs/starvla-heading/bin/python
P2N_TOKENIZER_PATH=
P2N_DATASET_PATH=
mode_args=()

while (( $# )); do
  case "$1" in
    --variant)
      require_value "$@"; P2N_VARIANT_SELECTION="$2"; shift 2 ;;
    --batch-size|--batch_size)
      require_value "$@"; P2N_BATCH_SIZE="$2"; shift 2 ;;
    --gpu|--devices)
      require_value "$@"; P2N_DEVICES="$2"; shift 2 ;;
    --convnext-frozen|--convnext_frozen)
      require_value "$@"; P2N_CONVNEXT_FROZEN="$2"; shift 2 ;;
    --gradient|--gradient-accumulation)
      require_value "$@"; P2N_GRADIENT_ACCUMULATION="$2"; shift 2 ;;
    --num-epochs|--num_epochs)
      require_value "$@"; P2N_NUM_EPOCHS="$2"; shift 2 ;;
    --lazy-eval|--lazy_eval)
      require_value "$@"; P2N_LAZY_EVAL="$2"; shift 2 ;;
    --eval-every|--eval_every)
      require_value "$@"; P2N_EVAL_EVERY="$2"; shift 2 ;;
    --eval-episodes|--eval_episodes)
      require_value "$@"; P2N_EVAL_EPISODES="$2"; shift 2 ;;
    --eval-envs|--eval_envs)
      require_value "$@"; P2N_EVAL_ENVS="$2"; shift 2 ;;
    --wandb-mode|--wandb_mode)
      require_value "$@"; P2N_WANDB_MODE="$2"; shift 2 ;;
    --tokenizer)
      require_value "$@"; P2N_TOKENIZER_PATH="$2"; shift 2 ;;
    --dataset)
      require_value "$@"; P2N_DATASET_PATH="$2"; shift 2 ;;
    --python)
      require_value "$@"; P2N_PYTHON="$2"; shift 2 ;;
    --dry-run|--preflight)
      (( ${#mode_args[@]} == 0 )) || die 'Choose only one of --dry-run and --preflight.'
      mode_args=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1. Use --help for available options." ;;
  esac
done

for P2N_INTEGER_OPTION in \
  "--batch-size:$P2N_BATCH_SIZE" \
  "--gradient:$P2N_GRADIENT_ACCUMULATION" \
  "--num-epochs:$P2N_NUM_EPOCHS" \
  "--eval-every:$P2N_EVAL_EVERY" \
  "--eval-episodes:$P2N_EVAL_EPISODES" \
  "--eval-envs:$P2N_EVAL_ENVS"; do
  [[ "${P2N_INTEGER_OPTION#*:}" =~ ^[1-9][0-9]*$ ]] \
    || die "${P2N_INTEGER_OPTION%%:*} must be a positive integer."
done
case "$P2N_VARIANT_SELECTION" in
  both) P2N_VARIANTS=(p2n_new p2n_state_gate_new) ;;
  p2n_new|p2n_state_gate_new) P2N_VARIANTS=("$P2N_VARIANT_SELECTION") ;;
  *) die '--variant must be both, p2n_new, or p2n_state_gate_new.' ;;
esac
case "$P2N_LAZY_EVAL" in
  true|false) ;;
  *) die '--lazy-eval must be true or false.' ;;
esac
case "$P2N_WANDB_MODE" in
  online|offline|disabled) ;;
  *) die '--wandb-mode must be online, offline, or disabled.' ;;
esac
case "$P2N_CONVNEXT_FROZEN" in
  true) ;;
  false) die 'Fine-tuning is not implemented; this version requires --convnext-frozen true.' ;;
  *) die '--convnext-frozen must be true; false is unsupported by this version.' ;;
esac

[[ -n "$P2N_DEVICES" && "$P2N_DEVICES" != ,* && "$P2N_DEVICES" != *, && "$P2N_DEVICES" != *,,* ]] \
  || die '--gpu requires a nonempty comma-separated list, such as 0 or 0,1.'
[[ ! "$P2N_DEVICES" =~ [[:space:]] ]] || die '--gpu identifiers must not contain spaces or newlines.'
IFS=',' read -r -a P2N_GPU_IDS <<< "$P2N_DEVICES"
declare -A P2N_SEEN_GPU_IDS=()
for P2N_GPU_ID in "${P2N_GPU_IDS[@]}"; do
  [[ "$P2N_GPU_ID" =~ ^[0-9]+$ || "$P2N_GPU_ID" =~ ^GPU-[[:xdigit:]-]+$ ]] \
    || die "Invalid GPU identifier: $P2N_GPU_ID. Use GPU indices or UUIDs without spaces."
  [[ -z "${P2N_SEEN_GPU_IDS[$P2N_GPU_ID]:-}" ]] || die "Duplicate GPU identifier: $P2N_GPU_ID."
  P2N_SEEN_GPU_IDS[$P2N_GPU_ID]=1
done
P2N_NUM_PROCESSES="${#P2N_GPU_IDS[@]}"

P2N_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$P2N_REPO_DIR"

# Renderer selection must be set before simulator imports; the Python launcher
# handles CUDA/EGL GPU identifiers for the installed renderer.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == egl || "$MUJOCO_GL" == osmesa ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
fi
P2N_NANO_PATH="/workspace/models/convnextv2_nano.fcmae_ft_in22k_in1k/aacb23b94d2adf0b206df6c8a75798b672183d6d"
P2N_NANO_REVISION="aacb23b94d2adf0b206df6c8a75798b672183d6d"
P2N_RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
source_args=()
if [[ -n "$P2N_TOKENIZER_PATH" ]]; then source_args+=(--tokenizer "$P2N_TOKENIZER_PATH"); fi
if [[ -n "$P2N_DATASET_PATH" ]]; then source_args+=(--dataset "$P2N_DATASET_PATH"); fi

for P2N_VARIANT in "${P2N_VARIANTS[@]}"; do
  "$P2N_PYTHON" scripts/train_p2n_new_convnext_libero10.py \
    --variant "$P2N_VARIANT" \
    --convnext "$P2N_NANO_PATH" \
    --convnext-revision "$P2N_NANO_REVISION" \
    --devices "$P2N_DEVICES" \
    --num-processes "$P2N_NUM_PROCESSES" \
    --output "output/training/${P2N_VARIANT}_nano_libero10_frozen_${P2N_RUN_ID}" \
    "${source_args[@]}" \
    "${mode_args[@]}" \
    -- \
    "policy.convnext_frozen=$P2N_CONVNEXT_FROZEN" \
    "dataloader.batch_size=$P2N_BATCH_SIZE" \
    "training.gradient_accumulate_every=$P2N_GRADIENT_ACCUMULATION" \
    "training.num_epochs=$P2N_NUM_EPOCHS" \
    "task.policy.lazy_eval=$P2N_LAZY_EVAL" \
    "training.rollout_every=$P2N_EVAL_EVERY" \
    "task.policy.env_runner.n_test=$P2N_EVAL_EPISODES" \
    "task.policy.env_runner.n_parallel_envs=$P2N_EVAL_ENVS" \
    "logging.mode=$P2N_WANDB_MODE" \
    seed=42
done
