#!/usr/bin/env bash
# Train base P2N, then state-gated P2N, with the pinned frozen Nano backbone.
# Run with --help for batch, device, accumulation, epoch and mode flags.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: train_p2n_new_convnext_frozen.sh [options]

Trains base P2N, then state-gated P2N, using the same settings.

  --batch-size N             Training batch per GPU (default: 8)
  --gpu IDS                  Comma-separated GPU indices or UUIDs (default: 0,1)
                             One training process is used per selected GPU.
  --convnext-frozen true     Keep the pretrained backbone frozen (default: true).
                             false is unsupported; fine-tuning needs code changes.
  --gradient N               Gradient accumulation steps (default: 4)
  --num-epochs N             Training epochs for each variant (default: 2001)
  --tokenizer PATH           OAT checkpoint (default: downloaded base-run epoch 1540)
  --wandb-mode MODE          W&B logging: online, offline, disabled (default: offline)
  --dry-run                  Resolve both configs without training or loading data
  --preflight                Check local weights and data without training
  -h, --help                 Show this help

Aliases: --batch_size, --devices, --convnext_frozen,
         --gradient-accumulation, --num_epochs, --wandb_mode.
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

P2N_BATCH_SIZE=8
P2N_DEVICES=0,1
P2N_CONVNEXT_FROZEN=true
P2N_GRADIENT_ACCUMULATION=4
P2N_NUM_EPOCHS=2001
P2N_WANDB_MODE=offline
P2N_TOKENIZER_PATH="/workspace/models/nut-washer-v3-N77-so3aug-20260924/9f8a05efea09baeb1f5fa304cf104faa3b5b0b57/nut_washer_v3_N77_p2n_so3aug_20260924_081003_468366670/frozen_tokenizer.ckpt"

mode_args=()
while (( $# )); do
  case "$1" in
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
    --tokenizer)
      require_value "$@"; P2N_TOKENIZER_PATH="$2"; shift 2 ;;
    --wandb-mode|--wandb_mode)
      require_value "$@"; P2N_WANDB_MODE="$2"; shift 2 ;;
    --dry-run|--preflight)
      (( ${#mode_args[@]} == 0 )) || die 'Choose only one of --dry-run and --preflight.'
      mode_args=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1. Use --help for available options." ;;
  esac
done

[[ "$P2N_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die '--batch-size must be a positive integer.'
[[ "$P2N_GRADIENT_ACCUMULATION" =~ ^[1-9][0-9]*$ ]] || die '--gradient must be a positive integer.'
[[ "$P2N_NUM_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die '--num-epochs must be a positive integer.'
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

P2N_PYTHON=/workspace/venvs/starvla-heading/bin/python
P2N_NANO_PATH="/workspace/models/convnextv2_nano.fcmae_ft_in22k_in1k/aacb23b94d2adf0b206df6c8a75798b672183d6d"
P2N_NANO_REVISION="aacb23b94d2adf0b206df6c8a75798b672183d6d"
P2N_RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"

for P2N_VARIANT in p2n_new p2n_state_gate_new; do
  "$P2N_PYTHON" scripts/train_p2n_new_convnext.py \
    --variant "$P2N_VARIANT" \
    --task real_robot \
    --vision convnext_nano \
    --tokenizer "$P2N_TOKENIZER_PATH" \
    --convnext "$P2N_NANO_PATH" \
    --convnext-revision "$P2N_NANO_REVISION" \
    --devices "$P2N_DEVICES" \
    --num-processes "$P2N_NUM_PROCESSES" \
    --output "output/training/${P2N_VARIANT}_nano_frozen_${P2N_RUN_ID}" \
    "${mode_args[@]}" \
    -- \
    "policy.convnext_frozen=$P2N_CONVNEXT_FROZEN" \
    "dataloader.batch_size=$P2N_BATCH_SIZE" \
    "training.gradient_accumulate_every=$P2N_GRADIENT_ACCUMULATION" \
    "training.num_epochs=$P2N_NUM_EPOCHS" \
    "logging.mode=$P2N_WANDB_MODE" \
    seed=42
done
