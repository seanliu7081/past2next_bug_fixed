#!/usr/bin/env bash
# Two-stage, from-scratch nut-washer training. Run p2n and gated separately.
set -euo pipefail

# Edit the default below: 0.1 = 90% train / 10% validation.
# Shared by the tokenizer and policy for both p2n and gated.
export VAL_RATIO="${VAL_RATIO:-0.05}"

usage() {
  cat <<'HELP'
Usage: bash train_nut_washer_v3.sh {p2n|gated} [--dry-run]

  p2n    SO(3)-augmented tokenizer -> original Past2Next self-past policy
         GPUs 0,1,2,3; policy epochs default to 1001.
  gated  SO(3)-augmented tokenizer -> learned state-history gate policy
         GPUs 4,5,6,7; policy epochs default to 2001.

Both variants train their own tokenizer from scratch for 3001 epochs, select
its best held-out reconstruction checkpoint, freeze it, and train a fresh policy.
Neither stage resumes an existing run. Existing output directories are refused.

Optional environment variables:
  VAL_RATIO           Validation fraction (default: 0.1); same split in both stages.
  TOKENIZER_EPOCHS     Override tokenizer epoch budget (default: 3001).
  POLICY_EPOCHS        Override the variant's policy epoch budget.
  POLICY_BATCH_SIZE    Training batch per GPU (default: 64 for both variants).
  NUT_WASHER_RUN_DIR   Fresh output path (default: unique timestamped directory).
  TRAIN_PY            Python executable (default: /venv/real_robot/bin/python).
  WANDB_MODE          online (default) or offline.

--dry-run validates the dataset and resolves both configs without training.
HELP
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if (( $# < 1 || $# > 2 )); then usage >&2; exit 2; fi
variant=$1
case "$variant" in
  p2n)
    gpus=0,1,2,3
    policy_config=train_past2next_scratch_all500
    default_policy_epochs=1001
    ;;
  gated)
    gpus=4,5,6,7
    policy_config=experimental/train_past2next_state_history_gate_real_robot
    default_policy_epochs=2001
    ;;
  *) usage >&2; exit 2 ;;
esac
extra=()
if (( $# == 2 )); then
  [[ $2 == --dry-run ]] || { usage >&2; exit 2; }
  extra+=(--dry-run)
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_dir=${NUT_WASHER_RUN_DIR:-"$repo/output/training/nut_washer_v3_N77_${variant}_so3aug_$(date -u +%Y%m%d_%H%M%S_%N)"}
export TOKENIZER_EPOCHS=${TOKENIZER_EPOCHS:-4001}
export POLICY_EPOCHS=${POLICY_EPOCHS:-$default_policy_epochs}
export POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-64}
exec bash "$repo/train_pen_cabinet.sh" \
  --dataset /workspace/ysk/zarr/nut_washer_v3_N77.zarr \
  --tokenizer-config train_oattok_so3aug \
  --policy-config "$policy_config" \
  --gpus "$gpus" \
  --output-dir "$run_dir" \
  "${extra[@]}"
