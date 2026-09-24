#!/usr/bin/env bash
# Trainable ResNet-18 direct action-flow Bash entrypoint: sequential plain/gate runs with explicit flag overrides.
set -euo pipefail
FLOW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$FLOW_ROOT"
FLOW_PYTHON="${FLOW_PYTHON:-${P2N_FLOW_PYTHON:-/venv/real_robot/bin/python}}"
FLOW_GPUS="${FLOW_GPUS:-2,3}"
FLOW_OUTPUT_ROOT="${FLOW_OUTPUT_ROOT:-$FLOW_ROOT/output/training}"
FLOW_TASK=nut_washer
FLOW_SELECTION=both
FLOW_OUTPUT=''
FLOW_RESUME=''
FLOW_PASSTHROUGH=()
FLOW_FLAG_OVERRIDES=()
FLOW_USER_OVERRIDES=()

flow_help() {
  cat <<'HELP'
Usage: bash train_p2n_action_flow_resnet18.sh [OPTIONS] [-- HYDRA_OVERRIDES...]

  --gpus 2,3              Physical GPU indices; checked before training
  --variant VARIANT       both (default), p2n_action_flow, or p2n_state_gate_action_flow
  --batch-size N          Training microbatch per GPU: >=4 and divisible by 4 (default 4)
  --val-batch-size N      Validation loader batch per rank (default 4)
  --grad-accum N          Microbatches per optimizer update (default 8)
  --save-every N          Save latest and periodic checkpoints every N completed epochs (default 20)
  --task TASK             nut_washer (default), pen_cabinet, fruits, fruits_v2, or libero
                          real_robot is a compatibility alias for nut_washer
  --python PATH           Python interpreter (default /venv/real_robot/bin/python)
  --output PATH           Single variant: exact output path; both: parent for separate run directories
  --resume CHECKPOINT     Resume a full artifact; requires one explicit variant
  --dry-run, --preflight  CPU source/schema checks only; no training or W&B run
  --num-processes N       Must match the selected GPU count
  --allow-busy-gpus       Explicitly allow GPUs reported as busy
  --help                  Show this help

Defaults: both variants run sequentially on nut_washer_v3_N77 (2001 epochs).
Select --task pen_cabinet, fruits, or fruits_v2 to change the real-robot dataset.
ResNet-18 is trainable and initialized from scratch.
W&B logging is online by default for fresh training and resume.
Effective batch = batch-size x GPU count x grad-accum.
Explicit Hydra overrides after -- take precedence over convenience flags.
HELP
}

while (($#)); do
  # Support both --flag VALUE and --flag=VALUE before the Hydra separator.
  if [[ "$1" == --*=* ]]; then
    set -- "${1%%=*}" "${1#*=}" "${@:2}"
  fi
  case "$1" in
    --gpus|--variant|--task|--python|--output|--resume|--num-processes|--batch-size|--val-batch-size|--grad-accum|--save-every)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      FLOW_FLAG="$1"
      FLOW_VALUE="$2"
      shift 2
      case "$FLOW_FLAG" in
        --gpus) FLOW_GPUS="$FLOW_VALUE" ;;
        --variant) FLOW_SELECTION="$FLOW_VALUE" ;;
        --task) FLOW_TASK="$FLOW_VALUE" ;;
        --python) FLOW_PYTHON="$FLOW_VALUE" ;;
        --output) FLOW_OUTPUT="$FLOW_VALUE" ;;
        --resume) FLOW_RESUME="$FLOW_VALUE" ;;
        --num-processes) FLOW_PASSTHROUGH+=(--num-processes "$FLOW_VALUE") ;;
        --batch-size|--val-batch-size|--grad-accum|--save-every)
          if [[ ! "$FLOW_VALUE" =~ ^[1-9][0-9]*$ ]]; then
            echo "$FLOW_FLAG requires a positive integer" >&2
            exit 2
          fi
          case "$FLOW_FLAG" in
            --batch-size)
              if ((FLOW_VALUE < 4 || FLOW_VALUE % 4 != 0)); then
                echo '--batch-size must be at least 4 and a multiple of 4 for the FM/CT split' >&2
                exit 2
              fi
              FLOW_FLAG_OVERRIDES+=("dataloader.batch_size=$FLOW_VALUE") ;;
            --val-batch-size) FLOW_FLAG_OVERRIDES+=("val_dataloader.batch_size=$FLOW_VALUE") ;;
            --grad-accum) FLOW_FLAG_OVERRIDES+=("training.gradient_accumulate_every=$FLOW_VALUE") ;;
            --save-every) FLOW_FLAG_OVERRIDES+=("training.checkpoint_every=$FLOW_VALUE" "training.snapshot_every=$FLOW_VALUE") ;;
          esac ;;
      esac ;;
    --dry-run|--preflight|--allow-busy-gpus)
      FLOW_PASSTHROUGH+=("$1"); shift ;;
    --)
      shift; FLOW_USER_OVERRIDES=("$@"); break ;;
    --help|-h)
      flow_help; exit 0 ;;
    *)
      echo "Unknown argument: $1 (see --help)" >&2
      exit 2 ;;
  esac
done

case "$FLOW_TASK" in
  nut_washer|pen_cabinet|fruits|fruits_v2|real_robot|libero) ;;
  *) echo "Unknown task: $FLOW_TASK" >&2; exit 2 ;;
esac
case "$FLOW_SELECTION" in
  both) FLOW_VARIANTS=(p2n_action_flow p2n_state_gate_action_flow) ;;
  p2n_action_flow|p2n_state_gate_action_flow) FLOW_VARIANTS=("$FLOW_SELECTION") ;;
  *) echo "Unknown variant: $FLOW_SELECTION" >&2; exit 2 ;;
esac
if [[ -n "$FLOW_RESUME" && "$FLOW_SELECTION" == both ]]; then
  echo '--resume requires --variant p2n_action_flow or --variant p2n_state_gate_action_flow' >&2
  exit 2
fi

FLOW_SOURCE_ARGS=()
if [[ -n "$FLOW_RESUME" ]]; then
  FLOW_SOURCE_ARGS+=(--resume "$FLOW_RESUME")
fi
FLOW_DEFAULT_OVERRIDES=(logging.mode=online)
export WANDB_MODE=online
[[ -z "${FLOW_DATA:-}" ]] || FLOW_DEFAULT_OVERRIDES+=("task.policy.dataset.zarr_path=$FLOW_DATA")
[[ -z "${FLOW_PROJECT:-}" ]] || FLOW_DEFAULT_OVERRIDES+=("logging.project=$FLOW_PROJECT")
FLOW_STAMP="$(date +%Y%m%d_%H%M%S)_$$"

for FLOW_VARIANT in "${FLOW_VARIANTS[@]}"; do
  FLOW_OUTPUT_ARGS=()
  if [[ "$FLOW_SELECTION" == both ]]; then
    FLOW_RUN="${FLOW_TASK}_${FLOW_VARIANT}_resnet18_${FLOW_STAMP}"
    FLOW_OUTPUT_ARGS=(--output "${FLOW_OUTPUT:-$FLOW_OUTPUT_ROOT}/$FLOW_RUN")
  elif [[ -n "$FLOW_OUTPUT" ]]; then
    FLOW_OUTPUT_ARGS=(--output "$FLOW_OUTPUT")
  elif [[ -z "$FLOW_RESUME" ]]; then
    FLOW_RUN="${FLOW_TASK}_${FLOW_VARIANT}_resnet18_${FLOW_STAMP}"
    FLOW_OUTPUT_ARGS=(--output "$FLOW_OUTPUT_ROOT/$FLOW_RUN")
  fi
  "$FLOW_PYTHON" "$FLOW_ROOT/scripts/train_p2n_action_flow_resnet18.py" \
    --variant "$FLOW_VARIANT" --task "$FLOW_TASK" --gpus "$FLOW_GPUS" \
    "${FLOW_SOURCE_ARGS[@]}" "${FLOW_OUTPUT_ARGS[@]}" "${FLOW_PASSTHROUGH[@]}" \
    -- "${FLOW_DEFAULT_OVERRIDES[@]}" "${FLOW_FLAG_OVERRIDES[@]}" "${FLOW_USER_OVERRIDES[@]}"
done
