#!/usr/bin/env bash
# Train offline self-past Past2Next with execution-acknowledged inference history.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash train_self_past_executed.sh [--dry-run] TOKENIZER_CHECKPOINT [OUTPUT_DIR]

Uses experimental/train_past2next_self_past_executed with model seed 42 and dataset split seed 42.
Preserves offline generated self-past training; inference requires execution acknowledgments.
Use a tokenizer trained with the corrected code and the same training split.
Relative paths are resolved from the caller's working directory.
OUTPUT_DIR defaults to a timestamped directory under output/training/.
TRAIN_PY overrides the Python executable (default: /venv/oat/bin/python).

--dry-run resolves and prints the training config without starting training.
HELP
}

DRY_RUN=false
case "${1:-}" in
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
esac
if (( $# < 1 || $# > 2 )); then
    usage >&2
    exit 2
fi
if [[ ! -f "$1" ]]; then
    printf 'Tokenizer checkpoint not found: %s\n' "$1" >&2
    exit 2
fi

REPO_DIR=$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")
TOKENIZER_PATH=$(realpath -- "$1")
RUN_DIR=$(realpath -m -- "${2:-$REPO_DIR/output/training/self_past_executed_seed42_$(date -u +%Y%m%d_%H%M%S_%N)}")
TRAIN_PY="${TRAIN_PY:-/venv/oat/bin/python}"

# Hydra parses overrides after the shell; quote paths for both parsers.
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
    "hydra.run.dir=$(hydra_string "$RUN_DIR")"
)
if [[ "$DRY_RUN" == true ]]; then
    ARGS+=(--cfg job --resolve)
fi

cd -- "$REPO_DIR"
printf 'Tokenizer: %s\nOutput directory: %s\n' "$TOKENIZER_PATH" "$RUN_DIR"
exec "$TRAIN_PY" scripts/run_workspace.py "${ARGS[@]}"
