#!/usr/bin/env bash
# Thin additive wrapper: all user flags, including Hydra overrides, pass through.
set -euo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" "$SCRIPT_ROOT/scripts/train_p2n_new_original_obs.py" "$@"
