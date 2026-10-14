#!/usr/bin/env bash
# Additive launcher: full vision + P2N action tuning, with Gemma frozen.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${P2N_PYTHON:-/venv/oat/bin/python}}"
args=()
while (($#)); do
  case "$1" in
    --python)
      (($# >= 2)) || { echo '--python requires an interpreter path' >&2; exit 2; }
      PYTHON_BIN="$2"; shift 2 ;;
    --)
      args+=("$@"); break ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not executable: $PYTHON_BIN (set PYTHON_BIN/P2N_PYTHON or pass --python)" >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd -- "$ROOT"
exec "$PYTHON_BIN" "$ROOT/scripts/train_p2n_vla_vision_action.py" "${args[@]}"
