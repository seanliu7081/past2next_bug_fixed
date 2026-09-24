#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${P2N_PYTHON:-python}"
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
cd -- "$ROOT"
exec "$PYTHON_BIN" "$ROOT/scripts/train_p2n_new.py" "${args[@]}"
