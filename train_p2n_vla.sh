#!/usr/bin/env bash
# P2N-VLA launcher wrapper (see docs/P2N_VLA.md). Every argument goes to scripts/train_p2n_vla.py:
#   bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --devices 0,1 --preflight
#   bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --devices 0,1 --probe
#   bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output output/training/p2n_vla_s42
#   bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output output/training/p2n_vla_libero10_s42 \
#       -- task.policy.lazy_eval=false logging.mode=online      # + official LIBERO-10 eval every 50 epochs, live W&B
#   bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output output/training/p2n_vla_s42 \
#       --resume output/training/p2n_vla_s42/checkpoints/latest.ckpt
# Hydra overrides follow `--`. Choose the interpreter with --python PATH or P2N_PYTHON (default /venv/oat).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${P2N_PYTHON:-/venv/oat/bin/python}"
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
  echo "Python interpreter not found or not executable: $PYTHON_BIN (set P2N_PYTHON or pass --python)" >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd -- "$ROOT"
exec "$PYTHON_BIN" "$ROOT/scripts/train_p2n_vla.py" "${args[@]}"
