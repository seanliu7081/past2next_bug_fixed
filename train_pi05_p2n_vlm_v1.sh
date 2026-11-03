#!/usr/bin/env bash
# Native VLM autoregressive P2N v1. Options are those of scripts/train_p2n_vla.py.
# bash train_pi05_p2n_vlm_v1.sh --devices 0,1 --dry-run
# bash train_pi05_p2n_vlm_v1.sh --devices 0,1 --probe --output /tmp/pi05_p2n_vlm_v1_probe
# bash train_pi05_p2n_vlm_v1.sh --devices 0,1 --output output/training/pi05_p2n_vlm_v1_seed42
# Hydra overrides follow --; the launcher keeps the default effective batch at 32.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
  [[ "$arg" != -- ]] || break
  if [[ "$arg" == --variant || "$arg" == --variant=* ]]; then
    echo 'train_pi05_p2n_vlm_v1.sh fixes --variant to pi05_p2n_vlm_v1' >&2
    exit 2
  fi
done
exec bash "$ROOT/train_p2n_vla.sh" --variant pi05_p2n_vlm_v1 --task libero \
  --spm "$ROOT/data/pretrained/p2n_vla/paligemma_tokenizer.model" \
  --tokenizer "$ROOT/data/pretrained/p2n_vla/oat_from_upd030000.ckpt" "$@"
