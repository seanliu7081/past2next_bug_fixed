#!/usr/bin/env bash
# pi05_p2n_vlm_v1 with an fp32 EMA of the trainable weights; every other setting is the v1 recipe.
# Decay 0.9998 with warmup power 0.85: d(n) = min(0.9998, 1 - (1+n)^-0.85), a time constant of
# ~2.5k updates at 10k and 5k from ~22.5k on (90% of the weight then spans ~11.5k updates, so the
# average lags the live weights and early rollouts can read low). Validation, in-training rollouts and
# snapshots/upd-NNNNNN_ema.ckpt use the EMA weights; checkpoints/latest.ckpt also keeps the live
# weights (evaluate with --weights ema or --weights model). See docs/PI05_P2N_VLM_V1.md.
# bash train_pi05_p2n_vlm_v1_ema.sh --devices 0,1,2,3 --dry-run
# bash train_pi05_p2n_vlm_v1_ema.sh --devices 0,1,2,3 --probe --output /tmp/pi05_p2n_vlm_v1_ema_probe
# bash train_pi05_p2n_vlm_v1_ema.sh --devices 0,1,2,3 --output output/training/pi05_p2n_vlm_v1_ema_libero10_seed42
# Resume with this launcher too (the checkpoint's EMA state must match the config).
# Hydra overrides after -- are applied after these settings, so e.g. `-- ema.decay=0.999` wins.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EMA=(training.use_ema=true ema.decay=0.9998 ema.warmup_power=0.85
     'logging.tags=[${variant},${task_name},pi05_base,native_vlm_ar,p2n_history_prefix,ema]')
args=()
overrides=()
while (($#)); do
  if [[ "$1" == -- ]]; then
    shift
    overrides=("$@")
    break
  fi
  args+=("$1")
  shift
done
for override in "${overrides[@]}"; do
  if [[ "$override" =~ ^[+]*training[.]use_ema= ]]; then
    echo 'train_pi05_p2n_vlm_v1_ema.sh always enables EMA; use train_pi05_p2n_vlm_v1.sh for the recipe without it' >&2
    exit 2
  fi
done
exec bash "$ROOT/train_pi05_p2n_vlm_v1.sh" "${args[@]}" -- "${EMA[@]}" "${overrides[@]}"
