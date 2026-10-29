#!/usr/bin/env bash
# Pi05_past launcher: pi05_base's original flow-matching action head plus the Past2Next condition
# (7 past commands, acc/jerk dynamic features, self-past) of oat/config/train_past2next_self_past.yaml.
# Design and evaluation: docs/PI05_PAST.md. Defaults live in oat/config/train_pi05_past.yaml (and the base
# oat/config/train_p2n_vla.yaml); flags override them and anything after `--` goes to Hydra verbatim.
# Relative paths are relative to the directory you launch from.
#
#   bash train_pi05_past.sh --help
#   bash train_pi05_past.sh --dry-run                             # resolve + validate the config only
#   bash train_pi05_past.sh --devices 0,1 --probe                 # memory/throughput go/no-go, no checkpoints
#   bash train_pi05_past.sh --devices 0,1,2,3                     # Pi05_past, 80k updates of batch 32
#   bash train_pi05_past.sh --devices 0,1                         # the same run on 2 GPUs
#   bash train_pi05_past.sh --devices 2,3 --use-past false        # pi05 no-past control, same regime
#   bash train_pi05_past.sh --devices 2,3 --self-past-p 0         # Pi05_past with ground-truth history only
#   bash train_pi05_past.sh --devices 0,1 --resume output/training/<run>/checkpoints/latest.ckpt
#   bash train_pi05_past.sh --devices 0,1 -- training.val_every=25 # any other Hydra override
#
# --variant pi05_p2n (or train_pi05_p2n.sh) swaps the flow head for the Past2Next self-past AR head over
# 8 OAT tokens and keeps everything else; see docs/PI05_P2N.md and oat/config/train_pi05_p2n.yaml.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${P2N_PYTHON:-/venv/oat/bin/python}"
UPDATES_PER_EPOCH=100   # train_pi05_past.yaml: max_train_steps = 100 x gradient_accumulate_every

usage() {
  cat <<'EOF'
Usage: bash train_pi05_past.sh [flags] [-- hydra.overrides ...]
Flags take a value as `--flag VALUE` or `--flag=VALUE`.

Run control
  --variant NAME          pi05_past (default: the pi05 flow head) or pi05_p2n (the Past2Next self-past AR head
                          over 8 OAT tokens; train_pi05_p2n.sh passes it). Everything else is shared
  --devices LIST          GPUs, e.g. 0,1; one process per GPU. Required to train or probe
  --output DIR            run directory; fresh runs need it empty
                          (default output/training/<variant>[_nopast|_gtpast]_libero10_seed<SEED>)
  --resume CKPT           continue from <run>/checkpoints/latest.ckpt with the same GPU count and the
                          original flags; the W&B run continues too (see --wandb-id)
  --seed N                training seed (default 42)
  --dry-run               compose and validate the config only (no data, weights or GPUs)
  --preflight             full CPU preflight incl. a model build (assets, split, prompts, parameter counts)
  --probe                 short memory/throughput probe; writes output/probe/... and no checkpoints
  --probe-steps N         probe optimizer steps (default 3; needs --probe)
  --allow-busy-gpus       skip the busy-GPU check
  --python PATH           interpreter (default $P2N_PYTHON or /venv/oat/bin/python)
  --pi05 PATH | --spm PATH | --tokenizer PATH   asset overrides (defaults: pinned HF cache / this checkout)
  --wandb-mode MODE       online | offline | disabled (default online; --probe defaults to disabled)
  --wandb-project NAME    W&B project (default p2n_vla)
  --wandb-name NAME       W&B run name
  --wandb-id ID           W&B run to continue (default with --resume: the run directory's latest W&B run)

Pi05_past condition (Past2Next)
  --use-past BOOL         true = Pi05_past (default); false = the pi05 no-past control
  --self-past-p P         per-sample self-past probability (default 0.5; 0 = ground-truth history)
  --self-past-warmup N    optimizer updates before self-past starts (default 1000)
  --self-past-ramp N      linear probability ramp after the warmup (default 0)
  --cond-dropout P        dropout on the condition embeddings in training (default 0.1)
  --cond-t0 T             constant flow time of the condition tokens' expert pass (default 0.6; pi05_p2n also
                          folds its adaRMS at this time)
  --flow-steps N          pi05_past only: Euler steps at inference and in self-past (default 10)

pi05_p2n head (pi05_p2n only)
  --temperature T         deployment sampler: in-training evaluation and the evaluator default (default 0 = greedy)
  --topk K                deployment top-k when temperature > 0 (default 10)
  --self-past-temperature T  self-past sampler (default 1.0, the Past2Next self-past sampler)
  --self-past-topk K      self-past top-k (default 10)

VLM regime
  --kv-grad-scale S       1 = the flow loss reaches the VLM (default); 0 = detached/insulated
  --lambda-ki L           OAT-token KI loss weight on the VLM side (default 0)
  --ki-init MODE          KI table init when lambda-ki > 0: embedding | random (default embedding)
  --train-vision BOOL     train SigLIP + projector (default true)
  --train-lora BOOL       train Gemma LoRA (default true)
  --lora-rank R           LoRA rank (default 32)
  --lora-alpha A          LoRA alpha (default 32)

Optimization
  --batch N               samples per optimizer update over all GPUs (default 32); the gradient
                          accumulation follows: N / (batch-size x GPUs), e.g. 2 on 4 GPUs, 4 on 2 GPUs
  --batch-size B          micro-batch per GPU, training and validation (default 4; 4 fills a 24 GB GPU)
  --grad-accum N          set the accumulation directly instead (batch = B x GPUs x N)
  --max-steps N           optimizer updates = hard stop and cosine horizon (default 80000); epochs of 100
                          updates follow. Short runs: lower --warmup and --self-past-warmup as well
  --warmup N              linear LR warmup updates (default 1000)
  --lr LR                 peak LR of expert, flow heads and SigLIP (default 5e-5)
  --new-lr LR             peak LR of condition modules, LoRA and KI table (default 1e-4)
  --min-lr-ratio R        cosine floor as a fraction of the peak (default 0.1)
  --weight-decay W        AdamW weight decay (default 1e-10)
  --grad-clip G           per-group max grad norm (default 1.0)
  --use-ema BOOL          keep an EMA (default false: an fp32 GPU EMA does not fit at micro-batch 4)
  --ema-decay D           EMA decay (default 0.999)
  --snapshot-every N      optimizer updates between periodic snapshots (only with --keep-snapshots all; 5000)

Evaluation and checkpoints
  --sim-eval-every N      LIBERO simulator evaluation on GPU 0 every N optimizer updates, a multiple of 100
                          (default 10000; 0 = none, evaluate offline with scripts/evaluate_p2n_vla.py)
  --sim-eval-episodes N   episodes per evaluation (default 500 = the official protocol, 50 per task)
  --keep-snapshots MODE   best = keep only the best evaluation's snapshot (snapshots/best.json; pi05_past default);
                          all = keep every --snapshot-every snapshot (default 5000 with this flag).
                          checkpoints/latest.ckpt is always kept. pi05_p2n's default keeps every evaluation's
                          snapshot and no periodic ones
  --checkpoint-every N    epochs between checkpoints/latest.ckpt (default 10)
  --val-every N           epochs between offline validations (default 50)
  --val-steps N           validation batches per GPU (default 200)
  --num-workers N         training dataloader workers per GPU (default 4)
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
parse_bool() {
  case "${2,,}" in
    true|1|yes|on) printf true ;;
    false|0|no|off) printf false ;;
    *) printf 'error: %s expects true or false, got %q\n' "$1" "$2" >&2; return 1 ;;
  esac
}
need_uint() { [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer, got '$2'"; }
need_posint() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer, got '$2'"; }
need_number() { [[ "$2" =~ ^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$ ]] || die "$1 must be a number, got '$2'"; }
abs_path() { if [[ "$1" == /* ]]; then printf '%s' "$1"; else printf '%s/%s' "$PWD" "$1"; fi; }
hydra_string() { local quoted=${1//\\/\\\\}; quoted=${quoted//\'/\\\'}; printf "'%s'" "$quoted"; }
set_mode() { [[ "$MODE" == train || "$MODE" == "$1" ]] || die "--$MODE and --$1 are mutually exclusive"; MODE="$1"; }

VALUE_FLAGS=" --devices --output --resume --seed --probe-steps --python --pi05 --spm --tokenizer --wandb-mode
 --wandb-project --wandb-name --wandb-id --use-past --self-past-p --self-past-warmup --self-past-ramp --cond-dropout
 --cond-t0 --flow-steps --kv-grad-scale --lambda-ki --ki-init --train-vision --train-lora --lora-rank --lora-alpha
 --batch --batch-size --grad-accum --max-steps --warmup --lr --new-lr --min-lr-ratio --weight-decay --grad-clip --use-ema
 --ema-decay --snapshot-every --checkpoint-every --val-every --val-steps --num-workers --sim-eval-every
 --sim-eval-episodes --keep-snapshots --variant --temperature --topk --self-past-temperature --self-past-topk "
VALUE_FLAGS=${VALUE_FLAGS//$'\n'/ }

VARIANT=""; FLOW_FLAG=""; AR_FLAGS=()
DEVICES=""; OUTPUT=""; RESUME=""; SEED=""; MODE="train"; PROBE_STEPS=""
ALLOW_BUSY=false; PI05=""; SPM=""; TOKENIZER=""; WANDB_MODE=""; WANDB_ID=""
USE_PAST=true; SELF_PAST_P=""; MAX_STEPS=""; MICRO=4; ACCUM=""; GLOBAL_BATCH=""
SIM_EVERY=""; KEEP=""; SNAPSHOT_EVERY=""
OVERRIDES=(); EXTRA=()

while (($#)); do
  flag="$1"
  case "$flag" in
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    --dry-run) set_mode dry-run; shift; continue ;;
    --preflight) set_mode preflight; shift; continue ;;
    --probe) set_mode probe; shift; continue ;;
    --allow-busy-gpus) ALLOW_BUSY=true; shift; continue ;;
    --*=*) value="${flag#*=}"; flag="${flag%%=*}"; consumed=1 ;;
    --*) value="${2-}"; consumed=2
         [[ "$VALUE_FLAGS" != *" $flag "* || ($# -ge 2 && "$value" != --*) ]] || die "$flag needs a value (see --help)" ;;
    *) die "unexpected argument '$flag' (Hydra overrides go after --)" ;;
  esac
  [[ "$VALUE_FLAGS" == *" $flag "* ]] || die "unknown flag $flag (see --help)"
  [[ -n "$value" ]] || die "$flag needs a value (see --help)"
  case "$flag" in
    --devices) [[ "$value" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--devices must be a comma-separated list of GPU indices, e.g. 2,3"
               DEVICES="$value" ;;
    --output) OUTPUT=$(abs_path "$value") ;;
    --resume) RESUME=$(abs_path "$value") ;;
    --seed) need_uint "$flag" "$value"; SEED="$value"; OVERRIDES+=("seed=$value") ;;
    --probe-steps) need_posint "$flag" "$value"; PROBE_STEPS="$value" ;;
    --python) if [[ "$value" == */* ]]; then PYTHON_BIN=$(abs_path "$value"); else PYTHON_BIN=$(command -v -- "$value" || true); fi ;;
    --pi05) PI05=$(abs_path "$value") ;;
    --spm) SPM=$(abs_path "$value") ;;
    --tokenizer) TOKENIZER=$(abs_path "$value") ;;
    --wandb-mode)
      case "$value" in online|offline|disabled) WANDB_MODE="$value" ;; *) die "--wandb-mode must be online, offline or disabled" ;; esac ;;
    --wandb-project) OVERRIDES+=("logging.project=$(hydra_string "$value")") ;;
    --wandb-name) OVERRIDES+=("logging.name=$(hydra_string "$value")") ;;
    --wandb-id) [[ "$value" =~ ^[A-Za-z0-9_-]+$ ]] || die "--wandb-id must be a W&B run id (letters, digits, - and _)"
                WANDB_ID="$value" ;;
    --use-past) USE_PAST=$(parse_bool "$flag" "$value") || exit 2 ;;
    --self-past-p) need_number "$flag" "$value"; SELF_PAST_P="$value"; OVERRIDES+=("policy.self_past_p=$value") ;;
    --self-past-warmup) need_uint "$flag" "$value"; OVERRIDES+=("policy.self_past_warmup_steps=$value") ;;
    --self-past-ramp) need_uint "$flag" "$value"; OVERRIDES+=("policy.self_past_ramp_steps=$value") ;;
    --cond-dropout) need_number "$flag" "$value"; OVERRIDES+=("policy.cond_dropout=$value") ;;
    --cond-t0) need_number "$flag" "$value"; OVERRIDES+=("policy.cond_t0=$value") ;;
    --flow-steps) need_posint "$flag" "$value"; FLOW_FLAG="$flag"; OVERRIDES+=("policy.flow_num_steps=$value") ;;
    --variant) [[ -z "$VARIANT" ]] || die "--variant given twice"
               case "$value" in pi05_past|pi05_p2n) VARIANT="$value" ;; *) die "--variant must be pi05_past or pi05_p2n" ;; esac ;;
    --temperature) need_number "$flag" "$value"; AR_FLAGS+=("$flag"); OVERRIDES+=("policy.temperature=$value") ;;
    --topk) need_posint "$flag" "$value"; AR_FLAGS+=("$flag"); OVERRIDES+=("policy.topk=$value") ;;
    --self-past-temperature) need_number "$flag" "$value"; AR_FLAGS+=("$flag"); OVERRIDES+=("policy.self_past_temperature=$value") ;;
    --self-past-topk) need_posint "$flag" "$value"; AR_FLAGS+=("$flag"); OVERRIDES+=("policy.self_past_topk=$value") ;;
    --kv-grad-scale) need_number "$flag" "$value"; OVERRIDES+=("policy.kv_grad_scale=$value") ;;
    --lambda-ki) need_number "$flag" "$value"; OVERRIDES+=("policy.lambda_ki=$value") ;;
    --ki-init) case "$value" in embedding|random) OVERRIDES+=("policy.ki_init=$value") ;; *) die "--ki-init must be embedding or random" ;; esac ;;
    --train-vision) bool=$(parse_bool "$flag" "$value") || exit 2; OVERRIDES+=("policy.train_vision=$bool") ;;
    --train-lora) bool=$(parse_bool "$flag" "$value") || exit 2; OVERRIDES+=("policy.train_lora=$bool") ;;
    --lora-rank) need_posint "$flag" "$value"; OVERRIDES+=("policy.lora_rank=$value") ;;
    --lora-alpha) need_number "$flag" "$value"; OVERRIDES+=("policy.lora_alpha=$value") ;;
    --batch) need_posint "$flag" "$value"; GLOBAL_BATCH="$value" ;;
    --batch-size) need_posint "$flag" "$value"; MICRO="$value"
                  OVERRIDES+=("dataloader.batch_size=$value" "val_dataloader.batch_size=$value") ;;
    --grad-accum) need_posint "$flag" "$value"; ACCUM="$value" ;;
    --max-steps) need_posint "$flag" "$value"; MAX_STEPS="$value" ;;
    --warmup) need_uint "$flag" "$value"; OVERRIDES+=("training.lr_warmup_steps=$value") ;;
    --lr) need_number "$flag" "$value"; OVERRIDES+=("optimizer.policy_lr=$value") ;;
    --new-lr) need_number "$flag" "$value"; OVERRIDES+=("optimizer.new_module_lr=$value") ;;
    --min-lr-ratio) need_number "$flag" "$value"; OVERRIDES+=("training.min_lr_ratio=$value") ;;
    --weight-decay) need_number "$flag" "$value"; OVERRIDES+=("optimizer.weight_decay=$value") ;;
    --grad-clip) need_number "$flag" "$value"; OVERRIDES+=("training.max_grad_norm=$value") ;;
    --use-ema) bool=$(parse_bool "$flag" "$value") || exit 2; OVERRIDES+=("training.use_ema=$bool") ;;
    --ema-decay) need_number "$flag" "$value"; OVERRIDES+=("ema.decay=$value") ;;
    --snapshot-every) need_uint "$flag" "$value"; SNAPSHOT_EVERY="$value" ;;
    --sim-eval-every) need_uint "$flag" "$value"; SIM_EVERY="$value" ;;
    --sim-eval-episodes) need_posint "$flag" "$value"; OVERRIDES+=("task.policy.env_runner.n_test=$value") ;;
    --keep-snapshots) case "$value" in best|all) KEEP="$value" ;; *) die "--keep-snapshots must be best or all" ;; esac ;;
    --checkpoint-every) need_posint "$flag" "$value"; OVERRIDES+=("training.checkpoint_every=$value") ;;
    --val-every) need_posint "$flag" "$value"; OVERRIDES+=("training.val_every=$value") ;;
    --val-steps) need_posint "$flag" "$value"; OVERRIDES+=("training.max_val_steps=$value" "training.max_reconst_steps=$value") ;;
    --num-workers) need_uint "$flag" "$value"; OVERRIDES+=("dataloader.num_workers=$value") ;;
  esac
  shift "$consumed"
done

# Settings that name the run directory or keep the schedule consistent have flags; refuse them after --.
for item in ${EXTRA[@]+"${EXTRA[@]}"}; do
  key="${item#+}"; key="${key#+}"; key="${key%%=*}"
  case "$key" in
    seed) die "use --seed instead of '-- $item' (it names the run directory)" ;;
    policy.use_past) die "use --use-past instead of '-- $item'" ;;
    policy.self_past_p) die "use --self-past-p instead of '-- $item'" ;;
    training.max_optimizer_steps|training.num_epochs|training.max_train_steps)
      die "use --max-steps instead of '-- $item' (it keeps the epochs and the cosine horizon consistent)" ;;
    training.gradient_accumulate_every|dataloader.batch_size|val_dataloader.batch_size)
      die "use --batch, --batch-size or --grad-accum instead of '-- $item'" ;;
  esac
done

VARIANT="${VARIANT:-pi05_past}"
if [[ "$VARIANT" == pi05_past ]]; then
  LABEL=Pi05_past
  ((${#AR_FLAGS[@]} == 0)) || die "${AR_FLAGS[0]} applies to the pi05_p2n token head only (the flow head has no token sampler)"
else
  LABEL=Pi05_p2n
  [[ -z "$FLOW_FLAG" ]] || die "$FLOW_FLAG applies to the pi05_past flow head only"
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die "Python interpreter not executable: ${PYTHON_BIN:-?} (set P2N_PYTHON or pass --python)"
[[ -z "$PROBE_STEPS" || "$MODE" == probe ]] || die "--probe-steps needs --probe"
if [[ -n "$DEVICES" ]]; then
  IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
  WORLD=${#DEVICE_LIST[@]}
elif [[ "$MODE" == train || "$MODE" == probe ]]; then
  die "--devices is required, e.g. --devices 0,1 (choose idle GPUs that are not reserved for other work)"
else
  WORLD=2  # dry-run and preflight describe the default 2-GPU run
fi

# The batch per update stays 32 on 1, 2, 4 or 8 GPUs; other counts need --batch or --grad-accum.
if [[ -n "$ACCUM" ]]; then
  [[ -z "$GLOBAL_BATCH" || "$GLOBAL_BATCH" -eq $((MICRO * WORLD * ACCUM)) ]] ||
    die "--batch $GLOBAL_BATCH != --batch-size $MICRO x $WORLD GPUs x --grad-accum $ACCUM"
else
  GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
  (( GLOBAL_BATCH % (MICRO * WORLD) == 0 )) ||
    die "--batch $GLOBAL_BATCH is not a multiple of --batch-size $MICRO x $WORLD GPUs (or pass --grad-accum)"
  ACCUM=$(( GLOBAL_BATCH / (MICRO * WORLD) ))
fi
OVERRIDES+=("training.gradient_accumulate_every=$ACCUM")
# Simulator evaluation and snapshot retention (best needs evaluations to rank and no periodic snapshots).
if [[ -n "$SIM_EVERY" ]]; then
  if (( SIM_EVERY == 0 )); then
    OVERRIDES+=("task.policy.lazy_eval=true")
    KEEP="${KEEP:-all}"
  else
    (( SIM_EVERY % UPDATES_PER_EPOCH == 0 )) || die "--sim-eval-every must be a multiple of $UPDATES_PER_EPOCH updates"
    OVERRIDES+=("task.policy.lazy_eval=false" "training.rollout_every=$((SIM_EVERY / UPDATES_PER_EPOCH))")
  fi
fi
case "${KEEP:-}" in
  best) [[ -z "$SNAPSHOT_EVERY" || "$SNAPSHOT_EVERY" == 0 ]] || die "--keep-snapshots best keeps no periodic snapshots"
        OVERRIDES+=("training.keep_snapshots=best" "training.snapshot_every=0") ;;
  all) OVERRIDES+=("training.keep_snapshots=all" "training.snapshot_every=${SNAPSHOT_EVERY:-5000}") ;;
  *) [[ -z "$SNAPSHOT_EVERY" ]] || OVERRIDES+=("training.snapshot_every=$SNAPSHOT_EVERY") ;;
esac
if [[ "$USE_PAST" == false ]]; then
  OVERRIDES+=("policy.use_past=false" "policy.self_past_p=0.0" "training.validate_generated_history=false")
fi
if [[ -n "$MAX_STEPS" ]]; then
  OVERRIDES+=("training.max_optimizer_steps=$MAX_STEPS"
              "training.num_epochs=$(( (MAX_STEPS + UPDATES_PER_EPOCH - 1) / UPDATES_PER_EPOCH ))")
fi
if [[ -n "$WANDB_MODE" ]]; then
  OVERRIDES+=("logging.mode=$WANDB_MODE")
elif [[ "$MODE" != "probe" ]]; then
  WANDB_MODE=online
  OVERRIDES+=("logging.mode=online")
fi
if [[ -n "$RESUME" && -z "$WANDB_ID" && "$MODE" != probe && "$WANDB_MODE" != disabled ]]; then
  # The workspace keeps W&B files in the run directory: <run>/wandb/latest-run -> [offline-]run-<time>-<id>
  latest=$(readlink -- "$(dirname -- "$(dirname -- "$RESUME")")/wandb/latest-run" 2>/dev/null || true)
  if [[ "$latest" =~ -([A-Za-z0-9]+)$ ]]; then
    WANDB_ID="${BASH_REMATCH[1]}"
    echo "W&B: continuing run $WANDB_ID from the run directory (pass --wandb-id to choose another)"
  else
    echo "warning: no W&B run found next to $RESUME; the resumed run logs to a new W&B run" >&2
  fi
fi
[[ -n "$WANDB_ID" ]] && OVERRIDES+=("logging.id=$WANDB_ID" "logging.resume=allow")

LAUNCH=(--variant "$VARIANT" --task libero)
[[ -n "$DEVICES" ]] && LAUNCH+=(--devices "$DEVICES")
LAUNCH+=(--spm "${SPM:-$ROOT/data/pretrained/p2n_vla/paligemma_tokenizer.model}")
LAUNCH+=(--tokenizer "${TOKENIZER:-$ROOT/data/pretrained/p2n_vla/oat_from_upd030000.ckpt}")
[[ -n "$PI05" ]] && LAUNCH+=(--pi05 "$PI05")
[[ "$ALLOW_BUSY" == true ]] && LAUNCH+=(--allow-busy-gpus)
case "$MODE" in
  dry-run) LAUNCH+=(--dry-run) ;;
  preflight) LAUNCH+=(--preflight) ;;
  probe) LAUNCH+=(--probe); [[ -n "$PROBE_STEPS" ]] && LAUNCH+=(--probe-steps "$PROBE_STEPS") ;;
esac
[[ -n "$RESUME" ]] && LAUNCH+=(--resume "$RESUME")
if [[ -z "$OUTPUT" && -z "$RESUME" && "$MODE" != "probe" ]]; then
  SUFFIX=""
  if [[ "$USE_PAST" == false ]]; then
    SUFFIX="_nopast"
  elif [[ -n "$SELF_PAST_P" ]] && awk -v p="$SELF_PAST_P" 'BEGIN { exit !(p + 0 == 0) }'; then
    SUFFIX="_gtpast"
  fi
  OUTPUT="$ROOT/output/training/${VARIANT}${SUFFIX}_libero10_seed${SEED:-42}"
fi
[[ -n "$OUTPUT" ]] && LAUNCH+=(--output "$OUTPUT")

echo "$LABEL: mode=$MODE devices=${DEVICES:-none} use_past=$USE_PAST" \
     "batch=$MICRO x $WORLD GPUs x $ACCUM = $((MICRO * WORLD * ACCUM))${OUTPUT:+ output=$OUTPUT}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd -- "$ROOT"
set -x
exec "$PYTHON_BIN" "$ROOT/scripts/train_p2n_vla.py" "${LAUNCH[@]}" -- \
  ${OVERRIDES[@]+"${OVERRIDES[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
