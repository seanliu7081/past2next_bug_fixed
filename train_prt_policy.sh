#!/usr/bin/env bash
# Launch Stage 2 with a user-selected Stage 1 PRT checkpoint.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash train_prt_policy.sh [--ckpt FILE] [--gpu_ids 0,1] [-- HYDRA_ARGS...]
Without --ckpt, choose a tokenizer checkpoint from the menu in an interactive terminal.
Relative checkpoint paths are resolved from the current directory. GPU IDs default to 0.
HELP
}

repo=$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")
ckpt=
gpu_ids=0
while (( $# )); do
    case $1 in
        --ckpt|--gpu_ids)
            if (( $# < 2 )) || [[ -z $2 || $2 == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            if [[ $1 == --ckpt ]]; then ckpt=$2; else gpu_ids=$2; fi
            shift 2 ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if [[ ! $gpu_ids =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'Expected comma-separated GPU IDs, e.g. --gpu_ids 0,1\n' >&2
    exit 2
fi
if [[ -z $ckpt ]]; then
    if [[ ! -t 0 ]]; then usage >&2; exit 2; fi
    shopt -s nullglob
    candidates=("$repo"/output/*/*_PRT_Tok_*/checkpoints/*.ckpt)
    if (( ${#candidates[@]} == 0 )); then
        printf 'No PRT checkpoints found under output/. Supply --ckpt FILE.\n' >&2
        exit 2
    fi
    PS3='Select a tokenizer checkpoint by number: '
    select ckpt in "${candidates[@]}"; do
        if [[ -n $ckpt ]]; then break; fi
    done
fi
if [[ ! -f $ckpt || ! -r $ckpt ]]; then
    printf 'Cannot read tokenizer checkpoint: %s\n' "$ckpt" >&2
    exit 2
fi

# run_workspace resolves this interpolation before constructing the workspace.
export PRT_POLICY_TOKENIZER_CKPT
PRT_POLICY_TOKENIZER_CKPT=$(realpath -- "$ckpt")
export MUJOCO_GL="${MUJOCO_GL:-egl}"
IFS=, read -r -a gpus <<< "$gpu_ids"
launch=(--gpu_ids "$gpu_ids" --num_processes "${#gpus[@]}" --num_machines 1)
if (( ${#gpus[@]} > 1 )); then launch+=(--multi_gpu); fi
printf 'PRT tokenizer: %s\nGPU IDs: %s\n' "$PRT_POLICY_TOKENIZER_CKPT" "$gpu_ids"
cd -- "$repo"
exec "${TRAIN_PY:-/venv/oat/bin/python}" -m accelerate.commands.launch "${launch[@]}" \
    scripts/run_workspace.py --config-name=PRT_Policy \
    'policy.action_tokenizer.checkpoint=${oc.env:PRT_POLICY_TOKENIZER_CKPT}' "$@"
