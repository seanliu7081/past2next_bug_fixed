# Frozen ConvNeXt Nano P2N on LIBERO-10

This version adds separate configs and launchers for base P2N and state-gated
P2N. The existing real-robot launchers and training code are unchanged.

From the repository root:

```bash
bash scripts/train_p2n_new_convnext_libero10.sh \
  --variant both \
  --batch-size 64 \
  --gpu 4,5,6,7 \
  --gradient 1 \
  --convnext-frozen true \
  --num-epochs 251 \
  --lazy-eval false \
  --eval-every 100 \
  --eval-episodes 100 \
  --eval-envs 10 \
  --wandb-mode online
```

Select idle GPUs appropriate to your run. `--variant both` trains base P2N,
then the gated variant sequentially, with separate output directories and W&B
runs. Select `--variant p2n_new` or `--variant p2n_state_gate_new` for one model.
The W&B project is `p2n_new_libero10`; online mode uses your existing login.

## Flags and defaults

| Flag | Default | Meaning |
| --- | --- | --- |
| `--variant` | `both` | Both variants sequentially, or one named variant |
| `--batch-size` | `8` | Samples per GPU |
| `--gpu` | `0,1` | Physical GPU indices or UUIDs from `nvidia-smi` |
| `--gradient` | `4` | Gradient accumulation steps |
| `--num-epochs` | `251` | Training epochs per variant |
| `--convnext-frozen` | `true` | Backbone stays frozen; `false` is unsupported |
| `--lazy-eval` | `false` | `false` enables simulator evaluation during training |
| `--eval-every` | `100` | Simulator evaluation interval in epochs |
| `--eval-episodes` | `100` | Total evaluation episodes across the ten tasks |
| `--eval-envs` | `10` | Maximum parallel simulator environments |
| `--wandb-mode` | `offline` | `online`, `offline`, or `disabled` |
| `--tokenizer` | Pinned local LIBERO tokenizer | Override action tokenizer checkpoint |
| `--dataset` | `data/libero/libero10_N500.zarr` | Override dataset; must match tokenizer provenance |
| `--python` | `/workspace/venvs/starvla-heading/bin/python` | Interpreter with model and simulator dependencies |
| `--dry-run` | Off | Resolve configs without opening data/models or contacting W&B |
| `--preflight` | Off | CPU model/schema checks plus simulator import/asset checks |

Effective batch size is batch per GPU × GPU count × accumulation. The example
uses 256. GPU memory at that batch size is not established by CPU preflight.

The existing workspace uses **zero-based epoch labels**: evaluation occurs after
epochs labeled **0, 100, 200, ...**, and there is no extra final rollout. The default
251 epochs therefore evaluate at labels 0, 100, and 200. Offline validation loss
continues every epoch; `--eval-every` controls simulator rollouts.

Evaluation inherits the existing corrected protocol: 100 episodes total gives
10 trials for each task. Both policies use their matching execution-history
runner; the gate additionally receives measured state history. No evaluation
videos are recorded by default.

## Dataset and weights

The default dataset has 500 demonstrations from all ten tasks. Seed 42 and
validation ratio 0.1 give 450 training episodes and 50 validation episodes.
`data/libero` is a new symlink to `/workspace/shared_data/libero`, preserving the
dataset path recorded inside the tokenizer checkpoint.

The frozen OAT tokenizer is the best saved checkpoint by `test_reconst_mse` from
the local LIBERO-10 SO3-augmented tokenizer run, epoch 4330, metric
`0.0010664978763088584`. It has action dimension 7, prediction horizon 16, and
eight latent action tokens. Both variants load its EMA weights and action
normalizer. Its SHA256 is
`547243abb8e93752589ff674121bf129eb7327d773c72818c27ed3b9aebb3325`.
The checkpoint and `provenance.json` are under
`/workspace/models/libero10-oat-so3aug/547243abb8e93752589ff674121bf129eb7327d773c72818c27ed3b9aebb3325/`.

The pretrained visual backbone uses the same downloaded ConvNeXt V2 Nano
revision `aacb23b94d2adf0b206df6c8a75798b672183d6d` as the real-robot version.
The backbone and tokenizer are frozen. Visual adapters, action transformer,
state projection, and the gate/history modules in the gated variant train.

## Verification and resume

Verified locally: 35 focused tests passed; both variants passed CPU preflight
with the real dataset/tokenizer/Nano weights; each variant completed one
10-step simulator episode on an idle GPU. Those short episodes check rendering,
policy inference, executed-action history and measured-state history. They do
not measure trained success rates or full-batch training memory. No full
training run was started.

CUDA GPU UUIDs and EGL renderer indices are mapped independently. The separate
runner adapter scopes numeric EGL settings to simulator construction and its
child processes, preserving the training process's GPU selection.

Append `--dry-run` or `--preflight` to the example command for validation without
starting training. Preflight does not execute a simulator rollout or measure
GPU memory. Training checks selected GPUs for existing compute jobs and fails
if they are occupied.

The dedicated Python launcher also supports an explicit compatible resume:

```bash
/workspace/venvs/starvla-heading/bin/python \
  scripts/train_p2n_new_convnext_libero10.py \
  --variant p2n_new --resume /absolute/path/latest.ckpt \
  --devices 4,5 --num-processes 2 --output /absolute/path/existing_run
```

Retain the matching dataset, epoch count, batch size, accumulation, and model
settings using Hydra overrides after `--`. Resume validates the embedded policy,
encoder, tokenizer, and training contract.
