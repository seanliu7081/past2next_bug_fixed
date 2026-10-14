# P2N-VLA: train vision and action, freeze language

This additive training mode updates SigLIP, its image projector, the P2N action
expert and action-conditioning modules, and the auxiliary KI token table. All
Gemma language parameters, including LoRA adapters, are frozen. The OAT action
tokenizer stays frozen.

The existing knowledge-insulation boundary is preserved: the action expert
reads detached language keys and values. Vision learns through the KI loss,
so `policy.lambda_ki` must be positive. This is not direct action-loss
backpropagation through the language backbone.

The new launcher is `train_p2n_vla_vision_action.sh`. Existing launchers, policies,
and configurations retain their behavior. The mode supports LIBERO-10 and the
base or state-gated P2N-VLA variant.

```bash
cd /workspace/ysk/past2next_bug_fixed
WANDB_MODE=online P2N_PYTHON=/venv/oat/bin/python \
bash train_p2n_vla_vision_action.sh \
  --finetune vision_action \
  --variant p2n_vla \
  --task libero \
  --devices 0,1 \
  --num-processes 2 \
  --num-train-epochs 300 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --use-ema=false \
  --lazy-eval=false \
  --eval-every 100 \
  --test-num 500 \
  --eval-parallel-envs 5 \
  --wandb-mode online \
  --output output/training/p2n_vla_vision_action_libero10_s42
```

Batch size is per GPU. The command has effective batch size `1 * 2 * 16 = 32`.
EMA is disabled and evaluation runs five simulators concurrently to leave
memory for training and rendering on 24 GB GPUs. Checkpoints and evaluation
use the current model weights. The total remains 500 episodes per evaluation.
For later standalone evaluation, pass `--weights model` to
`scripts/evaluate_p2n_vla.py`; this command produces `_model.ckpt` snapshots.
The default schedule uses 100 optimizer updates per epoch, so 300 epochs gives
30,000 optimizer updates. These are bounded training epochs, not complete
passes over the dataset. `--eval-every 100` schedules simulator evaluation after
100, 200, and 300 completed epochs; the final epoch is also evaluated if a
different epoch count is selected. Held-out loss validation has its own
`--val-every` setting (default 100). The original workspace uses zero-based
indices for that check, so held-out validation runs after epochs 1, 101, 201,
and the final epoch; simulator evaluation uses the completed epochs above.

Add `--dry-run` to resolve flags without reading models or launching training.
Add `--probe --probe-steps 2` to the command for a bounded training, generated-history,
validation, and short simulator memory check. W&B remains online for probes.
Use a separate output directory for each probe and fresh training run.

Vision activation checkpointing is enabled by default. Vision and the action
expert use the existing pretrained optimizer group's learning rate; the other
trainable modules use the new-module learning rate. Hydra overrides after `--`
remain available, for example `optimizer.policy_lr=1e-5`.

New artifacts include vision weights and record this training scope. Resume
from checkpoints produced by this mode; optimizer state from the previous
LoRA training mode has a different parameter set and cannot be resumed into
this mode. Fresh runs initialize from the configured PI0.5 base assets.

The default trained OAT asset is
`data/pretrained/p2n_vla/oat_from_upd030000.ckpt`, recovered exactly from the
frozen tokenizer embedded in the existing 30,000-update VLA EMA snapshot.
The original standalone OAT path no longer exists. The recovered file retains
the tokenizer's training-task metadata and records its source snapshot and
original tokenizer provenance. To reproduce that extraction into a new file:

```bash
/venv/oat/bin/python scripts/recover_p2n_vla_tokenizer.py \
  --checkpoint output/training/p2n_vla_libero10_s42/snapshots/upd-030000_ema.ckpt \
  --output /absolute/path/to/new_oat_checkpoint.ckpt
```

Validation on 2026-10-09: 69 focused/regression tests passed. A two-GPU probe
of the command above completed two optimizer updates, generated-history
backward, held-out validation, and ten official LIBERO episodes with W&B
online. Peak PyTorch reservation was 18.72 GiB; the simulator run peaked at
21.43 GB total GPU use, leaving 4.32 GB free. The probe passed its memory
checks. Report: `output/probe/p2n_vla_vision_action_2gpu_20261009_r3/probe.json`.
