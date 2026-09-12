# Real-robot Past2Next training

Each experiment trains its own action tokenizer, then a fresh single-task policy
using **`train_past2next_scratch_all500.yaml`**. The original training YAML remains
unchanged; real-robot dataset, validation, logging, and checkpoint overrides are
supplied by `scripts/train_real_robot.py`. Fruits and nut/washer data are never
combined, and tokenizer weights are never shared between experiments.

| Task | Variant | Tokenizer augmentation | Training / validation episodes |
| --- | --- | --- | --- |
| fruits | current | `conjugate`, `augment_position=true` | 46 / 5 |
| fruits | left_noise | `left_noise`, `augment_position=false` | 46 / 5 |
| nut_washer | current | `conjugate`, `augment_position=true` | 56 / 6 |
| nut_washer | left_noise | `left_noise`, `augment_position=false` | 56 / 6 |

All four tokenizers use the requested **`max_angle_deg=30.0`, `p=0.6`**. The
task-specific 3.2° / 4.6° alternatives are not part of this experiment matrix.
Augmentation is applied to raw actions before normalization, during tokenizer
training only. Validation reconstruction uses unaugmented held-out actions.

The variants have different geometric effects. `conjugate` maps each rotation
increment to `Q dR Q^T`, preserving its rotation angle, and rotates the translation
vector when `augment_position=true`. `left_noise` maps increments to `Q dR`, which
can add a rotation much larger than the original motion at 30°. It leaves
translation unchanged. The gripper channel remains unchanged in both variants.
The mixed base/body-frame action representation means the first variant should
be evaluated as an augmentation experiment; it is not automatically a physically
consistent global coordinate transformation of an entire trajectory.

The datasets and deployment operate at **30 Hz**. Seven action channels mean:

| Channels | Meaning | Units / convention |
| --- | --- | --- |
| 0–2 | Base-frame translation delta | metres; `p_next = p_previous + dp` |
| 3–5 | End-effector-frame rotation-vector delta | radians; `R_next = R_previous @ Exp(drotvec)` |
| 6 | Absolute gripper command | 0 = open, 1 = closed; no differencing |

`robot0_gripper_qpos` is measured width in millimetres, approximately 0–80, and is
different from the gripper action. Observations use `robot0_eef_rot6d [6]`,
`robot0_eef_pos [3]`, gripper width `[1]`, `task_uid [1]`, and two 128×128 RGB views
(`agentview_rgb` and `robot0_eye_in_hand_rgb`). Each independent dataset has
`task_uid=0`.

The seed-42 10% episode split is identical across both stages and both variants
for each task. Fruits contributes 15,493 training / 1,650 validation frames;
nut/washer contributes 51,535 / 5,185 frames. `manifest.json` records exact
zero-based episode IDs and boundaries before training. Numeric normalizers fit
only training episodes. RGB uses the fixed `[0,255]` → `[-1,1]` mapping.

The policy retains the requested architecture and recipe: two observations,
seven past actions, a 16-action prediction horizon, eight-action execution stride,
112×112 crops, policy and vision learning rates of `1e-5`, and the original
generated-history warmup/ramp. At 30 Hz the horizon is 0.533 s and execution
stride is 0.267 s. Simulator evaluation is disabled. Offline validation and
generated-history validation remain enabled.

The tokenizer defaults to 5,001 epochs, global batch 256; the policy defaults to
251 epochs, global batch 64. `--gpus` divides these global batches across 1, 2, 4,
or 8 GPUs. Both stages keep incomplete validation batches. Tokenizer selection
uses the best retained checkpoint by full-precision held-out reconstruction MSE.
The launcher copies that checkpoint to `frozen_tokenizer.ckpt` and supplies it
to Stage 2. Policy checkpoints are saved every 20 epochs, named
`ep-XXXX_mse-0.XXXXXX.ckpt`, and all scheduled checkpoints are retained.
Top-k retention is disabled, as are the previous 25-epoch snapshots.
`latest.ckpt` is updated at the same 20-epoch interval for resume.
Epoch labels follow the existing zero-based convention: 0, 20, 40, 60, etc.
MSE is action prediction error
over the configured held-out reconstruction batches (10 batches by default).
The held-out evaluation stays enabled to compute this metric.
These are offline metrics, not measured robot task-success rates.

Use the dedicated Python environment:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate real_robot
cd /workspace/ysk/past2next_bug_fixed
```

Inspect one complete experiment without creating an output directory or training:

```bash
python scripts/train_real_robot.py \
  --task fruits --variant current --gpus 0,1 \
  --output-dir /workspace/ysk/past2next_bug_fixed/output/real_robot/fruits_current \
  --dry-run
```

For a short diagnostic, use a fresh output directory and replace `--dry-run` with
`--smoke`. Each stage runs one epoch with two training batches, one validation
batch, and one reconstruction batch. Generated history is forced on during the
policy diagnostic. Full runs omit both flags. Use `--tokenizer-epochs` and
`--policy-epochs` only when intentionally changing the full experiment budget.

Run sustained training through the instance's Supervisor service; its command
should call `/venv/real_robot/bin/python` with the launcher arguments. Assign
disjoint GPU sets and output directories to concurrent experiments. The launcher
refuses an existing output directory and does not silently resume a previous
run. Stop/restart recovery requires explicitly handling the saved stage
checkpoints rather than reusing this fresh-run launcher blindly.

Every run writes:

- `manifest.json`: exact commands, input schema and split, action semantics,
  augmentation settings, key package versions, and relevant source-file hashes.
- `tokenizer.log`, `policy.log`, and stage subdirectories with Hydra configs,
  training metrics, checkpoints, and offline W&B logs.
- `frozen_tokenizer.ckpt`: the exact selected Stage 1 handoff checkpoint.
- `status.json`: current stage, selected checkpoints, metrics, and any failure.
- `checkpoint_check.json` / `checkpoint_check.log`: the final reload check.

The final check reloads the selected policy, verifies that tokenizer tensors
match the frozen Stage 1 checkpoint exactly and remain frozen, and exercises
explicit-history and stateful inference on a held-out observation window.
Outputs must be finite with executed action shape `[1,8,7]` and full prediction
shape `[1,16,7]`. A run is marked completed only after this check passes.

Environment exports are saved in `environments/real_robot/`. This instance's
`/workspace` is not a mounted persistent volume; copy checkpoints and manifests
off the instance before recycling or destroying it.


## Live W&B logging

The active four-run experiment uploads to
https://wandb.ai/andyliu7081-northeastern-university/real_robot through the
Supervisor service `real-robot-wandb-sync`. Training continues to write durable
local offline files; a separate `wandb beta sync --live` process uploads their
history and follows new records. The training processes require no restart.

`scripts/sync_real_robot_wandb.py` loads the experiment's `jobs.json` once at
startup. Every 15 seconds it scans those jobs' folders for tokenizer and policy
logs; it does not reload the registry on each scan. It ignores distributed ranks
that contain metadata but no training history. Each run retains its own W&B ID.
Failed uploads are retried; completion requires the W&B synced marker.
Service status, individual uploader logs and run links are saved under
`wandb_live_sync/` in the experiment output root. Uploading this experiment's
metrics, configuration, logs and system metadata was explicitly approved.

## Retrain only the fruits policies on one GPU each

`scripts/train_real_robot_policy.py` starts a fresh policy using an existing,
matching trained tokenizer. Fresh runs do not retrain the tokenizer. An explicit
`--resume` continues an existing policy-only run from its latest checkpoint.
Each invocation uses exactly one GPU, batch size 64, and 251 epochs by default.
Fresh runs use `train_past2next_scratch_all500.yaml` with `training.resume=false` and
`training.init_checkpoint=null`. The task's original seed-42 episode split is
preserved.

Keep the original experiment directories intact. Choose a new output root for
these policies and pair `current` with its own tokenizer and `left_noise` with
its own tokenizer. These examples perform read-only preflight checks:

```bash
REAL_ROBOT_SOURCE=/workspace/ysk/past2next_bug_fixed/output/training/real_robot_20260911T052052Z
REAL_ROBOT_NEW=/workspace/ysk/past2next_bug_fixed/output/training/fruits_policy_single_gpu_NEW

/venv/real_robot/bin/python scripts/train_real_robot_policy.py \
  --task fruits --variant current --gpu 0 \
  --tokenizer "$REAL_ROBOT_SOURCE/fruits/current/frozen_tokenizer.ckpt" \
  --output-dir "$REAL_ROBOT_NEW/current" \
  --entity andyliu7081-northeastern-university --project real_robot --dry-run

/venv/real_robot/bin/python scripts/train_real_robot_policy.py \
  --task fruits --variant left_noise --gpu 1 \
  --tokenizer "$REAL_ROBOT_SOURCE/fruits/left_noise/frozen_tokenizer.ckpt" \
  --output-dir "$REAL_ROBOT_NEW/left_noise" \
  --entity andyliu7081-northeastern-university --project real_robot --dry-run
```

For a short check, replace `--dry-run` with `--smoke` and use a separate fresh
smoke output directory. This trains two policy batches, validates and reconstructs
one batch, forces generated history, and verifies the resulting checkpoint.
For sustained training, run the same command through Supervisor without either
flag. A GPU list such as `--gpu 0,1` is rejected. Both this launcher and the
original two-stage launcher reject existing output directories by default.
For the policy-only launcher, repeat the original command with `--resume` to
continue the existing directory: this restores model, optimizer, scheduler, EMA,
and training counters, and reuses the original W&B run ID. The original
two-stage launcher still requires separate stage-aware recovery.

Full policy-only runs default to **native online W&B logging**, with a fresh run
ID and name and groups `fruits_current_single_gpu` /
`fruits_left_noise_single_gpu`. Smoke runs use offline logging; `--dry-run` starts
no W&B run. This differs from the original two-stage launcher's offline logging
and separate live uploader. The examples above show preflight commands; the
started replacement runs are identified below.

Before policy training, the launcher validates the source tokenizer's task,
augmentation variant, seven-channel action schema, horizon, and episode-split
settings, then copies it into the new output directory. Its manifest records the
original source path, matching source/copy SHA-256 hashes, exact command, split,
and W&B identity. Original tokenizer checkpoints and previous policy outputs
remain intact. Completion also requires the existing checkpoint checker to
confirm finite inference and unchanged, frozen tokenizer weights.

The replacement fruits policies were started under
`output/training/fruits_policy_single_gpu_20260911T090950Z`, using Supervisor
services `real-robot-fruits-current-1gpu` (GPU 0) and
`real-robot-fruits-left-noise-1gpu` (GPU 1), with native online W&B logging.
The two nut/washer jobs and their existing sync worker remain on the original
experiment. `output/real_robot_current_jobs.json` is the combined registry for
the two replacement fruits policies and the two original nut/washer jobs; the
original experiment's historical `jobs.json` remains intact.

The active fruits policy runs switched from validation-loss checkpoint selection
to action MSE on 2026-09-11. Their retained `_val-...` checkpoints were renamed
using the actual per-epoch MSE, and both services now invoke the policy-only
launcher with `--resume`. The original manifests and W&B IDs are preserved;
resume manifests record the effective MSE command and continuation checkpoint.
The change record is `output/training/fruits_policy_single_gpu_20260911T090950Z/launch/mse_checkpoint_change.json`.
The nut/washer tokenizer launchers retain their original in-memory policy
commands. The managed handoff described below intercepts those old commands
and starts the prepared single-GPU policies after each tokenizer completes.

The active fruits policy schedule was subsequently corrected to save every 20
epochs and retain all scheduled checkpoints. `checkpoint.save_all=true`,
`checkpoint.topk.k=0`, `training.checkpoint_every=20`, and
`training.snapshot_every=0` are applied to both fresh and resumed policy-only
launches. Existing checkpoints remain available. The final checker chooses one
retained checkpoint for inference verification without deleting the others.

## Nut/washer policies after the existing tokenizers finish

The prepared runs are under
`output/training/nut_washer_policy_single_gpu_20260911T102907Z`. Current uses
GPU 4 and left-noise uses GPU 6. Both use the same policy architecture, batch 64,
251 epochs, online W&B project, and every-20-epoch MSE checkpoint retention as
the fruits policies. Each keeps its matching fully trained frozen tokenizer.

`real-robot-nut-washer-policy-handoff` manages the transition. The two original
tokenizer jobs continue on their existing GPU pairs. When a tokenizer completes,
its queued old policy waits before model initialization. The handoff worker
checks both waiting ranks and the completed tokenizer, stops that old pipeline,
and starts `real-robot-nut-washer-current-1gpu` or
`real-robot-nut-washer-left-noise-1gpu`. The destination services start only
through this handoff; their output directories stay absent until then.

The handoff plan and live status are `handoff_plan.json` and
`handoff_status.json` in that run root. Per-source `policy_handoff.json` requests
record the exact redirection. The original training manifests and checkpoints
are preserved. The combined current-job registry is updated as each handoff
finishes. Unit tests, a two-rank gate check, and separate one-GPU policy smoke
runs verify the transition and the matching nut/washer datasets.
