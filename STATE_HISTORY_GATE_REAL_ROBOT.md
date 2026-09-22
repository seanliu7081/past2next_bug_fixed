# Pen/cabinet real-robot gate training

Use the separate `train_state_history_gate_real_robot.sh` entry. The existing
`train_state_history_gate.sh` remains a LIBERO recipe and is unchanged.

```bash
cd /workspace/ysk/past2next_bug_fixed
CUDA_VISIBLE_DEVICES=0,1 \
DATASET_PATH=/workspace/ysk/zarr/pen_cabinet_lp3_N67.zarr \
bash train_state_history_gate_real_robot.sh \
  /workspace/ysk/past2next_bug_fixed/output/training/pen_cabinet_lp3_N67_current_20260919_213053/frozen_tokenizer.ckpt
```

These tokenizer/data paths are also the new script's defaults. Add `--dry-run`
before the positional tokenizer argument to resolve configuration without
starting training. No training or robot execution is started by the checks.

Defaults: `/venv/real_robot/bin/python`, two GPUs, batch 32 per GPU, H8 with
seven past commands, 16 predicted steps/eight executed steps, learned gate
initialized at 0.9, 2001 epochs, online W&B project `real_robot`, validation each
epoch, checkpoints every 100 epochs. All scheduled checkpoints are retained as
`ep-XXXX_val-....ckpt`; there are no snapshots or `latest.ckpt` files. Override
`CHECKPOINT_EVERY`, `VAL_EVERY`, `NUM_EPOCHS`, `GATE_MODE`, or `GATE_INIT` as needed.
Checkpoint intervals must be divisible by validation intervals.

The data contains 67 episodes, split into 60 training and seven validation
episodes with seed 42. The original data and tokenizer files are read only.
No LIBERO runner is constructed, and no task-success-rate metric is assumed.
Checkpoint naming uses held-out token loss (`val_loss`); reconstruction error
and generated-history validation remain available through the existing trainer.
Offline metrics do not establish real-robot task success.

The real-robot observation schema is preserved: two RGB cameras, end-effector
position, raw rotation-6D, measured gripper width [1], and task ID. RGB remains
uint8 until fixed [0,255] normalization. Low-dimensional statistics are fit
using training episodes only. The frozen tokenizer keeps its saved statistics.

The new history encoder interprets `robot0_eef_rot6d` as two contiguous matrix
rows by default (`ROTATION_6D_LAYOUT=rows`). It reconstructs rotation matrices,
computes world-frame relative rotations `R[t] @ R[t-1].T`, and expresses absolute
and relative rotations in the original history encoder's column-based 6D
representation. It does not subtract normalized orientation components as if
they were geometric rotations. Current observation features retain the original
dataset representation. Position and gripper differences retain their original
normalization and are not divided by the time step.

Rotation-layout provenance: nearby robot-common conversion utilities use rows,
and the supplied dataset's adjacent pose changes are more consistent with its
body-frame rotation commands under the rows interpretation (median directional
cosine approximately 0.93 versus 0.25 for columns on moving transitions). This
supports the default but is not a recovered data-generation specification: the
Zarr only records crop/image-size metadata and its exact converter was not found.
If the data producer specifies two columns instead, set
`ROTATION_6D_LAYOUT=columns`; this choice is saved in the policy configuration.

The gate, executed-command acknowledgment contract, self-past schedule, and
discrete action tokenizer are inherited unchanged. Deployment requires an
external controller that supplies each actual control-step state history and
acknowledges executed commands. This training entry does not implement or start
robot deployment. The LIBERO runner is not the real-robot deployment adapter.

Validation includes synthetic episode-boundary/normalization checks, geometric
row/column rotation checks, and the complete configured model with the supplied
tokenizer on a small temporary subset of actual frames: expert/generated-history
forward and backward, gate gradients, and predicted action shapes. No optimizer
training run or W&B run was started during this verification.
