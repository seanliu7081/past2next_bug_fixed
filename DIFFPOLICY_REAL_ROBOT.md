# Real-robot diffusion policy

This standalone recipe adapts `past2next_clean/oat/config/train_diffpolicy.yaml`
and its transformer policy. It adds files only; the existing Past2Next recipes,
launchers, datasets, and training workspace are unchanged.

From this repository, train the LP3 pen/cabinet dataset with:

```bash
TRAIN_GPUS=7 bash train_diffpolicy_real_robot.sh
```

Select another compatible dataset or multiple GPUs:

```bash
TRAIN_GPUS=6,7 DATASET_PATH=/workspace/ysk/zarr/fruitV3_40.zarr \
  bash train_diffpolicy_real_robot.sh
```

Validate the dataset schema and resolve the config without training:

```bash
bash train_diffpolicy_real_robot.sh --dry-run
```

`POLICY_EPOCHS` (1001), `BATCH_SIZE` (64 per GPU), `VAL_BATCH_SIZE` (32 per GPU),
`WANDB_MODE` (online), `TRAIN_PY` (`/venv/real_robot/bin/python`), and `RUN_DIR`
can also be set as environment variables. Every normal launch requires a fresh
output directory. The launcher runs in the foreground; keep its terminal or
tmux session open.

Training has one stage: diffusion predicts continuous actions directly. No
tokenizer checkpoint or previous-action window is used. The dataset must have
the existing real-robot schema: two 128x128 RGB cameras, position, rotation-6D,
gripper width, task UID 0, and seven action channels. Action units and frame
conventions are unchanged.

The recipe preserves the source's 256-dimensional, four-layer, four-head
transformer, epsilon-prediction loss, 100 diffusion training steps, 10 DDIM
inference steps, and learning rates (policy 5e-5, observation encoder 1e-5).
It observes two frames, predicts 16 actions, and returns the first eight for
execution.

Real-robot adaptations use 112x112 crops, fixed evaluation crops, a 90/10 episode
split with seed 42, and the existing training-only real-robot normalizer. There
are no simulator rollouts. Relative to the source recipe, the defaults are
1001 epochs and batch sizes 64/32. Validation and sampled-action MSE run every
10 epochs; sampled-action MSE uses at most 10 validation batches per process.
EMA is enabled. Every 50 epochs, the workspace saves the latest checkpoint and
retains the best three checkpoints by `test_reconst_mse` (lower is better).
W&B logs to project `real_robot` under a dataset-specific diffusion group.

Outputs live under `output/<dataset>_diffpolicy_<timestamp>/`, including
`console.log`, `logs.json`, the resolved Hydra config, and `checkpoints/`.
Load a saved policy using the existing `BasePolicy.from_checkpoint(path)` API;
by default it selects the saved EMA weights and normalization statistics.
