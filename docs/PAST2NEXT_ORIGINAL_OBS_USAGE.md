# Modern Past2Next with the original observation encoder

These are new entry points; the older launchers and configurations are unchanged.
The modern 16-layer, 768-wide AR model consumes the original trainable ResNet18 /
SpatialSoftmax / identity-state fused features, with a 112×112 crop by default.
The base context has 11 tokens; the state-gate variant has 15.

Run from `/workspace/ysk/past2next_bug_fixed`. The existing `/venv/oat` environment
contains the repository dependencies. Set `PYTHON_BIN` to another prepared Python
interpreter if needed. The shell wrapper forwards every flag unchanged.

Training defaults to W&B **online** for both variants and both tasks.
Dry-run and preflight do not start a W&B run.

## Real robot: recorded Nut Washer N77 data

```bash
PYTHON_BIN=/venv/oat/bin/python bash train_p2n_new_original_obs.sh \
  --variant p2n_state_gate_new \
  --task real_robot \
  --tokenizer /absolute/path/to/the/N77/ep-1540_mse-0.000.ckpt \
  --lazy-eval=false \
  --num-train-epochs 2001 \
  --batch-size 8 \
  --val-batch-size 4 \
  --test-num 100 \
  --gradient-accumulation 4 \
  --devices 0,1 \
  --num-processes 2 \
  --output output/training/p2n_state_gate_new_original_obs_nut_washer_run1
```

The N77 dataset is present. The exact tokenizer path named in the guide is
currently missing on this instance; replace the `--tokenizer` placeholder with
the location of that matching checkpoint before training. The configured default
retains the guide path and preflight reports a clear missing-file error. Evaluation uses held-out recorded data. `--test-num` caps the total
validation windows across ranks. `--lazy-eval=false` is accepted, but does not
control a physical robot; no robot rollout runner is configured. Offline
validation remains controlled by `training.offline_validation_enabled` and
`--val-every`. Existing `training.max_val_steps` / `training.max_reconst_steps`
limits can further reduce the number of evaluated windows.

## LIBERO-10: simulator evaluation

```bash
PYTHON_BIN=/venv/oat/bin/python bash train_p2n_new_original_obs.sh \
  --variant p2n_new \
  --task libero \
  --lazy-eval=false \
  --num-train-epochs 251 \
  --batch-size 8 \
  --val-batch-size 4 \
  --test-num 100 \
  --eval-every 25 \
  --parallel-envs 10 \
  --gradient-accumulation 4 \
  --devices 0,1 \
  --num-processes 2 \
  --output output/training/p2n_new_original_obs_libero_run1
```

`--lazy-eval=false` enables LIBERO simulation. `--test-num` is the **total**
rollout episodes across the ten tasks, not episodes per task. The rollout
interval uses the existing workspace epoch labels, including epoch 0 after its
training epoch. `--lazy-eval=true` skips simulator evaluation while retaining
offline validation. LIBERO assets, simulator dependencies, a matching Zarr
file, and a task-matched OAT tokenizer checkpoint must already be installed.

Both variants work with both tasks. Use `--variant p2n_new` for the base policy
or `--variant p2n_state_gate_new` for measured-state history and gates.
`--batch-size` is per GPU/process; effective batch size is batch × processes ×
gradient accumulation. No pretrained ResNet/DINO/ConvNeXt weights are required.
The OAT tokenizer is frozen and must match the dataset and training split.

## Inspect, override, and resume

Append `--dry-run` to either command for configuration-only inspection. Append
`--preflight` for CPU tokenizer-source, dataset, encoder construction, and
simulator dependency checks. Neither starts training. Fresh training refuses a
nonempty output directory; GPU launch checks the selected devices are idle.

All flags accept `--name value` and `--name=value`. Additional examples:

```bash
# Dataset/tokenizer pair and standard Hydra overrides.
PYTHON_BIN=/venv/oat/bin/python bash train_p2n_new_original_obs.sh \
  --task libero --variant p2n_new \
  --dataset /absolute/path/libero10_N500.zarr \
  --tokenizer /absolute/path/matching_tokenizer.ckpt \
  --num-demo 500 --lazy-eval=false --num-train-epochs 251 \
  --batch-size 8 --test-num 100 --dry-run -- \
  policy.activation_checkpointing=true \
  training.max_reconst_steps=null

# Resume from the new original-observation artifact only.
PYTHON_BIN=/venv/oat/bin/python bash train_p2n_new_original_obs.sh \
  --task real_robot --variant p2n_state_gate_new \
  --resume output/training/p2n_state_gate_new_original_obs_nut_washer_run1/checkpoints/latest.ckpt \
  --output output/training/p2n_state_gate_new_original_obs_nut_washer_run1 \
  --lazy-eval=false --num-train-epochs 2001 --batch-size 8 --test-num 100 \
  --devices 0,1 --num-processes 2
```

Hydra overrides after `--` take precedence over the corresponding flag. Resume
uses embedded model/tokenizer/normalizer weights and rejects incompatible
architectures, variants, encoders, observation layouts, or dataset splits.
An explicit crop override is recorded, for example
`policy.original_obs_config.crop_shape=[76,76]`; the provided recipe defaults to
112×112. Additional supported flags are listed by `--help`.

`--seed` changes training RNG. The supplied real-robot recipe keeps dataset
split seed 42 fixed; changing `task.policy.dataset.seed` explicitly also requires
a tokenizer trained using that same split. LIBERO follows `--seed` for its split.
