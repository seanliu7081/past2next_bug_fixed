# Direct action flow with ResNet-18

The separate ResNet-18 version supports both `p2n_action_flow` and
`p2n_state_gate_action_flow`. Existing DINO configurations and launchers are
unchanged. The new policies retain continuous normalized `16 x 7` actions,
the 16-layer / 768-dimensional DiTX, FM + consistency training, EMA,
self-generated action history, dataset splits and action semantics.

## Training

```bash
cd /workspace/ysk/past2next_bug_fixed

bash train_p2n_action_flow_resnet18.sh \
  --task nut_washer \
  --gpus 2,3 \
  --batch-size 4 \
  --val-batch-size 4 \
  --grad-accum 8 \
  --save-every 20
```

Both variants run sequentially and receive separate output directories. The
effective batch is `4 x 2 x 8 = 64`. Defaults use
`/workspace/ysk/zarr/nut_washer_v3_N77.zarr`, 77 demonstrations, 2,001 epochs,
a 5% held-out episode split, and online W&B logging. As with the existing
launcher, selected GPUs are checked immediately before training. Online W&B is
also selected when resuming older checkpoints; no extra logging flag is needed.

Append `--variant p2n_action_flow` or
`--variant p2n_state_gate_action_flow` to train only one variant. Append
`--dry-run` for CPU dataset/schema, normalization and schedule checks; this
creates no model, GPU work, training run or output directory. LIBERO uses
`--task libero` and its task-specific dataset path. Python defaults to
`/venv/real_robot/bin/python`; use `--python PATH` to select another environment.

Periodic checkpoint saving defaults to every 20 completed epochs. Use
`--save-every N` to set both the latest-checkpoint and numbered-snapshot
intervals, including when resuming. For `--save-every 20`, numbered files
are `checkpoints/ep-0020.ckpt`, `ep-0040.ckpt`, and so on. `latest.ckpt`
is also saved at the end of training; metric-ranked best checkpoints can
still be saved between these intervals. Existing resumes retain their saved
intervals unless `--save-every` or explicit Hydra overrides are supplied.

Explicit Hydra overrides come after `--` and take precedence over convenience
flags:

```bash
bash train_p2n_action_flow_resnet18.sh \
  --task nut_washer \
  --gpus 2,3 --batch-size 4 --val-batch-size 4 --grad-accum 8 \
  --variant p2n_action_flow \
  -- training.num_epochs=501 logging.mode=online
```

## Choosing the task

The earlier command without `--task` trains `nut_washer_v3_N77`. Select the
named dataset with `--task`; both action-history variants use the same choice.

| Flag | Default dataset | Demonstrations | Validation ratio |
| --- | --- | ---: | ---: |
| `--task nut_washer` | `/workspace/ysk/zarr/nut_washer_v3_N77.zarr` | 77 | 0.05 |
| `--task pen_cabinet` | `/workspace/ysk/zarr/pen_cabinet_N67.zarr` | 67 | 0.10 |
| `--task fruits` | `/workspace/ysk/zarr/fruits_N51.zarr` | 51 | 0.10 |
| `--task fruits_v2` | `/workspace/ysk/zarr/fruits_v2_N49.zarr` | 49 | 0.10 |
| `--task libero` | Existing LIBERO-10 task configuration | Configured dataset | 0.10 |

`real_robot` remains a compatibility alias for `nut_washer`. Named real-robot
tasks retain the 2001-epoch action-flow recipe; ordinary Hydra overrides remain
available. Task selection does not load an OAT checkpoint or change the encoder.
Resume requires the same named task as its saved artifact, even though all four
robot datasets share `task_type: real_robot`.

`fruits` and `fruits_v2` use their canonical task configuration paths. These files
must be present, or provide their location after `--`, for example
`task.policy.dataset.zarr_path=/path/to/fruits_N51.zarr`. The launcher does not
silently substitute the separate `fruitV3_40` dataset.

For pen/cabinet training:

```bash
bash train_p2n_action_flow_resnet18.sh \
  --task pen_cabinet \
  --gpus 2,3 \
  --batch-size 4 \
  --val-batch-size 4 \
  --grad-accum 8 \
  --save-every 20
```

## Observation encoder

Each camera uses the repository's original trainable `ResNet18Conv` backbone
with GroupNorm, SpatialSoftmax, and a 64-dimensional visual feature. Its weights
are initialized randomly (`pretrained=False`); no ImageNet weights, DINO
snapshot, tokenizer checkpoint or download is required. A learned projection
maps each camera/frame feature to the 768-dimensional flow context.

The default two cameras and two observed frames produce four visual tokens.
The plain context contains four visual tokens, two current-state/task tokens,
seven past-command tokens and two command-difference tokens: 15 total. The
state-gated variant adds four history summaries for 19 tokens.

The default encoder configuration is:

```yaml
obs_encoder_type: resnet18
resnet_config:
  crop_shape: [112, 112]
  use_group_norm: true
  share_rgb_model: false
  eval_fixed_crop: true
```

The original dataset pixel normalizer maps byte-range RGB to `[-1, 1]`.
Training uses random crops from the 128 x 128 images and evaluation uses fixed
center crops. A crop is realized once for each condition and shared by student
and EMA feature extraction. Both networks compute their own visual features;
the student backbone receives gradients. Action/current-state normalization is
fitted only on training replay frames and remains frozen, as in the existing
direct action-flow implementation.

The formal configurations are:

- `oat/config/train_p2n_action_flow_resnet18.yaml`
- `oat/config/train_p2n_state_gate_action_flow_resnet18.yaml`
- `oat/config/experimental/train_p2n_action_flow_resnet18_real_robot.yaml`
- `oat/config/experimental/train_p2n_state_gate_action_flow_resnet18_real_robot.yaml`

## Resuming

```bash
bash train_p2n_action_flow_resnet18.sh \
  --task nut_washer \
  --gpus 2,3 \
  --variant p2n_action_flow \
  --resume /absolute/path/to/checkpoints/latest.ckpt
```

The saved configuration supplies batch and accumulation settings. Resume retains
the model, EMA, optimizer, scheduler, normalizers and per-rank RNG states, so
exact continuation requires the original GPU process count and training
contracts. ResNet checkpoints record `obs_encoder_type: resnet18`; cross-encoder
DINO resumes and DINO/Resampler options are rejected before training. The
separate workspace reuses the existing action-flow training loop and adds
ResNet artifact checks.

## Validation

The launcher tests exercise all four Hydra configurations, incompatible encoder
options, real synthetic Zarr schema and training-frame normalization, resume
contracts, CPU-only preflight, and both sequential shell variants with flag
precedence. Both default real-robot variants also pass CPU preflight against
the installed N77 dataset. Full GPU training and closed-loop robot quality
are not measured by these checks.


The full real-robot architectures were instantiated on CPU without downloads:

| Variant | Trainable parameters | Frozen normalization parameters | Total |
| --- | ---: | ---: | ---: |
| `p2n_action_flow` | 261,403,719 | 144 | 261,403,863 |
| `p2n_state_gate_action_flow` | 263,355,722 | 144 | 263,355,866 |

These counts include vector-sized action/state normalizers and the fixed RGB
normalizers. Vision adapters/backbones contain 22,456,384 trainable parameters.
The counts do not establish GPU peak memory.

The encoder, policy and launcher CPU tests cover actual ResNet gradients, all trainable gradients
through the fourth update after zero initialization, shared crops with independent
EMA backbones, normalization, deterministic evaluation, strict offline artifacts,
and launch/preflight contracts. Both variants also pass two-rank CPU DDP smoke
with accumulation, a partial last group, no unused parameters, complete EMA,
exact optimizer/EMA/RNG continuation and 8-/2-step predictions. That distributed
smoke uses small camera/flow modules; full model GPU behavior remains unmeasured.

```bash
/venv/real_robot/bin/python -m pytest -q \
  tests/test_p2n_action_flow_resnet_encoder.py \
  tests/test_p2n_action_flow_resnet_policy.py \
  tests/test_p2n_action_flow_resnet18_launch.py

/venv/real_robot/bin/python -m torch.distributed.run --standalone \
  --nproc_per_node=2 tests/p2n_action_flow_resnet_ddp_smoke.py \
  --variant p2n_action_flow

/venv/real_robot/bin/python -m torch.distributed.run --standalone \
  --nproc_per_node=2 tests/p2n_action_flow_resnet_ddp_smoke.py \
  --variant p2n_state_gate_action_flow
```

Named-task routing validation: 44 launcher/task tests passed. Explicit
`--task nut_washer` and `--task pen_cabinet` CPU preflight passed for both
variants: 73/4 and 60/7 train/validation episodes respectively. No training
was started. Canonical fruits datasets were not locally available for preflight.
