# Past2Next latent-flow implementation and launch commands

The flow implementation was added in new files. The existing AR implementations, configurations, launchers, tokenizer checkpoints and original implementation-plan Markdown are retained unchanged. Latent-flow launch commands are now consolidated into the single Bash entrypoint `train_p2n_latent_flow.sh`.

The two independent policies are `p2n_latent_flow` and `p2n_state_gate_latent_flow`, under artifact family `oat_latent_flow`. Both use the selected frozen OAT EMA tokenizer, 16-layer 768-wide DiT-X, and the specified FM+EMA-consistency objective. Fresh runs default to frozen DINOv3-S/16; `--obs-encoder resnet18` selects the original trainable ResNet-18 observation encoder. The gate policy additionally requires eight measured states and four history summaries. The plain policy constructs no state-history encoder or summary gate.

## Pen-cabinet: plain latent flow with ResNet-18

Use the existing single Bash entrypoint with an explicit pen-cabinet task:

```bash
cd /workspace/ysk/past2next_bug_fixed

bash train_p2n_latent_flow.sh \
  --variant p2n_latent_flow \
  --obs-encoder resnet18 \
  --task pen_cabinet \
  --gpus 2,3 \
  --batch-size 4 \
  --val-batch-size 4 \
  --grad-accum 8 \
  -- \
  training.num_epochs=2001 \
  logging.mode=online
```

This selects `/workspace/ysk/zarr/pen_cabinet_N67.zarr` and its frozen OAT EMA checkpoint at `output/training/pen_cabinet_N67_current_20260919_212551/tokenizer/checkpoints/ep-2950_mse-0.000.ckpt`. The checkpoint was selected by the best retained full-precision held-out reconstruction MSE (`1.3338282769836951e-05`). The policy preserves its tokenizer's seed 42 and validation ratio 0.1: 60 training / 7 validation episodes. The distinct `pen_cabinet_lp3_N67` dataset requires its own matching tokenizer.

The command starts only the plain variant with online W&B and an effective batch of 64. `--batch-size` is per GPU and must be a multiple of four. Add `--dry-run` before the `--` separator to check sources and configuration without starting training. `--task pen_cabinet` also supports the gated flow variant and either observation encoder; internally these recipes retain real-robot action/normalization semantics. `--task real_robot` retains the existing nut-washer recipe.

The exact plain ResNet command passed CPU source/schema preflight, and the full 16-layer, 768-wide model completed eight-step CPU inference on a real pen-cabinet sample using training-only normalizers. See [pen_cabinet_resnet18_preflight.json](../output/latent_flow_checks/pen_cabinet_resnet18_preflight.json) and [pen_cabinet_resnet18_cpu_inference.json](../output/latent_flow_checks/pen_cabinet_resnet18_cpu_inference.json). No production training or W&B run was started.

## Verified local inputs

CPU dry-runs passed for both real-robot configurations using:

- Data: `/workspace/ysk/zarr/nut_washer_v3_N77.zarr`.
- Split: 73 training episodes / 93,980 windows; 4 validation episodes / 5,269 windows. Validation episode IDs: `[6, 33, 49, 58]`.
- DINO snapshot: `/workspace/.hf_home/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots/114c1379950215c8b35dfcd4e90a5c251dde0d32`.
- Pinned DINO revision: `114c1379950215c8b35dfcd4e90a5c251dde0d32`.
- DINO weight digest (the existing encoder's filename+contents digest): `87bc0383a25e41c0b51ded893287fcb83f8ccf4f7d1698a3d239e6fba8888cba`.
- OAT source: `output/training/nut_washer_v3_N77_gated_so3aug_20260924_081008_317043427/tokenizer/checkpoints/ep-1540_mse-0.000.ckpt`.
- OAT checkpoint SHA256: `3ed93e731c35355194f7ab01a3c5c7cbd8a86cbb16b911d36a393d2f12f1e946`.

No GPU training, model downloads, full-size GPU memory measurements or robot trials were started during implementation. CPU tests and source/schema dry-runs do not establish the full-size model's GPU memory fit, training quality, two-step sampling quality or closed-loop success rate.

## Full-size DINO CPU construction and inference check

Both actual 16×768 policies completed an eight-step Euler inference on CPU with the real local pretrained DINO trunk, the selected frozen OAT EMA, and a raw sample from the nut-washer dataset. Outputs were finite with action shape `[1,8,7]` and full prediction shape `[1,16,7]`.

| Variant | Trainable parameters | Frozen parameters | Context shape |
|---|---:|---:|---|
| `p2n_latent_flow` | 260,716,805 | 27,401,428 | `[1,267,768]` |
| `p2n_state_gate_latent_flow` | 262,668,808 | 27,401,428 | `[1,271,768]` |

The machine-readable report is [full_model_cpu_inference.json](../output/latent_flow_checks/full_model_cpu_inference.json). These counts come from the actual instantiated modules. This was a construction/inference check with fresh untrained flow weights and identity policy state normalization; it does not measure task quality, trained action accuracy, CUDA throughput, GPU memory fit or optimizer/backward behavior.

The consulted ManiFlow source files are pinned to commit `ef2f116f1f90163ed36e657b8c5503740bb468af`, with URLs and SHA256 digests in [PAST2NEXT_LATENT_FLOW_REFERENCES.json](PAST2NEXT_LATENT_FLOW_REFERENCES.json). The implementation follows the local OAT-latent/history specification and does not equate its architecture or performance with the original direct-action ManiFlow model.

## Original ResNet-18 observation encoder

Use the same single Bash launcher for the ResNet version. This runs the plain and state-gate variants sequentially, with online W&B and the same frozen OAT checkpoint, dataset split, 2,001 epochs and latent-flow objective:

```bash
cd /workspace/ysk/past2next_bug_fixed

bash train_p2n_latent_flow.sh \
  --obs-encoder resnet18 \
  --gpus 2,3 \
  --batch-size 4 \
  --val-batch-size 4 \
  --grad-accum 8
```

Add `--variant p2n_latent_flow` to run only the plain policy, or `--variant p2n_state_gate_latent_flow` to run only the gate policy. Add `--dry-run` to check both real-robot configurations on CPU without training or starting W&B. The example uses effective batch 64. All batch-size flags retain their existing meaning and constraints.

The ResNet version directly reuses `RobomimicRgbEncoder`: separate trainable ResNet-18 backbones for each camera, GroupNorm, 32-keypoint SpatialSoftmax and 64-dimensional camera features; it retains the original pixel normalization, 112×112 random training crops and fixed center evaluation crops. The original current-state projection is reused. New projections and camera/frame embeddings adapt these features into the existing DiT-X context. With two cameras and two frames, vision contributes four tokens. There is no DINO trunk, visual resampler or DINO brightness/contrast augmentation.

ResNet weights initialize from scratch, matching the original encoder; no DINO files or external vision weights are needed. The frozen OAT EMA source and its action-normalization provenance are still required and validated. DINO-specific arguments such as `--dino` and `--dino-revision` are rejected for ResNet runs; an explicitly exported `FLOW_DINO` is also treated as a DINO source argument. Unset it before selecting ResNet if it was exported for a previous DINO run.

The encoder has its own student and EMA weights. Teacher and student reuse the same realized pixel crops for consistency training while evaluating their own trainable ResNet networks. Student vision parameters use the observation-encoder learning-rate group and the state projection uses the policy learning-rate group; the EMA teacher is updated by the existing EMA mechanism. Automatic output directories and ResNet W&B run names, groups and tags include `resnet18`; logging stays in `real_robot_p2n_latent_flow` by default.

Four additional recipes cover both policy variants and both tasks: `train_p2n_latent_flow_resnet18`, `train_p2n_state_gate_latent_flow_resnet18`, `experimental/train_p2n_latent_flow_resnet18_real_robot` and `experimental/train_p2n_state_gate_latent_flow_resnet18_real_robot`. Existing DINO recipes and original ResNet implementation files are unchanged. Direct Python launches accept the same `--obs-encoder resnet18` option.

For resume, omit `--obs-encoder` to infer it from the saved artifact, or specify its matching value explicitly. Switching encoder types during resume is rejected. The ResNet architecture, pixel normalizers, learned student/EMA vision weights and OAT are embedded in the artifact, so original external model paths are not required. Use the same single-variant resume syntax below.

Both full-size ResNet variants completed eight-step CPU inference on a real dataset sample with training-only state/action normalization and finite `[1,8,7]` executed-chunk outputs (`[1,16,7]` full prediction). The plain model has 269,719,343 total / 263,914,309 trainable parameters and context `[1,15,768]`; the gate model has 271,671,346 total / 265,866,312 trainable parameters and context `[1,19,768]`. Both retain 5,805,034 frozen OAT parameters. See [resnet18_full_model_cpu_inference.json](../output/latent_flow_checks/resnet18_full_model_cpu_inference.json).

The launcher tests cover source isolation, both variants/tasks, batch flags, default DINO behavior and encoder-aware resume. Both real-robot ResNet CPU source/schema dry-runs passed. The combined DINO/ResNet regression suite passed **97 CPU tests**. Both ResNet variants also passed two-rank CPU checks for accumulated updates, EMA synchronization and RNG restoration, plus explicit model/Adam checkpoint restoration checks. See [resnet18_implementation_verification.json](../output/latent_flow_checks/resnet18_implementation_verification.json). No production training or GPU acceptance was run for this addition.

## Real-robot commands

Run this command to train both variants sequentially with W&B online. The launcher uses the real-robot data, DINO snapshot and frozen OAT checkpoint listed above by default. Select GPUs that are free **when you launch**; `2,3` is the default selection, not an assertion that these GPUs are idle. The launcher checks GPU inventory and refuses visibly busy selections before starting worker processes.

```bash
cd /workspace/ysk/past2next_bug_fixed

bash train_p2n_latent_flow.sh \
  --gpus 2,3 \
  --batch-size 4 \
  --val-batch-size 4 \
  --grad-accum 8
```

`--batch-size` is the per-GPU training microbatch and must be at least 4 and divisible by 4 for the 3:1 FM/CT split. `--val-batch-size` is the per-rank validation batch size. `--grad-accum` is the number of microbatches accumulated per optimizer update. The example gives an effective training batch of **4 × 2 × 8 = 64**. These flag values are also the defaults.

The default `--variant both` runs `p2n_latent_flow` followed by `p2n_state_gate_latent_flow`, each on all selected GPUs. Use an explicit variant to run only one:

```bash
bash train_p2n_latent_flow.sh \
  --variant p2n_latent_flow --gpus 2,3 \
  --batch-size 4 --val-batch-size 4 --grad-accum 8 \
  --output output/training/nut_washer_p2n_latent_flow_seed42 \
  --dry-run
```

Remove `--dry-run` to start that variant. Change `--variant` to `p2n_state_gate_latent_flow` for the gate policy. `--dry-run` checks source/schema inputs without creating a model, touching CUDA, writing an output directory or starting training; `--preflight` is an alias. With `--variant both`, both variants are checked. `--output` names the exact run directory for a single variant; with `both`, it names the parent directory for separate variant run subdirectories.

The defaults are 2,001 epochs, EMA enabled, BF16 network autocast, 8-step Euler sampling and 8-step self-past generation. W&B online logging uses project `real_robot_p2n_latent_flow`; authenticate W&B before starting if needed. Explicit Hydra overrides after `--` take precedence over the batch flags and launcher defaults. For example:

```bash
bash train_p2n_latent_flow.sh --gpus 2,3 \
  --batch-size 8 --val-batch-size 4 --grad-accum 4 \
  -- training.num_epochs=101 logging.project=real_robot_p2n_latent_flow
```

`--python`, `--task`, `--obs-encoder`, `--dino`, `--tokenizer`, `--resume` and direct Hydra overrides remain available. The redundant `train_p2n_latent_flow_online.sh`, `train_nut_washer_latent_flow_online.sh` and `train_nut_washer_latent_flow_online_v2.sh` wrappers have been removed; use `train_p2n_latent_flow.sh` for all Bash commands.

The following variables are used by the advanced examples below; the main Bash command above does not require them:

```bash
FLOW_PYTHON=/venv/real_robot/bin/python
FLOW_DINO=/workspace/.hf_home/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots/114c1379950215c8b35dfcd4e90a5c251dde0d32
FLOW_GPUS=2,3
```

The Python entrypoint `"$FLOW_PYTHON" scripts/train_p2n_latent_flow.py` remains available for direct single-variant invocation; the Bash entrypoint supplies the convenience defaults documented above.

## Live progress and batched validation

The workspace reports setup stages, training batches, validation samples and checkpoint saving in the console and `progress.jsonl`. It reports the first training-batch loss immediately, then updates at `training.tqdm_interval_sec` (default one second), including loss, optimizer-update count, elapsed time, throughput and estimated remaining time for the current phase. Add `training.tqdm_interval_sec=5` after the Bash `--` separator for less frequent output. Epoch summaries remain in `logs.json` and now include validation/checkpoint durations.

Online W&B receives `train/*`, `validation/*` and `checkpoint/*` progress during those phases. W&B events advance independently of training batches, with charts using the explicit `global_step`; this lets validation progress arrive while training counters remain fixed. Distributed progress shows rank zero's local shard, while epoch metrics still reduce all ranks' sums/counts. Progress callbacks perform no distributed collectives, so unequal or empty validation shards are supported.

`--val-batch-size` now controls the actual observation-encoder, OAT and DiTX computation batch, including generated previous-window histories. For example, four valid samples require 26 batched DiTX forwards for expert/generated decoded validation instead of 104 batch-one forwards. Per-sample random draws retain the same seed, shape and draw order as the previous implementation; all held-out samples and action-validity masks remain included. Floating-point kernel results can differ slightly with batch shape. Training self-past chunks, loss, EMA, solver steps and checkpoint continuation settings are retained.

For the GPU 6 pen-cabinet run, the same command uses these fixes after restarting the Python process:

```bash
bash train_p2n_latent_flow.sh \
  --variant p2n_latent_flow \
  --obs-encoder resnet18 \
  --task pen_cabinet \
  --gpus 6 \
  --batch-size 16 \
  --val-batch-size 4 \
  --grad-accum 2 \
  -- training.num_epochs=2001 logging.mode=online
```

A process already running the older code must be restarted to load the changes. Use a complete saved checkpoint with `--resume` to continue saved work; the command above creates a fresh run. Editing source files does not update Python modules already loaded by the active worker.

Runtime-fix verification: **136 CPU regression tests passed**, including online-tracker stubs, accumulation/tail updates, and sequential-versus-batched validation. Actual two-rank CPU validation passed with uneven batches and an empty rank. The full 16×768 ResNet pen-cabinet model reduced four-sample expert/generated decoded validation from 104 batch-one DiTX calls to 26 batch-four calls: 15.270 seconds versus 6.312 seconds in one bounded two-thread CPU check. All metric counts matched and sums agreed within `rtol=atol=2e-5`. This is not a GPU speed measurement. See [pen_cabinet_batched_validation_cpu.json](../output/latent_flow_checks/pen_cabinet_batched_validation_cpu.json).

## Resume

Use a complete flow training artifact of the same variant, task and observation encoder, and select an explicit single `--variant`; resume does not accept `both`. Resume restores the saved resolved configuration first, then applies explicit overrides. Frozen DINO/OAT architecture, weights and normalizers are embedded in the artifact; their original external paths are not required.

```bash
bash train_p2n_latent_flow.sh --python "$FLOW_PYTHON" \
  --variant p2n_latent_flow --task real_robot --gpus "$FLOW_GPUS" \
  --resume output/training/nut_washer_p2n_latent_flow_seed42/checkpoints/latest.ckpt \
  --output output/training/nut_washer_p2n_latent_flow_seed42
```

For the gate checkpoint, change both the variant and output/checkpoint paths to `p2n_state_gate_latent_flow`. Exact resume retains the original world size, microbatch, accumulation, data split, optimizer and flow conventions. AR checkpoints, the other flow variant and mismatched observation encoders are rejected. Use `--dry-run` to validate resume metadata without constructing the networks.

## LIBERO entrypoints

Both LIBERO configs are covered by configuration and synthetic-data CPU tests. This instance does not contain the configured `data/libero/libero10_N500.zarr` or an approved matching LIBERO tokenizer, so source-level preflight and training for LIBERO were not claimed as completed.

Supply the actual LIBERO dataset and an OAT checkpoint trained with the same dataset, episode split, action semantics and normalizer. The nut-washer tokenizer is not a LIBERO substitute. These are command templates; replace the two placeholder paths first:

```bash
FLOW_LIBERO_DATA=/absolute/path/to/libero10_N500.zarr
FLOW_LIBERO_TOKENIZER=/absolute/path/to/libero_tokenizer_ema.ckpt

bash train_p2n_latent_flow.sh --python "$FLOW_PYTHON" \
  --variant p2n_latent_flow --task libero --gpus "$FLOW_GPUS" \
  --dino "$FLOW_DINO" --tokenizer "$FLOW_LIBERO_TOKENIZER" \
  --batch-size 4 --val-batch-size 4 --grad-accum 8 \
  --output output/training/libero_p2n_latent_flow_seed42 \
  -- task.policy.dataset.zarr_path="$FLOW_LIBERO_DATA"

bash train_p2n_latent_flow.sh --python "$FLOW_PYTHON" \
  --variant p2n_state_gate_latent_flow --task libero --gpus "$FLOW_GPUS" \
  --dino "$FLOW_DINO" --tokenizer "$FLOW_LIBERO_TOKENIZER" \
  --batch-size 4 --val-batch-size 4 --grad-accum 8 \
  --output output/training/libero_p2n_state_gate_latent_flow_seed42 \
  -- task.policy.dataset.zarr_path="$FLOW_LIBERO_DATA"
```

If the demonstration count, split seed or validation fraction differs, pass matching `training.num_demo`, `task.policy.dataset.seed` and `task.policy.dataset.val_ratio` overrides. Provenance checks reject mismatches with the tokenizer's normalization source. LIBERO uses quaternion history; real-robot history uses rotation-6D rows.

## New-only integration

Shared behavior is reused through new adapters rather than edits to old modules. Flow-specific observation/context and runner wrappers provide frozen-patch sharing, flow variant validation and execution acknowledgement while preserving old AR imports and code. Future-action validity and sample IDs are metadata only; sampled action bytes, repeated terminal padding, episode masks and normalizers are inherited unchanged.

`sample_id` is the absolute replay-buffer action anchor. The dataset identity is a SHA256 of source path, selected array schema, episode boundaries and raw action bytes; it is invariant to training/validation views and rank/batch ordering. It does not hash RGB payloads. Moving or changing a dataset creates a different identity and requires an explicit new run rather than silently resuming with changed validation noise.

CPU dataset/launcher contract tests:

```bash
/venv/real_robot/bin/python -m pytest -q tests/test_p2n_latent_flow_data_launch.py
```

The real full-size single-/dual-GPU acceptance stages from the plan remain a user-run step. Initial eight-step deployment and later two-step deployment must each be evaluated with decoded action metrics and robot trials.

## Optional bounded full-model GPU acceptance

The following commands **perform real, short optimizer runs** only when you execute them. They were not executed during implementation. The single-GPU entrypoint was syntax checked and its `--dry-run` passed for both real-robot variants using the local sources above. The frozen-codec CPU report is [oat_cpu_report.json](../output/latent_flow_checks/oat_cpu_report.json), including strict EMA loading, the saved normalizer hash, all 5,000 FSQ codes, exact decode/detokenize agreement and rejected nonfinite inputs.

Select one currently free GPU for the single-GPU check. This instantiates the full 16×768 model, uses microbatch 4 and the configured accumulation 8, runs at least three successful Adam/EMA updates with the curriculum at its maximum 0.5 self-past probability, then times full EMA `predict_action` at eight and two Euler steps. It records synchronized p50/p95 latency, per-phase peak memory, gradient checks and a single held-out sample's expert/generated-history diagnostics. Use a fresh output directory for each invocation.

```bash
FLOW_GPU=2

"$FLOW_PYTHON" scripts/smoke_p2n_latent_flow.py \
  --variant p2n_latent_flow --task real_robot --gpus "$FLOW_GPU" \
  --dino "$FLOW_DINO" --iterations 3 --latency-repeats 10 \
  --output output/latent_flow_checks/single_gpu_plain

"$FLOW_PYTHON" scripts/smoke_p2n_latent_flow.py \
  --variant p2n_state_gate_latent_flow --task real_robot --gpus "$FLOW_GPU" \
  --dino "$FLOW_DINO" --iterations 3 --latency-repeats 10 \
  --output output/latent_flow_checks/single_gpu_gate
```

Add `--dry-run` to either command to perform only CPU preflight. Actual runs write `resolved_config.yaml` and `smoke_report.json` beneath the supplied output directory. These short checks use constant configured optimizer learning rates; the main training workspace retains its complete scheduler. No trained policy checkpoint is produced. Inference memory is measured while student, EMA and optimizer state remain resident, and the held-out diagnostic is not the complete validation set.

The independent full-model two-rank check also tests synchronized fresh EMA, packed gradients with DDP `find_unused_parameters=False`, accumulated updates and a shorter final accumulation group. This diagnostic deliberately uses accumulation 2 plus a tail group of 1; use the normal training configuration to measure the production accumulation-8 schedule. Check the selected GPUs immediately before running these commands, then run the variants sequentially:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

CUDA_VISIBLE_DEVICES="$FLOW_GPUS" "$FLOW_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/p2n_latent_flow_ddp_smoke.py \
  --real --variant p2n_latent_flow --task real_robot --dino "$FLOW_DINO" \
  --updates 3 --latency-repeats 10

CUDA_VISIBLE_DEVICES="$FLOW_GPUS" "$FLOW_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/p2n_latent_flow_ddp_smoke.py \
  --real --variant p2n_state_gate_latent_flow --task real_robot --dino "$FLOW_DINO" \
  --updates 3 --latency-repeats 10
```

The two-rank script prints one JSON report per rank with losses, successful updates, hardware, peak allocated/reserved memory and synchronized eight-/two-step inference latency. Its default mode without `--real` is the separate tiny CPU contract test and cannot establish full-model GPU acceptance. Neither short smoke establishes convergence, two-step action quality or robot closed-loop performance.

## Final implementation verification

The original latent-flow implementation verification recorded **59 passing CPU tests** before the ResNet addition. Both variants also passed separate two-rank CPU smoke checks, including packed FM/CT gradients, generated history, synchronized EMA and an incomplete accumulation group. Real frozen DINO/OAT and full 16×768 CPU inference passed for both variants; these checks do not establish GPU fit or trained quality. Hash checks confirmed all 233 recorded original source/documentation files are unchanged.

The machine-readable evidence is [implementation_verification.json](../output/latent_flow_checks/implementation_verification.json). No production training or GPU work was started.
