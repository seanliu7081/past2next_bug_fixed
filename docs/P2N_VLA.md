# P2N-VLA: usage

P2N-VLA is a PI0.5-style VLA with a Past2Next action head, trained on LIBERO-10:

- **VLM.** PaliGemma: SigLIP So400m (frozen) and Gemma-2B with LoRA r16/α16.
- **Action expert.** The `lerobot/pi05_base` Gemma-300M expert, re-purposed as a Past2Next autoregressive head over the 8 tokens of the SO(3)-augmented OAT tokenizer.
- **Knowledge insulation.** The VLM also predicts the OAT tokens (KI loss), and the expert reads stop-gradient VLM K/V.
- **Self-past training.** The expert is also trained on histories it generated itself.

The design is in `/root/.claude/plans/give-me-a-plan-concurrent-knuth.md`, and the interface contract is in `docs/P2N_VLA_IMPLEMENTATION.md`.

| Variant | Policy class | Train config | Task config / runner |
|---|---|---|---|
| `p2n_vla` | `oat.policy.p2n_vla.P2NVLAPolicy` | `oat/config/train_p2n_vla.yaml` | `libero/libero10_vla` / `P2NNewLiberoRunner` |
| `p2n_vla_state_gate` | `oat.policy.p2n_vla_state_gate.P2NVLAStateGatePolicy` | `oat/config/train_p2n_vla_state_gate.yaml` | `libero/libero10_vla_state_history` / `P2NStateGateNewLiberoRunner` |
| `pi05_ki_flow` (baseline) | `oat.policy.pi05_ki_flow.PI05KIFlowPolicy` | `oat/config/train_pi05_ki_flow.yaml` | `libero/libero10_vla` / `P2NNewLiberoRunner` |

The variants:
- `p2n_vla` conditions on past commands and trains with self-past.
- `p2n_vla_state_gate` also reads 8 measured states. They are summarized into 4 HIST tokens behind a learned log-gate.
- `pi05_ki_flow` is the same stack with pi05's flow-matching head and no past input.

Entry points:

| File | Role |
|---|---|
| `train_p2n_vla.sh` → `scripts/train_p2n_vla.py` | Launcher: `--dry-run`, `--preflight`, `--probe`, or training via torchrun |
| `oat/workspace/train_p2n_vla.py` | Training workspace (Accelerate + DDP) |
| `oat/model/common/trainable_ema.py` | Trainable-parameter EMA |
| `scripts/evaluate_p2n_vla.py` | LIBERO-10 evaluation of snapshots |

All commands below run from the repository root, `/workspace/past2next_bug_fixed`, with `/venv/oat/bin/python`.

---

## 1. Assets (M0)

```bash
/venv/oat/bin/python scripts/fetch_p2n_vla_assets.py
```

This writes `data/pretrained/p2n_vla/assets.json`. The training configs pin exactly these files.

| Asset | Path | sha256 |
|---|---|---|
| pi05 weights | `/workspace/.hf_home/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors` (812 tensors, 14.47 GB) | `0eb11ca9…59b0f` |
| PaliGemma SentencePiece model | `/workspace/past2next_bug_fixed/data/pretrained/p2n_vla/paligemma_tokenizer.model` | `8986bb4f…fc6` |
| OAT tokenizer (LIBERO, SO(3) left_noise, EMA weights) | `/workspace/hf_upload/tokenizer_oattok_so3aug_ep4960_mse0.001.ckpt` | recorded by preflight |
| LIBERO-10 data | `/workspace/past_action/data/libero/libero10_N500.zarr` | 450 train / 50 validation episodes (seed 42, val_ratio 0.1) |

**OAT tokenizer.** Do not use `/workspace/frozen_tokenizer.ckpt`; that file is the nut_washer tokenizer. Preflight rejects any OAT tokenizer whose zarr, split, horizon, FSQ levels or EMA normalizer differ from these.

**Persistence.** `/workspace` is not persistent. Sync snapshots off the box after every run, and ask before publishing anything.

## 2. Tests

The fast suite uses tiny configs and runs on CPU. It includes the gloo DDP smokes, which are marked `slow`.

```bash
/venv/oat/bin/python -m pytest tests/test_p2n_vla_*.py -m "not requires_pi05 and not gpu"
```

Real assets and the GPU:

```bash
CUDA_VISIBLE_DEVICES=1 /venv/oat/bin/python -m pytest tests/test_p2n_vla_*.py -m "requires_pi05 or gpu"
```

The M4 suites alone:

```bash
/venv/oat/bin/python -m pytest tests/test_p2n_vla_ema.py tests/test_p2n_vla_workspace.py tests/test_p2n_vla_launcher.py -q
```

Markers are registered in `tests/conftest.py`:

| Marker | Meaning |
|---|---|
| `requires_pi05` | Needs the 14.5 GB pi05_base weights |
| `requires_data` | Needs the LIBERO-10 zarr |
| `gpu` | Needs CUDA; uses the first visible device, so select it with `CUDA_VISIBLE_DEVICES` |
| `slow` | Runs torchrun DDP smokes or real-policy integration |

Tests skip loudly when their resource is missing.

What the M4 tests cover:

`tests/test_p2n_vla_workspace.py` drives the workspace in two ways. First, through a stub policy that implements the M3 API exactly. Second, through the real tiny policies, using a LIBERO-schema zarr written on the fly. It checks:
- the LR schedule shape;
- per-group gradient clipping;
- counters that advance on optimizer steps only, including a tail accumulation group and a skipped update;
- the checkpoint and snapshot layout;
- EMA validation;
- exact resume;
- the hard stop and the probe;
- a 2-rank gloo DDP smoke, both through the stub and for each real variant;
- a CUDA run with fused AdamW;
- a `from_checkpoint` reload that gives bitwise-identical `predict_action` output, in-process and in a fresh process without the dataset (fp32 EMA artifact and bf16 snapshot);
- a 2-rank gloo resume that is bitwise identical to an uninterrupted run;
- that validation settings never change the training trajectory (validation runs on a forked RNG);
- the `use_ema=false` path (mode restoration after validation, `_model` snapshots, resume);
- that resume refuses a renamed, reordered or reshaped trainable set (AdamW state is positional).

The other two M4 files:
- `tests/test_p2n_vla_ema.py` covers the EMA formula, swap-in, and state round trips.
- `tests/test_p2n_vla_launcher.py` covers config composition and validation, the CLI, preflight checks against the real data and assets, and the eval script. The eval tests include dry-runs of real tiny gate and flow snapshots, unique default output folders, and the refusal of inference overrides the policy would ignore.

## 3. Launcher

```text
bash train_p2n_vla.sh --variant {p2n_vla,p2n_vla_state_gate,pi05_ki_flow} [--task libero]
    [--devices 0,1] [--num-processes N] [--output DIR] [--resume CKPT]
    [--tokenizer OAT.ckpt] [--pi05 model.safetensors] [--spm tokenizer.model]
    [--dry-run | --preflight | --probe [--probe-steps N] [--probe-self-past-step K]]
    [--cpu] [--allow-busy-gpus] [--full-hash] [--report FILE] [-- hydra.overrides ...]
```

`train_p2n_vla.sh` sets three defaults:
- `HF_HOME=/workspace/.hf_home`;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- the interpreter `/venv/oat/bin/python`. Override it with `P2N_PYTHON=...` or `--python PATH`.

The flags behave as follows:
- `--devices` sets `CUDA_VISIBLE_DEVICES`.
- `--num-processes` defaults to the number of devices.
- Before a real launch, the busy-GPU check refuses any GPU that is in use; `--allow-busy-gpus` overrides it.

### 3.1 Dry run: compose and validate the configuration

```bash
bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --dry-run
```

The dry run prints the resolved YAML and a JSON summary, which includes the update arithmetic. It does not read data, weights or GPUs.

It validates the following, and fails on any disagreement:
- the variant, the policy `_target_`, the dataset class and the runner class agree;
- the schema is horizon 16, execute 8, past 7, `n_obs_steps=1`;
- the policy and dataset prompt-state keys agree;
- the gate history keys and steps agree across policy, dataset and runner;
- the flow baseline has `use_past=false`;
- the `zarr_path` is absolute;
- the recipe numbers are in range.

### 3.2 Preflight: CPU checks and model build

```bash
bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --devices 0,1 --preflight \
    --report output/preflight/p2n_vla_state_gate.json
```

Preflight checks each of the following:
- **Dataset.** The zarr schema matches; the split is 124,600 train windows and 13,490 validation windows.
- **OAT provenance.** Through `scripts/train_p2n_latent_flow.validate_tokenizer_source`, it checks the zarr path, split and horizon, FSQ levels `[8,5,5,5,5]`, 16×7↔8×5, and the EMA action normalizer.
- **pi05 weights.** The sha256 comes from the HF blob name; `--full-hash` re-hashes all 14.5 GB instead. The header must hold 812 tensors.
- **SentencePiece model.** Its sha256 must match.
- **Prompt length.** The worst case must fit in 96 tokens. Every instruction is tried with every state bin at 3 digits; the longest is uid 34 at 62 tokens.
- **Update schedule.** It reports 200 micro-batches per rank per epoch, 100 updates per epoch, 300 epochs and 30,000 updates in total.
- **In-training rollouts** (only with `task.policy.lazy_eval=false`): the EGL renderer probe and the rollout plan.
- **W&B** (only with `logging.mode=online`): credentials from `WANDB_API_KEY` or `wandb login`'s `~/.netrc` entry.
- **Model build.** It builds the full policy on CPU and reports:
  - parameter counts;
  - optimizer groups and clip groups;
  - frozen base keys;
  - artifact size and estimated checkpoint sizes;
  - that every trainable parameter is an fp32 master weight.

  The build loads the bf16 backbone into RAM, which takes a few minutes.

A fresh run needs an empty `--output`.

### 3.3 Probe: memory and throughput go/no-go on 2 GPUs

Nothing else may run on the GPUs during the probe.

```bash
bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --devices 0,1 --probe            # self-past at p=0.5
bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --devices 0,1 --probe --probe-self-past-step 0   # p=0
# with in-training rollouts: also one official chunk (10 episodes) on rank 0, measuring the EGL renderer memory
bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --devices 0,1 --probe -- task.policy.lazy_eval=false
```

The probe launches torchrun and runs the following:
- `--probe-steps` optimizer steps (default 3), at accumulation 2 and micro-batch 8;
- the self-past counter set to warmup+ramp, so p=`self_past_p`;
- one worst-case forward and backward pass with `history_mode='generated'` (p=1);
- one validation batch under the EMA.

Results go to `output/probe/<variant>_libero_<timestamp>/probe.json`:
- samples/s, excluding the first update;
- per-rank peak `max_memory_reserved` for train, worst case and validation;
- an `nvidia-smi` table;
- `go_no_go`: at most 22.5 GB reserved and at least 2 GB of headroom (decimal GB; headroom is measured from `nvidia-smi`).

The probe writes no checkpoints.

### 3.4 Train

Run the training command inside tmux. Each run uses both GPUs, so run the variants one after the other.

**LIBERO-10 training with in-training official evaluation, logged live to W&B.**

```bash
tmux new -s p2n_vla
bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output output/training/p2n_vla_libero10_s42 \
    -- task.policy.lazy_eval=false logging.mode=online
# when it has finished:
bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --devices 0,1 --output output/training/p2n_vla_state_gate_libero10_s42 \
    -- task.policy.lazy_eval=false logging.mode=online
```

With `task.policy.lazy_eval=false`, the following happens after epochs 50, 100, …, 300 (`training.rollout_every=50`; one epoch is 100 updates, so every 5,000 updates, on the snapshot steps):
- **Who and what.** Rank 0 evaluates the EMA weights with LIBERO's official protocol: 50 episodes per task, which is each task's 50 fixed initial states once (500 episodes), with episode seeds 3000–3499. The settings live in `task.policy.env_runner`. The schedule is identical to `scripts/evaluate_p2n_vla.py --protocol official --n-test 500 --seed 44 --episode-start-seed 3000`.
- **Rendering and time.** The simulators render on rank 0's own GPU, and rank 1 waits. Episodes run in chunks of `n_parallel_envs` = 10, and a chunk lasts until its slowest episode ends: about 47 s at the 550-step budget. One evaluation is therefore 50 chunks, about 40 minutes plus a minute of simulator start-up. The launcher refuses settings whose expected time comes near `training.rollout_timeout_minutes` (100). Rank 1's barrier is bounded by the 2 h process-group timeout, so do not shrink `n_parallel_envs` far.
- **Memory.** The probe measured a 20.6 GB peak on GPU 0 with the training state plus 10 renderers (about 0.55 GB each), leaving 5.1 GB headroom.
- **Results.** Each evaluation writes `eval/rollout_epoch-EEEE_upd-NNNNNN/{summary.json,episodes.jsonl}`, linked to the snapshot of the same step (a snapshot is saved for every rollout step). The epoch record (training and validation) is written first. The rollout then writes its own record (`"event": "rollout"`, the same `optimizer_step`; `epoch` is 0-indexed in both) to `logs.jsonl` and W&B, with:
  - `rollout/success_rate`, `rollout/macro_task_success_rate` and its Wilson bounds;
  - `rollout/task/<task>` for every task;
  - `rollout/peak_gpu_used_gb`, `rollout/failed`;
  - `mean_success_rate`, the legacy key.
- **Isolation.** Training RNG streams, live weights and module modes are restored exactly, so rollouts do not change what training computes. On CPU the trajectory is bit-identical (tested). CUDA kernels such as SDPA backward are not deterministic, so GPU runs are never bitwise reproducible, with or without rollouts.
- **What is scored.** The EMA weights, rounded to bf16 exactly as the snapshot stores them. The in-training success rate therefore belongs to `snapshots/upd-NNNNNN_ema.ckpt`, and `scripts/evaluate_p2n_vla.py` on that snapshot loads bitwise the same weights.
- **Checkpoints.** A resume checkpoint is written right before every rollout except the final one. At the end training is complete, `latest.ckpt` is removed as usual, and the final snapshot remains.
- **Failures.** A simulator that crashes is killed, never waited on, and the traceback prints at once. One that hangs is killed by a watchdog after `training.rollout_timeout_minutes`. A failure during simulator start-up kills the workers already forked. With `training.rollout_failure=continue` (the default), the failure is recorded (`rollout/failed`, plus `summary.json` with `failed: true`) and training continues; re-evaluate that snapshot with `scripts/evaluate_p2n_vla.py`. With `raise`, the run stops. If the whole run dies during a rollout, its `eval/rollout_*` folder has no `summary.json`. The resume continues after that epoch, warns about it, and `training_summary.json` lists it as incomplete; re-evaluate that step's snapshot.
- **What "official" means here.** LIBERO's saved initial states 0–49 per task and LIBERO's 5 zero-action settling steps, with this repository's step budget of 550 policy steps (settling not counted). That is the budget every evaluation and baseline in this repo uses, including the 0.772-SR policy. LIBERO's own evaluator defaults to 600 steps, and openpi/OpenVLA use 520 plus 10 wait steps. State the budget when comparing with published numbers, or set `-- task.policy.env_runner.max_episode_steps=600`. `scripts/evaluate_p2n_vla.py` uses the run's own budget unless `--max-episode-steps` is given.
- **Total length.** About 24 h of training (11.2 samples/s once self-past has ramped up) plus 6 × about 41 min of evaluation, roughly 28 h per variant.

Selecting a checkpoint by these official-protocol numbers selects on the final test schedule. Report the last evaluation, or say that the best one was selected.

Without the override, `task.policy.lazy_eval` stays `true`, so there are no in-training rollouts; evaluate snapshots with `scripts/evaluate_p2n_vla.py` (section 5):

```bash
bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output output/training/p2n_vla_s42
```

**Resume.** Resume is epoch-granular. `latest.ckpt` is written every 10 epochs (1,000 updates), so a crash replays at most about 1,000 updates. It restarts at the epoch after the one `latest.ckpt` recorded, so any updates made after that checkpoint are replayed (with the same data order and RNG streams; bitwise on CPU) and `logs.jsonl` receives their records again. It needs the same world size, variant, architecture, trainable-parameter order, data split, optimizer recipe and update schedule. `training.max_optimizer_steps` (the cosine horizon) cannot change; `training.num_epochs` may, but training still stops at `max_optimizer_steps`. To continue the same W&B run, add `-- logging.id=<id> logging.resume=allow`.

```bash
bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output output/training/p2n_vla_s42 \
    --resume output/training/p2n_vla_s42/checkpoints/latest.ckpt
```

A resume rebuilds the configuration from the defaults, so pass every `--` override of the original run again; the launcher warns when `task.policy.lazy_eval`, `training.rollout_every` or `logging.mode` differ from the checkpoint's run. For the in-training-evaluation run, continuing the same W&B run (`<id>` is the suffix of `<run>/wandb/run-<timestamp>-<id>`; `logging.resume` only applies online):

```bash
bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output output/training/p2n_vla_libero10_s42 \
    --resume output/training/p2n_vla_libero10_s42/checkpoints/latest.ckpt \
    -- task.policy.lazy_eval=false logging.mode=online logging.id=<id> logging.resume=allow
```

Epochs replayed after a resume that had already been evaluated keep their earlier result: the new one goes to `eval/rollout_..._r1`. `training_summary.json` lists every rollout found on disk.

**Other variants.** These use the same recipe, seed and OAT tokenizer:

```bash
bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --devices 0,1 --output output/training/p2n_vla_state_gate_s42
bash train_p2n_vla.sh --variant pi05_ki_flow --task libero --devices 0,1 --output output/training/pi05_ki_flow_s42
```

**Pilot (plan, run order 1).** A gate run of about 4k updates, with self-past on from step 500 and snapshots every 2,000 updates. Then evaluate it with about 100 corrected-protocol trials.

```bash
bash train_p2n_vla.sh --variant p2n_vla_state_gate --task libero --devices 0,1 --output output/training/pilot_gate_s42 \
    -- training.max_optimizer_steps=4000 training.snapshot_every=2000 \
       policy.self_past_warmup_steps=500 policy.self_past_ramp_steps=0
```

**Tier-1 ablations (M7).** Each is a construction-time switch.

```bash
# GT past (self-past off)
bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output output/training/p2n_vla_gtpast_s42 -- policy.self_past_p=0.0

# Frozen VLM (lambda_KI = 0: LoRA and the KI table are frozen, and the KI pass is skipped)
bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output output/training/p2n_vla_noki_s42 -- policy.lambda_ki=0
```

**Logging.** W&B logs offline by default. Set `-- logging.mode=online` to stream (log in to W&B first), or `-- logging.mode=disabled` to turn it off.

**CPU smoke on the real dataset.** Each rank loads about 12.6 GiB into RAM.

```bash
bash train_p2n_vla.sh --variant p2n_vla --task libero --cpu --num-processes 2 --output /tmp/p2n_vla_cpu_smoke \
    -- policy.model_size=tiny policy.pi05_weights=null policy.activation_checkpointing=false \
       training.num_epochs=1 training.max_train_steps=4 training.max_optimizer_steps=4 training.lr_warmup_steps=1 \
       training.max_val_steps=1 dataloader.num_workers=0 dataloader.persistent_workers=false \
       val_dataloader.num_workers=0 val_dataloader.persistent_workers=false logging.mode=disabled
```

**Overfit on 2 episodes (M5 bring-up).** The launcher refuses this run: its OAT provenance check compares `max_train_episodes` with the OAT tokenizer's split. So compose and validate the config with the launcher's own function, then start the worker directly. This skips the preflight and the busy-GPU check. With one process the effective batch is 16.

```bash
RUN=output/training/overfit_2ep_p2n_vla && mkdir -p $RUN
/venv/oat/bin/python - "$RUN" <<'EOF'
import sys; sys.path.insert(0, ".")
from omegaconf import OmegaConf
import scripts.train_p2n_vla as launcher
cfg = launcher.compose_config("p2n_vla", "libero", [
    "task.policy.dataset.max_train_episodes=2", "training.max_train_steps=null", "training.num_epochs=100",
    "training.max_optimizer_steps=3000", "training.lr_warmup_steps=100", "training.val_every=25",
    "training.snapshot_every=1000", "logging.mode=disabled"])
OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), f"{sys.argv[1]}/config.yaml")
EOF
CUDA_VISIBLE_DEVICES=1 /venv/oat/bin/python scripts/train_p2n_vla.py --worker-config $RUN/config.yaml --output $RUN
```

With `max_train_episodes` set, the dataset's validation view is the complement of the 2 training episodes (498 episodes), so `val/*` measures generalization, not the overfit. Read the overfit from `train/loss_ar` and `train/ar_token_acc` in `logs.jsonl`.

Score the training windows themselves offline from a snapshot:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/oat/bin/python scripts/p2n_vla_bringup/overfit_recon.py --snapshot $RUN/snapshots/upd-003000_ema.ckpt
```

`scripts/p2n_vla_bringup/overfit_recon.py` re-instantiates the dataset config embedded in the snapshot. Pass `--split val` for the complement, and `--weights model` for a `training.use_ema=false` run, whose snapshots are `upd-NNNNNN_model.ckpt`. It writes `<run>/overfit_recon_<snapshot>_<weights>_<split>.json` with:
- teacher-forced `loss_ar`, `ar_token_acc`, `loss_ki` and `ki_token_acc`, using the dataset history;
- greedy generated-token accuracy, overall, per position k and as exact-chunk rate;
- the stateless `predict_action` reconstruction MSE, in raw and normalized action space;
- the OAT round-trip floor `detokenize(tokenize(action))` on the same windows;
- a check that `predict_action` decodes exactly the scored greedy tokens.

A short run with the default EMA (decay 0.999, a time constant of about 1,000 updates) lags the live weights, so set `training.use_ema=false` to score what the head can memorize.

**Hang diagnostics.** `py-spy` and `gdb` cannot attach in this unprivileged container. `scripts/p2n_vla_bringup/stack_dump/sitecustomize.py` is an opt-in hook that stays passive until signalled. Prefix any launch with `PYTHONPATH=scripts/p2n_vla_bringup/stack_dump P2N_STACK_DIR=<dir>`, then run `kill -USR1 <worker pid>` to append that process's Python thread stacks to `<dir>/stack_<pid>.txt`.

### 3.5 Run directory

```text
<run>/
  p2n_vla_resolved.yaml, p2n_vla_preflight.json    launcher (resumes add _resume_<timestamp>)
  resolved_config.yaml, training_report.json         workspace start-up (parameters, groups, schedule, DDP settings)
  dataset_split.json                                 episode ids and dataset identity hash
  logs.jsonl                                         train_step / skipped_update / epoch / rollout records
  eval/rollout_epoch-EEEE_upd-NNNNNN/                in-training official rollouts: summary.json, episodes.jsonl
  checkpoints/latest.ckpt                            resume payload; written every 10 epochs and before every rollout,
                                                     deleted after the final snapshot
  snapshots/upd-005000_ema.ckpt ... upd-030000_ema.ckpt   EMA artifacts (about 0.73 GB each)
  training_summary.json                              final counters, snapshot list, last validation, every rollout
  wandb/                                             offline W&B run (unless logging.mode=disabled)
  eval/<snapshot>_<weights>_<protocol>_n<N>_seed<S>_ep<E>[_k<k>][_T<t>][_gate-<mode>]...
                                                     default output of scripts/evaluate_p2n_vla.py (one suffix per
                                                     non-default result-affecting option, so paired runs never collide)
```

## 4. Training recipe

The recipe lives in `oat/config/train_p2n_vla.yaml`; the gate and flow configs inherit it.

**Batch and length**

| Setting | Value |
|---|---|
| Micro-batch | 8 per GPU |
| GPUs | 2 |
| Gradient accumulation | 2 |
| Effective batch | 32 |
| Epochs (`num_epochs`) | 300 |
| `max_train_steps` | 200 micro-batches per rank per epoch, so an epoch is 100 updates (3,200 windows). One pass over the 124,600 training windows is 3,894 updates, and 30k updates are 7.7 passes. Every epoch draws a fresh seeded permutation |
| Updates | 30,000 (`max_optimizer_steps`, also the cosine horizon) |
| Validation (`val_every`) | Every 50 epochs: after epochs 1, 51, …, 251 and the last |
| Resume checkpoint (`checkpoint_every`) | Every 10 epochs |
| Snapshots (`snapshot_every`) | Every 5,000 updates (`snapshots/upd-NNNNNN_ema.ckpt`) |
| In-training rollouts (`rollout_every`) | Every 50 epochs, after epochs 50 … 300, only with `task.policy.lazy_eval=false` |

**Optimizer and schedule**
- **AdamW:** betas (0.9, 0.95), eps 1e-8, weight decay 1e-10, fused when every parameter is on CUDA.
- **Peak LR by group:**

  | Group | Parameters | Peak LR |
  |---|---|---|
  | `pretrained` | expert, folded modulations, LoRA | 5e-5 |
  | `new` | KI table, `tok_emb`, RAW/ACC/JERK, history and gate modules | 1e-4 |

- **LR schedule:** one `LambdaLR` stepped on successful optimizer steps.
  - Linear warmup over 1,000 steps, from peak/(W+1) to the peak. The first update never uses LR 0, as in openpi.
  - Then a half-cosine down to 0.1×peak at 30,000 steps.
- **Gradient clipping:** max norm 1.0, applied separately to each `policy.clip_groups()` group: `ar` (expert side) and `ki` (LoRA and KI table). Both norms are logged as `grad_norm_ar` and `grad_norm_ki`. If any norm is non-finite on any rank, the update is skipped and counted. More than 20 consecutive skips abort the run.

**EMA**
- `TrainableEMA` with decay 0.999 keeps fp32 shadows of the trainable parameters only.
- It is updated once per successful optimizer step, after `scheduler.step()` and `policy.on_optimizer_step()`.
- For validation it is swapped into the live policy on every rank, under `torch.no_grad()` and never under `inference_mode`. Parameter values and every module's training flag are restored afterwards.

**Self-past**
- p=0.5, with 1,000 warmup steps and a 4,000-step ramp, chunk 4, T=1, top-k 10.
- The counter advances on successful optimizer steps only.
- Deployment decodes greedily (`temperature: 0.0`).

**Other policy settings**
- λ_KI = 1.0, LoRA r16/α16, `adarms_t0` = 0.6, `max_prompt_len` = 96, `model_size` = full, activation checkpointing on.

**Validation** runs every 50 epochs on the EMA weights, over 200 batches per rank. It draws a strided subset spanning all 50 validation episodes, with a fixed RNG stream that leaves the training RNG untouched. It reports:
- expert-history and generated-history losses and their components;
- stateless `predict_action` reconstruction MSE against ground-truth past commands, over all 16 steps and over the 8 executed steps;
- gate metrics.

**DDP and precision**
- Accelerate runs with `mixed_precision='no'`; the policy owns its bf16 autocast.
- DDP runs with `find_unused_parameters=False` and `gradient_as_bucket_view=True`, and gradients are zeroed with `zero_grad(set_to_none=False)`.
- At every epoch end, a per-rank parameter fingerprint verifies that the replicas are still bitwise identical.

**Data order** is a pure function of (seed, epoch), so all ranks shard the same permutation and a resume replays the uninterrupted order.

### Logged values (`logs.jsonl`; W&B receives the numeric subset)

**`train_step`**, written every `log_every` updates:
- `optimizer_step`, `epoch`, `global_step`;
- `train/<component>`, averaged over the accumulation group. The components are `loss`, `loss_ar` (`loss_flow` for the flow baseline), `loss_ki`, `ar_token_acc`, `ki_token_acc`, `self_past_p`, `self_past_rows`, `gate_mean`, `gate_min`, `gate_max` and `hist_attention_mass`;
- `lr/pretrained`, `lr/new`, giving the rate of the next update;
- `grad_norm_ar`, `grad_norm_ki`;
- `samples_per_sec`, the global value for the update;
- `max_memory_reserved_gib`, `max_memory_allocated_gib`;
- `self_past_step`.

**`epoch`**:
- `train_loss`, `train_samples_per_sec`, `train_epoch_seconds`;
- `val_loss`, which is the expert-history loss;
- `val_loss_generated_history`, `val_reconstruction_mse`;
- `val/expert/*`, `val/generated/*`, `val/gate/*`;
- `validation_seconds`, `replica_fingerprint`;
- the checkpoint and snapshot paths.

### Checkpoint payloads (`p2n_vla_checkpoint_v1`, written atomically with `torch.save`)

Both payload kinds share these keys:

```text
{'format': 'p2n_vla_checkpoint_v1',
 'cfg': resolved training config (plain dict),
 'policy_config': policy.export_config()                 # construction_mode='restore'
 'metadata': policy.artifact_metadata() + {'training_run': {...}},
 'state_dicts': ...}
```

`metadata['training_run']` records the kind, variant, optimizer step, epoch, EMA updates, world size, update schedule, dataset split, software versions and workspace source hashes.

| Kind | `state_dicts` | `training` block |
|---|---|---|
| **Resume:** `checkpoints/latest.ckpt` (about 5.7 GB) | `{'model': live artifact, 'ema_model': EMA artifact}`, all fp32 | `{optimizer, trainable_names, lr_scheduler, ema: {decay, warmup_power, updates, names}, rng_states: per rank, counters, world_size, update_schedule, dataset_split}` |
| **Snapshot:** `snapshots/upd-NNNNNN_ema.ckpt` | `{'ema_model': EMA artifact}`; trainable tensors are bf16, every other tensor keeps its native dtype | none |

An artifact is `policy.state_dict()` minus `policy.frozen_base_keys()`, the frozen pi05 backbone. It keeps:
- the normalizers and prompt-state statistics;
- the counters;
- the OAT tokenizer (fp32);
- the KI table, LoRA and the expert.

On resume, the EMA shadow is rebuilt from `state_dicts['ema_model']`, so it is not stored twice. AdamW state is matched to parameters by position, so resume refuses a checkpoint whose `trainable_names` differ from the rebuilt policy's (renamed, added, removed or reordered trainables), and checks every saved moment's shape.

Load a snapshot with:

```python
from oat.policy.p2n_vla import P2NVLAPolicy   # or the variant's class
policy = P2NVLAPolicy.from_checkpoint("<run>/snapshots/upd-030000_ema.ckpt", weights="ema", device="cuda:0")
# Any variant: oat.policy.p2n_vla_common.P2NVLACommonPolicy.from_checkpoint(...) restores the class named in the payload.
```

## 5. Evaluation

The evaluation script works as follows:
- `scripts/evaluate_p2n_vla.py` restores the snapshot with the class's `from_checkpoint`. The frozen backbone comes from the pi05 base weights (`--pi05`; default: the pinned HF cache path).
- It rebuilds the runner from the run's embedded task config.
- It reuses `scripts/evaluate_candidate.py` for the episode schedule, Wilson summaries and source/initial-state hashing.
- It writes `summary.json`, `episodes.jsonl`, `metadata.json`, `source_hashes.json`, `schedule.json`, `runner_config.json` and `resolved_config.json` into a new folder. Existing folders are never overwritten.
- There is no crop override, and decoding is greedy unless `--temperature` is given.
- Inference overrides are refused rather than silently ignored. `--use-k-tokens`, `--temperature` and `--topk` apply to the autoregressive heads only, so they are rejected for `pi05_ki_flow` (flow matching, 10 Euler steps). `--topk` also needs `--temperature` > 0, and `--force-gate` needs the gate variant. These checks run before the policy is restored.
- The default output folder is `<run>/eval/<snapshot>_<weights>_<protocol>_n<N>_seed<S>_ep<E>`. It gets one suffix per non-default result-affecting option: `_init<offset>`, `_tasks-<sel>`, `_steps<max>`, `_k<k>`, `_T<t>`, `_topk<k>` and `_gate-<mode>`. So the official run and a `--use-k-tokens 4` run of the same snapshot never collide.

Evaluate after training, with one process per GPU; nothing else may run on a GPU that is evaluating. Measured for one process with the default 10 parallel environments: 8.8 GB for the policy plus 5.5 GB for the simulator renderers, about 14.3 GB per GPU.

**Renderer GPU.** EGL device indices need not equal CUDA ordinals. On this 2×4090 host they are swapped: CUDA 0 is EGL 1. `evaluate_p2n_vla.py` therefore ignores `MUJOCO_EGL_DEVICE_ID`. It matches the policy GPU's CUDA UUID against `oat/common/libero_egl_devices.py`, run in an isolated process, and builds the simulators through the scoped runner wrappers in `oat/env_runner/p2n_new_convnext_libero10_runner.py`, so they render on the policy's own GPU. `metadata.json` records the chosen `renderer` and both runner targets. The legacy `evaluate_candidate.py` (context baseline below) still renders on the EGL index given by `MUJOCO_EGL_DEVICE_ID`, which on this host is the other GPU. Run it only when both GPUs are free.

**Single snapshot, final protocol** (official initial states, 500 trials, seed 44, episode seeds 3000–3499):

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py \
  --snapshot output/training/p2n_vla_s42/snapshots/upd-030000_ema.ckpt --protocol official --n-test 500 --seed 44 --episode-start-seed 3000
```

**Checkpoint selection.** Evaluate all six snapshots with 100 corrected-protocol trials each, two GPUs in parallel. Never select a checkpoint by token cross-entropy.

```bash
RUN=output/training/p2n_vla_s42
for GPU in 0 1; do
  ( for SNAP in $(ls $RUN/snapshots/upd-*_ema.ckpt | awk -v g=$GPU 'NR % 2 == g'); do
      CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py \
        --snapshot $SNAP --protocol corrected --n-test 100 --seed 45 --episode-start-seed 4000
    done ) &
done; wait
```

**Gate diagnostics.** These are eval-only and change only the HIST attention bias. `open` removes the bias; `closed` masks HIST completely.

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py \
  --snapshot output/training/p2n_vla_state_gate_s42/snapshots/upd-030000_ema.ckpt --protocol official --n-test 500 \
  --seed 44 --episode-start-seed 3000 --force-gate closed
```

**Partial OAT decoding** (eval only, AR variants; k ∈ {1, 2, 4, 8}):

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py \
  --snapshot output/training/p2n_vla_s42/snapshots/upd-030000_ema.ckpt --protocol official --n-test 500 \
  --seed 44 --episode-start-seed 3000 --use-k-tokens 4
```

**Dry run.** This loads the policy and writes provenance and the schedule without creating simulators:

```bash
/venv/oat/bin/python scripts/evaluate_p2n_vla.py --snapshot <snapshot> --device cpu --dry-run
```

**Context baseline.** The 0.772-SR self-past policy uses the same OAT tokenizer. Evaluate it on the identical schedule:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_candidate.py \
  --checkpoint /workspace/hf_upload/policy_past2next_self_past_ep0200_sr0.772.ckpt \
  --output-dir output/eval/past2next_self_past_0772_official --protocol official --n-test 500 --seed 44 --episode-start-seed 3000
```

**Paired comparison.** Runs on the same official schedule share `episode_index`, `task_name` and `init_state_id`. McNemar's test on the discordant pairs:

```python
import json, math
def load(path): return {r["episode_index"]: r for r in map(json.loads, open(path))}
a = load("output/training/p2n_vla_s42/eval/A/episodes.jsonl"); b = load("output/training/pi05_ki_flow_s42/eval/B/episodes.jsonl")
assert a.keys() == b.keys() and all(a[i]["init_state_id"] == b[i]["init_state_id"] for i in a)
n01 = sum(not a[i]["success"] and b[i]["success"] for i in a); n10 = sum(a[i]["success"] and not b[i]["success"] for i in a)
p = sum(math.comb(n01 + n10, k) for k in range(min(n01, n10) + 1)) * 2 / 2 ** (n01 + n10) if n01 + n10 else 1.0
print(dict(a_only=n10, b_only=n01, exact_mcnemar_p=min(1.0, p)))
```

## 6. Policy API used by M4

These are the calls the workspace, the launcher and the eval script make on a policy (contract section "M3: policy API").

**Workspace**

| Purpose | Calls |
|---|---|
| Construction | `hydra.utils.instantiate(cfg.policy)` for a fresh run. On resume, `hydra.utils.instantiate(payload['policy_config'] + {pi05_weights, spm_path})`, then `load_artifact_state(state_dicts['model'])` |
| Setup | `.to(device)`, `set_normalizer(dataset.get_normalizer())` (fresh runs only), `get_optimizer(**cfg.optimizer)` with named groups `pretrained`/`new` |
| Parameter contract | `trainable_named_parameters()`, `clip_groups()` (`ar`/`ki`), `parameters()`, `named_parameters()` |
| Training | `policy(batch)` through DDP, in train mode; `last_loss_components`; `on_optimizer_step()`; `self_past_step` |
| Probe | `set_self_past_step(k)`, `self_past_probability()`, `policy(batch, history_mode='generated')` |
| Validation | `policy(batch, history_mode='expert'\|'generated')` in eval mode; `predict_action(obs, past_actions=, past_action_valid=)` (through `predict_validation_action`); `get_history_gate_metrics()`; `reset()`; `n_action_steps` |
| Artifacts | `artifact_state_dict(trainable_override=)`, `frozen_base_keys()`, `export_config()` (which must use `construction_mode='restore'`), `artifact_metadata()` (a dict containing `variant`) |
| Capability flags | `supports_generated_history_validation`, `supports_explicit_past_actions`, `supports_explicit_past_action_valid`, `requires_state_history` (through `validate_history_batch`) |

**Launcher.** The same calls on a CPU-built policy, plus `dtype`.

**Eval script**

| Purpose | Calls |
|---|---|
| Loading | `Class.from_checkpoint(path, base_weights=, weights=, device=, spm_path=)` |
| Rollout | `predict_action` keyword introspection (`temperature`, `topk`, `use_k_tokens`); the class attribute `HAS_AR_HEAD` (or, failing that, `flow_num_steps`) decides whether the token overrides apply at all |
| Settings read | `n_action_steps`, `n_obs_steps`, `temperature`, `topk`, `max_seq_len`, `flow_num_steps`, `device`, `variant`, `pi05_sha256`, `spm_path`, `spm_sha256`, `_tokenizer_metadata` |
| Capability flags | `requires_state_history`, `supports_history_summary_gate` |
| `--force-gate` | `set_history_gate_mode(mode)` if defined, otherwise the `history_gate_mode` attribute |
| Runner protocol | `reset`, `predict_action`, `record_executed_actions`, `get_observation_ports`, `get_policy_name`, `shape_meta` |

## 7. M5 bring-up results (2026-10-06, 2× RTX 4090)

| Check | Result |
|---|---|
| Fast suite, `-m "not requires_pi05 and not gpu"` (CPU, includes the gloo DDP smokes) | 181 passed |
| Real-asset/GPU suite, `-m "requires_pi05 or gpu"` | 37 passed, after seeding the live-render orientation test (below) |
| fp32 parity with LeRobot PI0.5 (`tests/test_p2n_vla_parity.py`) | Every stage bitwise on CPU; on CUDA, at most 9e-6 relative (SDPA kernel choice, cuBLAS 12.9 vs 12.8) |
| 2-GPU probe, gate variant, p = 0.5 | 15.94 GB peak reserved per GPU (16.6 GB in nvidia-smi), 8.66 GB headroom: **GO**. 11.2 samples/s |
| 2-GPU probe, gate variant, p = 0 | Same memory, 16.0 samples/s. With self-past at p = 0.5 (11.2 samples/s) after its ramp, a 30k-update run takes about 24 h |
| 2-GPU probe with in-training rollouts (`-- task.policy.lazy_eval=false`; 10 official episodes on rank 0) | GPU 0 peaks at 20.6 GB with the training state plus 10 renderers, leaving 5.1 GB headroom: **GO**. 10 episodes take 55 s |
| Batch-1 bf16 `predict_action` latency (p50) | 184 ms (prefix 39.7, decode 141.6, detokenize 2.8), within the 267 ms budget |
| Step-0 losses on the full model | `L_AR` 8.53–8.57 (ln 5001 = 8.52); `L_KI` 15.6–17.5 (see the note below) |
| 2-episode overfit, `p2n_vla`, 3,000 updates, `use_ema=false`, scored on its 506 training windows | Teacher-forced `L_AR` 0.013, token accuracy 0.998; greedy tokens 0.992, exact chunks 0.986; reconstruction MSE 9.3e-4 vs the OAT round-trip floor 7.9e-4 (3.37e-3 vs 3.21e-3 normalized); `predict_action` decodes exactly the greedy tokens |
| The same runs with EMA, 2,000 updates (both variants) | EMA snapshots are 0.96 teacher-forced and 0.70 exact chunks, at 4.4–5.5e-3 MSE. The EMA (decay 0.999) lags: EMA@2000 matches the live weights @1500 |
| LIBERO smoke, 10 corrected episodes on tasks 0 and 3 (the training tasks) | `p2n_vla` EMA@2000: 1/10. `p2n_vla_state_gate` EMA@2000: 2/10. No-EMA@3000: 5/10. Outcomes are identical across reruns and across GPUs |

**Fixes made during bring-up**
- `evaluate_p2n_vla.py` now renders on the policy's GPU. On this host the EGL indices are swapped, so the documented `MUJOCO_EGL_DEVICE_ID=<cuda index>` put the 5.5 GB of renderers on the other GPU. See section 5.
- `test_live_libero_render_matches_training_frame_orientation_and_quaternion_sign` was flaky. `LiberoEnv` seeds only after its construction-time reset, so fixture placement differed per process. The wrist camera's upright/best-flip error ratio ranged from 0.15 to 0.26 against a 0.25 margin. The test now seeds numpy and uses a 0.35 wrist margin (seeded ratio 0.23; a flipped frame gives about 4).

**Watch in M6**
- **KI start.** `L_KI` starts near 16 because pi05_base learned FAST tokens in the same embedding rows (skip = 1152). Its gradient norm (about 15–20) is clipped separately, and on the overfit it fell to 0.12–0.3.
- **Gate movement.** On the 2-episode overfit the learned gate barely moved: mean 0.900 → 0.912 → 0.900, with A→HIST attention mass 0.04–0.08. This is the plan's early-abort signal, so check it in the pilot.
- **One unexplained probe hang.** The first 2-GPU probe hung once: rank 0 spun in a NCCL collective after rank 1 had destroyed its process group. Three later probes passed, one under the same CPU load. Run M6 with the stack-dump hook (section 3.4).
- **SigLIP precision.** SigLIP runs entirely in bf16. LeRobot keeps it fp32 under autocast, and the difference is about 1e-2 relative on image tokens. Keeping it fp32 costs about 0.8 GB.
