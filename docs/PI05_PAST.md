# Pi05_past: pi05's flow head with the Past2Next condition

Pi05_past keeps pi05_base's original flow-matching action head and adds the Past2Next condition of
`oat/config/train_past2next_self_past.yaml`: the 7 past commands, their acceleration and jerk, and self-past
training. Variant `pi05_past`, class `oat.policy.pi05_past.Pi05PastPolicy`, config
`oat/config/train_pi05_past.yaml`, launcher `train_pi05_past.sh`. Its sibling with the Past2Next AR token head
instead of the flow head is pi05_p2n ([PI05_P2N.md](PI05_P2N.md)).

## Design

| Part | Pi05_past |
|---|---|
| Observation and prompt | pi05 as is: agentview + wrist cameras at 224 px, "Task: ..., State: <8 bins>" prompt, one frame |
| Action head | pi05_base Gemma-300M flow expert with unfolded adaRMS and its flow heads, unchanged. Flow time ~ Beta(1.5, 1), 10 Euler steps, horizon 16, execute 8 |
| Condition features | `Past2NextPolicy._build_condition`: past commands normalized with the dataset action normalizer, zero padding normalized without masking, `acc = a[t-1]-a[t-2]`, `jerk = a[t-1]-2a[t-2]+a[t-3]`, tokens `[acc, jerk, raw x7]` through Linear-GELU-Linear MLPs, a zero-initialised slot embedding, dropout 0.1 in training |
| How the head reads them | Once per sample the 9 tokens run through the expert with self-only attention at a constant time `cond_t0` and one shared RoPE position after the chunk. Their per-layer K/V are appended to the VLM K/V, so every Euler step of the unchanged flow head reads `[prefix, condition]` keys. Masking those columns reproduces plain pi05 |
| Self-past | p 0.5 per sample with a valid previous window, warmup 1000 updates, no ramp. Generated with the deployment sampler, so self-past and rollout histories match |
| VLM regime (default) | pi05 fine-tune style: the flow loss reaches SigLIP (trainable, fp32 masters) and Gemma LoRA r32 through un-detached K/V; no KI loss. `kv_grad_scale=0` gives the insulated regime |

Deliberate departures from the original condition:
- One observation frame instead of two (pi05's input), so the comparison tests past commands only.
- The expert's own layers replace the original's shared Mish memory encoder.
- The 9 condition keys share the expert's attention with the VLM keys instead of a separate cross-attention.

Implementation notes:
- Without the KI block, the last Gemma layer's q/o/MLP LoRA adapters get no gradient. They are frozen
  (`policy.dead_lora`), otherwise DDP's unused-parameter check fails at step 1.
- Optimizer groups: `pretrained` (`optimizer.policy_lr`) = expert, flow heads, SigLIP; `new`
  (`optimizer.new_module_lr`) = condition modules, LoRA, KI table. Clip groups: `ki` = SigLIP, LoRA, KI table;
  `ar` = everything else.
- `use_ema=false` by default: an fp32 EMA of the 0.885 B trainable parameters does not fit next to training at
  micro-batch 4 on a 24 GB GPU. Snapshots are `snapshots/upd-NNNNNN_model.ckpt`.
- The condition MLPs and the condition tokens' expert pass run in fp32, so autocast does not keep another
  bf16 copy of the expert weights for backward. Their K/V are cast to the prefix dtype where they join it.
- `train/hist_attention_mass` is the attention mass the action tokens put on the 9 condition keys, averaged over
  every action row, head and layer.
- A fractional `kv_grad_scale` scales the VLM gradient in the K/V dtype (bf16), so it is exact only for powers
  of two. The forward pass is the same for every value.
- The training workspace runs DDP's one-time gradient-bucket rebuild before the first optimizer step instead of
  in the next forward. Done by DDP, the rebuild holds two fp32 gradient copies on top of the AdamW state, which
  raised the peak from 20.06 to 21.25 GiB. On resume, the loaded AdamW state is kept on the host during the
  rebuild. Resumed runs also define zero gradients before DDP wraps the model, so the first update accumulates
  into DDP's buckets instead of into a second gradient copy. Without that, a resumed run needs about 23.4 GiB and
  fails on a 24 GB GPU. Both changes apply to every P2N-VLA variant and leave training bitwise unchanged.

## Train

Flags override the configs; `bash train_pi05_past.sh --help` lists all of them. Flags take `--flag VALUE` or
`--flag=VALUE`, relative paths are relative to where you launch, and other Hydra overrides go after `--`.
`--devices` is required to train or probe.

```bash
cd /workspace/ysk/past2next_bug_fixed
bash train_pi05_past.sh --dry-run                            # resolve and validate the config
bash train_pi05_past.sh --devices 0,1,2,3 --probe            # memory/throughput go/no-go, writes output/probe/...
bash train_pi05_past.sh --devices 0,1,2,3                    # Pi05_past, 80k updates of batch 32 (~3 days)
bash train_pi05_past.sh --devices 0,1,2,3 --use-past false   # pi05 no-past control, same recipe
bash train_pi05_past.sh --devices 0,1,2,3 --self-past-p 0    # Pi05_past with ground-truth history only
bash train_pi05_past.sh --devices 0,1,2,3 --lr 3e-5 --batch 64 \
    --output output/training/pi05_past_lr3e-5_b64 --wandb-name pi05_past_lr3e-5_b64   # any knob; name the run
```

Defaults: 32 samples per update on 1, 2, 4 or 8 GPUs (`--batch`; micro-batch 4 per GPU, the most a 24 GB GPU
holds, and gradient accumulation 32 / (4 × GPUs): 2 on 4 GPUs, 4 on 2 GPUs), 1k warmup, cosine to 0.1× at 80k
updates (2.56M samples, about 20 passes over the 124k training windows), LR 5e-5 (expert, flow heads, SigLIP)
and 1e-4 (condition modules, LoRA), W&B online (training every update, validation every 5k updates, LIBERO
success every 10k).

Every 10k updates GPU 0 runs the official LIBERO-10 protocol (500 episodes, 10 environments, the same schedule
as `scripts/evaluate_p2n_vla.py --protocol official --n-test 500 --episode-start-seed 3000`) while the other
GPUs wait. It takes up to an hour, which is within the 100-minute watchdog. The simulators' renderers need about 5 GB there, so the
AdamW state waits in host memory meanwhile (the probe measured 5.8 GB headroom). Results go to
`eval/rollout_epoch-EEEE_upd-NNNNNN/` and to W&B as `rollout/*`. Only two checkpoints stay on disk: the
snapshot of the best evaluation so far (`snapshots/upd-NNNNNN_model.ckpt`, named in `snapshots/best.json`;
the earlier one wins a tie) and `checkpoints/latest.ckpt` (the latest weights, every 1k updates, resumable).
A snapshot whose evaluation failed is kept for an offline evaluation. `--sim-eval-every`,
`--sim-eval-episodes` and `--keep-snapshots all` change this. Default run directories are `output/training/pi05_past[_nopast|_gtpast]_libero10_seed<SEED>`. The seed, `use_past`,
`self_past_p`, the step count and the batch have flags and are refused after `--`, because they name the run
directory and set the schedule.

Measured with `--probe` on RTX 4090s (24 GB) at batch 32:

| Arm | GPUs | Peak allocated / reserved per GPU | Throughput | Per 40k updates |
|---|---|---|---|---|
| Pi05_past | 4 | 20.06 / 20.27 GiB (gate passes, 2.85 GB headroom) | 10.4 samples/s | ~34 h, ~136 GPU-h |
| Pi05_past | 2 | 20.06 / 20.27 GiB (gate passes, 2.85 GB headroom) | 8.3 samples/s | ~43 h, ~86 GPU-h |
| pi05 control (`--use-past false`) | 2 | 19.78 / 19.84 GiB (3.31 GB headroom) | 13.1 samples/s | ~27 h, ~54 GPU-h |
| ground-truth past (`--self-past-p 0`) | 2 | 20.06 / 20.27 GiB (2.85 GB headroom) | 11.7 samples/s | ~30 h, ~61 GPU-h |

One run on 4 GPUs is only 1.26× faster than on 2: all-reducing 3.3 GiB of fp32 gradients across the 4090s
takes about 1 s per update. Two 2-GPU runs side by side get more done.

Validation (every 50 epochs) and checkpoint writes add a little to these times; evaluation is separate.
Disk: an 11 GB `checkpoints/latest.ckpt` (another 11 GB free while it is rewritten) and one or two 1.8 GB
snapshots. With `--keep-snapshots all`, each periodic snapshot adds 1.8 GB.

The 4-GPU run takes about 3 days for 80k updates: ~62 h of training at ~2.8 s per update once self-past starts
(11.4 samples/s in the live run), plus eight LIBERO evaluations of up to an hour each. A resume must keep the
retention mode, and under `best` also the evaluation settings, because the best snapshot is chosen across every
evaluation of the run.

Resume from the latest checkpoint with the same GPU count and the original flags. The W&B run id is read from
the run directory (`<run>/wandb/latest-run`), or pass `--wandb-id`:

```bash
bash train_pi05_past.sh --devices 0,1,2,3 --resume output/training/pi05_past_libero10_seed42/checkpoints/latest.ckpt
```

## Evaluate

Evaluate snapshots standalone, one GPU each, always with 10 parallel environments: the initial states of
KITCHEN_SCENE4/6/8 depend on the worker schedule, so paired comparisons need the same `--n-parallel-envs`.
A training run uses about 20 GiB of each of its GPUs, so evaluate on GPUs that are not training.

The run scores the official 500 episodes at every 10k updates (`eval/rollout_*/summary.json`) and keeps the
best snapshot. Choosing it on those episodes makes its official score optimistic. Its unbiased score comes from the
corrected protocol on fresh seeds. To re-score a snapshot exactly as training did (for example one whose
in-training evaluation failed), add `--seed 44` (`training.rollout_seed`) to
`--protocol official --n-test 500 --episode-start-seed 3000`.

```bash
RUN=output/training/pi05_past_libero10_seed42
# best.json names the file (runs started before Oct 26 15:11 UTC store only the absolute "snapshot" path)
SNAP=$(/venv/oat/bin/python -c "import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); d = json.load(open(p)); print(p.parent / d.get('file', pathlib.Path(d['snapshot']).name))" $RUN/snapshots/best.json)
# final test: corrected 1500 (three runs of 500 keep each worker's reset count <= 50), with the command log
for START in 5000 5500 6000; do
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py --snapshot $SNAP --weights model \
      --protocol corrected --n-test 500 --seed 46 --episode-start-seed $START --n-parallel-envs 10 --log-commands
done
# the final weights: checkpoints/latest.ckpt (no _model snapshot of them is kept unless they were the best)
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py \
    --snapshot $RUN/checkpoints/latest.ckpt --weights model --protocol corrected --n-test 500 --seed 46 \
    --episode-start-seed 5000 --n-parallel-envs 10 --log-commands
```

### Copycat diagnostics (the best and the final weights of every arm)

Closed loop, from `--log-commands`: `commands.npz` holds every episode's executed commands with chunk
boundaries. `command_stats.jsonl` and `closed_loop` in `summary.json` give these, split by success and failure:

| Statistic | Meaning |
|---|---|
| `gripper_toggles` | gripper toggles per episode (the demonstrations average 3.47) |
| `boundary_ratio` | arm-command jump across chunk boundaries over the step within chunks |
| `final_motion` | arm-command size over the last 50 steps; low in stall-ending failures |
| `near_padding_fraction` | executed commands within 0.15 (normalized, max-abs) of the zero padding of a missing history slot; alarm above 1% |

History interventions change only what the head reads in the evaluator. The executed and recorded commands
stay the same, and training never sees them. Compare each against the plain run on the same episodes:

```bash
COMMON="--snapshot $SNAP --weights model --protocol corrected --n-test 200 --seed 45 --episode-start-seed 4000 --n-parallel-envs 10"
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py $COMMON --log-commands   # the plain run
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py $COMMON --history-intervention zero
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py $COMMON --history-intervention swap
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py $COMMON \
    --history-intervention noise --history-noise-std 0.1
```

`zero` shows the head no history, `swap` gives it the previous environment's history, and `noise` adds Gaussian
noise in normalized units to the valid commands. Output folders get a `_hist-<mode>` suffix.

Offline, on the run's held-out windows (about 2 minutes on one GPU), with the same sampling noise in every
condition:

```bash
CUDA_VISIBLE_DEVICES=1 /venv/oat/bin/python scripts/pi05_past_offline_diagnostics.py --snapshot $SNAP --weights model
```

It writes `<run>/eval/<snapshot>_model_offline_diagnostics.json` with these fields:
- `reliance.ratio`: the change of the executed chunk under a history swap, over its change under an
  observation swap.
- `gripper_toggles`: recall of demonstrated toggles within 2 steps, the timing error, and the false-toggle rate.
- `innovation_r2`: R² of `a[t+i] - a[t-1]` for each of the 16 chunk positions. The demonstrations copy the last
  command (R² 0.98 at position 0), so the plain R² and a first-action copy rate are uninformative.

## Experiment plan

| Run | Flags | Purpose | 2-GPU time at 80k updates (incl. 8 evaluations) |
|---|---|---|---|
| Pi05_past | (defaults) | the design | ~4 days, ~185 GPU-h |
| pi05 control | `--use-past false` | effect of the Past2Next condition on the flow head | ~2.5 days, ~120 GPU-h |
| ground-truth past | `--self-past-p 0` | effect of self-past | ~2.8 days, ~135 GPU-h |
| seed replicates | `--seed 43` for the first two | pre-committed unless the first gap is ≥ 6 points on both schedules | ~305 GPU-h |

GPUs 0-3 are available. The first run is Pi05_past alone on all four (~3 days, since Oct 26 14:30 UTC). For the
comparison, run every arm with the same batch and GPU count; two 2-GPU arms side by side get more done than 4-GPU
arms one after another, because one run gains only 1.26× from the second pair of GPUs.

Decision rules, fixed before launch, on 2,000 episodes per run (the best snapshot's official 500 from training,
which carries the same selection bias in every arm, and its corrected 1,500 on fresh seeds):
adopt Pi05_past only if both seeds agree in sign and the pooled gain is ≥ 3 points with p < 0.05 and the
official-only sign agrees. If the control wins by ≥ 4 points, investigate copycat behaviour. Within 2 points,
prefer the control. Keep self-past only if it beats ground-truth past by ≥ 3 points.

Add an arm with whole-history dropout (q = 0.3) or history noise if Pi05_past's toggle recall is ≥ 10 points
below the control's, or if it has at least twice the control's stall-ending failures.

Offline validation feeds ground-truth history and favours Pi05_past by construction: compare runs only in
closed loop.

## Tests

```bash
/venv/oat/bin/python -m pytest tests/test_pi05_past.py -q -m "not slow and not gpu"   # 20 tests, CPU, ~20 s
/venv/oat/bin/python -m pytest tests/test_pi05_past.py -q -m slow                      # 4 tiny workspace runs, ~40 s
CUDA_VISIBLE_DEVICES=0 /venv/oat/bin/python -m pytest tests/test_pi05_past.py -q -m gpu   # bf16 CUDA path
```
