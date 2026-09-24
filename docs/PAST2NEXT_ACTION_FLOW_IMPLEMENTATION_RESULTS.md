# Direct action-flow implementation and validation

Implemented 2026-09-24 from `PAST2NEXT_DITX_ACTION_FLOW_IMPLEMENTATION.md`.
Only new files were added; existing AR/latent-flow implementations and the
implementation plan remain unchanged.

## Launch

From `/workspace/ysk/past2next_bug_fixed`:

```bash
bash train_p2n_action_flow.sh \
  --gpus 2,3 \
  --batch-size 4 \
  --val-batch-size 4 \
  --grad-accum 8 \
  --save-every 20
```

This runs `p2n_action_flow` then `p2n_state_gate_action_flow`, each on both
selected GPUs, with separate timestamped output directories. Defaults use the
local DINOv3-S/16 snapshot, `nut_washer_v3_N77.zarr`, 2001 epochs, eight Euler
steps, and an effective training batch of 64. No tokenizer checkpoint is needed.
The launcher checks GPU availability immediately before each training run.
W&B logging defaults to online for fresh training and resume; the launcher also
exports `WANDB_MODE=online`. No extra logging flag is needed.

Use `--variant p2n_action_flow` or `--variant p2n_state_gate_action_flow` for
one variant. Add `--dry-run` for local CPU data/normalizer/DINO checks without
training. `--output`, `--dino`, `--task`, and `--resume` are supported; resume
requires one explicit variant and a full matching training checkpoint.
Overrides after `--` take precedence over convenience flags, for example:

```bash
bash train_p2n_action_flow.sh --variant p2n_action_flow \
  --gpus 2,3 --batch-size 4 --val-batch-size 4 --grad-accum 8 \
  -- training.num_epochs=100 logging.mode=online
```

## Implementation

- `oat/policy/p2n_action_flow*.py` and `p2n_state_gate_action_flow.py` implement
  normalized 16×7 continuous outputs, single inverse normalization, shared
  frozen image features with independent student/EMA adapters, self-past,
  summary-only gating, and acknowledged execution history.
- `oat/model/flow/ditx_action.py` provides the 16-layer, 768-wide, 12-head
  LayerNorm/GELU/AdaLN-Zero network with time/relative-step-only modulation.
  `action_euler_sampler.py` integrates in FP32 and reuses cross-attention KV.
- `oat/model/action_flow_state_history.py` isolates rotation6D history geometry
  from the legacy tokenizer policies. Direct policies do not import OAT/FSQ.
- `oat/dataset/action_flow_dataset.py` adds evaluation-only future masks and
  stable physical sample IDs to all four existing history dataset types.
- `oat/workspace/train_p2n_action_flow.py` fits train-frame normalizers once,
  initializes EMA after DDP synchronization, updates curricula/EMA/scheduler on
  successful optimizer updates, and saves complete rank-specific RNG state.
  Validation uses disjoint shards and globally reduced error sums/counts.
- Four new task/variant configs and the Python/Bash launchers keep this family
  separate. Strict artifacts embed DINO configuration/weights, normalizers,
  action semantics, student/EMA states and continuation metadata.

Shared consistency arithmetic and existing vision/history/execution helpers
are reused without editing them. The upstream reference is pinned to ManiFlow
commit `ef2f116f1f90163ed36e657b8c5503740bb468af`; this is a history/vision
extension with the capacity and training choices documented in the plan.

## Local validation

Both real-data Bash preflights passed. Dataset split: 73 training episodes,
4 validation episodes, 93,980 real training frames and 5,269 validation frames.
Normalizer fitting excludes validation, overlapping windows and padding.
The pinned local DINO configuration, weight digest and processor were checked.

All 73 CPU tests passed, including two isolated runs with every OAT tokenizer
import blocked. Tests cover both policy layouts, all four dataset/config combinations,
flow/consistency reference arithmetic, masked/closed memory containing NaNs,
BF16 network/FP32 arithmetic, static-cache equivalence, optimizer ownership,
normalizer round trips, offline restore, exact RNG continuation, execution
acknowledgements, and launcher overrides. All trainable parameters receive
nonzero gradients after the zero-initialized paths open over several updates.

```bash
/venv/real_robot/bin/python -m pytest -q tests/test_p2n_action_flow_*.py

/venv/real_robot/bin/python -m torch.distributed.run --standalone \
  --nproc_per_node=2 tests/p2n_action_flow_ddp_smoke.py --variant p2n_action_flow

/venv/real_robot/bin/python -m torch.distributed.run --standalone \
  --nproc_per_node=2 tests/p2n_action_flow_ddp_smoke.py --variant p2n_state_gate_action_flow
```

Both two-rank CPU smoke runs passed: three updates with accumulation and a
partial final group, no unused parameters, synchronized full EMA, uneven/empty
validation shards, and identical next-update loss/student/EMA hashes after
restoring student, EMA, Adam, scheduler, augmentation RNG and flow RNG streams.

Exact full real-robot architecture counts after fitting the normalizer:

| Variant | Trainable | Frozen | Total |
| --- | ---: | ---: | ---: |
| `p2n_action_flow` | 258,206,215 | 21,596,652 | 279,802,867 |
| `p2n_state_gate_action_flow` | 260,158,218 | 21,596,652 | 281,754,870 |

Each DiT-X has 238,919,687 parameters; frozen DINO has 21,596,544 and the
normalizer has 108. These counts use full CPU constructors and local DINO
architecture. CPU integration tests use a small network and a mocked backbone.

## Remaining empirical acceptance

No GPU probe, GPU training, model download or robot execution was run during
implementation. Full two-4090 BF16 peak memory, actual training, held-out action
quality, two/eight-step latency and robot closed-loop quality remain unmeasured.
The opt-in DDP smoke script has a `--real --dino PATH` mode that exercises the
full network, CT/self-past/Adam/EMA, expert/generated validation, and both
sampling step counts, reporting synchronized p50/p95 timings and memory.
Deployment defaults remain eight steps; two-step quality must be measured.

Periodic checkpoint saving defaults to every 20 completed epochs. Use
`--save-every N` to set both the latest-checkpoint and numbered-snapshot
intervals, including when resuming. For `--save-every 20`, numbered files
are `checkpoints/ep-0020.ckpt`, `ep-0040.ckpt`, and so on. `latest.ckpt`
is also saved at the end of training; metric-ranked best checkpoints can
still be saved between these intervals. Existing resumes retain their saved
intervals unless `--save-every` or explicit Hydra overrides are supplied.
