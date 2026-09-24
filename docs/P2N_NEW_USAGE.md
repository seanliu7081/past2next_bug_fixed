# p2n_new and p2n_state_gate_new

These are new, independent implementations of the DINOv3-S/16 plan. Existing
policies, configurations, launchers, and their execution behavior are unchanged.
`p2n_new` has no state-history encoder, observation-summary gate, or summary
parameters. `p2n_state_gate_new` additionally consumes eight measured states and
adds four gated history summaries. Both accept explicit action validity, use
self-past, and record online history only after execution acknowledgement.

The default architecture is a frozen DINOv3-S/16 backbone and frozen OAT tokenizer,
64 visual queries per camera/frame, two Resampler layers, and a 16-layer,
768-dimensional action decoder. Two cameras and two frames produce 267 condition
tokens for `p2n_new` and 271 for `p2n_state_gate_new`.

## Configuration checks without training

Choose an interpreter explicitly; the launch script does not switch Python
installations, download models, or install dependencies. The inspected environment
is `/venv/real_robot/bin/python`. `--dry-run` only resolves configuration. It
requires no DINO snapshot, tokenizer checkpoint, dataset, CUDA context, or output
files, and makes no claim about instantiated parameter counts.

All four configuration entries can be checked with:

```bash
bash train_p2n_new.sh --python /venv/real_robot/bin/python \
  --variant p2n_new --task libero --dry-run

bash train_p2n_new.sh --python /venv/real_robot/bin/python \
  --variant p2n_state_gate_new --task libero --dry-run

bash train_p2n_new.sh --python /venv/real_robot/bin/python \
  --variant p2n_new --task real_robot --dry-run

bash train_p2n_new.sh --python /venv/real_robot/bin/python \
  --variant p2n_state_gate_new --task real_robot --dry-run
```

Arguments following `--` are Hydra overrides and retain their values. For example,
`-- dataloader.batch_size=4 training.gradient_accumulate_every=8
training.num_epochs=10 val_dataloader.batch_size=2` is valid. The launcher never
changes these values based on a configuration name. `--devices 2,3` sets
`CUDA_VISIBLE_DEVICES`; it does not reserve GPUs or stop existing work.
`--num-processes` defaults to two. For one GPU, choose `--num-processes 1` and
`training.gradient_accumulate_every=8` to retain the default effective batch 64.

## Local weights and preflight

Fresh runs require a local, authorized snapshot of
`facebook/dinov3-vits16-pretrain-lvd1689m`. Supply the exact commit revision with
`--dino-revision`, or use a Hugging Face `snapshots/<40-character-commit>` directory.
The DINO loader validates the local architecture and processor configuration and
does not substitute DINOv2 or randomly initialize a missing pretrained backbone.

A full CPU preflight constructs the actual model, checks local tokenizer
provenance and dataset schema/split, and prints total/trainable/frozen parameter
counts, optimizer groups, context metadata, and the planned update schedule:

```bash
bash train_p2n_new.sh --python /venv/real_robot/bin/python \
  --variant p2n_state_gate_new --task real_robot \
  --dino /path/to/dinov3-s-snapshot --dino-revision YOUR_EXACT_COMMIT \
  --output output/training/p2n_state_gate_new_nut_washer_seed42 \
  --devices 2,3 --num-processes 2 --preflight
```

LIBERO requires `--tokenizer /path/to/libero-tokenizer.ckpt`; matching action
width alone is insufficient. Preflight checks dataset identity, action field,
action horizon, split seed, validation ratio, maximum training episodes, and EMA
availability against the tokenizer checkpoint. Changing those fields requires a
compatible tokenizer. Images remain byte-range RGB and bypass the old RGB
normalizer; policy state/history normalization and the frozen tokenizer's own
action normalization remain separate.

The real-robot defaults explicitly select:

- Dataset: `/workspace/ysk/zarr/nut_washer_v3_N77.zarr`.
- 77 episodes, seed 42, validation ratio 0.05.
- Two 128×128 cameras; rot6d observations with row layout; 7-dimensional commands.
- Horizon 16, execution prefix 8, past commands 7, two observation frames.
- The fixed EMA tokenizer at
  `output/training/nut_washer_v3_N77_gated_so3aug_20260924_081008_317043427/tokenizer/checkpoints/ep-1540_mse-0.000.ckpt`.
- 2001 epochs for both real-robot variants; LIBERO retains 251 epochs and a 0.1
  validation ratio. Explicit epoch overrides take precedence.

Read-only checks on the available real dataset found 73 training episodes / 93,980
windows and four validation episodes / 5,269 windows. Validation episode indices
are `[6, 33, 49, 58]`. Both real-robot configurations passed tokenizer-provenance
checks. Neither real-robot configuration constructs a simulator runner; offline
loss and reconstruction metrics do not measure robot task success.

## Training and continuation

Removing `--preflight` from a command launches training after the same preflight
checks pass. Fresh output directories must be empty. No training of the full
model was launched during implementation. Each variant uses its own output,
logging name, checkpoints, optimizer, EMA, and curriculum counter.

Defaults use microbatch eight per rank, two ranks, accumulation four, validation
microbatch four, and self-past generation chunks of four. Warmup is computed as
5% of planned successful optimizer updates after distributed sharding. The count
includes incomplete accumulation groups at epoch ends and respects
`training.max_train_steps`. LR, EMA, and self-past advance only after a successful
optimizer update. A final latest checkpoint is saved even when an explicit epoch
limit does not coincide with the periodic checkpoint interval.

Resume requires an explicit compatible artifact:

```bash
bash train_p2n_new.sh --python /venv/real_robot/bin/python \
  --variant p2n_state_gate_new --task real_robot \
  --resume output/training/p2n_state_gate_new_nut_washer_seed42/checkpoints/latest.ckpt \
  --output output/training/p2n_state_gate_new_nut_washer_seed42 \
  --devices 2,3 --num-processes 2 --preflight
```

Resume and deployment reconstruct DINO and OAT from structure embedded in the
artifact, then strictly load embedded weights. Their original external weight
paths are unnecessary. Training continuation additionally restores optimizer,
EMA, scheduler, completed epoch/batch/update counts, and per-rank Python/NumPy/
PyTorch RNG. Resume requires the original world size, matching variant and
architecture, matching data configuration, and identical saved episode masks and
action/episode-boundary digest. It rejects old policy checkpoints and cross-variant
resumes. Changing the data path is treated as a new data configuration.

## Deployment and execution acknowledgement

Use the **new policy class** loader. Existing `BasePolicy.from_checkpoint` remains
unchanged and is not the new artifact restore entrypoint:

```python
from oat.policy.p2n_new import P2NNewPolicy
from oat.policy.p2n_state_gate_new import P2NStateGateNewPolicy

policy = P2NNewPolicy.from_checkpoint("/path/to/p2n_new.ckpt", weights="ema")
# For the gate artifact, use P2NStateGateNewPolicy.from_checkpoint(...).
policy = policy.to("cuda").eval()
policy.reset()

prediction = policy.predict_action(observations)
# Control code executes some prefix and reports commands actually submitted.
policy.record_executed_actions(actual_commands, executed_lengths=actual_lengths)
```

Do not make another stateful prediction while acknowledgement is pending. A zero
execution length leaves history invalid; a confirmed zero-valued command is valid.
Reset at episode boundaries and before changing the environment batch size. The
gate variant's measured state history is supplied by the controller at every
control step; predicted actions never create synthetic measured states.

For stateless offline calls, pass both `past_actions` and `past_action_valid`.
Neither one alone is accepted, and those calls do not alter online execution
history. For LIBERO, the new runner adapters preserve the old simulator rollout
implementation while restoring RGB bytes and Boolean validity after the old
runner's floating-point conversion. They submit confirmed execution lengths via
the existing execution adapter.

## Verification boundary

`tests/test_p2n_new_integration.py` checks all four configurations, capability
fallbacks/explicit disabling, validity alignment, update counts, capped
accumulation tails, new artifact restore, and a tiny CPU training loop through
the new workspace. The tiny loop verifies four updates from two capped epochs
and the final latest checkpoint. Continuation from the first epoch checkpoint
matches uninterrupted weights, EMA, scheduler, counters, and RNG exactly; a
simulated skipped optimizer update leaves the scheduler, EMA, and curriculum
unchanged. It is not a run of the 16×768 production model.

Before long training, complete real-DINO/OAT single-device forward/backward,
optimizer-initialized memory measurements, generated-history validation, and a
two-rank communication/self-past smoke. Record peak allocated and reserved memory
separately. A CPU preflight or configuration dry run cannot establish GPU memory
capacity, throughput, convergence, or robot success rate.

The repeatable CPU/gloo smoke additionally runs two ranks with
`find_unused_parameters=False`, two AdamW updates for each variant, expert and
generated self-past, and checks every trainable gradient for presence/finiteness:

```bash
/venv/real_robot/bin/python -m pytest -q tests/test_p2n_new_integration.py
/venv/real_robot/bin/python tests/p2n_new_ddp_smoke.py
```

That smoke uses small real OAT/FSQ, decoder, and Resampler modules plus an explicit
DINO test double. It does not replace production-width pretrained GPU validation.

Production-width parameter counts were instantiated on CPU with the actual
selected OAT EMA and training normalizer. The DINO backbone was constructed from
its exact architecture for counting; approved pretrained DINO weights were not
available. The [machine-readable report](P2N_NEW_PARAMETER_COUNTS.json) records:

| Real-robot variant | Total parameters | Trainable | Frozen |
|---|---:|---:|---:|
| p2n_new | 201,571,654 | 174,170,112 | 27,401,542 |
| p2n_state_gate_new | 203,523,657 | 176,122,115 | 27,401,542 |

The frozen count includes DINO, OAT, and 144 policy normalizer elements. The
additional gate/history/pooling structures account for 1,952,003 trainable
parameters. These are instantiated architecture counts, not memory or task
performance measurements.

After the approved local DINO snapshot is supplied, the bounded real-weight probe
can check a complete forward/backward, Adam-initialized memory, expert/generated
validation, and complete prediction latency without launching long training:

```bash
/venv/real_robot/bin/python scripts/smoke_p2n_new.py \
  --variant p2n_new --task real_robot \
  --dino /path/to/dinov3-s-snapshot --dino-revision YOUR_EXACT_COMMIT \
  --device cuda:0 --batch-size 8 --val-batch-size 4 \
  --output /tmp/p2n_new_gpu_probe.json
```

Repeat with `--variant p2n_state_gate_new` for the independent gate variant. For
distributed probing, use `torchrun --nproc_per_node=2` and add `--distributed`.
Select currently free devices explicitly in `CUDA_VISIBLE_DEVICES`. No pretrained
GPU probe was run as part of this implementation.

Architecture-only counting and the two-rank real-weight probe have explicit
entrypoints:

```bash
/venv/real_robot/bin/python scripts/smoke_p2n_new.py \
  --variant p2n_new --task real_robot --count-only \
  --output /tmp/p2n_new_parameter_count.json

CUDA_VISIBLE_DEVICES=2,3 /venv/real_robot/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/smoke_p2n_new.py \
  --variant p2n_state_gate_new --task real_robot \
  --dino /path/to/dinov3-s-snapshot --dino-revision YOUR_EXACT_COMMIT \
  --device cuda --distributed --batch-size 8 --val-batch-size 4 \
  --output /tmp/p2n_state_gate_new_ddp_probe.json
```

The device numbers above are examples, not reserved GPUs. `--count-only` uses no
DINO pretrained weights and does not validate their behavior. Both variants still
require their own pretrained production smoke and memory acceptance before a long
training run.

Final implementation checks passed: 64 tests across the five new test modules,
55 selected legacy generation/execution/gate/EMA regression tests, and the
separate two-rank CPU DDP smoke. Run the new suite with:

```bash
/venv/real_robot/bin/python -m pytest -q tests/test_p2n_new_*.py
```

The installed Transformers DINOv3 implementation also produced finite
`[1, 196, 384]` patch features in a CPU architecture test with explicitly synthetic
weights. Approved pretrained DINO weights were unavailable locally, so full-model
pretrained GPU steps, memory/latency measurements, and task training/evaluation
remain pending. No production training was started.
