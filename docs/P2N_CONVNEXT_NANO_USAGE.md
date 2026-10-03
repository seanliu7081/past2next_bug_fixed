# Additive ConvNeXt V2-Nano implementation

The implementation adds new modules, configurations and entry points. Existing
DINO policies, scripts and configuration files remain unchanged. Use
`scripts/train_p2n_new_convnext.py` to select the new backend; the original
`scripts/train_p2n_new.py` does not gain new arguments.

The two new configurations inherit their corresponding real-robot configuration:

| Variant | Configuration under `oat/config/experimental` |
| --- | --- |
| `p2n_new` | `train_p2n_new_convnext_nano_real_robot.yaml` |
| `p2n_state_gate_new` | `train_p2n_state_gate_new_convnext_nano_real_robot.yaml` |

Both retain 77 demonstrations, seed 42, validation ratio 0.05, the selected OAT
checkpoint, 2001 epochs and the original per-rank batch/accumulation defaults.
The policy class names and variants remain `P2NNewPolicy`/`p2n_new` and
`P2NStateGateNewPolicy`/`p2n_state_gate_new`; their new import module is
`oat.policy.p2n_new_convnext`. The additive workspace is
`oat.workspace.train_p2n_new_convnext.TrainP2NNewWorkspace`.

The AR remains 16 layers, width 768, 12 heads and SwiGLU intermediate width 2048.
The selected frozen OAT must produce eight latent action tokens. Visual context
has 256 tokens at width 768, current state has two tokens, and complete context
contains 267 tokens for the base policy or 271 for the state gate policy.
The original seven-action history, gate formula, execution confirmation protocol,
self-past sampling behavior and normalizer ownership are preserved.

## Local dependencies and weights

Use the repository's established Python environment plus the additional pinned
dependency in `requirements-convnext-nano.txt`. The implementation was inspected
with `timm==1.0.29`. No dependency installation, weight download or training run
is performed by a configuration dry-run.

Fresh Nano construction requires an explicit local weight file or local snapshot
directory for `convnextv2_nano.fcmae_ft_in22k_in1k`, plus its pinned 40-character
hexadecimal source revision.
The loader checks backbone key coverage, tensor shapes and metadata; only the
known classifier keys can be removed. Missing weights never select a random
frozen backbone. Artifacts record the revision, SHA256, timm version, construction
parameters and deterministic preprocessing recipe. Snapshot directories prefer
`model.safetensors`, then `pytorch_model.bin`; otherwise exactly one `.pt`, `.pth`,
`.bin` or `.safetensors` weight file is required. Refer to
`oat/perception/convnext_feature_encoder.py` for format checks.

The Bash launcher uses the frozen tokenizer selected by the user from
[the base P2N run](https://huggingface.co/SeanLiu0272/nut-washer-v3-N77-so3aug-20260924/tree/9f8a05efea09baeb1f5fa304cf104faa3b5b0b57/nut_washer_v3_N77_p2n_so3aug_20260924_081003_468366670).
Its `tokenizer_selection.json` selects epoch 1540 (`test_reconst_mse`
`3.0012812203494832e-05`). The downloaded `frozen_tokenizer.ckpt` and that
epoch checkpoint have the same published SHA256:
`11213d4fff6c5efb63f80b4d46852acf348dc2b4967e25a0701fb664f43ac6c7`.
The local file is:

```text
/workspace/models/nut-washer-v3-N77-so3aug-20260924/9f8a05efea09baeb1f5fa304cf104faa3b5b0b57/nut_washer_v3_N77_p2n_so3aug_20260924_081003_468366670/frozen_tokenizer.ckpt
```

Both variants receive this checkpoint through the Bash launcher's `--tokenizer`
argument. To choose another checkpoint, pass `--tokenizer /absolute/path.ckpt`
to either launcher. Direct Python invocations must explicitly select this local
checkpoint; the inherited configuration still points to the original gated-run
checkpoint. Fresh preflight verifies EMA weights, action schema and dataset/split
provenance before training.

## Configuration-only dry-runs

Run from the repository root. These commands work before pretrained weights are
available and do not open model files, the dataset or GPU devices:

```bash
python scripts/train_p2n_new_convnext.py \
  --variant p2n_new --task real_robot --vision convnext_nano --dry-run

python scripts/train_p2n_new_convnext.py \
  --variant p2n_state_gate_new --task real_robot --vision convnext_nano --dry-run
```

Use `--convnext /absolute/local/path --convnext-revision PINNED_REVISION` to include
the chosen source in the resolved configuration. Hydra overrides after `--` take
precedence over generated CLI overrides, for example:

```bash
python scripts/train_p2n_new_convnext.py \
  --variant p2n_new --task real_robot --vision convnext_nano --dry-run \
  -- dataloader.batch_size=8 training.gradient_accumulate_every=4
```

The new launcher defaults to `--vision dinov3` and delegates those arguments to
the unchanged original launcher. `--dino` and `--dino-revision` retain their
existing meaning. Nano rejects conflicting DINO sources, unsupported LIBERO
configurations and changes to the fixed visual/AR recipe.

## Preflight and training

The Bash launcher accepts `--wandb-mode online` to stream both variants' metrics
to W&B during training. The default is `offline`; `disabled` turns W&B logging
off. Online mode uses the account authenticated in the training environment;
if needed, run `/workspace/venvs/starvla-heading/bin/wandb login` once. Dry-run
and preflight do not initialize W&B or upload metrics.

```bash
bash scripts/train_p2n_new_convnext_frozen.sh \
  --batch-size 64 --gpu 0,1,2,3 --gradient 1 \
  --convnext-frozen true --num-epochs 2001 --wandb-mode online
```

Preflight constructs the selected policy on CPU, checks local Nano/OAT weights,
dataset schema, split and optimizer contracts, and reports actual parameters:

```bash
python scripts/train_p2n_new_convnext.py \
  --variant p2n_new --task real_robot --vision convnext_nano \
  --tokenizer /absolute/local/frozen_tokenizer.ckpt \
  --convnext /absolute/local/nano_snapshot \
  --convnext-revision PINNED_REVISION \
  --output output/training/p2n_new_convnext_nano_real_robot_seed42 \
  --preflight
```

For an explicitly requested training run, omit `--preflight` and select idle
devices with `--devices 0,1 --num-processes 2` (indices are from `nvidia-smi`). Repeat with
`--variant p2n_state_gate_new` and a distinct output directory for the gate model.
The launcher checks selected GPUs for existing compute processes, resolves
selected indices to UUIDs, performs CPU preflight, then starts DDP. It leaves
existing processes untouched and fails if the selected devices are busy. The
default output name and logging tags include `convnext_nano`.

Resume a complete new training checkpoint using `--resume CHECKPOINT`, retaining
the matching Nano configuration and training contract. Restore uses the embedded
architecture and state, so original Nano and tokenizer source paths are not
required. Encoder/backend, preprocessing, fusion and resampler changes fail
resume/override checks. DINO-to-Nano and base-to-gate resume are rejected. Existing
DINO entry points remain available for old artifacts.

## Verification status and parameter counts

Configuration and synthetic contract tests run without loading the selected OAT
checkpoint or acquiring pretrained Nano weights. CPU tests cover both variants,
strict artifact restoration, preprocessing, gradients, optimizer groups and
launcher configuration. They do not establish
pretrained-weight quality, real-weight dual-GPU training or closed-loop task
acceptance. Run the new checks with the repository Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_p2n_new_convnext_*.py
```

With the pinned production architecture, the feature-only backbone contains
14,981,520 parameters. Trainable fusion contains 246,528; the narrow resampler
contains 2,254,080; the output projection contains 197,376. These sum to 2,697,984
adapter parameters, plus 3,072 camera/frame embedding parameters; the current-state
projection is counted separately. These counts describe modules, not a whole-policy
reduction or measured training speedup. Hardware performance and task success
remain unverified until the explicit real-weight acceptance runs are completed.
