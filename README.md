# Past2Next: 015, 043 and 046

This checkout retains the three LIBERO-10 policy versions and the shared code needed
to train, load and evaluate them. Their recorded settings and 20 historical results
are in [COMPARISON_015_043_046.md](COMPARISON_015_043_046.md).

| Version | Training config | Initialization | Train / validation demos | Policy / vision LR | Task residual LR | Generated-history temperature |
|---|---|---|---|---|---|---|
| 015 | `train_past2next_finetune` **with overrides below** | Original Aug-24 EMA | 450 / 50 | `1e-5 / 1e-5` | — | 0 |
| 043 | `train_past2next_finetune_all500` | 015 epoch 9 EMA | 500 / 0 | `1e-5 / 1e-5` | — | 1 |
| 046 | `train_past2next_finetune_tasklr` | 015 epoch 9 EMA | 450 / 50 | `1e-5 / 2e-6` | `1e-3` | 1 |

All three use two observation frames, agent/wrist images with 112×112 crops, seven
past action commands, explicit acceleration/jerk features, and an autoregressive
transformer with eight blocks, eight heads and width 256. The frozen Aug-18 action
tokenizer has FSQ levels `[8,5,5,5,5]` and eight ordered tokens; it decodes 16 actions,
of which eight are executed. Version 046 adds a zero-initialized `10×138` task table
to the fused observation features while retaining the scalar task UID.

The retained config dependency chain is `finetune → improve → self_past →
libero10_with_prev_window`; 043 and 046 inherit `finetune`. The SO(3) tokenizer
training config and its LIBERO task config are also retained. Shared model, dataset,
normalizer, checkpoint, rendering and evaluation utilities support these recipes.

## Setup and data

Run commands from this repository's root. Use the existing configured Python
environment on this instance, or install the project and LIBERO dependencies:

```bash
git submodule update --init --recursive
uv sync
source .venv/bin/activate
python -c "import torch, oat, libero; print(torch.__version__, torch.cuda.is_available())"
```

For headless simulation, set `MUJOCO_GL=egl` before starting Python. Training logs
can run offline with `logging.mode=offline`; online logging requires W&B credentials.

Policy configs expect `data/libero/libero10_N500.zarr`, containing 50 expert
demonstrations for each of the ten tasks. If starting from LIBERO HDF5 files, place
the ten `*_demo.hdf5` files in `data/libero/hdf5_datasets/`, then run:

```bash
python scripts/convert_libero_dataset.py -n 50
python scripts/compose_libero_multitask_dataset.py -mt libero10
```

`training.num_demo=500` selects the dataset filename. The validation ratio determines
the 450/50 split; 043 explicitly sets it to zero and disables offline validation.
Simulator evaluation episodes are separate from these training demonstrations.

## Checkpoint prerequisites

The historical default tokenizer path is:

```text
/workspace/past2next_clean/output/20260818/073434_train_oattok_so3aug_libero10_N500/checkpoints/ep-4960_mse-0.001.ckpt
```

The historical 015 source is the original Aug-24 EMA checkpoint:

```text
/workspace/past2next_clean/output/20260824/093317_train_past2next_self_past_libero10_N500/checkpoints/ep-0200_sr-0.806.ckpt
```

The default source for 043/046 is
`output/training/015_finetune112_greedy_history/checkpoints/ep-0009.ckpt`.
Historical 015/043/046 fine-tune checkpoints and their evaluation artifacts are
absent from this checkout. Restore them from your archive for historical inference
or continuation. Set the following variables to existing files before training:

```bash
TOKENIZER=/absolute/path/to/ep-4960_mse-0.001.ckpt
ORIGINAL=/absolute/path/to/ep-0200_sr-0.806.ckpt
SOURCE_015=/absolute/path/to/015/checkpoints/ep-0009.ckpt
```

Policy construction still needs the tokenizer checkpoint, including when policy
weights are initialized from scratch. The retained tokenizer training entry point
is `python scripts/run_workspace.py --config-name=train_oattok_so3aug`; training a
new tokenizer produces a different experiment from the recorded frozen-tokenizer
results.

## Training the three versions

These commands encode the known recipe settings. The original run's saved Hydra
config and overrides remain authoritative for exact historical reproduction; the
archived 015 run recipe is not present locally.

015 uses crop112 and vision LR `1e-5`, which must override the generic fine-tune
config's crop76 and vision LR `2e-6`. Spatial-coordinate buffer adaptation allows
initialization from the original crop76 checkpoint:

```bash
python scripts/run_workspace.py --config-name=train_past2next_finetune \
    policy.action_tokenizer.checkpoint="$TOKENIZER" \
    training.init_checkpoint="$ORIGINAL" \
    'policy.obs_encoder.vision_encoder.crop_shape=[112,112]' \
    optimizer.obs_enc_lr=1e-5 training.init_allow_spatial_resize=true \
    hydra.run.dir=output/training/015_finetune112_greedy_history
```

043 and 046 start independently from 015 epoch 9 EMA, with fresh optimizers and
ten epochs of training:

```bash
python scripts/run_workspace.py --config-name=train_past2next_finetune_all500 \
    policy.action_tokenizer.checkpoint="$TOKENIZER" \
    training.init_checkpoint="$SOURCE_015" \
    hydra.run.dir=output/training/043_all500

python scripts/run_workspace.py --config-name=train_past2next_finetune_tasklr \
    policy.action_tokenizer.checkpoint="$TOKENIZER" \
    training.init_checkpoint="$SOURCE_015" \
    hydra.run.dir=output/training/046_tasklr
```

All three use cosine LR decay, 100 LR warmup updates, EMA, and an optimizer-step
history curriculum: 1,000 warmup updates followed by a 4,000-update ramp to a maximum
self-generated-history probability of 0.5. Fine-tuning restores the source
normalizers, whose statistics were computed from all 500 demonstrations.

The recipes default to supervised training with `task.policy.lazy_eval=true`.
Per-epoch snapshots are saved as `checkpoints/ep-XXXX.ckpt`. To resume an existing
run, use the same output directory and set `training.resume=true`; its
`checkpoints/latest.ckpt` must exist. Use a fresh directory when starting a new run.

## Retained 046 training from scratch

`output/training/046_scratch_2gpu_20260910_092050/` is a separate run with randomly
initialized policy/vision weights and the same frozen tokenizer. It retains the
046 architecture, 450/50 split and learning rates, and trains for 251 epochs. Its
saved `.hydra/config.yaml` and `.hydra/overrides.yaml` record the actual recipe.
This run is separate from the historical fine-tuned 046 epoch-4 comparison.

The launch settings below match that recipe; choose a new output directory:

```bash
MUJOCO_GL=egl accelerate launch --multi_gpu --num_processes 2 \
    scripts/run_workspace.py --config-name=train_past2next_finetune_tasklr \
    policy.action_tokenizer.checkpoint="$TOKENIZER" \
    training.init_checkpoint=null training.resume=false training.num_epochs=251 \
    task.policy.lazy_eval=false training.rollout_every=25 \
    training.checkpoint_every=25 training.snapshot_every=0 checkpoint.topk.k=3 \
    task.policy.env_runner.protocol=official task.policy.env_runner.n_test=500 \
    +task.policy.env_runner.test_start_seed=3000 \
    ++policy.obs_encoder.vision_encoder.eval_fixed_crop=true \
    dataloader.batch_size=32 logging.mode=online logging.resume=false \
    logging.name=046_scratch_2gpu \
    hydra.run.dir=output/training/046_scratch_new
```

## Evaluation

Use the retained evaluator with a checkpoint and a new output directory. This
command matches the recorded official protocol: 500 episodes, saved states 0–49
per task, policy seed 44, reset seeds 3000–3499, greedy EMA inference, center112
crops, zero initial history and a 550-policy-step limit:

```bash
MUJOCO_GL=egl python scripts/evaluate_candidate.py \
    --checkpoint /absolute/path/to/checkpoints/ep-0009.ckpt \
    --output-dir output/evaluations/new_official \
    --protocol official --n-test 500 --n-parallel-envs 10 \
    --seed 44 --episode-start-seed 3000 --init-state-offset 0 \
    --weights ema --temperature 0 --topk 10 --use-k-tokens 8 \
    --crop-mode center \
    --max-episode-steps 550
```

For the corrected development protocol, change the protocol to `corrected`, policy
seed to 45 and episode start seed to 4000. See the comparison document for all five
recorded schedules and the limitations of historical legacy resets. These schedules
are already known and reused results should be reported as retrospective checks.
