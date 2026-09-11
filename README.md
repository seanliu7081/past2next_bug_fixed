# Past2Next: two-stage training

Stage 1 trains an action tokenizer. Stage 2 loads that tokenizer, freezes it, and
trains the policy and vision encoders from scratch. No policy checkpoint is needed.
The three standalone Stage 2 recipes retain the designs of 015, 043 and 046:

| Design reference | Stage 2 config | Train / validation demos | Policy / vision LR | Task residual LR | Generated-history temperature |
|---|---|---|---|---|---|
| 015 | `train_past2next_scratch` | 450 / 50 | `1e-5 / 1e-5` | — | 0 |
| 043 | `train_past2next_scratch_all500` | 500 / 0 | `1e-5 / 1e-5` | — | 1 |
| 046 | `train_past2next_scratch_tasklr` | 450 / 50 | `1e-5 / 2e-6` | `1e-3` | 1 |

All three start with fresh policy/vision weights and optimizers. They use two
observation frames, agent/wrist images with 112×112 crops, seven past commands,
acceleration/jerk features, and a transformer with eight blocks, eight heads and
width 256. The action tokenizer uses FSQ levels `[8,5,5,5,5]` and eight ordered
tokens to reconstruct 16 actions, of which eight are executed. The 046 design adds
a zero-initialized `10×138` task table alongside the scalar task UID.

Each training config explicitly declares its model, training, optimizer, loader,
logging and checkpoint settings. No training config inherits another. Hydra selects
the shared LIBERO task config before applying each recipe's task overrides.
Historical fine-tuning results remain in [COMPARISON_015_043_046.md](COMPARISON_015_043_046.md);
the new scratch recipes produce separate runs.

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

## Stage 1: train the tokenizer

Train the action-only tokenizer from random weights:

```bash
python scripts/run_workspace.py --config-name=train_oattok_so3aug \
    logging.mode=offline \
    hydra.run.dir=output/training/tokenizer
```

This stage uses only action chunks, with SO(3) augmentation, and saves tokenizer
checkpoints under `output/training/tokenizer/checkpoints/`. The default training
length is 5,001 epochs; override `training.num_epochs` to choose a different length.
After training, select the saved tokenizer checkpoint for Stage 2, for example:

```bash
TOKENIZER="$PWD/output/training/tokenizer/checkpoints/latest.ckpt"
```

The checkpoint loader selects its EMA weights when Stage 1 used EMA. Use the same
selected tokenizer checkpoint for all three policy recipes when comparing them.

## Stage 2: train a policy from scratch

Every policy config requires `policy.action_tokenizer.checkpoint`. It has no default
path to an old run. The loaded tokenizer's weights and normalizer remain fixed;
it stays in evaluation mode during policy training so dropout cannot change its
token targets. Policy and vision weights, optimizer, EMA, and history curriculum
start fresh. Policy normalizers are initialized from the dataset.

Choose one of these commands, using a fresh output directory:

```bash
# 015 design: 450 demonstrations, greedy generated history.
python scripts/run_workspace.py --config-name=train_past2next_scratch \
    policy.action_tokenizer.checkpoint="$TOKENIZER" \
    hydra.run.dir=output/training/015_scratch

# 043 design: all 500 demonstrations, stochastic generated history.
python scripts/run_workspace.py --config-name=train_past2next_scratch_all500 \
    policy.action_tokenizer.checkpoint="$TOKENIZER" \
    hydra.run.dir=output/training/043_scratch

# 046 design: 450 demonstrations and a learned task residual.
python scripts/run_workspace.py --config-name=train_past2next_scratch_tasklr \
    policy.action_tokenizer.checkpoint="$TOKENIZER" \
    hydra.run.dir=output/training/046_scratch
```

All three default to `training.init_checkpoint=null`, `training.resume=false`,
`logging.resume=false`, and 251 epochs. They use cosine LR decay, 100 LR warmup
updates, EMA, and a history curriculum with 1,000 warmup updates followed by a
4,000-update ramp to a maximum generated-history probability of 0.5.

Training and validation use the demonstration dataset by default
(`task.policy.lazy_eval=true`). Simulator evaluation is a separate command below.
The all500 recipe explicitly disables offline validation because it has no holdout.
Policy snapshots and `latest.ckpt` are saved every 25 epochs; the last configured
epoch is 250, producing `ep-0250.ckpt`. All retained entrypoints start fresh by default.
To resume a run intentionally, use its original config, tokenizer checkpoint and
output directory with `training.resume=true`.

The additional `train_past2next_self_past` and `train_past2next_improve` configs are
also standalone scratch recipes with their own architecture and schedule settings;
both require a Stage 1 tokenizer checkpoint.

## Multiple GPUs and simulator evaluation during training

This 046 example uses two GPUs and evaluates all 500 official episodes every
25 epochs. Batch size 32 is per process:

```bash
MUJOCO_GL=egl accelerate launch --multi_gpu --num_processes 2 \
    scripts/run_workspace.py --config-name=train_past2next_scratch_tasklr \
    policy.action_tokenizer.checkpoint="$TOKENIZER" \
    task.policy.lazy_eval=false training.rollout_every=25 \
    task.policy.env_runner.protocol=official task.policy.env_runner.n_test=500 \
    +task.policy.env_runner.test_start_seed=3000 \
    ++policy.obs_encoder.vision_encoder.eval_fixed_crop=true \
    dataloader.batch_size=32 \
    hydra.run.dir=output/training/046_scratch_2gpu
```

The existing `046_scratch_2gpu_20260910_092050` artifact directory retains its original
saved configuration and overrides. It used the earlier tokenizer checkpoint and
training code; its settings and results are historical records.

## Evaluation

Use the retained evaluator with a checkpoint and a new output directory. This
command matches the recorded official protocol: 500 episodes, saved states 0–49
per task, policy seed 44, reset seeds 3000–3499, greedy EMA inference, center112
crops, zero initial history and a 550-policy-step limit:

```bash
MUJOCO_GL=egl python scripts/evaluate_candidate.py \
    --checkpoint output/training/046_scratch/checkpoints/ep-0250.ckpt \
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
