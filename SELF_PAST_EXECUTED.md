This variant preserves the scratch recipe's **offline self-past training** and changes how stateful inference records action history. It uses the same tokenizer, discrete autoregressive action policy, past-action inputs, and command-difference features.

During training, `ZarrDatasetWithPrevWindow` supplies recorded demonstrations and the preceding observation/action window. The policy sometimes generates a synthetic previous chunk and substitutes its history into the current training example, following the original self-past schedule. **These generated histories are not executed in an environment during training.** This remains offline history augmentation; it does not supply observations from hypothetical rollouts.

During stateful inference, generating a proposed chunk does not update history. The execution layer acknowledges commands confirmed executed at the simulator API boundary, and only their executed prefixes enter the history buffer. Raw past-action inputs and acceleration/jerk-style command differences use that same acknowledged buffer. These features describe changes in commands, rather than measured physical acceleration or actuator motion.

The standalone recipe is [experimental/train_past2next_self_past_executed.yaml](oat/config/experimental/train_past2next_self_past_executed.yaml). It preserves the scratch model and training settings: eight transformer layers, eight heads, width 256, 112-pixel crops, a 16-action prediction horizon, eight executed actions per normal chunk, seven past actions, two observations, model/dataset seed 42, and validation ratio 0.1. Both learning rates remain `1e-5`. The self-past probability reaches 0.5 using the existing 1,000-update warmup and 4,000-update ramp, driven by the persisted optimizer-step counter. Generated-history validation remains enabled. Use the same frozen tokenizer checkpoint as the comparison baseline, including its eight-token representation.

Train from a tokenizer checkpoint:

```bash
bash train_self_past_executed.sh /path/to/tokenizer.ckpt
```

Optionally supply a new output directory:

```bash
bash train_self_past_executed.sh /path/to/tokenizer.ckpt /path/to/new_training_output
```

The default output directory has a fresh UTC timestamp under `output/training/`. Relative paths are interpreted from the caller's working directory. `TRAIN_PY` selects the Python executable and defaults to `/venv/oat/bin/python`.

Check configuration without loading checkpoint weights, creating training output, or starting training:

```bash
bash train_self_past_executed.sh --dry-run /path/to/tokenizer.ckpt
```

The tokenizer file must exist even for a dry run. The equivalent Hydra config name is `experimental/train_past2next_self_past_executed`; the config is standalone, with its LIBERO task supplied by the existing `libero10_with_prev_window` task group.

Evaluate a checkpoint created with this new config using the existing candidate evaluator:

```bash
MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_candidate.py \
    --checkpoint /path/to/policy.ckpt \
    --output-dir /path/to/new_evaluation_output \
    --protocol corrected --n-test 100 --n-parallel-envs 10 \
    --seed 42 --episode-start-seed 1000 \
    --weights ema --temperature 0 --topk 10 --use-k-tokens 8 \
    --crop-mode center --max-episode-steps 550
```

Checkpoints created with this new config save `LiberoExecutedPastRunner` as their runner target, and `evaluate_candidate.py` preserves that target. Existing checkpoints still select their saved policy and runner; changing the current YAML does not rewrite an old checkpoint. The runner acknowledges each worker's executed prefix after the simulator step, including partial chunks and zero-length prefixes for completed workers. It supports corrected and official protocols; legacy autoresets are incompatible with execution accounting. Use the same protocol, episode schedule, and inference settings when comparing policies. The older `eval_policy_sim.py` helper does not recognize this runner subclass for its LIBERO-specific protocol override; use `evaluate_candidate.py` for this variant.

Existing self-past checkpoints with the same architecture have compatible weights; retraining is unnecessary solely to change inference history accounting. For programmatic evaluation, explicitly override the policy when loading and update the returned runner configuration before constructing the evaluation runner:

```python
from oat.policy.base_policy import BasePolicy

policy, cfg = BasePolicy.from_checkpoint(
    "/path/to/existing_self_past.ckpt",
    weights="ema",
    policy_overrides={
        "_target_": (
            "oat.policy.past2next_self_past_executed."
            "Past2NextSelfPastExecutedPolicy"
        ),
    },
    return_configuration=True,
)
cfg.task.policy.env_runner._target_ = (
    "oat.env_runner.executed_action_runner.LiberoExecutedPastRunner"
)
cfg.task.policy.env_runner.protocol = "corrected"
# Construct the evaluation runner from cfg.task.policy.env_runner.
```

These overrides apply to this in-memory policy and configuration; they do not modify the checkpoint. The CLI command above does not apply these overrides to old checkpoints automatically. Keep the same evaluation schedule and inference settings as the comparison baseline.

External real-robot callers load the new policy through the existing `BasePolicy.from_checkpoint` API, then call `policy.eval()` and `policy.reset()` at the episode boundary. After each stateful `predict_action`, they must call `record_executed_actions(commands, executed_lengths=counts)` using commands confirmed by their execution layer, before requesting another stateful prediction. Supply raw command coordinates matching the training dataset; if the controller changes commands, acknowledge the changed commands. Do not acknowledge an unexecuted suffix or a proposed chunk whose execution failed. Explicit `past_actions` remains a stateless interface for offline evaluation, and does not require an execution acknowledgment.

This variant is separate from `train_executed_past.sh`, which trains only on expert history. Existing recipes, launchers, and checkpoints are unchanged. The existing real-robot checkpoint checker assumes automatic history updates from predictions and is not the checker for this acknowledgment-based contract. Training, deployment, and success-rate validation are separate steps; the new recipe does not imply a performance improvement.
