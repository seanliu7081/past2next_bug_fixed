# Past2Next with executed-command history

This is a separate variant. Existing policies, configurations, evaluators, and
launchers are unchanged.

## What supplies history

During offline training and validation, `past_action` contains the demonstration
commands recorded immediately before the current observation/action window. The
variant does not generate substitute self-past actions and does not require
previous-window observations. It uses the existing Zarr format without conversion.

During rollout, `predict_action()` proposes actions without appending them to
history. The execution layer acknowledges the commands actually executed, then
the policy appends only those prefixes. With the defaults, a complete eight-step
execution keeps its last seven commands. A three-step execution appends three
commands; an unexecuted suffix and the remaining eight future predictions never
enter history. Unavailable initial history is zero, and each episode starts fresh.

The raw-action condition and the first/second normalized command differences
(`acc` and `jerk` in the original policy) all use this same history. These features
describe executed **commands**, not measured physical velocity or acceleration.
No measured-state-history encoder is added in this variant.

## Train the new baseline

From the repository root, provide a tokenizer checkpoint trained with the corrected
code and the matching episode split:

```bash
bash train_executed_past.sh /absolute/path/to/tokenizer.ckpt
```

An optional second argument sets the output directory. The default is a fresh
timestamped directory under `output/training/`. Check configuration without
loading the checkpoint or starting training:

```bash
bash train_executed_past.sh --dry-run /absolute/path/to/tokenizer.ckpt
```

The standalone recipe is `oat/config/experimental/train_past2next_executed_past.yaml` with task
`libero/libero10_executed_past`. It preserves the scratch baseline's architecture,
learning rates, 251 epochs, 112-pixel crops, horizon 16, eight executed steps,
seven past commands, model/split seeds 42, and validation ratio 0.1. It removes the
self-past replacement curriculum and generated-history validation. The existing
workspace still performs ordinary offline validation and supports EMA/checkpoints.

This trains on executed **demonstration** commands. It does not collect new
on-policy trajectories or obtain expert labels for newly visited states.

## Evaluate LIBERO

The new checkpoint saves the execution-aware runner target. Use the existing
candidate evaluator, which honors that target:

```bash
MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_candidate.py \
  --checkpoint /absolute/path/to/policy.ckpt \
  --output-dir output/evaluations/executed_past_new \
  --protocol corrected --n-test 500 --n-parallel-envs 10 \
  --seed 45 --episode-start-seed 4000 \
  --weights ema --temperature 0 --topk 10 --use-k-tokens 8 \
  --crop-mode center --max-episode-steps 550
```

`LiberoExecutedPastRunner` also supports `official`. Legacy autoresets are rejected
because a submitted chunk can reset an environment without executing its actions.
The older `eval_policy_sim.py` protocol switch recognizes only the original runner
name; use `evaluate_candidate.py` for this variant.

The adapter waits for `env.step()` to return, reads each environment's `cur_step`,
and acknowledges the count increase. It handles different terminal-prefix lengths
across vector workers, including zero. It reuses the existing runner's scheduling,
metrics, video handling, and episode records. A `RoboCasaExecutedPastRunner` adapter
is also provided; the supplied training recipe and command above are for LIBERO.

The extra scalar-counter RPC occurs once per action chunk. No latency or success
improvement has been established by the CPU regression tests.

## External controllers and real robots

The new policy class can be loaded through the existing `BasePolicy.from_checkpoint`.
An external execution loop must implement the acknowledgement contract. In pseudocode:

```python
policy.reset()
prediction = policy.predict_action(observation)
executed_commands, lengths = controller.execute_and_report(prediction["action"])
policy.record_executed_actions(executed_commands, executed_lengths=lengths)
# The next prediction uses the updated history and newly observed state.
```

`executed_commands` has shape `(batch, steps, action_dim)` and uses the dataset's raw
command units. `lengths` has one executed-prefix count per batch member. A controller
that clips or changes commands must return the changed commands in that same command
space. Omitting `lengths` asserts that every supplied command executed. Simulator
counts establish execution at the simulator command API; they do not measure actuator
torques or guarantee the commanded motion occurred.

A second stateful prediction without acknowledgement raises an error. Call `reset()`
at episode/batch-identity boundaries; history is not stored in checkpoints. Explicit
`past_actions` remains a stateless inference interface for recorded-history validation
and does not alter an ongoing rollout's buffer or pending acknowledgement.

Existing real-robot launchers and `check_real_robot_checkpoint.py` expect the original
automatic predicted-history update. They are unchanged and must not be used as the
execution loop/checker for this new contract without an acknowledgement-aware caller.
