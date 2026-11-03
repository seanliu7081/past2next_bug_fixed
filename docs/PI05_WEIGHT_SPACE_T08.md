# PI05-past weight-space edits on the frozen t08 bottle scene

This experiment edits the final flow-velocity projection of the 50,000-update
PI05-past checkpoint directly. It uses development-rollout evidence to construct
a fixed tensor change, then measures the edited policy. There is no backward
pass, optimizer, training loop, or runtime action correction. This is an
experiment on one custom scene, not an official LIBERO benchmark result.

## Method

The target is `flow_heads.action_out_proj`, which computes `v = W h + b`.
`oat/model/vla/weight_space_edit.py` applies an explicit rank-one update to selected
real action rows, plus an optional bias change:

```text
W'[rows, :] = W[rows, :] + coefficients[:, None] * direction[None, :]
b'[rows]    = b[rows]    + bias_delta
```

Padding rows are protected. Source tensor hashes pin the starting weights; the
application records exact before/after hashes, changed-value counts and delta
norms. The undo handle restores the exact source tensors and refuses to overwrite
a later edit.

`scripts/build_pi05_weight_edit.py` builds a direction from manually selected
risk and reference chunks in the development seed-42 rollout. It averages
captured projection inputs across all denoising steps and selected action-horizon
positions, then takes the difference between the two context means. The bias
centers the response on the reference mean. With
`--protect-reference-variation`, one SVD projects this contrast into the
orthogonal complement of variation among reference chunk means. This preserves
those means in the fixed-feature estimate; it does not preserve every token or
the full denoising trajectory.

The sampler integrates from noise at `t=1` to actions at `t=0`, with
`dt = -1 / flow_num_steps`. A negative velocity correction therefore initially
raises the corresponding action value. The builder accounts for this sign and
the checkpoint's action normalizer. `--raw-action-change` uses LIBERO command
units, **not metres**. Its requested change is a risk-mean estimate holding the
baseline features fixed. Later denoising can change every action dimension, and
the same static edit can affect unselected stages. Replay and fresh rollouts are
required to measure these effects.

The context choices use development observations, activations and evaluator
diagnostics. They are data-informed edits, even though they use no gradient
training. Freeze the candidate before confirmation seeds 43, 44 and 45; those
seeds vary policy sampling on the same layout, not object placement.

## Frozen evaluation contract

| Setting | Value |
|---|---|
| Source | `output/training/pi05_past_libero10_seed42/snapshots/upd-050000_model.ckpt`, `--weights model` |
| Scene | `/workspace/gpt_field/output/layouts/libero_10_t8_bottle_obstacle_20261103_001` |
| Task | Zero-based LIBERO-10 task 8, UID 38: put both moka pots on the stove |
| Reset | Environment seed 7, saved `custom_task.bddl` **and** `initial_state.npy`, 10 settling steps |
| Episode budget | 550 policy actions, excluding settling |
| Policy inputs | Native 128px agentview/wrist RGB, vertically flipped, uint8; raw robot proprioception and task UID |
| Policy processing | Checkpoint's image preprocessing, state transforms, normalizers and flow sampler |
| History | Only acknowledged executed action prefixes; unused predictions never enter history |
| Controller | Stock OSC_POSE, translation output scale 0.05 m and rotation scale 0.5 rad |

`scripts/evaluate_pi05_weight_edit.py` loads the policy in `/venv/oat` and runs a
separate simulator worker in `/workspace/venvs/libero-plus/bin/python`, with the
scene-authoring LIBERO checkout at `/workspace/heading_flow_pub/third_party/LIBERO`.
It validates settled object poses, joint addresses and the stove against the
layout manifest. The policy's 128px images are intentionally different from the
scene preview's 512px observation hash. Policy RNG is reset after simulator setup
so setup does not consume sampling noise.

Object poses and contacts are evaluator-only records. They never enter
`predict_action`. The contact monitor inspects physics substeps and control
endpoints for bottle-versus-robot/gripper or bottle-versus-pot penetration.
Table support is excluded. Contacts are recorded without stopping the rollout.
Results distinguish raw LIBERO success from `obstacle_collision_free_success`;
the latter does not certify absence of other scene collisions. Contact counts
are contact-point samples, not independent impacts.

## Commands

Run from the repository with an available CUDA device. Use the instance's
supervisor job mechanism for managed GPU runs; these scripts run in the
foreground. Output directories and exported snapshots must be new. The existing
experiment's exact launches are archived under `supervisor/*.launch.json`.

```bash
cd /workspace/ysk/past2next_bug_fixed
WSP_PY=/venv/oat/bin/python
WSP_TRAIN=/workspace/ysk/past2next_bug_fixed/output/training/pi05_past_libero10_seed42
WSP_SNAPSHOT="$WSP_TRAIN/snapshots/upd-050000_model.ckpt"
WSP_RUN="$WSP_TRAIN/weight_space_t08_20261103_001"
```

Capture an unedited rollout. The archived development run is
`baseline_dev_s42`; this example uses a separate destination:

```bash
"$WSP_PY" scripts/evaluate_pi05_weight_edit.py \
  --snapshot "$WSP_SNAPSHOT" --weights model --device cuda:0 \
  --policy-seed 42 --env-seed 7 --settle-steps 10 \
  --max-episode-steps 550 --capture-head \
  --output-dir "$WSP_RUN/baseline_recheck_s42"
```

Construct an **experimental** candidate using the archived development captures.
The following selectors and magnitude illustrate the `transport_lift_b` design;
the new name prevents overwriting that artifact. Ranges are half-open. Remove
`--protect-reference-variation` to use the plain centered mean contrast.

```bash
"$WSP_PY" scripts/build_pi05_weight_edit.py \
  --snapshot "$WSP_SNAPSHOT" --weights model \
  --trace "$WSP_RUN/baseline_dev_s42/head_captures" \
  --risk-chunks 14:18 --reference-chunks 0:14,20:24,27:35,43:47 \
  --horizon-positions 0:8 --row 2 --raw-action-change 0.35 \
  --protect-reference-variation \
  --output-dir "$WSP_RUN/edits/transport_lift_example" \
  --export-snapshot "$WSP_RUN/snapshots/transport_lift_example_model.ckpt"
WSP_PATCH="$WSP_RUN/edits/transport_lift_example/patch.json"
```

The builder writes `patch.json`, `direction.npz`, diagnostic `responses.*` and
`patch_provenance.json`. The optional exporter creates a compatible
`p2n_vla_checkpoint_v1` snapshot, preserving the source and all other tensors.
It stores edited head tensors in fp32 to reproduce the runtime patch without
additional bf16 rounding. The export carries edit provenance and source hashes.
An exported snapshot can be loaded by the existing evaluator; do not also apply
the same `--patch-json` to it.

Replay stored observations, executed history and exact CPU/CUDA Torch RNG states:

```bash
"$WSP_PY" scripts/replay_pi05_weight_edit.py \
  --snapshot "$WSP_SNAPSHOT" \
  --capture-dir "$WSP_RUN/baseline_dev_s42/head_captures" \
  --chunks 0,8,11,12,13,15,16,20,32,40,46 \
  --patch-json "$WSP_PATCH" --device cuda:0 \
  --output-dir "$WSP_RUN/replay_transport_lift_example"
```

Replay first reproduces the stored unedited chunk and full prediction, then
repeats them with the same RNG. It reports both bitwise equality and numerical
tolerance checks; a failed baseline check aborts before applying the patch to
that context. Only then does it compare edited actions using the same inputs and
noise. Replay establishes action effects, not task success or collision avoidance.

Evaluate the fixed candidate in closed loop:

```bash
"$WSP_PY" scripts/evaluate_pi05_weight_edit.py \
  --snapshot "$WSP_SNAPSHOT" --patch-json "$WSP_PATCH" \
  --policy-seed 42 --env-seed 7 --settle-steps 10 \
  --max-episode-steps 550 --device cuda:0 --capture-head \
  --output-dir "$WSP_RUN/transport_lift_example_dev_s42"
```

For confirmation, run both the original and frozen candidate at each of policy
seeds 43, 44 and 45, using new output directories. Omit `--patch-json` for the
original. Keep the environment seed, scene, step budget and all other settings
fixed. Do not use confirmation outcomes to revise a candidate and still describe
those outcomes as held out.

## Evidence and current results

Each rollout writes source/input hashes, reset validation, simulator metadata,
`trajectory.jsonl`, `commands.npz`, `obstacle_contacts.jsonl`, video and
`summary.json`. `--capture-head` additionally saves hidden features, velocities,
policy observations, action history and RNG states for offline replay.

The final frozen candidate is **C**, `edits/transport_lift_slow_c/patch.json`.
It reuses B's protected latent direction and edits output rows y and z plus
their biases, requesting first-order raw command changes **+0.25 y / +0.70 z**.
These are estimates in command units, not physical displacements. The candidate
was frozen before confirmation and was not revised using confirmation outcomes.

Development used policy seed 42 throughout:

| Policy | Raw goal | Actions | First bottle contact index | Robot/bottle samples | Pot/bottle samples |
|---|---:|---:|---:|---:|---:|
| Original | no | 550 | 96 | 57 | 25,789 |
| A | no | 550 | 85 | 452 | 23,693 |
| B | no | 550 | 126 | 0 | 3,410 |
| C, selected | yes | 424 | 93 | 79 | 0 |

All six reserved episodes are complete:

| Policy seed | Original raw goal / actions | C raw goal / actions | Original robot / pot samples | C robot / pot samples |
|---|---|---|---:|---:|
| 43 | no / 550 | yes / 479 | 0 / 37,620 | 0 / 1 |
| 44 | no / 550 | yes / 508 | 133 / 33,556 | 1,850 / 138 |
| 45 | no / 550 | no / 550 | 0 / 28,280 | 870 / 12 |

Thus reserved raw goal completion was **0/3 original versus 2/3 edited**.
Bottle-contact-free goal completion was **0/3 for both**. Candidate C still
contacts the bottle; even the seed-43 success recorded one pot/bottle contact
sample. Development C's success is not pooled into the reserved results. These
three sampling seeds on one frozen layout support descriptive outcomes, not a
statistical claim or general obstacle-avoidance capability. Failed episodes stop
at 550 actions, so their action counts are not completion-time measurements.

The experiment-local `summarize_results.py` independently checks input/source
hashes, the frozen patch, every contact log and trajectory, and the 25 physics
substeps per action. Initial policy observations and complete post-reset
simulator states match bitwise within all three original/edited pairs. It
requires all six reserved runs to be complete before writing
[comparison.json](../output/training/pi05_past_libero10_seed42/weight_space_t08_20261103_001/comparison.json)
and [RESULTS.md](../output/training/pi05_past_libero10_seed42/weight_space_t08_20261103_001/RESULTS.md).
The report keeps all ten development/confirmation episodes separate.

The original checkpoint, 264 frozen runtime source files and five protected
inputs remain unchanged. The recorded test suite passed **24 tests**. The
compatible export is
`weight_space_t08_20261103_001/upd-050000_model_t08_weight_edit_experimental.ckpt`.
Real checkpoint reload reproduced the runtime patch's predictions bitwise on
three saved contexts; all 1,071 unrelated tensors were unchanged. See
`export_validation.json` and `integrity_verification.json` beside the export.
This validates the artifact, not safety or task success on additional scenes.

To evaluate the frozen artifact, use the exported snapshot as `--snapshot`
without `--patch-json`, or use the original snapshot with
`--patch-json "$WSP_RUN/edits/transport_lift_slow_c/patch.json"`.
A [seed-43 side-by-side video](../output/training/pi05_past_libero10_seed42/weight_space_t08_20261103_001/comparison_seed43.mp4)
shows the original and edited rollouts.
