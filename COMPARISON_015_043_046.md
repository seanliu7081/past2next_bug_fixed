# Single-checkpoint comparison: 015, 043 and 046

Historical results recorded on 2026-09-10. The original report stated that all 20 evaluation cells were audited. The corresponding historical checkpoint and evaluation directories are absent from this checkout; the figures below are preserved records, not newly verified measurements.

Every cell uses one checkpoint across all ten LIBERO-10 tasks, with 50 trials per task (500 total). The separate retained `046_scratch_2gpu_20260910_092050` run does not supply these historical scores.

| Architecture / training setting | 015 | 043 | 046 |
|---|---|---|---|
| Cameras / crop | Agent + wrist; 112×112 | Same | Same |
| Visual backbone | Two independent ResNet18 + GroupNorm + SpatialSoftmax encoders, 64 features each | Same | Same |
| Policy transformer | 8 decoder blocks, 8 heads, width 256 | Same | Same |
| Observation and past context | 2 frames; 7 past commands plus command-derived acceleration/jerk | Same | Same |
| Condition sequence | 11×138, projected to width 256 | Same | Same |
| Tokenizer / action horizon | Frozen Aug-18 FSQ, levels [8,5,5,5,5], 8 tokens, 16 decoded / 8 returned actions | Same | Same |
| Task conditioning | Scalar task UID in state | Same | Scalar UID plus learned 10×138 task residual |
| Training / validation demonstrations | 450 / 50 | 500 / 0 | 450 / 50 |
| Initialization | Original Aug-24 EMA | 015 epoch 9 EMA | 015 epoch 9 EMA |
| Policy / vision learning rate | 1e-5 / 1e-5 | 1e-5 / 1e-5 | 1e-5 / 2e-6; task table 1e-3 |
| Generated-history training temperature | 0 | 1 | 1 |

All use EMA inference, greedy temperature 0, top-k 10, all eight tokens, native center112 crops, zero initial command history and a 550-policy-step limit. The task residual in 046 is added to the fused observation features of one shared policy.

| Checkpoint | Official matched | Official alternate seeds | Corrected development | Corrected final seed schedule† | Historical legacy‡ |
|---|---:|---:|---:|---:|---:|
| 015, epoch 9 | 85.8% (429/500) | 84.8% (424/500) | 86.6% (433/500) | 86.0% (430/500) | 86.6% (433/500) |
| 015, epoch 19 | 86.0% (430/500) | 87.2% (436/500) | 85.4% (427/500) | 86.6% (433/500) | 86.0% (430/500) |
| 043, epoch 9 | 88.0% (440/500) | 87.4% (437/500) | 89.8% (449/500) | 85.4% (427/500) | 87.6% (438/500) |
| 046, epoch 4 | 88.2% (441/500) | 89.2% (446/500) | 86.4% (432/500) | 87.8% (439/500) | 89.6% (448/500) |

† Only the existing 043 evaluation was a checkpoint-frozen, prospective final test on this schedule. New 015/046 runs are retrospective comparisons using the same now-known seeds, not new untouched final tests. These retrospective measurements do not constitute independent final validation.

‡ Legacy retains historical reset/autoreset behavior. Its nominal episode seeds are not applied and retained step counts are unreliable. Matching nominal schedules does not make legacy comparisons paired. The limit remains configured at 550 steps.

| Condition | Reset procedure | Policy seed | Episode reset schedule | Parallel environments |
|---|---|---:|---|---:|
| Official matched | LIBERO saved initial states 0–49 per task; 5 zero-action settling steps | 44 | 3000–3499 | 10 |
| Official alternate | Same saved-state procedure | 44 | 1000–1499 | 10 |
| Corrected development | Seeded environment resets; 10 open-gripper settling steps; post-settle observations; no autoreset | 45 | 4000–4499 | 10 |
| Corrected final schedule | Same corrected reset procedure | 46 | 5000–5499 | 10 |
| Historical legacy | Historical reset/autoreset procedure | 42 | Nominal 1000–1499; not applied | 20 |

The 450/50 split describes expert demonstrations used for gradient training and offline validation. It does not describe the 500 simulator evaluation episodes. Original normalizer statistics are preserved and were computed from all 500 demonstrations. The branches differ in training lineage and settings, so their scores do not isolate the effect of training on 450 versus 500 demonstrations.

The archived evidence was recorded under `output/evaluations/` and
`output/analysis/table_015_043_046_completion/` (plan, dispatch, complete results,
summary matrix and per-task counts). Those directories are not available locally.
The original audit reportedly covered episode counts, unique indices, task balance,
protocol schedules, checkpoint/tokenizer identities, source hashes and inference
settings. Use the source archives to recheck those claims.

See [README.md](README.md) for retained configs, known recipe overrides and evaluation
commands. The generic fine-tuning config requires explicit crop/LR overrides for 015;
it is not itself the complete historical 015 recipe.
