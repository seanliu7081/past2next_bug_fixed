# Original-observation implementation verification

Date: 2026-10-04. Existing tracked files and the supplied implementation guide were not edited.
All implementation, configuration, launcher, test, and documentation files are new.

## Results

- Final feature suite: **63 passed** (encoder, policy, workspace, CLI).
- Existing AR/DINO/ConvNeXt regression selection plus the earlier feature tests: 80 passed,
  13 skipped. The skipped ConvNeXt tests require the optional `timm` dependency absent
  from `/venv/oat`; those skips are not counted as passes.
- Two-rank CPU smoke: both variants, self-past probabilities 0 and 0.5, gradient
  accumulation, every trainable gradient, optimizer/EMA state, and strict continuation.
- Two idle RTX 4090 GPUs (6 and 7), BF16: both variants and both probabilities passed
  with the production 16-layer/768-width/12-head/2048-FFN AR, real original ResNet18/GN
  encoders, batch 2 per rank, accumulation 2. The tokenizer and synthetic action horizon
  in this smoke are deliberately small (2 OAT tokens, 4 actions); this is a correctness
  check, not acceptance of production training speed or robot quality.
- Both complete LIBERO production configurations passed CPU preflight using the local
  production OAT checkpoint, expected eight action tokens, real dataset schema/split,
  encoder construction, and simulator dependency/assets checks. LIBERO fused width is
  138; the real-robot schema is 139 because its actual state dimensions differ.
- Real-robot preflight for both variants correctly rejected the missing exact OAT
  checkpoint requested by the implementation guide. No replacement was silently chosen.
- `git diff --exit-code` and shell syntax checks passed. GPU smoke workers exited;
  existing jobs on GPUs 0–5 were left untouched. No full training or simulator/robot
  rollouts were launched.

## Covered contracts

The tests exercise original encoder equivalence before the bridge, seeded random
crops, ordered camera/state fusion including task UID, normalization exactly once,
strict completeness of restored normalizer statistics, both camera CNN gradients,
bridge/policy versus CNN learning rates, 11/15-token context layouts, summary-only
gates including open/closed modes, BF16 generated history, temporary evaluation mode
restoration, frozen OAT, cached/full AR consistency through the legacy tests, exact
checkpoint predictions, incompatible layout/crop/field-order rejection, no normalizer
refit on resume, and successful-update-only scheduler/EMA/curriculum progression.
The original ContextBatch implementation remains unchanged; the new layout is an
additive subclass with segment 5 and schema version 2.

## Reproduce

```bash
cd /workspace/ysk/past2next_bug_fixed
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /venv/oat/bin/python -m pytest \
  tests/test_p2n_new_original_obs_encoder.py \
  tests/test_p2n_new_original_obs_policy.py \
  tests/test_p2n_new_original_obs_workspace.py \
  tests/test_p2n_new_original_obs_launcher.py \
  -q --disable-warnings -p no:cacheprovider

# Select two currently idle GPUs before running this bounded smoke.
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /venv/oat/bin/python tests/p2n_new_original_obs_ddp_smoke.py \
  --device cuda --bf16 --production-ar --updates 3 \
  --output /tmp/p2n_new_original_obs_ddp_cuda.json
```

## Recorded reports and limits

- [CUDA synthetic correctness/timings](original_obs_verification/ddp_cuda.json)
- [CPU synthetic correctness/timings](original_obs_verification/ddp_cpu.json)
- [LIBERO base production preflight](original_obs_verification/libero_base_preflight.json)
- [LIBERO gate production preflight](original_obs_verification/libero_gate_preflight.json)
- [Training commands and flag meanings](PAST2NEXT_ORIGINAL_OBS_USAGE.md)
- [Tested recipe dependency pins](../requirements-original-obs.txt)

Smoke timings include synchronization and correctness instrumentation. Only two
post-initialization updates per CUDA case contribute to aggregate timings; the report
is not a stable throughput benchmark and does not isolate encoder/self-past/AR costs.
A full production-OAT, real-data performance benchmark and closed-loop quality
assessment remain separate runs. Normalizers are fitted on real training data only
when a fresh workspace starts; preflight reports that fitting is deferred.
