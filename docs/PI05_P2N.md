# pi05_p2n: Pi05_past with the Past2Next self-past action head

pi05_p2n is Pi05_past ([PI05_PAST.md](PI05_PAST.md)) with one change: the action head. pi05_base's flow-matching
head is replaced by the Past2Next self-past head, an autoregressive decoder over the 8 frozen-OAT tokens trained
with cross-entropy. Everything else is Pi05_past's code with Pi05_past's config values, so a run pairs with the
Pi05_past run episode by episode. Variant `pi05_p2n`, class `oat.policy.pi05_p2n.Pi05P2NPolicy` (a subclass
of `Pi05PastPolicy`), config `oat/config/train_pi05_p2n.yaml` (inherits `train_pi05_past.yaml`), launcher
`train_pi05_p2n.sh` (runs `train_pi05_past.sh --variant pi05_p2n`).

## Design

| Part | Pi05_past | pi05_p2n |
|---|---|---|
| Observation, prompt, VLM regime | SigLIP trainable (fp32 masters), Gemma-2B LoRA r32, flow loss through un-detached K/V, no KI | same (inherited code) |
| Past2Next condition | `[acc, jerk, raw x7]` GELU MLPs + slot embedding + dropout 0.1; one fp32 self-only expert pass at `pos0 + 16`; K/V appended to the VLM K/V | same; identical K/V at initialisation |
| Action head | flow: 16 noisy-action rows, Beta(1.5, 1) time, 10 Euler steps | 8 causal OAT-token rows `[BOS, z0..z6]` at `pos0 + k`, tied `tok_emb` 5001 x 1024 (new), cross-entropy, KV-cached decoding, OAT decoder to 16 x 7 actions |
| Expert norms | adaRMS: 37 Dense layers driven by the flow time (116.5M parameters) | adaRMS folded at `adarms_t0 = cond_t0 = 0.6` into per-segment modulations (0.34M): condition rows `SEG_RAWDIFF`, token rows `SEG_AR` |
| Deployment sampler | fresh noise + 10 Euler steps | greedy (`temperature 0`) |
| Self-past | p 0.5, warmup 1000, no ramp, chunk 4, deployment flow sampler | same schedule; tokens sampled at T=1 / top-k 10 (the Past2Next self-past sampler) |
| Data, batch, schedule, evaluation | libero10_N500, 32 per update, 80k cosine updates, LR 5e-5 / 1e-4, official 500 episodes every 10k | same |
| Snapshot retention | best evaluation + `latest.ckpt` | every evaluation's snapshot + `latest.ckpt` (the final test needs the best of the steps both runs share) |
| Trainable parameters | 885,276,432 | 772,067,056 |

The composed config differs from the Pi05_past run's stored config (`p2n_vla_resolved.yaml`) in exactly 9 of 216
keys: naming (5), the two flow knobs (null), the policy target and `training.keep_snapshots`
(`test_config_matches_pi05_past_except_the_head` checks it).

Differences that come with the head:
- The folded condition modulation is its own parameter. In Pi05_past it is the adaRMS Dense at `time_mlp(0.6)`,
  shared with the flow rows. It is equal at initialisation, but under Adam a Dense fed a constant `c0` moves its
  output about `||c0||_1 + 1 ≈ 21` times faster than the folded vector does.
- The token embedding starts cold, while Pi05_past's flow head was pretrained. Earlier AR-VLA heads in this repo
  reached 0.35 at 10k, 0.42-0.56 at 20k and 0.71 at 40k (insulated VLM, frozen SigLIP); Pi05_past was 0.764 at
  10k. Self-past starts at update 1000 as in Pi05_past, when the token head is still weak, so the early generated
  histories are mostly noise: `train/self_past_hist_mse` (normalized-action MSE between the generated and the
  replaced ground-truth history) shows when they become useful.
- The initial LoRA and condition-MLP draws differ from Pi05_past's at the same seed (the two constructions consume
  the random stream differently). The data order is identical (seeded per epoch); augmentation and self-past
  draws diverge.

Differences from the original Past2Next (`Past2NextSelfPastPolicy`): the observation pathway is pi05's (SigLIP +
Gemma-2B prefix, one frame, language and discretized-state prompt), the decoder is the pi05_base expert (18 x 1024,
RoPE, pretrained) reading the condition through joint attention instead of a 256-d cross-attention transformer,
the recipe is Pi05_past's, and deployment is greedy (the repo's scored Past2Next runs were all evaluated greedily;
the original yaml writes T=1). The tokenizer is the same 5000-code [8,5,5,5,5] OAT family, and the self-past
mechanics are the original's: one level of unrolling from the previous window with its ground-truth history,
`pred[1:8]` replacing the valid past slots.

## Train

```bash
cd /workspace/ysk/past2next_bug_fixed
bash train_pi05_p2n.sh --dry-run                            # resolve and validate the config
bash train_pi05_p2n.sh --devices 0,1,2,3 --probe            # memory/throughput go/no-go, writes output/probe/...
bash train_pi05_p2n.sh --devices 0,1,2,3                    # pi05_p2n, 80k updates of batch 32 (~3 days)
bash train_pi05_p2n.sh --devices 0,1,2,3 --use-past false   # AR no-past control, same recipe
```

`bash train_pi05_p2n.sh --help` lists the flags; they are `train_pi05_past.sh`'s, except that `--flow-steps` is
refused and `--temperature`, `--topk`, `--self-past-temperature` and `--self-past-topk` set the two token
samplers (refused for pi05_past). `--cond-t0` moves the adaRMS fold with it. The default run directory is
`output/training/pi05_p2n[_nopast|_gtpast]_libero10_seed<SEED>`, the W&B group `pi05_p2n_libero`.

Every 10k updates GPU 0 scores the official LIBERO-10 protocol (500 episodes, seed 44, episode seeds 3000-3499,
10 environments): the same episodes as every Pi05_past evaluation. Each evaluation keeps the snapshot it scored
(`snapshots/upd-NNNNNN_model.ckpt`, about 1.6 GB, 8 in all); `checkpoints/latest.ckpt` holds the latest weights.
After the comparison, prune to the best snapshot and `latest.ckpt`. This is the config's default; do not pass
`--keep-snapshots all`, which also writes a periodic snapshot every 5,000 updates (`--keep-snapshots all
--snapshot-every 0` equals the default).

The dataset path in the config and in the frozen OAT's provenance is
`/workspace/past_action/data/libero/libero10_N500.zarr`. That checkout is gone; the symlink
`/workspace/past_action/data/libero -> /workspace/shared_data/libero` restores it (content identity
`05dcceae...` and the same 450/50 split as the Pi05_past run's `dataset_split.json`). The OAT records this path and
the provenance check compares resolved paths, so the symlink must exist whatever path the config names; keep the
symlinked path so the config equals the Pi05_past run's.

Keep `/workspace/past_action` for the whole run. Besides the data symlink (read at launch and on resume), every
LIBERO evaluation imports the simulator through `/workspace/past_action/third_party/LIBERO`, a symlink to this
repo's `third_party/LIBERO` that `/venv/oat`'s editable LIBERO install and `~/.libero/config.yaml` point at.
Removing it mid-run makes every later in-training evaluation fail (Pi05_past lost its 10k evaluation that way);
`rollout_failure=continue` keeps training and the snapshot, which then has to be re-scored offline.

## Evaluate

The same evaluator and commands as Pi05_past ([PI05_PAST.md](PI05_PAST.md#evaluate)), always with 10 parallel
environments. The token head also accepts `--temperature`, `--topk` and `--use-k-tokens`:

```bash
RUN=output/training/pi05_p2n_libero10_seed42
SNAP=$RUN/snapshots/upd-050000_model.ckpt                     # the chosen snapshot (see the protocol below)
for START in 5000 5500 6000; do                                # final test: corrected 1,500 on fresh seeds
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py --snapshot $SNAP --weights model \
      --protocol corrected --n-test 500 --seed 46 --episode-start-seed $START --n-parallel-envs 10 --log-commands
done
# secondary readouts of the same snapshot on the official 500 (re-score as training did: --seed 44)
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl /venv/oat/bin/python scripts/evaluate_p2n_vla.py --snapshot $SNAP --weights model \
    --protocol official --n-test 500 --seed 44 --episode-start-seed 3000 --n-parallel-envs 10 --temperature 1.0 --topk 10
```

History interventions (`--history-intervention zero|swap|noise`), `--log-commands` and
`scripts/pi05_past_offline_diagnostics.py` work unchanged.

## Comparison protocol

Fixed before launch (2026-10-29), against the Pi05_past run `output/training/pi05_past_libero10_seed42`
(0.764 / 0.848 / 0.930 / 0.912 / 0.956 / 0.954 / 0.948 at 10k-70k; its 10k was re-scored offline because the
in-training evaluation failed):

1. Per step, 10k-70k: exact McNemar on the official 500, after asserting that `(task_name, init_state_id,
   episode_seed)` match. Report each step; do not average across steps (10k-30k mostly measure how fast a cold
   head learns).
2. Final test: each run's best snapshot among 10k-70k by official success (Pi05_past: 50k), scored on the
   corrected 1,500 (seed 46, episode seeds 5000/5500/6000). This carries the decision; the official and pooled
   numbers are secondary. Pi05_past@50k, recorded 2026-10-29 before pi05_p2n was launched: 1445/1500 = 96.3%
   (Wilson 95% 95.3-97.2; blocks 96.4 / 95.8 / 96.8; lowest task KITCHEN_SCENE8 84.0%), in
   `output/training/pi05_past_libero10_seed42/eval/upd-050000_model_model_corrected_n500_seed46_ep{5000,5500,6000}`.
3. Decision: pi05_p2n is non-inferior if the lower 95% bound of its paired difference is above -3 points; it is
   better only at >= +2 points with p < 0.05, confirmed by a seed-43 pair of both runs. On 1,500 paired episodes
   the detectable difference is about 2.5-3 points. McNemar ignores training-seed variance: one seed per run.
4. Never stop or select on validation loss (Pi05_past's held-out loss rose while its success rose). Stop only
   for mechanical failures: non-finite loss, skipped updates, `train/loss_ar` not below ~7 by update 3k (a healthy
   AR-VLA head was at 5.7, chance is 8.5), or `self_past_rows` far from ~1.9 per micro-batch. Never stop on success
   before 40k.
5. Secondary: the chosen snapshot at T=1 / top-k 10 and with `--use-k-tokens 4`, history interventions, command
   statistics, offline reliance. Follow-up arms: the AR no-past control (`--use-past false`), then a seed-43 pair
   if the gap is under 6 points.

Validation losses and reconstruction MSE are not comparable across the two heads (a flow sample carries sampling
noise; greedy decoding is a mode): compare runs only in closed loop.

`scripts/compare_paired_evals.py` runs steps 1 and 2. It matches episodes on `(task_name, init_state_id,
episode_seed)`, refuses different episode sets or schedules, and reports both rates, the discordant counts, the
paired difference with a 95% interval and the exact McNemar p (`--per-task` adds the per-task lines):

```bash
/venv/oat/bin/python scripts/compare_paired_evals.py --runs output/training/pi05_past_libero10_seed42 \
    output/training/pi05_p2n_libero10_seed42                       # step 1: every step both runs scored
E=eval/upd-050000_model_model_corrected_n500_seed46_ep             # step 2: the chosen snapshots' corrected 1,500
/venv/oat/bin/python scripts/compare_paired_evals.py \
    --a output/training/pi05_past_libero10_seed42/${E}{5000,5500,6000} \
    --b output/training/pi05_p2n_libero10_seed42/eval/<chosen>_model_model_corrected_n500_seed46_ep{5000,5500,6000}
```

## Tests

```bash
/venv/oat/bin/python -m pytest tests/test_pi05_p2n.py -q -m "not slow and not gpu"   # 27 tests, CPU, ~25 s
/venv/oat/bin/python -m pytest tests/test_pi05_p2n.py -q -m slow                      # 4 tiny workspace runs
CUDA_VISIBLE_DEVICES=0 /venv/oat/bin/python -m pytest tests/test_pi05_p2n.py -q -m gpu   # bf16 CUDA path
```

They cover:
- the trainable scope, which is Pi05_past's apart from the head;
- the condition K/V, equal to Pi05_past's under the fold;
- gradient coverage;
- cached decoding logits equal to the teacher-forcing logits, on weights whose segment rows and slot embeddings are
  perturbed (the tiny init makes argmax checks vacuous);
- the sampler at every decoding step: T=1 / top-k 10 in self-past, greedy at deployment, and evaluator overrides;
- the sampler equal to the original Past2Next rule;
- the warmup and probability schedule on the run's own path;
- self-past content, which is actions 1..7 of the previous window's decode;
- export/restore and the config diff against the Pi05_past run;
- the shell launchers, resume, retention and 2-rank DDP.
