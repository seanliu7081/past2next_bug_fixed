# Pi05 P2N VLM v1

`pi05_p2n_vlm_v1` uses the pretrained Gemma-2B VLM to autoregressively predict the same eight frozen-OAT action tokens as `pi05_p2n`. There is no Gemma-300M expert or flow head. OAT decodes the eight tokens into a 16-step action chunk; the runner executes eight steps and acknowledges the commands actually executed.

The input is `[image, language/state, history, BOS, previous action tokens]`. Seven executed actions plus their latest first and second differences form nine continuous history embeddings projected to the VLM width. Nine learned slot embeddings identify both feature type and temporal lag; validity masks hide unavailable history. Observation queries read observations; history queries read observations and valid history; action queries also read preceding action inputs. History is not quantized. Teacher forcing predicts eight ground-truth OAT IDs, with loss only at action positions. Inference prefills the observation/history cache and then decodes with the VLM cache.

History order is `[acc, jerk, a(t-7), ..., a(t-1)]`. The two differences require two and three valid recent actions, respectively. RoPE reserves all nine history positions even in the no-history control. The tied action table has 5001 rows at VLM width (5000 OAT codes and BOS), initialized with the expert-AR baseline's `normal(0, 0.76/sqrt(width))`. Inputs scale by `sqrt(width)`; the baseline's 5001-class CE is retained, with BOS forbidden during sampling. All VLM LoRA projections, including the final layer, can receive the main action loss. OAT remains frozen and retains its own normalizer; history uses the training-set action normalizer.

The initial comparison preserves the `pi05_p2n` dataset, OAT checkpoint, image augmentation, LoRA rank 32, trainable vision encoder, batch 32, 80k optimizer updates and no EMA. Self-past starts at update 1,000 with probability 0.5 and T=1/top-k=10; deployment is greedy. A greedy self-past sampler is a separate experiment. The new variant has its own checkpoint identity and cannot resume an expert-AR checkpoint.

```bash
# Compose and validate without loading assets or touching GPUs.
bash train_pi05_p2n_vlm_v1.sh --devices 0,1 --dry-run

# Short memory/throughput probe, including generated history and a small rollout.
bash train_pi05_p2n_vlm_v1.sh --devices 0,1 --probe \
  --output /tmp/pi05_p2n_vlm_v1_probe

# Train only after the probe passes. Choose available GPUs explicitly.
bash train_pi05_p2n_vlm_v1.sh --devices 0,1 \
  --output output/training/pi05_p2n_vlm_v1_seed42

# Continue a run; preserve its GPU count and recipe.
bash train_pi05_p2n_vlm_v1.sh --devices 0,1 \
  --resume output/training/pi05_p2n_vlm_v1_seed42/checkpoints/latest.ckpt
```

The launcher automatically chooses accumulation to keep batch 32 on 1/2/4/8 GPUs. An explicit Hydra `training.gradient_accumulate_every` override takes precedence. Other overrides follow `--`; e.g. `-- policy.use_past=false policy.self_past_p=0 training.validate_generated_history=false` selects the no-history control. Logging defaults to offline; use `logging.mode=online` when desired. The inherited `lambda_ki`, `ki_skip` and `adarms_t0` fields are null placeholders; non-null values are rejected.

Evaluate saved weights using `scripts/evaluate_p2n_vla.py` with the same task schedule, initial states and seeds as the baseline. Offline validation also records `val/free_running_token_acc`, `val/full_sequence_acc` and `val/normalized_executed_action_mse` from the existing reconstruction decode with expert history. Compare these with held-out teacher-forced token accuracy, closed-loop success and inference latency. A higher training token accuracy alone does not establish improvement. Delete disposable probe output once its results have been recorded; retain real training checkpoints and regression tests.
