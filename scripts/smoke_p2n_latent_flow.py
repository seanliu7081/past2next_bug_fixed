#!/usr/bin/env python
"""User-invoked, bounded acceptance of the full real latent-flow model on ONE GPU.

This runs real optimizer updates. It is separate from launcher --dry-run and
never substitutes tiny models or synthetic observations for production assets.
No policy checkpoint is saved: results are diagnostics, not trained artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides = []
    if "--" in argv:
        split = argv.index("--")
        overrides, argv = argv[split + 1:], argv[:split]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("p2n_latent_flow", "p2n_state_gate_latent_flow"), required=True)
    parser.add_argument("--task", choices=("real_robot", "libero"), required=True)
    parser.add_argument("--gpus", required=True, help="Exactly one explicit physical GPU index, e.g. 2")
    parser.add_argument("--dino", required=True)
    parser.add_argument("--dino-revision")
    parser.add_argument("--tokenizer", help="Required for LIBERO; real_robot uses the approved default")
    parser.add_argument("--output", type=Path, required=True, help="New, empty diagnostic output directory")
    parser.add_argument("--iterations", type=int, default=3, help="Successful optimizer updates (minimum three)")
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="CPU preflight only; do not create a GPU model or run updates")
    args = parser.parse_args(argv)
    if not args.gpus.isdigit():
        parser.error("This acceptance script takes exactly one physical GPU in --gpus; use the separate DDP smoke for two ranks")
    if args.iterations < 3:
        parser.error("--iterations must be at least three to check gradients after zero initialization")
    if args.latency_repeats < 3:
        parser.error("--latency-repeats must be at least three")
    return args, overrides


def main(argv=None):
    args, overrides = parse_args(argv)
    # Select visibility before importing torch or any policy dependency.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    from scripts.train_p2n_latent_flow import compose_config, preflight, check_gpu_selection
    from oat.workspace.train_p2n_latent_flow import (
        make_fresh_ema, validate_optimizer_ownership, successful_update,
        _move, _slice_sample, stable_validation_seed,
    )
    from oat.policy.p2n_latent_flow_common import prepare_flow_training_batch
    from oat.model.diffusion.ema_model import EMAModel
    from omegaconf import OmegaConf
    import hydra
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from accelerate.utils import set_seed

    generated = [f"policy.dino_path={json.dumps(str(args.dino))}"]
    for value, key in ((args.tokenizer, "policy.tokenizer_checkpoint"), (args.dino_revision, "policy.dino_revision")):
        if value is not None:
            generated.append(f"{key}={json.dumps(str(value))}")
    cfg = compose_config(args.variant, args.task, [*generated, *overrides])
    for key, value in {"embed_dim": 768, "n_layers": 16, "n_heads": 12, "ffn_dim": 2048,
                       "num_visual_queries": 64, "resampler_depth": 2}.items():
        if cfg.policy[key] != value:
            raise ValueError(f"Full-model acceptance requires policy.{key}={value}")
    if cfg.dataloader.batch_size != 4 or not cfg.policy.activation_checkpointing:
        raise ValueError("Full-model acceptance starts at microbatch four with activation checkpointing enabled")
    if cfg.policy.flow.self_past_steps != 8 or cfg.policy.flow.inference_steps != 8:
        raise ValueError("Initial acceptance uses eight-step inference and self-past; two-step inference is measured separately")
    if cfg.policy.self_past_p != 0.5:
        raise ValueError("Full-model acceptance uses maximum curriculum self-past probability 0.5")
    report = preflight(cfg, args.output, world_size=1)
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return report
    check_gpu_selection([args.gpus], args.allow_busy_gpus)
    if not torch.cuda.is_available():
        raise RuntimeError("The explicitly selected GPU is not available")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not cfg.training.allow_bf16 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full-model acceptance requires the configured BF16-capable GPU")
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), args.output / "resolved_config.yaml")
    report.update(status="running", scope="bounded single-GPU full-model acceptance", training_started=True,
                  physical_gpu=args.gpus, hardware={"name": torch.cuda.get_device_name(device),
                  "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                  "compute_capability": list(torch.cuda.get_device_capability(device))},
                  precision="bf16 network autocast; fp32 objective/projection/Euler", world_size=1,
                  limitations=["This is a bounded runtime check, not trained quality or robot closed-loop acceptance.",
                               "Single-GPU results do not validate two-rank DDP.",
                               "Inference memory includes resident student, EMA, optimizer and training batches."])

    def memory():
        torch.cuda.synchronize(device)
        return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}

    def visible_window_batch(dataset, count):
        # Real samples only. Randomized selection avoids taking exclusively
        # episode-start windows that cannot exercise generated past commands.
        rng = torch.Generator().manual_seed(int(cfg.training.seed) + 81)
        selected = []
        for index in torch.randperm(len(dataset), generator=rng).tolist():
            example = dataset[index]
            if bool(example["prev_window_valid"]):
                selected.append(index)
                if len(selected) == count:
                    break
        if len(selected) != count:
            raise ValueError(f"Dataset needs {count} samples with valid previous windows for this acceptance check")
        return next(iter(DataLoader(Subset(dataset, selected), batch_size=count, num_workers=0, shuffle=False)))

    try:
        set_seed(int(cfg.training.seed))
        dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
        validation_dataset = dataset.get_validation_dataset()
        batch = _move(visible_window_batch(dataset, 4), device)
        validation_batch = _move(visible_window_batch(validation_dataset, 1), device)
        student = hydra.utils.instantiate(cfg.policy)
        student.set_normalizer(dataset.get_normalizer())
        student.to(device).train()
        # Explicitly reach the final curriculum probability, while preserving
        # the real recipe's 0.5 random choice in every training preparation.
        curriculum_start = student.self_past_warmup_steps + student.self_past_ramp_steps
        student.set_self_past_step(curriculum_start)
        teacher = make_fresh_ema(student)
        optimizer = student.get_optimizer(**cfg.optimizer)
        validate_optimizer_ownership(student, teacher, optimizer)
        ema_config = dict(OmegaConf.to_container(cfg.ema, resolve=True))
        ema_config.pop("_target_", None)
        ema = EMAModel(teacher, **ema_config)
        # Constant configured LR is intentional for this short gradient check;
        # the full training workspace owns the long-run LR schedule.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        train_generator = torch.Generator(device=device).manual_seed(int(cfg.training.seed) + 11)
        past_generator = torch.Generator(device=device).manual_seed(int(cfg.training.seed) + 29)
        accumulation = int(cfg.training.gradient_accumulate_every)
        report.update(parameters=student.parameter_counts(), activation_checkpointing=True,
                      microbatch=4, accumulation=accumulation, bounded_lr_schedule="constant configured optimizer rates",
                      selected_sample_ids=batch["sample_id"].tolist(), self_past_probability=0.5,
                      self_past_steps=8, self_past_chunk_size=student.self_past_chunk_size)
        teacher_calls = [0]
        def count_teacher_call(module, inputs, output):
            if torch.is_grad_enabled() or teacher.training:
                raise RuntimeError("EMA teacher unexpectedly ran in training/grad mode")
            teacher_calls[0] += 1
        handle = teacher.model.register_forward_hook(count_teacher_call)
        records, changed_rows = [], 0
        for update in range(args.iterations):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            losses = []
            for _ in range(accumulation):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prepared = prepare_flow_training_batch(batch, student=student, teacher=teacher,
                        generator=train_generator, self_past_generator=past_generator)
                    changed_rows += int((prepared.past_actions != batch["past_action"]).flatten(1).any(1).sum())
                    loss = student(prepared)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite full-model flow loss")
                (loss / accumulation).backward()
                losses.append(float(loss.detach()))
                del prepared, loss
            trainable = [p for p in student.parameters() if p.requires_grad]
            if any(p.grad is None for p in trainable):
                raise RuntimeError("Some student trainable parameters did not participate in the packed forward")
            norm = torch.nn.utils.clip_grad_norm_(trainable, float(cfg.training.max_grad_norm))
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite full-model gradient norm")
            active_parameters = sum(p.numel() for p in trainable if bool((p.grad != 0).any()))
            optimizer.step()
            successful_update(student, ema, scheduler)
            if teacher.training or any(p.grad is not None for p in teacher.parameters()):
                raise RuntimeError("Teacher accumulated gradients or changed to train mode")
            record = {"successful_update": update + 1, "mean_loss": sum(losses) / len(losses),
                      "gradient_norm_before_clip": float(norm), "parameter_elements_with_nonzero_gradient": active_parameters,
                      "seconds": 0.0, "ema_updates": ema.optimization_step, **memory()}
            record["seconds"] = time.perf_counter() - started
            records.append(record)
            print(json.dumps(record), flush=True)
        handle.remove()
        if teacher_calls[0] == 0 or changed_rows == 0:
            raise RuntimeError("Bounded run did not exercise both EMA CT and generated previous commands")
        if records[-1]["parameter_elements_with_nonzero_gradient"] == 0:
            raise RuntimeError("No active gradients after several zero-initialized updates")
        report.update(updates=records, successful_optimizer_updates=args.iterations,
                      ema_teacher_forward_calls=teacher_calls[0], generated_history_row_count=changed_rows)
        optimizer.zero_grad(set_to_none=True)
        teacher.eval().reset()
        sample = _slice_sample(batch, 0)
        latency = {}
        for steps in (8, 2):
            rng = torch.Generator(device=device).manual_seed(int(cfg.training.seed) + 100 + steps)
            durations = []
            torch.cuda.reset_peak_memory_stats(device)
            for iteration in range(args.latency_repeats + 1):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = teacher.predict_action(sample["obs"], past_actions=sample["past_action"],
                        past_action_valid=sample["past_action_valid"], num_flow_steps=steps, generator=rng)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                if prediction["action_pred"].shape != (1, 16, 7) or not torch.isfinite(prediction["action_pred"]).all():
                    raise RuntimeError("Malformed/nonfinite full-model prediction")
                if iteration:
                    durations.append(elapsed)
            latency[str(steps)] = {"steps": steps, "weights": "EMA after bounded updates", "batch_size": 1,
                "camera_count": 2, "frames_per_camera": 2, "repeats": args.latency_repeats,
                "p50_seconds": float(np.percentile(durations, 50)), "p95_seconds": float(np.percentile(durations, 95)),
                "durations_seconds": durations, **memory()}
        report["full_predict_action_latency"] = latency
        validation = {}
        for mode in ("expert", "generated"):
            seed = stable_validation_seed(int(cfg.training.seed), validation_dataset.dataset_identity,
                                          int(validation_batch["sample_id"].item()))
            rng = torch.Generator(device=device).manual_seed(seed)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                measured = teacher.validation_metrics(validation_batch, rng, history_mode=mode)
            validation[mode] = {key: {"sum": float(value["sum"]), "count": float(value["count"])}
                                for key, value in measured.items()}
            validation[mode]["memory"] = memory()
        report["single_heldout_sample_diagnostic"] = validation
        report["status"] = "passed bounded single-GPU production runtime checks"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        try:
            report["failure_memory"] = memory()
        except Exception:
            pass
        raise
    finally:
        path = args.output / "smoke_report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Full-model diagnostic report: {path}", flush=True)
    return report


if __name__ == "__main__":
    main()
