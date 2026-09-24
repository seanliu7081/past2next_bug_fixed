#!/usr/bin/env python
"""Opt-in two-rank smoke. Default is a tiny CPU flow, never a full-model claim.

CPU execution:
  /venv/real_robot/bin/python -m torch.distributed.run --standalone \
    --nproc_per_node=2 tests/p2n_latent_flow_ddp_smoke.py

Real model (explicitly allocates the user's selected GPUs):
  CUDA_VISIBLE_DEVICES=2,3 /venv/real_robot/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=2 tests/p2n_latent_flow_ddp_smoke.py \
    --real --variant p2n_latent_flow --task real_robot --dino /local/dino/snapshot

Repeat --real with p2n_state_gate_latent_flow for the other variant. Real mode
builds the unmodified 16x768 recipe, runs FM+CT/self-past/Adam/EMA and full 8/2-step
predictions, measures synchronized latency and memory. CPU mode also checks a
strict EMA state/RNG round trip. It does not write a long training checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate

from oat.model.diffusion.ema_model import EMAModel
from oat.workspace.train_p2n_latent_flow import (
    MetricSums, NonPaddingDistributedSampler, _module_digest,
    assert_teacher_synchronized, make_fresh_ema, successful_update,
    validate_optimizer_ownership,
)


def toy_mode(args, rank, device):
    # Reuse the integration fixtures' tiny production modules and actual OAT/FSQ.
    # Only the external frozen DINO backbone is replaced by a local mock.
    sys.path.insert(0, str(ROOT / "tests"))
    import transformers
    from test_p2n_new_policy import MockDINOBackbone
    from test_p2n_latent_flow_policy import make_flow, make_batch
    from oat.policy.p2n_latent_flow_common import prepare_flow_training_batch
    transformers.DINOv3ViTModel = MockDINOBackbone
    student = make_flow(args.variant == "p2n_state_gate_latent_flow").to(device)
    optimizer = student.get_optimizer(policy_lr=.005, obs_enc_lr=.005)
    batch = make_batch(student, 4)
    def prepare(student, teacher, generator, self_past_generator):
        return prepare_flow_training_batch(batch, student=student, teacher=teacher,
            generator=generator, self_past_generator=self_past_generator)
    return student, optimizer, prepare, batch


def real_mode(args, rank, device):
    import hydra
    from scripts.train_p2n_latent_flow import compose_config, validate_dino_source, validate_tokenizer_source
    from oat.policy.p2n_latent_flow_common import prepare_flow_training_batch
    overrides = [f"policy.dino_path={json.dumps(args.dino)}", "dataloader.num_workers=0",
                 "dataloader.persistent_workers=false", "val_dataloader.num_workers=0",
                 "val_dataloader.persistent_workers=false"]
    if args.tokenizer:
        overrides.append(f"policy.tokenizer_checkpoint={json.dumps(args.tokenizer)}")
    if args.data:
        overrides.append(f"task.policy.dataset.zarr_path={json.dumps(args.data)}")
    cfg = compose_config(args.variant, args.task, overrides)
    validate_dino_source(cfg)
    validate_tokenizer_source(cfg)
    student = hydra.utils.instantiate(cfg.policy).to(device)
    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    student.set_normalizer(dataset.get_normalizer())
    optimizer = student.get_optimizer(**cfg.optimizer)
    candidates = []
    # Every smoke row exercises a valid previous window at the maximum curriculum.
    for index in range(rank, len(dataset), torch.distributed.get_world_size()):
        sample = dataset[index]
        if bool(sample["prev_window_valid"]):
            candidates.append(sample)
        if len(candidates) == int(cfg.dataloader.batch_size):
            break
    if len(candidates) != int(cfg.dataloader.batch_size):
        raise ValueError("Not enough valid previous windows for a full real-model smoke microbatch")
    from oat.workspace.train_p2n_latent_flow import _move
    batch = _move(default_collate(candidates), device)
    student.set_self_past_step(student.self_past_warmup_steps + student.self_past_ramp_steps)
    def prepare(student, teacher, generator, self_past_generator):
        return prepare_flow_training_batch(batch, student=student, teacher=teacher,
                                          generator=generator, self_past_generator=self_past_generator)
    return student, optimizer, prepare, batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--variant", choices=("p2n_latent_flow", "p2n_state_gate_latent_flow"), default="p2n_latent_flow")
    parser.add_argument("--task", choices=("real_robot", "libero"), default="real_robot")
    parser.add_argument("--dino")
    parser.add_argument("--tokenizer")
    parser.add_argument("--data")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.updates < 3:
        parser.error("--updates must be at least three to test beyond zero initialization")
    if args.latency_repeats < 1:
        parser.error("--latency-repeats must be positive")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.real:
        if not args.dino or not torch.cuda.is_available():
            parser.error("--real requires --dino and explicitly selected CUDA GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)
    torch.distributed.init_process_group("nccl" if args.real else "gloo")
    try:
        if torch.distributed.get_world_size() != 2:
            raise ValueError("Smoke expects exactly two ranks")
        # Deliberately different initialization seeds reveal pre-sync EMA bugs.
        torch.manual_seed(231 + rank)
        student, optimizer, prepare, batch = real_mode(args, rank, device) if args.real else toy_mode(args, rank, device)
        wrapped = DDP(student, device_ids=[local_rank] if args.real else None, find_unused_parameters=False)
        teacher = make_fresh_ema(student)
        assert _module_digest(student) == _module_digest(teacher)
        assert_teacher_synchronized(teacher)
        validate_optimizer_ownership(student, teacher, optimizer)
        ema = EMAModel(teacher, power=.75)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        generator = torch.Generator(device=device).manual_seed(998 + rank)
        past_generator = torch.Generator(device=device).manual_seed(1898 + rank)
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if args.real else nullcontext()
        if args.real:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.monotonic()
        losses = []
        for update_index in range(args.updates):
            wrapped.train()
            optimizer.zero_grad(set_to_none=True)
            group_length = 1 if update_index == args.updates - 1 else 2
            for microbatch in range(group_length):
                with autocast():
                    prepared = prepare(student, teacher, generator, past_generator)
                with wrapped.no_sync() if microbatch < group_length - 1 else nullcontext():
                    with autocast():
                        loss = wrapped(prepared)
                    assert torch.isfinite(loss)
                    (loss / group_length).backward()
                losses.append(float(loss.detach()))
            missing = [name for name, p in student.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing, f"Unused trainable parameters: {missing}"
            assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.requires_grad)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
            successful_update(student, ema, scheduler)
        assert ema.optimization_step == args.updates
        assert_teacher_synchronized(teacher)
        if args.real:
            torch.cuda.synchronize(device)
        report = {"mode": "real_full_model" if args.real else "cpu_tiny_flow_contract_only",
                  "variant": args.variant, "rank": rank,
                  "successful_updates": ema.optimization_step, "accumulation": 2, "tail_group_size": 1,
                  "seconds": time.monotonic() - start, "losses": losses}
        # Strict EMA roundtrip preserves a teacher that differs from the student.
        # A real-model duplicate would contaminate the memory measurement, so do
        # this serialization check only in CPU toy mode.
        if not args.real:
            stream = io.BytesIO()
            torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(),
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            generator=generator.get_state(), past_generator=past_generator.get_state(),
                            ema_step=ema.optimization_step, ema_decay=ema.decay), stream)
            stream.seek(0)
            saved = torch.load(stream, weights_only=False)
            teacher_copy = make_fresh_ema(student)
            teacher_copy.load_state_dict(saved["teacher"], strict=True)
            assert _module_digest(teacher_copy) == _module_digest(teacher)
            next_noise = torch.randn(8, generator=generator)
            generator.set_state(saved["generator"])
            assert torch.equal(next_noise, torch.randn(8, generator=generator))
        # Deliberately uneven validation: one rank receives no samples.
        sampler = NonPaddingDistributedSampler(range(1), rank, 2)
        sums = MetricSums(["masked_mse"], device)
        for _ in sampler:
            sums.add({"masked_mse": {"sum": torch.tensor(21., device=device), "count": 7}})
        assert sums.reduce()["masked_mse"] == 3.
        if args.real:
            import numpy as np
            report["train_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
            report["train_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2 ** 20
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad(), autocast():
                for mode in ("expert", "generated"):
                    evaluated = teacher.validation_metrics(batch, generator=generator, history_mode=mode)
                    report[f"validation_{mode}"] = {key: float(value["sum"] / value["count"])
                                                     for key, value in evaluated.items() if value["count"] > 0}
            torch.cuda.synchronize(device)
            report["validation_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
            report["validation_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2 ** 20
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad(), autocast():
                sample = {key: value[:1] for key, value in batch["obs"].items()}
                for steps in (8, 2):
                    times = []
                    for repeat in range(args.latency_repeats + 1):
                        torch.cuda.synchronize(device)
                        started = time.monotonic()
                        output = teacher.predict_action(sample, past_actions=batch["past_action"][:1],
                            past_action_valid=batch["past_action_valid"][:1], num_flow_steps=steps,
                            generator=generator)
                        torch.cuda.synchronize(device)
                        assert output["action_pred"].shape == (1, 16, 7)
                        assert torch.isfinite(output["action_pred"]).all()
                        if repeat:
                            times.append(time.monotonic() - started)
                    report[f"inference_{steps}_steps_ms"] = {"p50": float(np.percentile(times, 50) * 1000),
                                                            "p95": float(np.percentile(times, 95) * 1000)}
            report.update(gpu=torch.cuda.get_device_name(device), precision="bf16 network/fp32 flow",
                          peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 2 ** 20,
                          peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 2 ** 20,
                          cameras=2, observation_frames=2, weights="ema",
                          activation_checkpointing=True)
        print(json.dumps(report), flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
