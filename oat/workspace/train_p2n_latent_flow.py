"""Independent latent-flow training, strict artifacts, and unbiased validation.

Only the student is DDP wrapped. Target preparation runs before its single
packed forward; the complete EMA conditioner remains an ordinary eval module.
Checkpoints are written at epoch boundaries, after accumulation has flushed.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Sampler
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from oat.common.hydra_util import register_new_resolvers
from oat.common.checkpoint_util import TopKCheckpointManager
from oat.common.p2n_new_capabilities import resolve_update_schedule
from oat.model.common.lr_scheduler import get_scheduler
from oat.model.diffusion.ema_model import EMAModel
from oat.workspace.base_workspace import BaseWorkspace, _atomic_torch_save, _copy_to_cpu

register_new_resolvers()

POLICY_FAMILY = "oat_latent_flow"
ARTIFACT_SCHEMA_VERSION = 1
VARIANTS = {"p2n_latent_flow", "p2n_state_gate_latent_flow"}
METRICS = ("fm_loss", "decoded_action_mse", "translation_mse", "rotation_mse",
           "gripper_mse", "oat_autoencoding_mse", "projection_distance",
           "projection_out_of_bounds", "projection_legal")


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _slice_sample(value, index):
    if isinstance(value, torch.Tensor):
        return value[index:index + 1]
    if isinstance(value, dict):
        return {key: _slice_sample(item, index) for key, item in value.items()}
    raise TypeError(f"Unsupported validation batch value: {type(value).__name__}")


def stable_validation_seed(seed: int, dataset_identity: str, sample_id: int) -> int:
    """Stable across processes, views, batch sizes, and Python hash seeds."""
    payload = f"latent-flow-validation-v1\0{seed}\0{dataset_identity}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


class NonPaddingDistributedSampler(Sampler):
    """Disjoint rank-strided indices, including legitimately empty ranks."""
    def __init__(self, dataset, rank=0, world_size=1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid rank/world size")
        self.size, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return max(0, (self.size - self.rank + self.world_size - 1) // self.world_size)


class _LimitedBatchSampler:
    def __init__(self, sampler, maximum):
        if maximum < 1:
            raise ValueError("max_train_steps must be positive")
        self.sampler, self.maximum = sampler, int(maximum)
        self.batch_size, self.drop_last = sampler.batch_size, sampler.drop_last

    def __len__(self):
        return min(len(self.sampler), self.maximum)

    def __iter__(self):
        from itertools import islice
        yield from islice(self.sampler, self.maximum)


def _cap_loader(loader, maximum):
    kwargs = dict(dataset=loader.dataset,
                  batch_sampler=_LimitedBatchSampler(loader.batch_sampler, maximum),
                  num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                  pin_memory=loader.pin_memory, persistent_workers=loader.persistent_workers,
                  worker_init_fn=loader.worker_init_fn, generator=loader.generator)
    if loader.num_workers:
        kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(**kwargs)


def _module_digest(module):
    digest = hashlib.sha256()
    for key, tensor in module.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Unexpected non-tensor policy state {key}")
        value = tensor.detach().contiguous().cpu()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_fresh_ema(synchronized_student):
    """Call only AFTER DDP has synchronized the student, on every rank."""
    teacher = copy.deepcopy(synchronized_student)
    teacher.requires_grad_(False).eval()
    if hasattr(teacher, "reset"):
        teacher.reset()
    return teacher


def assert_teacher_synchronized(teacher):
    """Exact state hashes establish first-CT equality without a teacher DDP."""
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("EMA teacher must be eval and have no trainable parameters")
    digest = _module_digest(teacher)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        digests = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(digests, digest)
        if len(set(digests)) != 1:
            raise RuntimeError("EMA teachers differ across ranks before the first CT batch")
    return digest


def validate_optimizer_ownership(student, teacher, optimizer):
    expected = {id(p) for p in student.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Optimizer must contain each trainable student parameter exactly once")
    if set(actual) & {id(p) for p in teacher.parameters()}:
        raise ValueError("Teacher parameters cannot appear in the student optimizer")
    if any("teacher" in key.split(".") for key in student.state_dict()):
        raise ValueError("Student cannot register its EMA teacher")


def successful_update(student, ema, scheduler):
    """One call per successful optimizer step; never call for microbatches."""
    scheduler.step()
    student.on_optimizer_step()
    ema.step(student)
    # The curriculum is a counter, and all other buffers are state, not averages.
    with torch.no_grad():
        source = dict(student.named_buffers())
        for name, target in ema.averaged_model.named_buffers():
            target.copy_(source[name])
    ema.averaged_model.set_self_past_step(student.self_past_step)
    ema.averaged_model.requires_grad_(False).eval()


class MetricSums:
    """Accumulate numerators/denominators in FP64 before the final reduction."""
    def __init__(self, names, device="cpu"):
        self.names = tuple(names)
        self.values = torch.zeros((len(self.names), 2), dtype=torch.float64, device=device)

    def add(self, metrics):
        unknown = set(metrics) - set(self.names)
        if unknown:
            raise KeyError(f"Unregistered metric names: {sorted(unknown)}")
        for name, pair in metrics.items():
            if isinstance(pair, dict):
                numerator, denominator = pair["sum"], pair["count"]
            else:
                numerator, denominator = pair
            numerator = torch.as_tensor(numerator, device=self.values.device, dtype=torch.float64)
            denominator = torch.as_tensor(denominator, device=self.values.device, dtype=torch.float64)
            if numerator.numel() != 1 or denominator.numel() != 1:
                raise ValueError("Metric sums/counts must be scalars")
            if not torch.isfinite(numerator) or not torch.isfinite(denominator) or denominator < 0:
                raise ValueError(f"Nonfinite or negative metric accumulation: {name}")
            row = self.names.index(name)
            self.values[row, 0] += numerator.reshape(())
            self.values[row, 1] += denominator.reshape(())

    def reduce(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.values, op=torch.distributed.ReduceOp.SUM)
        return {name: (float(row[0] / row[1]) if row[1] > 0 else None)
                for name, row in zip(self.names, self.values)}


def _software_versions():
    result = {}
    for name in ("torch", "transformers", "accelerate", "hydra-core"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class TrainP2NLatentFlowWorkspace(BaseWorkspace):
    include_keys = ("epoch", "global_step", "completed_optimizer_steps", "ema_state",
                    "lr_scheduler_state", "rng_states", "update_schedule", "dataset_split",
                    "best_metric", "skipped_optimizer_steps")

    def __init__(self, cfg, output_dir=None, lazy_instantiation=True):
        super().__init__(cfg, output_dir)
        self.model = self.ema_model = self.optimizer = None
        self.epoch = self.global_step = self.completed_optimizer_steps = 0
        self.skipped_optimizer_steps = 0
        self.ema_state = self.lr_scheduler_state = self.rng_states = None
        self.update_schedule = self.dataset_split = None
        self.best_metric = math.inf
        self.generators = {}
        if not lazy_instantiation:
            self.model = hydra.utils.instantiate(cfg.policy)
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)
            self.ema_model = make_fresh_ema(self.model)

    @staticmethod
    def validate_resume_payload(payload, cfg):
        metadata = payload.get("metadata", {})
        if metadata.get("policy_family") != POLICY_FAMILY:
            raise ValueError("Resume policy_family must be oat_latent_flow; AR artifacts are incompatible")
        if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported latent-flow artifact schema version")
        if metadata.get("variant") != cfg.variant or cfg.variant not in VARIANTS:
            raise ValueError("Resume variant does not match the requested flow variant")
        if metadata.get("task_type", metadata.get("task")) != cfg.task_type:
            raise ValueError("Resume task/action semantics do not match")
        if not cfg.training.use_ema:
            raise ValueError("Consistency training requires use_ema=true")
        required = {"model", "ema_model", "optimizer"}
        if not required.issubset(payload.get("state_dicts", {})):
            raise ValueError("Resume needs student, EMA, and optimizer state")
        if not set(TrainP2NLatentFlowWorkspace.include_keys).issubset(payload.get("pickles", {})):
            raise ValueError("Resume artifact lacks complete training continuation state")
        config = payload.get("policy_config", {})
        if config.get("construction_mode") != "restore":
            raise ValueError("Resume requires a self-contained restore policy configuration")
        saved = OmegaConf.create(_plain(payload["cfg"]))
        # Comparing the full resolved policy catches flow timing, solver, gate,
        # schema, self-past, and normalization contracts. Sources may move because
        # restore always constructs from the embedded offline configuration.
        ignored = {"dino_path", "dino_revision", "tokenizer_checkpoint", "construction_mode"}
        for key in set(saved.policy) | set(cfg.policy):
            if key not in ignored and _plain(saved.policy.get(key)) != _plain(cfg.policy.get(key)):
                raise ValueError(f"Resume policy contract differs: policy.{key}")
        for section in ("ema", "optimizer"):
            if _plain(saved.get(section)) != _plain(cfg.get(section)):
                raise ValueError(f"Resume cannot change {section} configuration")
        for key in ("gradient_accumulate_every", "lr_scheduler", "lr_warmup_steps", "lr_warmup_ratio", "seed"):
            if saved.training.get(key) != cfg.training.get(key):
                raise ValueError(f"Resume training contract differs: training.{key}")
        if saved.dataloader.batch_size != cfg.dataloader.batch_size:
            raise ValueError("Resume cannot change per-rank training microbatch")
        old_data, new_data = saved.task.policy.dataset, cfg.task.policy.dataset
        for key in set(old_data) | set(new_data):
            if key != "zarr_path" and _plain(old_data.get(key)) != _plain(new_data.get(key)):
                raise ValueError(f"Resume data schema/split differs: dataset.{key}")
        return payload

    @staticmethod
    def _read_payload(path):
        return torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)

    @staticmethod
    def _split_metadata(dataset, validation):
        train = np.asarray(dataset.train_mask, dtype=np.bool_)
        val = np.asarray(validation.train_mask, dtype=np.bool_)
        if train.shape != val.shape or np.any(train & val):
            raise ValueError("Training and validation episode masks overlap or differ in shape")
        identity = str(dataset.dataset_identity)
        if str(validation.dataset_identity) != identity:
            raise ValueError("Validation must preserve its source dataset identity")
        return {"schema_version": 1, "identity": identity,
                "sample_id_schema": "absolute_action_anchor_v1",
                "train_episode_ids": np.flatnonzero(train).tolist(),
                "validation_episode_ids": np.flatnonzero(val).tolist(),
                "train_windows": len(dataset), "validation_windows": len(validation)}

    def _capture_rng(self):
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                "generators": {name: value.get_state() for name, value in self.generators.items()}}

    def _gather_rng(self, accelerator):
        state = self._capture_rng()
        if accelerator.num_processes > 1:
            self.rng_states = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(self.rng_states, state)
        else:
            self.rng_states = [state]

    def _restore_rng(self, rank=0, world_size=1):
        if self.rng_states is None or len(self.rng_states) != world_size:
            raise ValueError("Exact resume requires the original world size and every rank's RNG")
        state = self.rng_states[rank]
        if set(state["generators"]) != set(self.generators):
            raise ValueError("Saved dedicated RNG streams differ from this run")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
        for name, generator in self.generators.items():
            generator.set_state(state["generators"][name])

    def _load_continuation(self, payload):
        for key in self.include_keys:
            setattr(self, key, dill.loads(payload["pickles"][key]))
        if self.ema_state["optimization_step"] != self.completed_optimizer_steps:
            raise ValueError("EMA schedule and successful update counters disagree")
        if self.model.self_past_step != self.completed_optimizer_steps:
            raise ValueError("Student curriculum and successful update counters disagree")

    def save_checkpoint(self, path=None, tag="latest", **_):
        if self.rng_states is None:
            self.rng_states = [self._capture_rng()]
        path = Path(path) if path is not None else self.get_checkpoint_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.model.artifact_metadata()
        root = Path(__file__).resolve().parents[2]
        for relative in ("oat/workspace/train_p2n_latent_flow.py", "oat/dataset/latent_flow_dataset.py",
                         "scripts/train_p2n_latent_flow.py"):
            metadata.setdefault("source_sha256", {})[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        metadata.update(policy_family=POLICY_FAMILY, variant=self.cfg.variant,
                        task_type=self.cfg.task_type, artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                        software=_software_versions(), successful_optimizer_updates=self.completed_optimizer_steps,
                        dataset_split=copy.deepcopy(self.dataset_split), update_schedule=self.update_schedule,
                        evaluation={"weights": "ema", "solver": "euler",
                                    "seed": int(self.cfg.training.seed),
                                    "steps": int(self.cfg.policy.flow.inference_steps)})
        state = {key: getattr(self, key) for key in self.include_keys}
        payload = {"cfg": OmegaConf.create(_plain(self.cfg)),
                   "policy_config": _plain(self.model.export_config()), "metadata": metadata,
                   "state_dicts": {name: _copy_to_cpu(getattr(self, name).state_dict())
                                   for name in ("model", "ema_model", "optimizer")},
                   "pickles": {key: dill.dumps(value) for key, value in state.items()},
                   "training_state": _copy_to_cpu(state)}
        _atomic_torch_save(payload, path)
        return str(path.absolute())

    @classmethod
    def create_from_checkpoint(cls, path, output_dir=None, **_):
        payload = cls._read_payload(path)
        cfg = OmegaConf.create(_plain(payload["cfg"]))
        cfg.training.resume = True
        cfg.training.resume_checkpoint = str(path)
        cls.validate_resume_payload(payload, cfg)
        instance = cls(cfg, output_dir)
        instance.model = hydra.utils.instantiate(OmegaConf.create(payload["policy_config"]))
        instance.model.load_state_dict(payload["state_dicts"]["model"], strict=True)
        instance.optimizer = instance.model.get_optimizer(**cfg.optimizer)
        instance.optimizer.load_state_dict(payload["state_dicts"]["optimizer"])
        instance.ema_model = make_fresh_ema(instance.model)
        instance.ema_model.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
        instance._load_continuation(payload)
        instance.model.eval().reset()
        instance.ema_model.eval().reset()
        return instance

    @torch.no_grad()
    def _validate(self, accelerator, loader, dataset, *, generated, decode):
        policy = self.ema_model
        policy.eval()
        modes = ("expert", "generated") if generated else ("expert",)
        names = [f"{mode}_{name}" for mode in modes for name in METRICS]
        metrics = MetricSums(names, accelerator.device)
        sample_ids = []
        for batch in loader:
            batch = _move(batch, accelerator.device)
            # Individual seeds deliberately decouple all stochastic evaluation
            # from validation partitioning. Batched production inference remains
            # available independently through predict_action.
            for index in range(batch["action"].shape[0]):
                sample = _slice_sample(batch, index)
                sample_id = int(sample["sample_id"].item())
                sample_ids.append(sample_id)
                seed = stable_validation_seed(int(self.cfg.training.seed), dataset.dataset_identity, sample_id)
                for mode in modes:
                    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                    with accelerator.autocast():
                        measured = policy.validation_metrics(sample, generator=generator, history_mode=mode, compute_decoded=decode)
                    metrics.add({f"{mode}_{key}": value for key, value in measured.items()})
        # Different ranks may have different batch counts, including zero.
        # There are no collectives in the loop; every rank reduces exactly once.
        values = metrics.reduce()
        gathered = [sample_ids]
        if accelerator.num_processes > 1:
            gathered = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(gathered, sample_ids)
        ids = [value for group in gathered for value in group]
        expected = [int(row[0] + dataset.pad_before - row[2]) for row in dataset.seq_sampler.indices]
        if sorted(ids) != sorted(expected) or len(set(ids)) != len(ids):
            raise RuntimeError("Validation sample IDs are duplicated, missing, or do not cover the dataset")
        result = {f"val_{key}": value for key, value in values.items()}
        result["val_loss"] = values["expert_fm_loss"]
        result["val_action_mse"] = values["expert_decoded_action_mse"]
        result["validation_samples"] = len(ids)
        result["validation_solver_steps"] = int(self.cfg.policy.flow.inference_steps)
        return result

    def run(self):
        from oat.policy.p2n_latent_flow_common import prepare_flow_training_batch

        cfg = self.cfg
        if cfg.get("policy_family") != POLICY_FAMILY or cfg.variant not in VARIANTS:
            raise ValueError("This workspace only accepts explicit latent-flow variants")
        if not cfg.training.use_ema:
            raise ValueError("FM+CT training requires use_ema=true")
        if cfg.training.get("init_checkpoint"):
            raise ValueError("Use a compatible full resume artifact; warm-start is not supported")
        batch_size = int(cfg.dataloader.batch_size)
        if batch_size < 4 or batch_size % 4 or not cfg.dataloader.drop_last:
            raise ValueError("Per-rank microbatch must be a positive multiple of four and drop_last=true")
        if cfg.val_dataloader.drop_last:
            raise ValueError("Validation requires drop_last=false")
        # Partial validation would invalidate the complete, nonpadding coverage
        # contract. Diagnostic subsets should be separate datasets.
        if cfg.training.get("max_val_steps") is not None or cfg.training.get("max_reconst_steps") is not None:
            raise ValueError("max_val_steps/max_reconst_steps must be null for complete validation coverage")
        accumulation = int(cfg.training.gradient_accumulate_every)
        use_bf16 = bool(cfg.training.allow_bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        log_with = "wandb" if cfg.get("logging") and cfg.logging.get("mode", "offline") != "disabled" else None
        accelerator = Accelerator(
            gradient_accumulation_steps=accumulation, mixed_precision="bf16" if use_bf16 else "no",
            log_with=log_with,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False),
                             InitProcessGroupKwargs(timeout=timedelta(hours=2))])
        device = accelerator.device
        set_seed(int(cfg.training.seed), device_specific=True)
        self.generators = {
            "train": torch.Generator(device=device).manual_seed(int(cfg.training.seed) + 1009 * accelerator.process_index + 11),
            "self_past": torch.Generator(device=device).manual_seed(int(cfg.training.seed) + 1009 * accelerator.process_index + 29),
            "dataloader": torch.Generator().manual_seed(int(cfg.training.seed) + 43),
        }
        payload = None
        if cfg.training.resume:
            path = cfg.training.get("resume_checkpoint")
            if not path:
                raise ValueError("Resume requires training.resume_checkpoint")
            payload = self.validate_resume_payload(self._read_payload(path), cfg)
        construction = cfg.policy if payload is None else OmegaConf.create(payload["policy_config"])
        self.model = hydra.utils.instantiate(construction)
        dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
        val_dataset = dataset.get_validation_dataset()
        split = self._split_metadata(dataset, val_dataset)
        validation_enabled = bool(cfg.training.get("offline_validation_enabled", True))
        if validation_enabled and not len(val_dataset):
            raise ValueError("Offline validation is enabled but the held-out dataset is empty")
        if not len(dataset):
            raise ValueError("Training dataset is empty")
        if payload is None:
            self.model.set_normalizer(dataset.get_normalizer())
        else:
            self.model.load_state_dict(payload["state_dicts"]["model"], strict=True)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        if payload is not None:
            self.optimizer.load_state_dict(payload["state_dicts"]["optimizer"])
            self._load_continuation(payload)
            if self.dataset_split != split:
                raise ValueError("Resume data identity or actual episode masks differ")
        self.dataset_split = split
        train_kwargs = dict(_plain(cfg.dataloader))
        train_kwargs["generator"] = self.generators["dataloader"]
        if hasattr(dataset, "get_training_sampler"):
            train_kwargs.update(sampler=dataset.get_training_sampler(), shuffle=False)
        train_loader = DataLoader(dataset, **train_kwargs)
        maximum = cfg.training.get("max_train_steps")
        if maximum is not None:
            train_loader = _cap_loader(train_loader, int(maximum) * accelerator.num_processes)
        val_kwargs = dict(_plain(cfg.val_dataloader))
        val_kwargs.update(shuffle=False, drop_last=False,
                          sampler=NonPaddingDistributedSampler(val_dataset, accelerator.process_index, accelerator.num_processes))
        val_loader = DataLoader(val_dataset, **val_kwargs)
        # The EMA must not exist until prepare's DDP constructor has broadcast
        # all student parameters and buffers from rank zero.
        self.model, self.optimizer, train_loader = accelerator.prepare(self.model, self.optimizer, train_loader)
        student = accelerator.unwrap_model(self.model)
        self.ema_model = make_fresh_ema(student)
        if payload is not None:
            self.ema_model.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
        assert_teacher_synchronized(self.ema_model)
        validate_optimizer_ownership(student, self.ema_model, self.optimizer)
        ema_kwargs = dict(_plain(cfg.ema))
        ema_kwargs.pop("_target_", None)
        ema = EMAModel(self.ema_model, **ema_kwargs)
        schedule = resolve_update_schedule(
            len(train_loader), int(cfg.training.num_epochs), accumulation,
            warmup_steps=cfg.training.get("lr_warmup_steps"), warmup_ratio=cfg.training.get("lr_warmup_ratio", 0.05))
        if payload is not None and self.update_schedule != schedule:
            raise ValueError("Resume must preserve the resolved successful-update LR schedule")
        self.update_schedule = schedule
        scheduler = get_scheduler(cfg.training.lr_scheduler, optimizer=self.optimizer,
                                  num_warmup_steps=schedule["lr_warmup_steps"],
                                  num_training_steps=schedule["planned_optimizer_updates"], last_epoch=-1)
        if payload is not None:
            scheduler.load_state_dict(self.lr_scheduler_state)
            for group, rate in zip(self.optimizer.param_groups, self.lr_scheduler_state["_last_lr"]):
                group["lr"] = rate
            ema.optimization_step = int(self.ema_state["optimization_step"])
            ema.decay = float(self.ema_state["decay"])
        del payload
        output = Path(self.output_dir)
        if accelerator.is_main_process:
            output.mkdir(parents=True, exist_ok=True)
            (output / "dataset_split.json").write_text(json.dumps(split, indent=2) + "\n")
            OmegaConf.save(OmegaConf.create(_plain(cfg)), output / "resolved_config.yaml")
            counts = {"trainable": sum(p.numel() for p in student.parameters() if p.requires_grad),
                      "frozen": sum(p.numel() for p in student.parameters() if not p.requires_grad)}
            accelerator.print(json.dumps({"parameters": counts, "update_schedule": schedule, "dataset_split": split}, indent=2))
        if log_with:
            logging = dict(_plain(cfg.logging))
            project = logging.pop("project")
            logging["dir"] = str(output)
            accelerator.init_trackers(project, config=_plain(cfg), init_kwargs={"wandb": logging})
        topk = None
        if accelerator.is_main_process and cfg.checkpoint.get("topk"):
            topk = TopKCheckpointManager(save_dir=str(output / "checkpoints"), **cfg.checkpoint.topk)
            if cfg.training.resume:
                topk.restore_from_logs(str(output / "logs.json"), self.epoch)
        runner = None
        if not cfg.task.policy.get("lazy_eval", True) and accelerator.is_main_process:
            runner = hydra.utils.instantiate(cfg.task.policy.env_runner, output_dir=self.output_dir)
        # Restore after constructors, tracking, and DDP have consumed RNG, before
        # the next data iterator or target/self-past random draw is created.
        if cfg.training.resume:
            self._restore_rng(accelerator.process_index, accelerator.num_processes)
        self.optimizer.zero_grad(set_to_none=True)
        while self.epoch < int(cfg.training.num_epochs):
            if hasattr(train_loader, "set_epoch"):
                train_loader.set_epoch(self.epoch)
            self.model.train()
            self.ema_model.eval()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.monotonic()
            epoch_loss = torch.zeros(2, dtype=torch.float64, device=device)
            n_batches = len(train_loader)
            for batch_index, batch in enumerate(train_loader):
                batch = _move(batch, device)
                # All teacher and self-past temporary activations disappear
                # before the one trainable context+network graph is created.
                with accelerator.autocast():
                    prepared = prepare_flow_training_batch(
                        batch, student=student, teacher=self.ema_model,
                        generator=self.generators["train"], self_past_generator=self.generators["self_past"])
                with accelerator.accumulate(self.model):
                    with accelerator.autocast():
                        loss = self.model(prepared)
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise FloatingPointError("Packed flow loss must be a finite scalar")
                    # Accelerate divides by the nominal accumulation length.
                    # Correct the incomplete last group to preserve mean grads.
                    group_length = min(accumulation, n_batches - (batch_index // accumulation) * accumulation)
                    accelerator.backward(loss * (accumulation / group_length))
                    if accelerator.sync_gradients:
                        params = [p for p in student.parameters() if p.requires_grad]
                        norm = accelerator.clip_grad_norm_(params, float(cfg.training.get("max_grad_norm") or math.inf))
                        finite = torch.isfinite(torch.as_tensor(norm, device=device)).to(torch.int32)
                        if accelerator.num_processes > 1:
                            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                        if finite.item():
                            self.optimizer.step()
                            if not accelerator.optimizer_step_was_skipped:
                                successful_update(student, ema, scheduler)
                                self.completed_optimizer_steps += 1
                            else:
                                self.skipped_optimizer_steps += 1
                        else:
                            self.skipped_optimizer_steps += 1
                        self.optimizer.zero_grad(set_to_none=True)
                    epoch_loss[0] += loss.detach().double() * batch["action"].shape[0]
                    epoch_loss[1] += batch["action"].shape[0]
                    self.global_step += 1
            epoch_loss = accelerator.reduce(epoch_loss, reduction="sum")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            log = {"epoch": self.epoch, "global_step": self.global_step,
                   "successful_optimizer_updates": self.completed_optimizer_steps,
                   "skipped_optimizer_updates": self.skipped_optimizer_steps,
                   "train_loss": float(epoch_loss[0] / epoch_loss[1]),
                   "train_epoch_seconds": time.monotonic() - started,
                   "lr": scheduler.get_last_lr()[0]}
            if device.type == "cuda":
                log.update(gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 2 ** 20,
                           gpu_peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 2 ** 20)
            validation_due = self.epoch % int(cfg.training.get("val_every", 1)) == 0
            decoded_due = self.epoch % int(cfg.training.get("sample_every", 1)) == 0
            if validation_enabled and (validation_due or decoded_due):
                log.update(self._validate(accelerator, val_loader, val_dataset,
                                         generated=bool(cfg.training.get("validate_generated_history", True)), decode=decoded_due))
            if not cfg.task.policy.get("lazy_eval", True) and self.epoch % int(cfg.training.rollout_every) == 0:
                accelerator.wait_for_everyone()
                if runner is not None:
                    log.update(runner.run(self.ema_model))
                accelerator.wait_for_everyone()
            score = log.get("val_expert_decoded_action_mse")
            is_best = score is not None and score < self.best_metric
            if is_best:
                self.best_metric = score
            completed_epoch = self.epoch
            self.epoch += 1
            self.ema_state = {"optimization_step": ema.optimization_step, "decay": ema.decay}
            self.lr_scheduler_state = copy.deepcopy(scheduler.state_dict())
            self._gather_rng(accelerator)
            checkpoint_due = completed_epoch % int(cfg.training.checkpoint_every) == 0 or self.epoch == int(cfg.training.num_epochs)
            periodic = int(cfg.training.get("snapshot_every", 0))
            if accelerator.is_main_process:
                wrapped = self.model
                self.model = student
                try:
                    if checkpoint_due and cfg.checkpoint.get("save_last_ckpt", True):
                        self.save_checkpoint(tag="latest")
                    if periodic > 0 and completed_epoch % periodic == 0:
                        self.save_checkpoint(tag=f"ep-{completed_epoch:04d}")
                    if checkpoint_due and cfg.checkpoint.get("save_last_snapshot", False):
                        self.save_checkpoint(path=output / "snapshots" / "latest.ckpt")
                    if checkpoint_due and cfg.checkpoint.get("save_all", False):
                        self.save_checkpoint(path=output / "checkpoints" / cfg.checkpoint.topk.format_str.format(**log))
                    elif topk is not None and topk.k > 0 and log.get(topk.monitor_key) is not None:
                        ranked_path = topk.get_ckpt_path(log)
                        if ranked_path is not None:
                            self.save_checkpoint(path=ranked_path)
                    elif is_best:
                        self.save_checkpoint(tag="best")
                finally:
                    self.model = wrapped
                with (output / "logs.json").open("a") as stream:
                    stream.write(json.dumps(log, allow_nan=False) + "\n")
                if log_with:
                    accelerator.log({key: value for key, value in log.items() if value is not None}, step=self.global_step)
                accelerator.print(json.dumps(log))
            accelerator.wait_for_everyone()
        if runner is not None:
            runner.close()
        accelerator.wait_for_everyone()
        accelerator.end_training()


@hydra.main(version_base=None, config_path="../config", config_name="train_p2n_latent_flow")
def main(cfg):
    TrainP2NLatentFlowWorkspace(cfg).run()


if __name__ == "__main__":
    main()
