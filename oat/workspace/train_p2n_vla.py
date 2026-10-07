"""Training workspace for the P2N-VLA policies (contract: docs/P2N_VLA_IMPLEMENTATION.md, M4).

Derived from ``train_p2n_new.py`` and reusing its helpers (``_cap_dataloader``, the
offline-validation split metadata, atomic saves, capability routing). Differences:

- Accelerate runs with ``mixed_precision='no'``: the policy owns its bf16 autocast.
  DDP uses ``find_unused_parameters=False`` and ``gradient_as_bucket_view=True``, and
  gradients are zeroed with ``set_to_none=False`` so they stay bucket views.
- Setup order: instantiate -> ``.to(device)`` -> ``set_normalizer`` -> ``get_optimizer``
  -> ``accelerator.prepare(policy, optimizer, train_loader, val_loader)``. The EMA is
  created afterwards, once DDP has broadcast rank 0's parameters.
- :class:`TrainableEMA` averages trainable parameters only and is swapped into the
  live policy for validation, on every rank, at an optimizer-step boundary.
- One LambdaLR on successful optimizer steps: linear warmup from ``1/(W+1)`` to the
  peak over ``lr_warmup_steps``, then cosine to ``min_lr_ratio`` at
  ``max_optimizer_steps`` (openpi's ``warmup_cosine_decay_schedule``).
- Gradients are clipped separately per ``policy.clip_groups()`` ('ar', 'ki').
- ``max_optimizer_steps`` hard-stops training; ``max_train_steps`` caps micro-batches
  per rank per epoch. Resume is epoch-granular and needs the original world size.
- ``checkpoints/latest.ckpt`` (every ``checkpoint_every`` epochs) holds the artifacts
  (state dict minus the frozen pi05 backbone), optimizer, scheduler, EMA counters,
  per-rank RNG and counters. ``snapshots/upd-NNNNNN_ema.ckpt`` (every
  ``snapshot_every`` optimizer steps and at the end) holds the EMA artifact with
  trainable tensors in bf16 and no training state.
- No simulator rollouts during training; evaluate snapshots with
  ``scripts/evaluate_p2n_vla.py``.
"""
from __future__ import annotations

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import contextlib
import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, SequentialSampler, Subset
import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs, set_seed as accelerate_set_seed

from oat.common.hydra_util import register_new_resolvers
from oat.common.p2n_new_capabilities import (
    policy_capability, predict_validation_action, resolve_update_schedule, validate_history_batch)
from oat.model.common.trainable_ema import TrainableEMA
from oat.workspace.base_workspace import BaseWorkspace, _atomic_torch_save, _copy_to_cpu
from oat.workspace.train_p2n_new import TrainP2NNewWorkspace, _cap_dataloader

register_new_resolvers()

CHECKPOINT_FORMAT = "p2n_vla_checkpoint_v1"
POLICY_FAMILY = "p2n_vla"
VARIANTS = ("p2n_vla", "p2n_vla_state_gate", "pi05_ki_flow")
CLIP_GROUP_NAMES = ("ar", "ki")
OPTIMIZER_GROUP_NAMES = ("pretrained", "new")
OPTIMIZER_KEYS = ("policy_lr", "new_module_lr", "weight_decay", "betas", "eps", "fused")
# Asset locations may move between machines; everything else in cfg.policy must match on resume.
RESUME_IGNORED_POLICY_KEYS = ("pi05_weights", "spm_path", "tokenizer_checkpoint", "construction_mode")
RESUME_TRAINING_KEYS = ("gradient_accumulate_every", "max_train_steps", "max_optimizer_steps",
                        "lr_warmup_steps", "min_lr_ratio", "max_grad_norm", "seed", "use_ema")
RESUME_TOP_LEVEL_KEYS = ("variant", "task_type", "horizon", "n_action_steps", "n_obs_steps", "past_n")
SOURCE_FILES = ("oat/workspace/train_p2n_vla.py", "oat/model/common/trainable_ema.py",
                "scripts/train_p2n_vla.py", "oat/config/train_p2n_vla.yaml",
                "oat/config/train_p2n_vla_state_gate.yaml", "oat/config/train_pi05_ki_flow.yaml")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GIB = float(2 ** 30)


# --------------------------------------------------------------------------- helpers
def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(item, device) for item in value)
    return value


def _sanitize(value):
    """JSON-safe copy: non-finite floats become None, tensors/numpy scalars become Python."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        value = value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return str(value)


def scalar_items(values: Optional[Mapping], prefix: str = "") -> Dict[str, float]:
    """Flatten ``last_loss_components``-style dicts into scalar floats (None entries dropped)."""
    result: Dict[str, float] = {}
    if not values:
        return result
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                result[name] = float(value.detach())
            else:
                for index, item in enumerate(value.detach().reshape(-1).tolist()):
                    result[f"{name}/{index}"] = float(item)
        elif isinstance(value, bool):
            result[name] = float(value)
        elif isinstance(value, numbers.Real):
            result[name] = float(value)
        elif isinstance(value, (list, tuple)) and all(isinstance(item, numbers.Real) for item in value):
            for index, item in enumerate(value):
                result[f"{name}/{index}"] = float(item)
    return result


def _numeric(record: Mapping) -> Dict[str, float]:
    """Tracker-safe subset: finite real numbers only (bools and strings dropped)."""
    result = {}
    for key, value in _sanitize(dict(record)).items():
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            result[key] = value
    return result


def _software_versions():
    versions = {}
    for package in ("torch", "transformers", "accelerate", "hydra-core", "numpy", "sentencepiece"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_sha256():
    hashes = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return hashes


def _positive_int(value, name, *, allow_none=False, minimum=1):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}" + (" or null" if allow_none else ""))
    return int(value)


@contextlib.contextmanager
def _preserved_modes(module: torch.nn.Module):
    """Restore every submodule's ``training`` flag on exit (the no-EMA validation path)."""
    modes = [(child, child.training) for child in module.modules()]
    try:
        yield module
    finally:
        for child, mode in modes:
            child.training = mode


class EpochSeededRandomSampler(Sampler):
    """Shuffle as a pure function of (seed, epoch).

    Every rank draws the same permutation (Accelerate then shards it), and resume
    reproduces the data order of an uninterrupted run regardless of how many RNG
    draws happened elsewhere. Call ``set_epoch`` before iterating.
    """

    def __init__(self, data_source, seed: int):
        self.num_samples = len(data_source)
        if self.num_samples < 1:
            raise ValueError("Cannot shuffle an empty training dataset")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def permutation_seed(self) -> int:
        payload = f"p2n-vla-sampler\0{self.seed}\0{self.epoch}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.permutation_seed())
        return iter(torch.randperm(self.num_samples, generator=generator).tolist())

    def __len__(self):
        return self.num_samples


class WarmupCosineSchedule:
    """LR multiplier as a function of completed optimizer steps.

    ``step < warmup``: linear from ``1/(warmup+1)`` to 1 (the first update never uses
    LR 0, as in openpi). Afterwards a half cosine from 1 to ``min_lr_ratio`` over
    ``total_steps - warmup`` steps, then constant at ``min_lr_ratio``.
    """

    def __init__(self, warmup_steps: int, total_steps: int, min_lr_ratio: float):
        warmup_steps = _positive_int(warmup_steps, "lr_warmup_steps", minimum=0)
        total_steps = _positive_int(total_steps, "max_optimizer_steps")
        min_lr_ratio = float(min_lr_ratio)
        if warmup_steps > total_steps:
            raise ValueError("lr_warmup_steps cannot exceed the cosine horizon (max_optimizer_steps)")
        if not math.isfinite(min_lr_ratio) or not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must lie in [0, 1]")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step: int) -> float:
        step = int(step)
        if step < 0:
            raise ValueError("Optimizer step must be nonnegative")
        if step < self.warmup_steps:
            start = 1.0 / (self.warmup_steps + 1)
            return start + (1.0 - start) * step / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        progress = 1.0 if decay_steps <= 0 else min(1.0, (step - self.warmup_steps) / decay_steps)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio):
    """One LambdaLR shared by every parameter group (peak LRs come from the groups)."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, WarmupCosineSchedule(warmup_steps, total_steps, min_lr_ratio))


def clip_gradients_per_group(groups: Mapping[str, Sequence[torch.Tensor]], max_norm) -> Dict[str, torch.Tensor]:
    """Clip each group to ``max_norm`` independently; returns the pre-clip global norm per group.

    ``max_norm=None`` only measures. An empty group reports a zero norm.
    """
    limit = math.inf if max_norm is None else float(max_norm)
    if not limit > 0:
        raise ValueError("max_grad_norm must be positive or null")
    norms = {}
    for name, parameters in groups.items():
        with_grad = [parameter for parameter in parameters if parameter.grad is not None]
        if not with_grad:
            device = parameters[0].device if len(parameters) else torch.device("cpu")
            norms[name] = torch.zeros((), device=device)
            continue
        norms[name] = torch.nn.utils.clip_grad_norm_(with_grad, limit, foreach=True)
    return norms


def validate_parameter_partition(policy, optimizer):
    """Check the M3 ownership contract; returns (named trainable parameters, clip groups).

    Every trainable parameter appears exactly once in the optimizer, the named
    optimizer groups are 'pretrained'/'new', and ``clip_groups()`` partitions the
    trainable parameters into 'ar' and 'ki'.
    """
    named = list(policy.trainable_named_parameters())
    by_id = {}
    for name, parameter in named:
        if not parameter.requires_grad:
            raise ValueError(f"trainable_named_parameters returned frozen parameter {name}")
        if id(parameter) in by_id:
            raise ValueError(f"trainable_named_parameters repeats parameter {name}")
        by_id[id(parameter)] = name
    actual = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    if actual != set(by_id):
        raise ValueError("trainable_named_parameters must list every trainable parameter exactly once")
    owned = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(owned) != len(set(owned)) or set(owned) != set(by_id):
        raise ValueError("The optimizer must own every trainable parameter exactly once")
    for group in optimizer.param_groups:
        if group.get("name") not in OPTIMIZER_GROUP_NAMES:
            raise ValueError(f"Optimizer groups must be named {OPTIMIZER_GROUP_NAMES}, got {group.get('name')!r}")
    clip = policy.clip_groups()
    if set(clip) != set(CLIP_GROUP_NAMES):
        raise ValueError(f"clip_groups() must return exactly {CLIP_GROUP_NAMES}")
    clip = {name: list(clip[name]) for name in CLIP_GROUP_NAMES}
    seen = []
    for name in CLIP_GROUP_NAMES:
        seen.extend(id(parameter) for parameter in clip[name])
    if len(seen) != len(set(seen)):
        raise ValueError("clip_groups() must be disjoint")
    if set(seen) != set(by_id):
        raise ValueError("clip_groups() must cover every trainable parameter")
    return named, clip


def _validation_subset(dataset, max_batches, batch_size, world_size):
    """Evenly spaced windows so a capped validation still spans every held-out episode."""
    if max_batches is None:
        return dataset
    wanted = int(max_batches) * int(batch_size) * int(world_size)
    if wanted >= len(dataset):
        return dataset
    indices = np.unique(np.linspace(0, len(dataset) - 1, wanted).round().astype(np.int64))
    return Subset(dataset, indices.tolist())


class _MetricSums:
    """Weighted means reduced once across ranks; ranks may report different keys."""

    def __init__(self):
        self.sums: Dict[str, List[float]] = {}

    def add(self, name: str, value, weight: float) -> None:
        if value is None:
            return
        value = float(value)
        entry = self.sums.setdefault(name, [0.0, 0.0, 0.0])
        entry[0] += value * weight
        entry[1] += weight
        entry[2] += 0.0 if math.isfinite(value) else 1.0

    def add_all(self, values: Mapping[str, float], weight: float, prefix: str = "") -> None:
        for name, value in values.items():
            self.add(prefix + name, value, weight)

    def reduce(self, accelerator) -> Dict[str, Optional[float]]:
        names = sorted(self.sums)
        if accelerator.num_processes > 1:
            gathered = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(gathered, names)
            names = sorted({name for group in gathered for name in group})
        table = torch.zeros((len(names), 3), dtype=torch.float64, device=accelerator.device)
        for row, name in enumerate(names):
            if name in self.sums:
                table[row] = torch.tensor(self.sums[name], dtype=torch.float64)
        if accelerator.num_processes > 1 and len(names):
            torch.distributed.all_reduce(table, op=torch.distributed.ReduceOp.SUM)
        result = {}
        for name, (total, count, nonfinite) in zip(names, table.tolist()):
            result[name] = total / count if count > 0 else None
            if nonfinite:
                result[f"{name}/nonfinite_batches"] = nonfinite
        return result


class _JsonlLog:
    def __init__(self, path, enabled: bool):
        self.path = pathlib.Path(path)
        self.enabled = bool(enabled)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Mapping) -> None:
        if not self.enabled:
            return
        line = json.dumps(_sanitize(dict(record)), allow_nan=False)
        with self.path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()


@dataclass
class _Run:
    accelerator: Any
    model: Any
    policy: Any
    optimizer: Any
    scheduler: Any
    ema: Optional[TrainableEMA]
    named: List[Tuple[str, torch.nn.Parameter]]
    clip: Dict[str, List[torch.nn.Parameter]]
    train_loader: Any
    val_loader: Any
    log: _JsonlLog
    trackers: bool
    validation_enabled: bool
    group_names: List[str] = field(default_factory=list)
    checked_gradients: bool = False


# ------------------------------------------------------------------------- workspace
class TrainP2NVLAWorkspace(BaseWorkspace):
    """Accelerate/DDP training for P2N-VLA policies (see the module docstring)."""

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.model = None             # the unwrapped policy
        self.optimizer = None
        self.ema: Optional[TrainableEMA] = None
        self.lr_scheduler = None
        self.epoch = 0
        self.global_step = 0
        self.completed_optimizer_steps = 0
        self.skipped_optimizer_steps = 0
        self.consecutive_skipped_steps = 0
        self.last_snapshot_step: Optional[int] = None
        self.rng_states = None
        self.update_schedule = None
        self.dataset_split = None
        self.world_size = 1
        self.generators: Dict[str, torch.Generator] = {}
        self.ddp_settings = None
        self.last_validation = None
        self.probe_report = None
        self.saved_snapshots: List[str] = []
        self._trainable_names: List[str] = []
        self._sampler = None

    # ------------------------------------------------------------- configuration
    @staticmethod
    def check_run_config(cfg) -> None:
        """Fail fast on settings the loop depends on (the launcher validates the full recipe)."""
        variant = cfg.get("variant")
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if cfg.policy.get("variant", variant) != variant:
            raise ValueError("policy.variant must equal the selected variant")
        training = cfg.training
        if training.get("init_checkpoint"):
            raise ValueError("P2N-VLA starts from pi05_base; init_checkpoint is unsupported (use resume)")
        if training.get("resume") and not training.get("resume_checkpoint"):
            raise ValueError("Resume requires training.resume_checkpoint")
        if cfg.task.policy.get("lazy_eval", True) is not True:
            raise ValueError("P2N-VLA training runs no simulator rollouts; set task.policy.lazy_eval=true "
                             "and evaluate snapshots with scripts/evaluate_p2n_vla.py")
        for key in ("num_epochs", "gradient_accumulate_every", "checkpoint_every", "val_every"):
            _positive_int(training.get(key), f"training.{key}")
        _positive_int(training.get("log_every", 1), "training.log_every")
        _positive_int(training.get("snapshot_every", 0), "training.snapshot_every", minimum=0)
        for key in ("max_train_steps", "max_optimizer_steps", "max_val_steps"):
            _positive_int(training.get(key), f"training.{key}", allow_none=True)
        _positive_int(training.get("max_reconst_steps"), "training.max_reconst_steps", allow_none=True, minimum=0)
        _positive_int(training.get("lr_warmup_steps", 0), "training.lr_warmup_steps", minimum=0)
        _positive_int(training.get("max_consecutive_skipped_updates", 20),
                      "training.max_consecutive_skipped_updates")
        ratio = float(training.get("min_lr_ratio", 0.1))
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("training.min_lr_ratio must lie in [0, 1]")
        norm = training.get("max_grad_norm")
        if norm is not None and not float(norm) > 0:
            raise ValueError("training.max_grad_norm must be positive or null")
        if not isinstance(training.get("use_ema", True), bool):
            raise ValueError("training.use_ema must be a boolean")
        unknown = set(cfg.optimizer) - set(OPTIMIZER_KEYS)
        if unknown:
            raise ValueError(f"Unknown optimizer settings {sorted(unknown)}; expected {OPTIMIZER_KEYS}")
        mode = cfg.logging.get("mode", "offline") if cfg.get("logging") else "disabled"
        if mode not in ("online", "offline", "disabled"):
            raise ValueError("logging.mode must be online, offline or disabled")
        _positive_int(cfg.dataloader.get("batch_size"), "dataloader.batch_size")
        _positive_int(cfg.val_dataloader.get("batch_size"), "val_dataloader.batch_size")

    @staticmethod
    def read_payload(path):
        path = pathlib.Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        # mmap keeps multi-GB resume payloads out of RAM until tensors are copied.
        return torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)

    @staticmethod
    def validate_resume_payload(payload, cfg, world_size=None):
        """A resume needs the same variant, architecture, data, optimizer recipe and world size."""
        if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Not a {CHECKPOINT_FORMAT} checkpoint")
        training = payload.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("EMA snapshots carry no optimizer/scheduler/RNG state and cannot resume; "
                             "resume from checkpoints/latest.ckpt")
        for key in ("optimizer", "lr_scheduler", "ema", "rng_states", "counters", "world_size",
                    "update_schedule", "dataset_split"):
            if key not in training:
                raise ValueError(f"Resume checkpoint is missing training.{key}")
        metadata = payload.get("metadata") or {}
        run = metadata.get("training_run") or {}
        if run.get("variant", metadata.get("variant")) != cfg.variant:
            raise ValueError("Resume variant does not match the selected variant")
        policy_config = payload.get("policy_config") or {}
        if policy_config.get("_target_") != cfg.policy.get("_target_"):
            raise ValueError("Resume policy target does not match the selected policy")
        if policy_config.get("construction_mode") != "restore":
            raise ValueError("Resume needs an exported restore-mode policy configuration")
        if world_size is not None and (int(training["world_size"]) != int(world_size)
                                       or len(training["rng_states"]) != int(world_size)):
            raise ValueError(f"Resume requires the original world size {training['world_size']} "
                             f"(per-rank RNG streams); got {world_size}")
        state_dicts = payload.get("state_dicts") or {}
        if "model" not in state_dicts:
            raise ValueError("Resume checkpoint is missing the live model artifact")
        use_ema = bool(cfg.training.get("use_ema", True))
        if use_ema != (training["ema"] is not None) or use_ema != ("ema_model" in state_dicts):
            raise ValueError("Resume cannot change the EMA configuration")
        saved = OmegaConf.create(payload["cfg"])
        current_policy, saved_policy = _plain(cfg.policy), _plain(saved.policy)
        for key in sorted(set(current_policy) | set(saved_policy)):
            if key not in RESUME_IGNORED_POLICY_KEYS and current_policy.get(key) != saved_policy.get(key):
                raise ValueError(f"Resume architecture/recipe mismatch: policy.{key}")
        for key in RESUME_TOP_LEVEL_KEYS:
            if _plain(cfg.get(key)) != _plain(saved.get(key)):
                raise ValueError(f"Resume schema mismatch: {key}")
        new_data, old_data = _plain(cfg.task.policy.dataset), _plain(saved.task.policy.dataset)
        for key in sorted(set(new_data) | set(old_data)):
            a, b = new_data.get(key), old_data.get(key)
            if key == "zarr_path" and a and b:
                a, b = str(pathlib.Path(a).expanduser().resolve()), str(pathlib.Path(b).expanduser().resolve())
            if a != b:
                raise ValueError(f"Resume data schema/split mismatch: dataset.{key}")
        for section in ("ema", "optimizer"):
            if _plain(cfg.get(section)) != _plain(saved.get(section)):
                raise ValueError(f"Resume cannot change the {section} configuration")
        for key in RESUME_TRAINING_KEYS:
            if _plain(cfg.training.get(key)) != _plain(saved.training.get(key)):
                raise ValueError(f"Resume training recipe mismatch: training.{key}")
        if int(cfg.dataloader.batch_size) != int(saved.dataloader.batch_size):
            raise ValueError("Resume cannot change the per-rank micro-batch")
        return payload

    @staticmethod
    def restore_policy_config(payload, cfg):
        """The exported restore-mode constructor kwargs, with this machine's asset paths."""
        config = copy.deepcopy(payload["policy_config"])
        if config.get("construction_mode") != "restore":
            raise ValueError("Exported policy configuration must use construction_mode='restore'")
        for key in ("pi05_weights", "spm_path"):
            value = cfg.policy.get(key)
            if value is not None:
                config[key] = str(value)
        return config

    @staticmethod
    def load_optimizer_state(optimizer, training, trainable_names):
        """Load saved AdamW state, refusing any change in the parameter order it is keyed by.

        ``Optimizer.load_state_dict`` matches state to parameters by position and does not
        check shapes, so a reordered or renamed trainable set would silently swap moments.
        """
        saved = training.get("trainable_names")
        if saved is not None and list(saved) != list(trainable_names):
            missing = sorted(set(saved) - set(trainable_names))
            unexpected = sorted(set(trainable_names) - set(saved))
            raise ValueError("Resume trainable parameters differ from the checkpoint (optimizer state is "
                             f"positional): missing={missing[:5]} unexpected={unexpected[:5]} "
                             f"same_set_reordered={not missing and not unexpected}")
        optimizer.load_state_dict(training["optimizer"])
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                for key, value in optimizer.state.get(parameter, {}).items():
                    if (isinstance(value, torch.Tensor) and value.ndim
                            and tuple(value.shape) != tuple(parameter.shape)):
                        raise ValueError(f"Saved optimizer state {key!r} has shape {tuple(value.shape)} for a "
                                         f"parameter of shape {tuple(parameter.shape)}")

    def _instantiate_policy(self, cfg, payload):
        if payload is None:
            return hydra.utils.instantiate(cfg.policy)
        policy = hydra.utils.instantiate(self.restore_policy_config(payload, cfg))
        policy.load_artifact_state(payload["state_dicts"]["model"])
        return policy

    # ---------------------------------------------------------------- data loaders
    @staticmethod
    def _loader_kwargs(settings) -> dict:
        kwargs = dict(_plain(settings))
        for key in ("sampler", "batch_sampler", "generator"):
            if key in kwargs:
                raise ValueError(f"Dataloader config must not set {key}; the workspace owns it")
        if not int(kwargs.get("num_workers", 0) or 0):
            kwargs["num_workers"] = 0
            kwargs["persistent_workers"] = False
            kwargs.pop("prefetch_factor", None)
        return kwargs

    def _make_train_loader(self, dataset, cfg, world_size):
        kwargs = self._loader_kwargs(cfg.dataloader)
        shuffle = bool(kwargs.pop("shuffle", True))
        custom = getattr(dataset, "get_training_sampler", None)
        if custom is not None:
            sampler = custom()
        elif shuffle:
            sampler = EpochSeededRandomSampler(dataset, int(cfg.training.seed))
        else:
            sampler = SequentialSampler(dataset)
        self._sampler = sampler
        loader = DataLoader(dataset, sampler=sampler, generator=self.generators["dataloader"], **kwargs)
        cap = cfg.training.get("max_train_steps")
        if cap is not None:
            # Cap before Accelerate wraps the loader so its end-of-loader signal still flushes.
            loader = _cap_dataloader(loader, int(cap) * int(world_size))
        if len(loader) == 0:
            raise ValueError("The training loader is empty (dataset smaller than one micro-batch?)")
        return loader

    def _make_val_loader(self, val_dataset, cfg, world_size):
        kwargs = self._loader_kwargs(cfg.val_dataloader)
        kwargs["shuffle"] = False
        subset = _validation_subset(val_dataset, cfg.training.get("max_val_steps"),
                                    int(kwargs["batch_size"]), world_size)
        return DataLoader(subset, **kwargs), len(subset)

    # ------------------------------------------------------------------------ RNG
    def _capture_rng(self, device):
        cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": cuda, "generators": {name: generator.get_state()
                                             for name, generator in self.generators.items()}}

    def _gather_rng(self, accelerator):
        state = self._capture_rng(accelerator.device)
        if accelerator.num_processes > 1:
            states = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(states, state)
            self.rng_states = states
        else:
            self.rng_states = [state]

    def _restore_rng(self, rank, world_size, device):
        if self.rng_states is None or len(self.rng_states) != world_size:
            raise ValueError("Exact resume requires the original world size and every rank's RNG state")
        state = self.rng_states[rank]
        if set(state["generators"]) != set(self.generators):
            raise ValueError("Saved dedicated RNG streams differ from this run")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            if device.type != "cuda":
                raise ValueError("Checkpoint holds CUDA RNG state but this run is not on CUDA")
            torch.cuda.set_rng_state(state["cuda"], device)
        for name, generator in self.generators.items():
            generator.set_state(state["generators"][name])

    # ------------------------------------------------------------------ artifacts
    def _artifact(self, policy, override=None, trainable_dtype=None):
        state = policy.artifact_state_dict(trainable_override=override)
        if not isinstance(state, Mapping):
            raise TypeError("artifact_state_dict must return a mapping")
        frozen = set(policy.frozen_base_keys())
        leaked = sorted(frozen & set(state))
        if leaked:
            raise ValueError(f"Artifacts must exclude frozen pi05 base keys: {leaked[:5]}")
        missing = [name for name in self._trainable_names if name not in state]
        if missing:
            raise ValueError(f"Artifact lacks trainable parameters {missing[:5]}")
        trainable = set(self._trainable_names)
        result = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Artifact entry {key} is not a tensor")
            value = value.detach()
            if trainable_dtype is not None and key in trainable and value.is_floating_point():
                value = value.to(trainable_dtype)
            result[key] = value.to("cpu", copy=True)
        return result

    def _check_frozen_keys(self, policy):
        parameters = dict(policy.named_parameters())
        trainable = [key for key in policy.frozen_base_keys()
                     if key in parameters and parameters[key].requires_grad]
        if trainable:
            raise ValueError(f"Frozen base keys must not require grad: {trainable[:5]}")

    def _run_metadata(self, kind, trainable_dtype):
        cfg = self.cfg
        ema = self.ema
        return {
            "workspace": f"{type(self).__module__}.{type(self).__name__}",
            "checkpoint_format": CHECKPOINT_FORMAT, "kind": kind,
            "policy_family": POLICY_FAMILY, "variant": cfg.variant, "task_type": cfg.get("task_type"),
            "trainable_dtype": str(trainable_dtype).replace("torch.", ""),
            "optimizer_step": int(self.completed_optimizer_steps),
            "skipped_optimizer_steps": int(self.skipped_optimizer_steps),
            "epoch": int(self.epoch), "global_step": int(self.global_step),
            "ema": None if ema is None else {"decay": ema.decay, "warmup_power": ema.warmup_power,
                                             "updates": int(ema.updates)},
            "world_size": int(self.world_size),
            "update_schedule": copy.deepcopy(self.update_schedule),
            "dataset_split": copy.deepcopy(self.dataset_split),
            "seed": int(cfg.training.seed),
            "data_path": str(cfg.task.policy.dataset.get("zarr_path")),
            "software": _software_versions(), "source_sha256": _source_sha256(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

    def build_payload(self, kind):
        """``kind='resume'``: live + EMA artifacts and the training block; ``'snapshot'``: bf16 EMA only."""
        if kind not in ("resume", "snapshot"):
            raise ValueError("kind must be resume or snapshot")
        policy, ema = self.model, self.ema
        self._check_frozen_keys(policy)
        override = None if ema is None else ema.trainable_override()
        if kind == "resume":
            trainable_dtype = torch.float32
            state_dicts = {"model": self._artifact(policy)}
            if ema is not None:
                state_dicts["ema_model"] = self._artifact(policy, override)
        else:
            trainable_dtype = torch.bfloat16
            key = "model" if ema is None else "ema_model"
            state_dicts = {key: self._artifact(policy, override, torch.bfloat16)}
        metadata = _plain(policy.artifact_metadata())
        if not isinstance(metadata, dict):
            raise TypeError("artifact_metadata must return a dict")
        metadata["training_run"] = self._run_metadata(kind, trainable_dtype)
        payload = {"format": CHECKPOINT_FORMAT, "cfg": OmegaConf.to_container(self.cfg, resolve=True),
                   "policy_config": _plain(policy.export_config()), "metadata": metadata,
                   "state_dicts": state_dicts}
        if payload["policy_config"].get("construction_mode") != "restore":
            raise ValueError("export_config() must use construction_mode='restore'")
        if kind == "resume":
            payload["training"] = {
                "optimizer": _copy_to_cpu(self.optimizer.state_dict()),
                # Optimizer state is matched to parameters by position: record the order it was built in.
                "trainable_names": list(self._trainable_names),
                "lr_scheduler": copy.deepcopy(self.lr_scheduler.state_dict()),
                # The shadow itself is state_dicts['ema_model'][name] for these names.
                "ema": None if ema is None else {"decay": ema.decay, "warmup_power": ema.warmup_power,
                                                 "updates": int(ema.updates), "names": list(ema.names)},
                "rng_states": self.rng_states,
                "counters": {"epoch": int(self.epoch), "global_step": int(self.global_step),
                             "completed_optimizer_steps": int(self.completed_optimizer_steps),
                             "skipped_optimizer_steps": int(self.skipped_optimizer_steps),
                             "consecutive_skipped_steps": int(self.consecutive_skipped_steps),
                             "last_snapshot_step": self.last_snapshot_step},
                "world_size": int(self.world_size),
                "update_schedule": copy.deepcopy(self.update_schedule),
                "dataset_split": copy.deepcopy(self.dataset_split),
            }
        return payload

    def save_checkpoint(self, path=None, tag="latest", **_):
        """Synchronous, atomic resume checkpoint (main process only)."""
        path = pathlib.Path(path) if path is not None else self.get_checkpoint_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.rng_states is None:
            self.rng_states = [self._capture_rng(torch.device("cpu"))]
        _atomic_torch_save(self.build_payload("resume"), path)
        return str(path.absolute())

    def snapshot_path(self, step=None):
        step = self.completed_optimizer_steps if step is None else int(step)
        suffix = "model" if self.ema is None else "ema"
        return pathlib.Path(self.output_dir) / "snapshots" / f"upd-{step:06d}_{suffix}.ckpt"

    def save_snapshot(self, tag=None):
        """EMA artifact (trainable tensors bf16) at the current optimizer step (main process only)."""
        path = self.snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(self.build_payload("snapshot"), path)
        self.saved_snapshots.append(str(path))
        return str(path.absolute())

    # -------------------------------------------------------------------- report
    def training_report(self, policy, optimizer, clip, ema):
        def count(parameters):
            unique = {id(parameter): parameter for parameter in parameters}
            return sum(parameter.numel() for parameter in unique.values())
        parameters = list(policy.parameters())
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        groups = [{"name": group.get("name"), "lr": group["lr"], "weight_decay": group.get("weight_decay", 0.0),
                   "tensors": len(group["params"]), "elements": count(group["params"])}
                  for group in optimizer.param_groups]
        metadata = _plain(policy.artifact_metadata())
        return {
            "variant": self.cfg.variant, "policy_family": POLICY_FAMILY,
            "parameters": {"total": count(parameters), "trainable": count(trainable),
                           "frozen": count(parameters) - count(trainable),
                           "frozen_base_keys": len(policy.frozen_base_keys())},
            "optimizer_groups": groups,
            "clip_groups": {name: {"tensors": len(clip[name]), "elements": count(clip[name])}
                            for name in CLIP_GROUP_NAMES},
            "ema": None if ema is None else {"decay": ema.decay, "warmup_power": ema.warmup_power,
                                             "tensors": len(ema.names), "elements": ema.num_elements()},
            "update_schedule": self.update_schedule, "dataset_split": self.dataset_split,
            "ddp": self.ddp_settings, "world_size": self.world_size,
            "policy_metadata": metadata,
        }

    # -------------------------------------------------------------- one update
    def _optimizer_update(self, run: _Run, max_norm):
        """Clip per group, step if every norm is finite on every rank, then advance counters."""
        accelerator = run.accelerator
        norms = clip_gradients_per_group(run.clip, max_norm)
        finite = torch.ones((), dtype=torch.int32, device=accelerator.device)
        for value in norms.values():
            finite = finite * torch.isfinite(value).all().to(device=accelerator.device, dtype=torch.int32)
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        success = False
        if bool(finite.item()):
            run.optimizer.step()
            success = not accelerator.optimizer_step_was_skipped
        run.optimizer.zero_grad(set_to_none=False)
        if success:
            run.scheduler.step()
            run.policy.on_optimizer_step()
            if run.ema is not None:
                run.ema.step(run.named)
            self.completed_optimizer_steps += 1
            self.consecutive_skipped_steps = 0
        else:
            self.skipped_optimizer_steps += 1
            self.consecutive_skipped_steps += 1
            limit = int(self.cfg.training.get("max_consecutive_skipped_updates", 20))
            if self.consecutive_skipped_steps > limit:
                raise FloatingPointError(f"{self.consecutive_skipped_steps} consecutive optimizer updates had "
                                         "non-finite gradients; training diverged")
        return success, norms

    def _check_loss(self, loss):
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.is_floating_point():
            raise TypeError("policy(batch) must return a floating-point scalar tensor")
        if not loss.requires_grad:
            raise RuntimeError("The training loss does not require grad; no trainable parameter is connected")

    def _check_gradient_coverage(self, run: _Run):
        missing = [name for name, parameter in run.named if parameter.grad is None]
        if missing:
            raise RuntimeError("Trainable parameters received no gradient (DDP find_unused_parameters=False "
                               f"would fail): {missing[:10]}")

    def _lr_values(self, run: _Run):
        values, counts = {}, {}
        for group in run.optimizer.param_groups:
            name = str(group.get("name"))
            counts[name] = counts.get(name, 0) + 1
            key = f"lr/{name}" if counts[name] == 1 else f"lr/{name}_{counts[name] - 1}"
            values[key] = float(group["lr"])
        return values

    @staticmethod
    def _memory(device):
        if device.type != "cuda":
            return {}
        return {"max_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
                "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB}

    def _micro_step(self, run: _Run, batch, group_length, accumulation, history_mode=None):
        """Forward + backward of one micro-batch inside ``accelerator.accumulate``."""
        validate_history_batch(run.policy, batch)
        loss = run.model(batch) if history_mode is None else run.model(batch, history_mode=history_mode)
        self._check_loss(loss)
        # Accelerate divides by the nominal accumulation length; rescale a short tail
        # group so its gradient is still the mean over the micro-batches it contains.
        run.accelerator.backward(loss * (accumulation / group_length))
        if not run.checked_gradients:
            self._check_gradient_coverage(run)
            run.checked_gradients = True
        return loss.detach()

    # ------------------------------------------------------------------- training
    def _train_epoch(self, run: _Run, *, stop_at=None):
        cfg, accelerator = self.cfg, run.accelerator
        device = accelerator.device
        accumulation = int(cfg.training.gradient_accumulate_every)
        max_norm = cfg.training.get("max_grad_norm")
        snapshot_every = int(cfg.training.get("snapshot_every", 0) or 0)
        log_every = int(cfg.training.get("log_every", 1))
        run.model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        n_batches = len(run.train_loader)
        epoch_loss = torch.zeros(2, dtype=torch.float64, device=device)
        nonfinite = torch.zeros((), dtype=torch.float64, device=device)
        components: Dict[str, List[float]] = {}
        group_samples = 0
        epoch_started = time.monotonic()
        last_update = time.perf_counter()
        updates_this_epoch, samples_this_epoch, batches = 0, 0, 0
        stopped = False
        progress = tqdm.tqdm(run.train_loader, desc=f"Training epoch {self.epoch}", leave=False,
                             disable=not accelerator.is_local_main_process,
                             mininterval=float(cfg.training.get("tqdm_interval_sec", 1.0)))
        for batch_index, batch in enumerate(progress):
            batch = _move(batch, device)
            group_start = (batch_index // accumulation) * accumulation
            group_length = min(accumulation, n_batches - group_start)
            batch_size = int(batch["action"].shape[0])
            with accelerator.accumulate(run.model):
                loss = self._micro_step(run, batch, group_length, accumulation)
                loss_value = float(loss)
                if math.isfinite(loss_value):
                    epoch_loss += torch.tensor((loss_value * batch_size, float(batch_size)),
                                               dtype=torch.float64, device=device)
                else:
                    nonfinite += 1
                items = scalar_items(getattr(run.policy, "last_loss_components", None))
                items["loss"] = loss_value
                for name, value in items.items():
                    entry = components.setdefault(name, [0.0, 0.0])
                    entry[0] += value
                    entry[1] += 1.0
                group_samples += batch_size
                batches += 1
                self.global_step += 1
                if accelerator.sync_gradients:
                    success, norms = self._optimizer_update(run, max_norm)
                    now = time.perf_counter()
                    samples = group_samples * accelerator.num_processes
                    if success:
                        updates_this_epoch += 1
                        samples_this_epoch += samples
                        if snapshot_every and self.completed_optimizer_steps % snapshot_every == 0:
                            if accelerator.is_main_process:
                                self.save_snapshot()
                            self.last_snapshot_step = self.completed_optimizer_steps
                    if accelerator.is_main_process and (not success or self.completed_optimizer_steps % log_every == 0):
                        record = {"event": "train_step" if success else "skipped_update",
                                  "optimizer_step": self.completed_optimizer_steps, "epoch": self.epoch,
                                  "global_step": self.global_step,
                                  "grad_norm_ar": float(norms["ar"]), "grad_norm_ki": float(norms["ki"]),
                                  "samples_per_sec": samples / max(now - last_update, 1e-9),
                                  "self_past_step": getattr(run.policy, "self_past_step", None),
                                  **{f"train/{name}": total / count for name, (total, count) in components.items()},
                                  **self._lr_values(run), **self._memory(device)}
                        run.log.write(record)
                        if run.trackers:
                            accelerator.log(_numeric(record))
                        progress.set_postfix(loss=record.get("train/loss"), refresh=False)
                    components.clear()
                    group_samples = 0
                    last_update = now
                    if success and stop_at is not None and self.completed_optimizer_steps >= stop_at:
                        stopped = True
            if stopped:
                break
        progress.close()
        totals = accelerator.reduce(torch.cat((epoch_loss, nonfinite[None])), reduction="sum")
        seconds = time.monotonic() - epoch_started
        stats = {"train_loss": (float(totals[0] / totals[1]) if totals[1] > 0 else None),
                 "train_nonfinite_losses": int(totals[2]), "train_batches": batches,
                 "train_updates": updates_this_epoch, "train_epoch_seconds": seconds,
                 "train_samples_per_sec": samples_this_epoch / max(seconds, 1e-9),
                 **{f"epoch_{key}": value for key, value in self._memory(device).items()}}
        return stats, stopped

    # ----------------------------------------------------------------- validation
    @torch.no_grad()
    def _validate(self, run: _Run, max_batches=None):
        """Expert/generated-history losses and stateless reconstruction MSE under the EMA."""
        cfg, accelerator, policy = self.cfg, run.accelerator, run.policy
        device = accelerator.device
        generated = (bool(cfg.training.get("validate_generated_history", True))
                     and policy_capability(policy, "supports_generated_history_validation"))
        modes = ("expert", "generated") if generated else ("expert",)
        recon_cap = cfg.training.get("max_reconst_steps") if max_batches is None else max_batches
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        sums = _MetricSums()
        batches = 0
        # Both paths restore every module's training flag; the EMA path also restores the live weights.
        swap = run.ema.swap_in(policy) if run.ema is not None else _preserved_modes(policy)
        fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
            if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            # Common random numbers across epochs; the training RNG streams are untouched.
            validation_seed = int(cfg.training.seed) + 104729 + 7919 * accelerator.process_index
            torch.default_generator.manual_seed(validation_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(validation_seed)
            with swap:
                policy.eval()
                for batch_index, batch in enumerate(run.val_loader):
                    if max_batches is not None and batch_index >= max_batches:
                        break
                    batch = _move(batch, device)
                    validate_history_batch(policy, batch)
                    weight = float(batch["action"].shape[0])
                    for mode in modes:
                        loss = policy(batch, history_mode=mode)
                        items = scalar_items(getattr(policy, "last_loss_components", None))
                        items["loss"] = float(loss)
                        sums.add_all(items, weight, prefix=f"{mode}/")
                    if recon_cap is None or batch_index < int(recon_cap):
                        prediction = predict_validation_action(policy, batch)["action_pred"]
                        target = batch["action"]
                        if prediction.shape != target.shape:
                            raise ValueError(f"predict_action returned {tuple(prediction.shape)}, "
                                             f"expected {tuple(target.shape)}")
                        error = (prediction.float() - target.float()).square()
                        sums.add("reconstruction_mse", float(error.mean()), weight)
                        executed = int(getattr(policy, "n_action_steps", cfg.n_action_steps))
                        sums.add("reconstruction_mse_executed", float(error[:, :executed].mean()), weight)
                    gate = getattr(policy, "get_history_gate_metrics", None)
                    if callable(gate):
                        sums.add_all(scalar_items(gate()), weight, prefix="gate/")
                    batches += 1
            policy.reset()
        values = sums.reduce(accelerator)
        counted = accelerator.reduce(torch.tensor([batches], dtype=torch.float64, device=device), reduction="sum")
        result = {f"val/{name}": value for name, value in values.items()}
        result.update({
            "val_loss": values.get("expert/loss"),
            "val_loss_expert_history": values.get("expert/loss"),
            "val_loss_generated_history": values.get("generated/loss"),
            "val_reconstruction_mse": values.get("reconstruction_mse"),
            "validation_batches": int(counted.item()),
            "validation_seconds": time.monotonic() - started,
            **{f"validation_{key}": value for key, value in self._memory(device).items()},
        })
        return result

    @torch.no_grad()
    def _replica_check(self, run: _Run):
        """DDP replicas must stay bitwise identical; compare cheap per-rank fingerprints."""
        if run.accelerator.num_processes == 1:
            return None
        device = run.accelerator.device
        fingerprint = torch.zeros(2, dtype=torch.float64, device=device)
        for _, parameter in run.named:
            value = parameter.detach().double()
            fingerprint += torch.stack((value.sum(), value.square().sum()))
        gathered = run.accelerator.gather(fingerprint[None])
        if not torch.equal(gathered, gathered[:1].expand_as(gathered)):
            raise RuntimeError(f"DDP replicas diverged: per-rank parameter fingerprints {gathered.tolist()}")
        return float(fingerprint[0])

    # ----------------------------------------------------------------------- probe
    def _probe(self, run: _Run):
        """Memory/throughput probe: N updates at the configured self-past probability,
        one worst-case ``history_mode='generated'`` pass, and a short validation pass."""
        cfg, accelerator, policy = self.cfg, run.accelerator, run.policy
        device = accelerator.device
        probe = cfg.training.probe
        steps = _positive_int(probe.get("optimizer_steps", 3), "training.probe.optimizer_steps")
        accumulation = int(cfg.training.gradient_accumulate_every)
        target = probe.get("self_past_step")
        if target is None:
            target = int(cfg.policy.get("self_past_warmup_steps", 0)) + int(cfg.policy.get("self_past_ramp_steps", 0))
        policy.set_self_past_step(int(target))
        probability = float(policy.self_past_probability())
        run.model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        max_norm = cfg.training.get("max_grad_norm")
        update_seconds, update_samples = [], []
        components: Dict[str, float] = {}
        n_batches = len(run.train_loader)
        last_batch = None
        started = time.perf_counter()
        group_samples = 0
        for batch_index, batch in enumerate(run.train_loader):
            batch = _move(batch, device)
            group_start = (batch_index // accumulation) * accumulation
            group_length = min(accumulation, n_batches - group_start)
            with accelerator.accumulate(run.model):
                self._micro_step(run, batch, group_length, accumulation)
                components = scalar_items(getattr(policy, "last_loss_components", None))
                group_samples += int(batch["action"].shape[0])
                if accelerator.sync_gradients:
                    success, norms = self._optimizer_update(run, max_norm)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    now = time.perf_counter()
                    update_seconds.append(now - started)
                    update_samples.append(group_samples * accelerator.num_processes)
                    started, group_samples = now, 0
                    if not success:
                        raise FloatingPointError("Probe update had non-finite gradients")
            last_batch = batch
            if len(update_seconds) >= steps:
                break
        if len(update_seconds) < steps:
            raise ValueError(f"Probe needs {steps} updates but the loader ended after {len(update_seconds)}")
        timed = slice(1, None) if len(update_seconds) > 1 else slice(None)
        samples_per_sec = sum(update_samples[timed]) / max(sum(update_seconds[timed]), 1e-9)
        memory = {"train": self._memory(device)}
        worst = None
        if bool(probe.get("worst_case_pass", True)) and last_batch is not None:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            began = time.perf_counter()
            loss = run.model(last_batch, history_mode="generated")
            self._check_loss(loss)
            accelerator.backward(loss)
            run.optimizer.optimizer.zero_grad(set_to_none=False)
            worst = {"history_mode": "generated", "loss": float(loss.detach()),
                     "seconds": time.perf_counter() - began,
                     "components": scalar_items(getattr(policy, "last_loss_components", None))}
            memory["worst_case_generated"] = self._memory(device)
        validation = None
        batches = int(probe.get("validation_batches", 1) or 0)
        if batches and run.validation_enabled:
            validation = self._validate(run, max_batches=batches)
            memory["validation"] = self._memory(device)
        local = {"rank": accelerator.process_index, "device": str(device), "memory": memory,
                 "total_memory_gib": (torch.cuda.get_device_properties(device).total_memory / GIB
                                      if device.type == "cuda" else None)}
        ranks = [local]
        if accelerator.num_processes > 1:
            ranks = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(ranks, local)
        report = None
        if accelerator.is_main_process:
            peak = max((item.get("max_memory_reserved_gib", 0.0) for rank in ranks
                        for item in rank["memory"].values()), default=0.0)
            total = min((rank["total_memory_gib"] for rank in ranks if rank["total_memory_gib"]), default=None)
            limit = float(probe.get("max_reserved_gb", 22.5))
            headroom_needed = float(probe.get("min_headroom_gb", 2.0))
            smi = _nvidia_smi()
            used = _visible_memory_used_gb(smi)
            headroom = None
            if total is not None:
                # nvidia-smi includes the CUDA context and NCCL buffers that PyTorch does not reserve.
                headroom = (total * GIB / 1e9 - used) if used is not None else (total - peak) * GIB / 1e9
            go = None
            if total is not None:
                go = bool(peak * GIB / 1e9 <= limit and headroom >= headroom_needed)
            report = {
                "variant": cfg.variant, "world_size": accelerator.num_processes,
                "micro_batch_per_rank": int(cfg.dataloader.batch_size), "accumulation": accumulation,
                "effective_batch": int(cfg.dataloader.batch_size) * accelerator.num_processes * accumulation,
                "optimizer_steps": steps, "self_past_step": int(target), "self_past_p": probability,
                "seconds_per_update": update_seconds, "samples_per_sec": samples_per_sec,
                "last_train_components": components, "worst_case_pass": worst,
                "validation": validation, "ranks": ranks,
                "peak_reserved_gb": peak * GIB / 1e9, "peak_reserved_gib": peak,
                "go_no_go": {"max_reserved_gb": limit, "min_headroom_gb": headroom_needed,
                             "headroom_gb": headroom, "pass": go},
                "nvidia_smi": smi, "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            path = pathlib.Path(self.output_dir) / "probe.json"
            path.write_text(json.dumps(_sanitize(report), indent=2) + "\n")
            accelerator.print(json.dumps(_sanitize({key: report[key] for key in (
                "variant", "world_size", "samples_per_sec", "peak_reserved_gb", "go_no_go", "self_past_p")}), indent=2))
        self.probe_report = report
        accelerator.wait_for_everyone()
        return report

    # ------------------------------------------------------------------------ run
    def run(self):
        cfg = self.cfg
        self.check_run_config(cfg)
        if torch.cuda.is_available() and not torch.cuda.is_initialized():
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        training = cfg.training
        accumulation = int(training.gradient_accumulate_every)
        mode = cfg.logging.get("mode", "offline") if cfg.get("logging") else "disabled"
        log_with = "wandb" if mode in ("online", "offline") else None
        accelerator = Accelerator(
            mixed_precision="no",  # the policy owns bf16 autocast
            gradient_accumulation_steps=accumulation, log_with=log_with,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False,
                                                           gradient_as_bucket_view=True),
                             InitProcessGroupKwargs(timeout=timedelta(hours=2))])
        device = accelerator.device
        self.world_size = accelerator.num_processes
        seed = int(training.seed)
        accelerate_set_seed(seed, device_specific=True)
        # Worker base seeds only; the data order comes from EpochSeededRandomSampler.
        self.generators = {"dataloader": torch.Generator().manual_seed(seed + 43)}
        probe = bool(training.get("probe", {}).get("enabled", False))

        payload = None
        if training.get("resume"):
            if probe:
                raise ValueError("The probe always starts fresh; drop --resume")
            payload = self.read_payload(training.resume_checkpoint)
            self.validate_resume_payload(payload, cfg, world_size=self.world_size)
            counters = payload["training"]["counters"]
            max_steps = training.get("max_optimizer_steps")
            if int(counters["epoch"]) >= int(training.num_epochs) or (
                    max_steps is not None and int(counters["completed_optimizer_steps"]) >= int(max_steps)):
                accelerator.print(f"Checkpoint already completed {counters['epoch']} epochs / "
                                  f"{counters['completed_optimizer_steps']} updates; nothing to do.")
                return None

        # 1-2. policy on its device
        policy = self._instantiate_policy(cfg, payload)
        policy.to(device)
        # 3. data and normalizer (a resumed artifact already carries its normalizers)
        dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
        val_dataset = dataset.get_validation_dataset()
        if payload is None:
            policy.set_normalizer(dataset.get_normalizer())
        # 4. optimizer, built after the move
        optimizer = policy.get_optimizer(**_plain(cfg.optimizer))
        named, clip = validate_parameter_partition(policy, optimizer)
        self._trainable_names = [name for name, _ in named]
        if payload is not None:
            self.load_optimizer_state(optimizer, payload["training"], self._trainable_names)
        train_loader = self._make_train_loader(dataset, cfg, self.world_size)
        val_loader, val_windows = self._make_val_loader(val_dataset, cfg, self.world_size)
        # 5. prepare (DDP broadcasts rank 0's parameters here)
        model, optimizer, train_loader, val_loader = accelerator.prepare(policy, optimizer, train_loader, val_loader)
        if accelerator.unwrap_model(model) is not policy:
            raise RuntimeError("Accelerate replaced the policy object; the EMA must track the live module")
        self.model, self.optimizer = policy, optimizer
        self.ddp_settings = {"wrapper": type(model).__name__,
                             "find_unused_parameters": getattr(model, "find_unused_parameters", None),
                             "gradient_as_bucket_view": getattr(model, "gradient_as_bucket_view", None),
                             "broadcast_buffers": getattr(model, "broadcast_buffers", None),
                             "mixed_precision": accelerator.mixed_precision}
        split = TrainP2NNewWorkspace._offline_validation_metadata(
            dataset, val_dataset, training, train_loader, val_loader)
        split["validation"]["evaluated_windows"] = int(val_windows)
        if payload is not None:
            saved_identity = (payload["training"]["dataset_split"] or {}).get("identity")
            if saved_identity != split.get("identity"):
                raise ValueError("Resume dataset identity or episode split differs from the checkpoint")
        self.dataset_split = split
        validation_enabled = bool(split["offline_validation_enabled"])

        ema = None
        if bool(training.get("use_ema", True)):
            ema_cfg = _plain(cfg.ema)
            ema = TrainableEMA(named, decay=ema_cfg.get("decay", 0.999), warmup_power=ema_cfg.get("warmup_power"))

        n_batches = len(train_loader)
        # Update arithmetic (tail groups counted once per epoch). The warmup is validated against the
        # cosine horizon by WarmupCosineSchedule, not against the planned updates.
        schedule = resolve_update_schedule(n_batches, int(training.num_epochs), accumulation,
                                           max_train_steps=training.get("max_train_steps"), warmup_steps=0)
        schedule["lr_warmup_steps"] = int(training.get("lr_warmup_steps", 0))
        max_steps = training.get("max_optimizer_steps")
        horizon = int(max_steps) if max_steps is not None else schedule["planned_optimizer_updates"]
        micro = int(cfg.dataloader.batch_size)
        schedule.update({
            "max_optimizer_steps": None if max_steps is None else int(max_steps),
            "cosine_horizon": horizon, "min_lr_ratio": float(training.get("min_lr_ratio", 0.1)),
            "expected_optimizer_updates": (schedule["planned_optimizer_updates"] if max_steps is None
                                           else min(int(max_steps), schedule["planned_optimizer_updates"])),
            "micro_batch_per_rank": micro, "world_size": self.world_size,
            "effective_batch": micro * self.world_size * accumulation})
        scheduler = make_lr_scheduler(optimizer.optimizer, schedule["lr_warmup_steps"], horizon,
                                      schedule["min_lr_ratio"])
        if payload is not None:
            saved = payload["training"]["update_schedule"]
            for key in ("batches_per_rank", "updates_per_epoch", "lr_warmup_steps", "cosine_horizon",
                        "min_lr_ratio", "effective_batch"):
                if saved.get(key) != schedule.get(key):
                    raise ValueError(f"Resume must preserve the update schedule: {key} "
                                     f"{saved.get(key)} != {schedule.get(key)}")
            state = payload["training"]["lr_scheduler"]
            scheduler.load_state_dict(state)
            rates = state.get("_last_lr")
            if rates is None or len(rates) != len(optimizer.param_groups):
                raise ValueError("Saved scheduler rates do not match the optimizer groups")
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate
            if ema is not None:
                saved_ema = payload["training"]["ema"]
                if list(saved_ema["names"]) != list(ema.names):
                    raise ValueError("Resume EMA parameter names differ from this policy")
                ema_artifact = payload["state_dicts"]["ema_model"]
                ema.load_state_dict({"decay": saved_ema["decay"], "updates": saved_ema["updates"],
                                     "warmup_power": saved_ema.get("warmup_power"),
                                     "shadow": {name: ema_artifact[name] for name in ema.names}})
            counters = payload["training"]["counters"]
            self.epoch = int(counters["epoch"])
            self.global_step = int(counters["global_step"])
            self.completed_optimizer_steps = int(counters["completed_optimizer_steps"])
            self.skipped_optimizer_steps = int(counters["skipped_optimizer_steps"])
            self.consecutive_skipped_steps = int(counters.get("consecutive_skipped_steps", 0))
            self.last_snapshot_step = counters.get("last_snapshot_step")
            self.rng_states = payload["training"]["rng_states"]
            if ema is not None and ema.updates != self.completed_optimizer_steps:
                raise ValueError("EMA update count and successful optimizer updates disagree")
            step = getattr(policy, "self_past_step", None)
            if step is not None and int(step) != self.completed_optimizer_steps:
                raise ValueError("Policy self-past counter and successful optimizer updates disagree")
        self.update_schedule = schedule
        self.ema, self.lr_scheduler = ema, scheduler
        del payload

        output = pathlib.Path(self.output_dir)
        run = _Run(accelerator=accelerator, model=model, policy=policy, optimizer=optimizer, scheduler=scheduler,
                   ema=ema, named=named, clip=clip, train_loader=train_loader, val_loader=val_loader,
                   log=_JsonlLog(output / "logs.jsonl", accelerator.is_main_process),
                   trackers=log_with is not None, validation_enabled=validation_enabled)
        report = self.training_report(policy, optimizer, clip, ema)
        if accelerator.is_main_process:
            output.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(_plain(cfg)), output / "resolved_config.yaml")
            (output / "dataset_split.json").write_text(json.dumps(_sanitize(split), indent=2) + "\n")
            (output / "training_report.json").write_text(json.dumps(_sanitize(report), indent=2) + "\n")
            accelerator.print(json.dumps(_sanitize({key: report[key] for key in (
                "variant", "parameters", "optimizer_groups", "clip_groups", "ema", "update_schedule", "ddp")}),
                indent=2))
        if log_with is not None:
            logging = dict(_plain(cfg.logging))
            project = logging.pop("project")
            logging["dir"] = str(output)
            accelerator.init_trackers(project, config=_sanitize(_plain(cfg)), init_kwargs={"wandb": logging})
            if accelerator.is_main_process:
                tracker = accelerator.get_tracker("wandb", unwrap=True)
                tracker.define_metric("optimizer_step")
                tracker.define_metric("*", step_metric="optimizer_step")

        if probe:
            try:
                return self._probe(run)
            finally:
                accelerator.end_training()

        if training.get("resume"):
            # After every constructor, DDP broadcast and tracker; before the first draw.
            self._restore_rng(accelerator.process_index, accelerator.num_processes, device)

        num_epochs = int(training.num_epochs)
        keep_resume = bool(training.get("keep_resume_checkpoint", False))
        stopped = False
        while self.epoch < num_epochs and not stopped:
            if hasattr(train_loader, "set_epoch"):
                train_loader.set_epoch(self.epoch)
            if hasattr(self._sampler, "set_epoch"):
                self._sampler.set_epoch(self.epoch)
            stats, stopped = self._train_epoch(run, stop_at=None if max_steps is None else int(max_steps))
            completed_epoch = self.epoch
            self.epoch += 1
            final = stopped or self.epoch >= num_epochs
            record = {"event": "epoch", "epoch": completed_epoch, "optimizer_step": self.completed_optimizer_steps,
                      "global_step": self.global_step, "skipped_optimizer_steps": self.skipped_optimizer_steps,
                      **stats, **self._lr_values(run)}
            if validation_enabled and (completed_epoch % int(training.val_every) == 0 or final):
                self.last_validation = self._validate(run)
                record.update(self.last_validation)
            record["replica_fingerprint"] = self._replica_check(run)
            self._gather_rng(accelerator)
            checkpoint_due = completed_epoch % int(training.checkpoint_every) == 0 or final
            if accelerator.is_main_process:
                saved_started = time.monotonic()
                if checkpoint_due and (keep_resume or not final):
                    record["checkpoint"] = self.save_checkpoint()
                if final:
                    if self.last_snapshot_step != self.completed_optimizer_steps:
                        record["snapshot"] = self.save_snapshot()
                    latest = self.get_checkpoint_path()
                    if not keep_resume and latest.exists():
                        latest.unlink()
                record["checkpoint_seconds"] = time.monotonic() - saved_started
            if final:
                self.last_snapshot_step = self.completed_optimizer_steps
            if accelerator.is_main_process:
                run.log.write(record)
                if run.trackers:
                    accelerator.log(_numeric(record))
                accelerator.print(json.dumps(_sanitize(record)))
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            summary = {"variant": cfg.variant, "epochs": self.epoch, "optimizer_steps": self.completed_optimizer_steps,
                       "skipped_optimizer_steps": self.skipped_optimizer_steps, "global_step": self.global_step,
                       "stopped_by_max_optimizer_steps": stopped,
                       "snapshots": sorted(str(path) for path in (output / "snapshots").glob("upd-*.ckpt")),
                       "resume_checkpoint_kept": self.get_checkpoint_path().exists(),
                       "last_validation": self.last_validation}
            (output / "training_summary.json").write_text(json.dumps(_sanitize(summary), indent=2) + "\n")
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return self.last_validation


def _nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append({"index": parts[0], "name": parts[1], "memory_used_mib": float(parts[2]),
                         "memory_total_mib": float(parts[3]), "utilization_pct": float(parts[4])})
    return rows


def _visible_memory_used_gb(rows):
    """Largest nvidia-smi memory use (decimal GB) among this job's visible GPUs."""
    if not rows:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = rows
    if visible:
        indices = [item.strip() for item in visible.split(",")]
        if all(item.isdigit() for item in indices):
            selected = [row for row in rows if row["index"] in indices] or rows
    return max(row["memory_used_mib"] for row in selected) * 2 ** 20 / 1e9


@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
            config_name="train_p2n_vla")
def main(cfg):
    TrainP2NVLAWorkspace(cfg).run()


if __name__ == "__main__":
    main()
