"""Isolated training workspace for p2n_new variants; legacy workspace is unchanged.

The established training loop is retained with explicit capability routing,
self-contained restore, successful-update scheduling, and RNG continuation.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import time
import json
import numbers
import hydra
from datetime import timedelta
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pathlib
import copy
import dill
import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import (
    set_seed as accelerate_set_seed, DistributedDataParallelKwargs)
from typing import Union

from oat.workspace.base_workspace import BaseWorkspace
from oat.dataset.base_dataset import BaseDataset
from oat.env_runner.base_runner import BaseRunner
from oat.common.checkpoint_util import TopKCheckpointManager
from oat.common.json_logger import JsonLogger
from oat.common.hydra_util import register_new_resolvers
from oat.common.pytorch_util import dict_apply, maybe_to_device
from oat.model.common.lr_scheduler import get_scheduler
from oat.model.common.misc import detect_bf16_support
from oat.policy.base_policy import BasePolicy
from oat.common.p2n_new_capabilities import (policy_capability, predict_validation_action,
    validate_history_batch, resolve_update_schedule)
from oat.workspace.base_workspace import _copy_to_cpu, _atomic_torch_save
import random
import hashlib
import numpy as np
import importlib.metadata

register_new_resolvers()

class _LimitedBatchSampler:
    def __init__(self, sampler, maximum):
        if maximum < 1:
            raise ValueError("max_train_steps must be positive")
        self.sampler = sampler
        self.maximum = maximum
        self.batch_size = sampler.batch_size
        self.drop_last = sampler.drop_last

    def __len__(self):
        return min(len(self.sampler), self.maximum)

    def __iter__(self):
        from itertools import islice
        yield from islice(self.sampler, self.maximum)


def _cap_dataloader(loader, maximum):
    # Cap before Accelerator wraps the loader, so its end-of-loader signal flushes
    # the final incomplete accumulation group, including explicitly capped epochs.
    kwargs = {"dataset": loader.dataset, "batch_sampler": _LimitedBatchSampler(loader.batch_sampler, maximum),
              "num_workers": loader.num_workers, "collate_fn": loader.collate_fn,
              "pin_memory": loader.pin_memory, "persistent_workers": loader.persistent_workers,
              "worker_init_fn": loader.worker_init_fn, "generator": loader.generator}
    if loader.num_workers:
        kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(**kwargs)



class TrainP2NNewWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'checkpoint_version',
                    'completed_optimizer_steps', 'ema_state', 'lr_scheduler_state',
                    'resume_migration', 'rng_states', 'update_schedule', 'dataset_split']

    def __init__(self, cfg: OmegaConf, output_dir=None, lazy_instantiation=True):
        super().__init__(cfg, output_dir=output_dir)

        """
        Lazy instantiation allows deferring model, optimizer, and ema creation
        until after the seed has been set and the accelerator device has been chosen.
        1. If lazy_instantiation is False, model, optimizer, and ema are created immediately.
           This is useful for checkpoint loading, where we need to create the model
           before loading the state dict.
        2. If lazy_instantiation is True, model, optimizer, and ema are created in run().
           This is useful for normal training, where we want to seed the creation of these
           objects.
        """
        if lazy_instantiation:
            self.model = None
            self.ema_model = None
            self.optimizer = None
        else:
            self.model = hydra.utils.instantiate(cfg.policy)
            if cfg.training.use_ema:
                self.ema_model = copy.deepcopy(self.model)
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0
        # Version 2 saves the NEXT epoch and completed batch/update counts.
        self.checkpoint_version = 3
        self.completed_optimizer_steps = 0
        self.ema_state = None
        self.lr_scheduler_state = None
        self.resume_migration = None
        self.rng_states = None
        self.update_schedule = None
        self.dataset_split = None

    @staticmethod
    def _make_training_dataloader(dataset, loader_kwargs):
        kwargs = dict(loader_kwargs)
        sampler_factory = getattr(dataset, "get_training_sampler", None)
        if sampler_factory is not None:
            kwargs.update(sampler=sampler_factory(), shuffle=False)
        return DataLoader(dataset, **kwargs)

    @staticmethod
    def _offline_validation_metadata(dataset, val_dataset, training,
                                     train_dataloader, val_dataloader):
        """Describe the actual split and reject undefined held-out metrics.

        Zero-validation diagnostic runs must explicitly opt out. Disabling
        offline validation also disables reconstruction on validation data;
        simulator rollouts remain controlled separately by task.policy.lazy_eval.
        """
        enabled = training.get("offline_validation_enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("training.offline_validation_enabled must be a boolean")
        reason = ("enabled" if enabled else training.get(
            "offline_validation_reason", "explicitly_disabled_by_configuration"))
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("training.offline_validation_reason must be a nonempty string")
        if enabled and (len(val_dataset) == 0 or len(val_dataloader) == 0):
            raise ValueError(
                "Offline validation is enabled but the validation dataset/loader is empty. "
                "Provide held-out data (and usable batches), or explicitly set "
                "training.offline_validation_enabled=false for a run without offline validation."
            )

        def counts(data):
            result = {"windows": int(len(data)), "episodes": None}
            mask = getattr(data, "train_mask", None)
            if mask is not None:
                result["episodes"] = int(sum(bool(value) for value in mask))
            return result

        train_mask = np.asarray(dataset.train_mask, dtype=np.bool_)
        validation_mask = np.asarray(val_dataset.train_mask, dtype=np.bool_)
        ends = np.asarray(dataset.replay_buffer.episode_ends)
        digest = hashlib.sha256()
        digest.update(ends.tobytes())
        digest.update(np.asarray(dataset.replay_buffer[dataset.action_key]).tobytes())
        identity = {"episode_and_action_sha256": digest.hexdigest(),
                    "train_episode_ids": np.flatnonzero(train_mask).tolist(),
                    "validation_episode_ids": np.flatnonzero(validation_mask).tolist()}
        return {
            "schema_version": 1,
            "identity": identity,
            "offline_validation_enabled": enabled,
            "offline_validation_reason": reason,
            "train": {**counts(dataset), "batches_per_epoch": len(train_dataloader)},
            "validation": {**counts(val_dataset), "batches": len(val_dataloader)},
        }


    def _capture_training_state(self, ema, lr_scheduler):
        self.checkpoint_version = 3
        self.lr_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())
        self.ema_state = (None if ema is None else {
            "optimization_step": int(ema.optimization_step), "decay": float(ema.decay),
        })

    def _complete_epoch(self, ema, lr_scheduler):
        completed_epoch = self.epoch
        self.epoch += 1
        self._capture_training_state(ema, lr_scheduler)
        return completed_epoch

    def _restore_training_helpers(self, ema, lr_scheduler):
        if self.lr_scheduler_state is not None:
            lr_scheduler.load_state_dict(self.lr_scheduler_state)
            # Loading scheduler state alone does not undo constructor LR changes.
            rates = self.lr_scheduler_state.get("_last_lr")
            if rates is None or len(rates) != len(self.optimizer.param_groups):
                raise ValueError("Saved scheduler learning rates do not match optimizer groups")
            for group, rate in zip(self.optimizer.param_groups, rates):
                group["lr"] = rate
        if ema is not None:
            state = self.ema_state or {"optimization_step": self.completed_optimizer_steps}
            ema.optimization_step = int(state["optimization_step"])
            ema.decay = float(state.get("decay", ema.get_decay(max(0, ema.optimization_step - 1))))



    _predict_validation_action = staticmethod(predict_validation_action)

    @staticmethod
    def validate_resume_payload(payload, cfg):
        if "policy_config" not in payload or "metadata" not in payload:
            raise ValueError("Resume requires a self-contained p2n_new training artifact")
        saved = payload["cfg"]
        if payload["metadata"].get("variant") != cfg.variant:
            raise ValueError("Checkpoint variant does not match selected variant")
        required = ("epoch", "global_step", "completed_optimizer_steps", "ema_state", "lr_scheduler_state", "rng_states")
        if any(key not in payload.get("pickles", {}) for key in required):
            raise ValueError("Training checkpoint is missing continuation state")
        if "model" not in payload["state_dicts"] or "optimizer" not in payload["state_dicts"]:
            raise ValueError("Training checkpoint is missing model or optimizer state")
        if bool(saved.training.use_ema) != bool(cfg.training.use_ema):
            raise ValueError("Resume cannot change EMA configuration")
        if cfg.training.use_ema and "ema_model" not in payload["state_dicts"]:
            raise ValueError("Training checkpoint is missing EMA weights")
        architecture = ("_target_", "variant", "shape_meta", "n_action_steps", "n_obs_steps", "past_n",
                        "horizon", "embed_dim", "n_layers", "n_heads", "ffn_dim", "num_visual_queries",
                        "resampler_depth", "task", "state_history_steps", "state_history_keys",
                        "history_summary_tokens", "rotation_6d_layout", "history_embed_dim",
                        "history_n_heads", "history_n_layers")
        def plain(value):
            return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        for key in architecture:
            if plain(saved.policy.get(key)) != plain(cfg.policy.get(key)):
                raise ValueError(f"Resume architecture/schema mismatch: policy.{key}")
        for key in ("zarr_path", "obs_keys", "action_key", "n_obs_steps", "n_action_steps", "past_n", "n_exec_steps", "seed", "val_ratio", "max_train_episodes"):
            if plain(saved.task.policy.dataset.get(key)) != plain(cfg.task.policy.dataset.get(key)):
                raise ValueError(f"Resume data split/schema mismatch: dataset.{key}")
        return payload

    def _policy_config_for_run(self, cfg):
        if not cfg.training.resume:
            return cfg.policy
        path = cfg.training.get("resume_checkpoint")
        if not path:
            raise ValueError("Resume requires training.resume_checkpoint explicitly")
        payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
        self.validate_resume_payload(payload, cfg)
        return OmegaConf.create(payload["policy_config"])

    def _resume_training_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
        self.validate_resume_payload(payload, self.cfg)
        self.load_payload(payload, exclude_keys=(), include_keys=self.include_keys)
        if self.checkpoint_version != 3 or self.lr_scheduler_state is None:
            raise ValueError("Only complete version-3 p2n_new training checkpoints may resume")
        self.model.reset()
        if self.ema_model is not None:
            self.ema_model.reset()
        return payload

    def training_report(self, model):
        groups = [{"name": group.get("name", str(index)), "lr": group["lr"],
                   "weight_decay": group.get("weight_decay", 0),
                   "elements": sum(parameter.numel() for parameter in group["params"])}
                  for index, group in enumerate(self.optimizer.param_groups)]
        return {"variant": self.cfg.variant,
                "parameters": {"total": sum(p.numel() for p in model.parameters()),
                               "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                               "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad)},
                "optimizer_groups": groups, "update_schedule": self.update_schedule,
                "dataset_split": self.dataset_split, "policy_metadata": model.artifact_metadata()}

    @staticmethod
    def _local_rng():
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}

    def _gather_rng(self, accelerator):
        local = self._local_rng()
        if accelerator.num_processes > 1:
            states = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(states, local)
            self.rng_states = states
        else:
            self.rng_states = [local]

    def _restore_rng(self, rank=0, world_size=1):
        if self.rng_states is None or len(self.rng_states) != world_size:
            raise ValueError("Resume requires the original world size and per-rank RNG state")
        state = self.rng_states[rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def save_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, use_thread=True):
        # Synchronous atomic snapshots avoid queued copies of these large models.
        path = pathlib.Path(path) if path is not None else self.get_checkpoint_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        policy_config = self.model.export_config()
        if OmegaConf.is_config(policy_config):
            policy_config = OmegaConf.to_container(policy_config, resolve=True)
        cfg = copy.deepcopy(self.cfg)
        payload = {"cfg": cfg, "policy_config": policy_config,
                   "metadata": self.model.artifact_metadata(), "state_dicts": {}, "pickles": {}}
        payload["metadata"].update({
            "software": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "accelerate", "hydra-core")},
            "successful_optimizer_updates": self.completed_optimizer_steps,
            "dataset_split": self.dataset_split,
            "data_path": str(cfg.task.policy.dataset.zarr_path),
            "seed": int(cfg.training.seed), "update_schedule": self.update_schedule,
        })
        if self.rng_states is None:
            self.rng_states = [self._local_rng()]
        excluded = set(exclude_keys or ())
        for key in ("model", "ema_model", "optimizer"):
            value = getattr(self, key, None)
            if value is not None and key not in excluded:
                payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
        for key in include_keys or self.include_keys:
            payload["pickles"][key] = dill.dumps(getattr(self, key))
        _atomic_torch_save(payload, path)
        return str(path.absolute())

    @classmethod
    def create_from_checkpoint(cls, path, output_dir=None, **kwargs):
        payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
        cfg = copy.deepcopy(payload["cfg"])
        cfg.training.resume = True
        cfg.training.resume_checkpoint = str(path)
        instance = cls(cfg, output_dir=output_dir)
        instance.model = hydra.utils.instantiate(instance._policy_config_for_run(cfg))
        instance.ema_model = copy.deepcopy(instance.model) if cfg.training.use_ema else None
        instance.optimizer = instance.model.get_optimizer(**cfg.optimizer)
        instance._resume_training_checkpoint(path)
        instance.model.eval()
        if instance.ema_model is not None:
            instance.ema_model.eval()
        return instance

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        init_checkpoint = cfg.training.get("init_checkpoint")
        if init_checkpoint:
            raise ValueError("New architectures require fresh initialization or an explicit compatible resume; init_checkpoint is unsupported")

        # configure accelerator
        accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=False),
                InitProcessGroupKwargs(timeout=timedelta(hours=2)), # sim eval can take long time
            ],
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            mixed_precision="bf16" if cfg.training.allow_bf16 and detect_bf16_support() else "no",
        )
        device = accelerator.device

        # set seed
        seed = int(cfg.training.seed)
        accelerate_set_seed(seed, device_specific=True)

        # configure model, ema, and optimizer after seeding
        self.model: BasePolicy = hydra.utils.instantiate(self._policy_config_for_run(cfg))
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # configure dataset
        dataset: BaseDataset = hydra.utils.instantiate(
            cfg.task.policy.dataset)
        train_dataloader = self._make_training_dataloader(dataset, cfg.dataloader)
        if cfg.training.max_train_steps is not None:
            train_dataloader = _cap_dataloader(train_dataloader, int(cfg.training.max_train_steps) * accelerator.num_processes)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        dataset_split = self._offline_validation_metadata(
            dataset, val_dataset, cfg.training, train_dataloader, val_dataloader)
        offline_validation_enabled = dataset_split["offline_validation_enabled"]

        # configure normalizer
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure checkpoint
        if accelerator.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **cfg.checkpoint.topk
            )

        # configure env
        lazy_eval = cfg.task.policy.lazy_eval  # don't eval during training
        if (not lazy_eval) and accelerator.is_main_process:
            env_runner: BaseRunner = hydra.utils.instantiate(
                cfg.task.policy.env_runner,
                output_dir=self.output_dir
            )

        # resume training
        if cfg.training.resume:
            resume_path = cfg.training.get("resume_checkpoint")
            latest_ckpt_path = pathlib.Path(resume_path) if resume_path else self.get_checkpoint_path()
            if resume_path and not latest_ckpt_path.is_file():
                raise FileNotFoundError(latest_ckpt_path)
            if not latest_ckpt_path.is_file():
                raise FileNotFoundError(latest_ckpt_path)
            if latest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {latest_ckpt_path}")
                self._resume_training_checkpoint(latest_ckpt_path)
                if self.dataset_split is None or self.dataset_split.get("identity") != dataset_split.get("identity"):
                    raise ValueError("Resume actual dataset identity or episode masks differ from saved training split")
                if accelerator.is_main_process:
                    restored = topk_manager.restore_from_logs(
                        os.path.join(self.output_dir, 'logs.json'), self.epoch)
                    accelerator.print(f"Restored {restored} retained checkpoint rankings")
                if self.epoch >= cfg.training.num_epochs:
                    accelerator.print(f"Already trained for {self.epoch} epochs. Exiting.")
                    return

        # prepare with accelerator
        (
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
        ) = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
        )
        # Report the batches each process actually executes after sharding.
        dataset_split = self._offline_validation_metadata(
            dataset, val_dataset, cfg.training, train_dataloader, val_dataloader)
        if accelerator.is_main_process:
            split_path = pathlib.Path(self.output_dir) / "dataset_split.json"
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(json.dumps(dataset_split, indent=2) + "\n")
            accelerator.print(f"Dataset split and offline validation: {json.dumps(dataset_split)}")

        ema = None
        if cfg.training.use_ema:
            self.ema_model = accelerator.prepare(self.ema_model)
            ema = hydra.utils.instantiate(cfg.ema, model=accelerator.unwrap_model(self.ema_model))

        # configure lr scheduler
        len_train_dataloader = len(train_dataloader)
        self.update_schedule = resolve_update_schedule(
            len_train_dataloader, cfg.training.num_epochs, cfg.training.gradient_accumulate_every,
            max_train_steps=cfg.training.max_train_steps,
            warmup_steps=cfg.training.get("lr_warmup_steps"),
            warmup_ratio=cfg.training.get("lr_warmup_ratio", 0.05))
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler, optimizer=self.optimizer,
            num_warmup_steps=self.update_schedule["lr_warmup_steps"],
            num_training_steps=self.update_schedule["planned_optimizer_updates"],
            last_epoch=-1)
        self._restore_training_helpers(ema, lr_scheduler)

        self.dataset_split = dataset_split
        accelerator.print(json.dumps(self.training_report(accelerator.unwrap_model(self.model)), indent=2))
        # configure logging
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop("project")
        wandb_cfg['dir'] = str(self.output_dir)
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )
        if accelerator.is_main_process:
            wandb_run = accelerator.get_tracker("wandb").run
            wandb_run.config.update({
                "output_dir": str(self.output_dir), "dataset_split": dataset_split,
            }, allow_val_change=True)
            # W&B history can extend beyond the restored checkpoint. Let W&B
            # advance its own history step; plot against true training progress.
            wandb_run.define_metric("global_step")
            wandb_run.define_metric("*", step_metric="global_step")

        if cfg.training.resume:
            self._restore_rng(accelerator.process_index, accelerator.num_processes)

        # training loop
        with JsonLogger(os.path.join(self.output_dir, 'logs.json'), filter_fn=lambda key, value:
                        isinstance(value, numbers.Number) or key == "offline_validation_reason") as json_logger:
            while self.epoch < cfg.training.num_epochs:

                if accelerator.is_main_process:
                    step_log = dict()

                train_started = time.monotonic()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                # model to train mode
                self.model.train()
                if cfg.training.use_ema:
                    self.ema_model.train()

                loss_info = torch.zeros(2, device=device)   # [total loss, total batch_size]
                with tqdm.tqdm(
                    train_dataloader, 
                    desc=f"Training epoch {self.epoch}",
                    leave=False, 
                    disable=not accelerator.is_local_main_process,
                    mininterval=cfg.training.tqdm_interval_sec
                ) as tepoch:

                    for batch_idx, batch in enumerate(tepoch):
                        with accelerator.accumulate(self.model):
                            # device transfer
                            batch = dict_apply(batch, lambda x: maybe_to_device(x, device))

                            validate_history_batch(accelerator.unwrap_model(self.model), batch)

                            # forward pass
                            with accelerator.autocast():
                                loss = self.model(batch)

                            # backward pass
                            accelerator.backward(loss)

                            # log loss
                            batch_size = batch['action'].shape[0]
                            loss_info[0] += loss.detach() * batch_size
                            loss_info[1] += batch_size

                            # step optimizer
                            if accelerator.sync_gradients:
                                # clip grad norm
                                if cfg.training.max_grad_norm is not None:
                                    accelerator.clip_grad_norm_(
                                        self.model.parameters(), 
                                        cfg.training.max_grad_norm
                                    )
                            
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)

                                if not accelerator.optimizer_step_was_skipped:
                                    lr_scheduler.step()
                                    self.completed_optimizer_steps += 1
                                    online_policy = accelerator.unwrap_model(self.model)
                                    if hasattr(online_policy, "on_optimizer_step"):
                                        online_policy.on_optimizer_step()
                                    # EMA updates parameters, so synchronize the curriculum
                                    # counter explicitly instead of averaging it.
                                    if cfg.training.use_ema:
                                        ema.step(online_policy)
                                        ema_policy = accelerator.unwrap_model(self.ema_model)
                                        if hasattr(ema_policy, "set_self_past_step"):
                                            ema_policy.set_self_past_step(online_policy.self_past_step)

                            # logging
                            is_last_batch = (batch_idx == (len_train_dataloader-1))
                            if accelerator.is_main_process:
                                loss_cpu = loss.item()
                                tepoch.set_postfix(loss=loss_cpu, refresh=False)
                                step_log = {
                                    'train_loss': loss_cpu,
                                    'global_step': self.global_step,
                                    'epoch': self.epoch,
                                    'lr': lr_scheduler.get_last_lr()[0],
                                }
                                if not is_last_batch:
                                    accelerator.log(step_log)
                                    json_logger.log(step_log)

                            # Count every completed batch exactly once, including
                            # the final batch and explicitly capped epochs.
                            self.global_step += 1

                            # break if reach max training steps
                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                # at the end of each epoch
                # replace train_loss with epoch average
                accelerator.wait_for_everyone()
                loss_info = accelerator.reduce(loss_info, reduction='sum')
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    step_log['train_loss'] = (loss_info[0] / loss_info[1]).item()
                    step_log['train_epoch_seconds'] = time.monotonic() - train_started
                    step_log['train_batches'] = batch_idx + 1
                    step_log.update({
                        "offline_validation_enabled": offline_validation_enabled,
                        "offline_validation_reason": dataset_split["offline_validation_reason"],
                        "train_dataset_windows": dataset_split["train"]["windows"],
                        "validation_dataset_windows": dataset_split["validation"]["windows"],
                    })
                    for split in ("train", "validation"):
                        count = dataset_split[split]["episodes"]
                        if count is not None:
                            step_log[f"{split}_dataset_episodes"] = count
                    if device.type == 'cuda':
                        step_log['gpu_peak_allocated_mb'] = torch.cuda.max_memory_allocated(device) / 2**20
                        step_log['gpu_peak_reserved_mb'] = torch.cuda.max_memory_reserved(device) / 2**20

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = accelerator.unwrap_model(self.ema_model)
                policy.eval()

                # run policy rollout
                if not lazy_eval:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and (self.epoch % cfg.training.rollout_every) == 0:
                        runner_log = env_runner.run(policy)
                        step_log.update(runner_log)
                    accelerator.wait_for_everyone()

                # run validation
                if offline_validation_enabled and (self.epoch % cfg.training.val_every) == 0:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    generated_validation = (
                        cfg.training.get("validate_generated_history", False)
                        and policy_capability(policy, "supports_generated_history_validation")
                    )
                    loss_info = torch.zeros(3, device=device)  # expert loss, count, generated loss
                    with torch.inference_mode(), accelerator.autocast():
                        with tqdm.tqdm(
                            val_dataloader, 
                            desc=f"Validation epoch {self.epoch}",
                            leave=False, 
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:
                            
                            for batch_idx, batch in enumerate(tepoch):
                                # device transfer
                                batch = dict_apply(batch, lambda x: maybe_to_device(x, device, non_blocking=True))

                                validate_history_batch(policy, batch)
                                # Keep expert-history and generated-history metrics separate.
                                if policy_capability(policy, "supports_generated_history_validation"):
                                    loss = policy(batch, history_mode="expert").item()
                                else:
                                    loss = policy(batch).item()

                                batch_size = batch['action'].shape[0]
                                loss_info[0] += loss * batch_size
                                loss_info[1] += batch_size
                                if generated_validation:
                                    generated_loss = policy(batch, history_mode="generated").item()
                                    loss_info[2] += generated_loss * batch_size

                                # break if reach max val steps
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                    
                    # logging
                    accelerator.wait_for_everyone()
                    loss_info = accelerator.reduce(loss_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        if device.type == 'cuda':
                            step_log['validation_peak_allocated_mb'] = torch.cuda.max_memory_allocated(device) / 2**20
                            step_log['validation_peak_reserved_mb'] = torch.cuda.max_memory_reserved(device) / 2**20
                        step_log['val_loss'] = (loss_info[0] / loss_info[1]).item()
                        if policy_capability(policy, "supports_explicit_past_actions"):
                            step_log['val_loss_expert_history'] = step_log['val_loss']
                        if generated_validation:
                            step_log['val_loss_generated_history'] = (loss_info[2] / loss_info[1]).item()

                # action prediction eval
                if offline_validation_enabled and self.epoch % cfg.training.sample_every == 0:
                    loss_info = torch.zeros(2, device=device)   # [total loss, total batch_size]
                    with torch.inference_mode(), accelerator.autocast():
                        with tqdm.tqdm(
                            val_dataloader, 
                            desc=f"Reconstruction epoch {self.epoch}",
                            leave=False, 
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:

                            for batch_idx, batch in enumerate(tepoch):
                                # device transfer
                                batch = dict_apply(batch, lambda x: maybe_to_device(x, device, non_blocking=True))

                                # action prediction
                                gt_action = batch['action']     # [B, Ta, Da]
                                result = self._predict_validation_action(policy, batch)
                                pred_action = result['action_pred']  # [B, Ta, Da]
                                mse = F.mse_loss(pred_action, gt_action).item()

                                # log loss
                                batch_size = batch['action'].shape[0]
                                loss_info[0] += mse * batch_size
                                loss_info[1] += batch_size

                                # early stop if reach max samples
                                if (cfg.training.max_reconst_steps is not None) \
                                    and batch_idx >= (cfg.training.max_reconst_steps-1):
                                    break

                    # logging
                    accelerator.wait_for_everyone()
                    loss_info = accelerator.reduce(loss_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        step_log['test_reconst_mse'] = (loss_info[0] / loss_info[1]).item()

                # Save NEXT-epoch progress, while filenames/metrics label the
                # epoch just completed. Resuming must never repeat that epoch.
                completed_epoch = self._complete_epoch(ema, lr_scheduler)
                self._gather_rng(accelerator)
                checkpoint_due = ((completed_epoch % cfg.training.checkpoint_every) == 0
                                  or self.epoch >= cfg.training.num_epochs)
                snapshot_every = int(cfg.training.get("snapshot_every", 0))
                snapshot_due = snapshot_every > 0 and (completed_epoch % snapshot_every) == 0
                if accelerator.is_main_process and (checkpoint_due or snapshot_due):
                    # unwrap
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.training.use_ema:
                        ema_model_ddp = self.ema_model
                        self.ema_model = accelerator.unwrap_model(self.ema_model)

                    # checkpointing
                    if checkpoint_due and cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if checkpoint_due and cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    if snapshot_due:
                        self.save_checkpoint(tag=f"ep-{completed_epoch:04d}")

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    if checkpoint_due and cfg.checkpoint.get("save_all", False):
                        # Keep each scheduled checkpoint, regardless of its score.
                        checkpoint_path = pathlib.Path(self.output_dir) / "checkpoints" / (
                            cfg.checkpoint.topk.format_str.format(**metric_dict))
                        self.save_checkpoint(path=checkpoint_path)
                    else:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                    # restore
                    self.model = model_ddp
                    if cfg.training.use_ema:
                        self.ema_model = ema_model_ddp

                # end of epoch
                # log of last step is combined with validation and rollout
                if accelerator.is_main_process:
                    accelerator.log(step_log)
                    json_logger.log(step_log)

        # clean up
        if not lazy_eval:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and (not lazy_eval):
                env_runner.close()
            accelerator.wait_for_everyone()
        accelerator.end_training()



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainP2NNewWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()