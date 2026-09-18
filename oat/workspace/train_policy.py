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
from oat.policy.past2next import Past2NextPolicy
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy

register_new_resolvers()


class TrainPolicyWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'checkpoint_version',
                    'completed_optimizer_steps', 'ema_state', 'lr_scheduler_state',
                    'resume_migration']

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
        self.checkpoint_version = 2
        self.completed_optimizer_steps = 0
        self.ema_state = None
        self.lr_scheduler_state = None
        self.resume_migration = None

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

        return {
            "schema_version": 1,
            "offline_validation_enabled": enabled,
            "offline_validation_reason": reason,
            "train": {**counts(dataset), "batches_per_epoch": len(train_dataloader)},
            "validation": {**counts(val_dataset), "batches": len(val_dataloader)},
        }

    def _initialize_policy_weights(self, checkpoint, weights="ema", allow_spatial_resize=False):
        """Initialize a new run's policy weights without restoring optimizer or progress."""
        if weights not in ("ema", "model"):
            raise ValueError("training.init_weights must be 'ema' or 'model'")
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill,
                             weights_only=False)
        key = "ema_model" if weights == "ema" else "model"
        if key not in payload["state_dicts"]:
            raise ValueError(f"Initialization checkpoint has no {key} weights")
        state_dict = payload["state_dicts"][key]
        regenerated = []
        if allow_spatial_resize:
            from robomimic.models.base_nets import SpatialSoftmax

            target_state = self.model.state_dict()
            # VisualCore registers the same pool as both `pool` and `nets.1`;
            # state_dict retains both paths, so check every module alias.
            allowed_buffers = {
                f"{name}.{axis}"
                for name, module in self.model.named_modules(remove_duplicate=False)
                if name.startswith("obs_encoder.") and isinstance(module, SpatialSoftmax)
                for axis in ("pos_x", "pos_y")
                if axis in module._buffers and axis not in module._parameters
            }
            mismatches = {
                name: (tuple(value.shape), tuple(target_state[name].shape))
                for name, value in state_dict.items()
                if name in target_state and value.shape != target_state[name].shape
            }
            unsupported = set(mismatches) - allowed_buffers
            if unsupported:
                raise ValueError(
                    "Spatial resize only permits observation SpatialSoftmax pos_x/pos_y "
                    f"buffer mismatches; found {sorted(unsupported)}"
                )
            state_dict = state_dict.copy()
            for name, (source_shape, target_shape) in sorted(mismatches.items()):
                # Keep coordinates constructed for the new feature-map dimensions.
                # No learned tensors, temperature, or normalizers are replaced.
                state_dict[name] = target_state[name].detach().clone()
                regenerated.append(name)
                print(f"Regenerated SpatialSoftmax buffer {name}: "
                      f"{source_shape} -> {target_shape}")
        self.model.load_state_dict(state_dict)
        if hasattr(self.model, "set_self_past_step"):
            self.model.set_self_past_step(0)
        if self.ema_model is not None:
            self.ema_model.load_state_dict(state_dict)
            if hasattr(self.ema_model, "set_self_past_step"):
                self.ema_model.set_self_past_step(0)
        # The source weights include tokenizer and observation normalizers; keep them.
        return regenerated

    def _capture_training_state(self, ema, lr_scheduler):
        self.checkpoint_version = 2
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

    def _resume_training_checkpoint(self, path):
        """Load completed-epoch checkpoints; migrate the historical counter layout.

        Historical GPU checkpoints could retain references to CPU Adam step
        tensors during asynchronous saves. Only counters exceeding a trustworthy
        completed-update count are clamped; smaller per-parameter counts remain
        untouched because parameters may legitimately receive fewer updates.
        """
        payload = torch.load(path, map_location="cpu", pickle_module=dill,
                             weights_only=False)
        saved = {key: dill.loads(value) for key, value in payload["pickles"].items()}
        version = int(saved.get("checkpoint_version", 1))
        if version not in (1, 2):
            raise ValueError(f"Unsupported training checkpoint version {version}")
        self.load_payload(payload)
        migration = {"source": str(path), "source_version": version}
        if version == 1:
            if "epoch" not in saved or "global_step" not in saved:
                raise ValueError("Legacy resume requires saved epoch and global_step counters")
            completed_epochs = int(saved["epoch"]) + 1
            completed_batches = int(saved["global_step"]) + 1
            if completed_epochs < 1 or completed_batches < 1:
                raise ValueError("Legacy checkpoint is not a completed training epoch")
            source_training = payload["cfg"].training
            cap = source_training.get("max_train_steps")
            if cap is not None and completed_batches % completed_epochs == 0:
                # The historical early-break path counted its capped last batch
                # once in the loop and once again at the epoch boundary.
                completed_batches = completed_epochs * min(
                    completed_batches // completed_epochs, int(cap))
            model_state = payload["state_dicts"]["model"]
            history_counter = model_state.get("_self_past_optimizer_step")
            if history_counter is not None:
                updates = int(history_counter.item())
                evidence = "persisted model optimizer-step history counter"
            else:
                accumulation = int(source_training.get("gradient_accumulate_every", 1))
                if accumulation < 1 or completed_batches % completed_epochs:
                    raise ValueError("Cannot conservatively infer legacy optimizer progress")
                batches_per_epoch = completed_batches // completed_epochs
                cap = source_training.get("max_train_steps")
                if cap is not None:
                    # Old capped loops counted one extra global step per epoch.
                    batches_per_epoch = min(batches_per_epoch, int(cap))
                    if accumulation != 1:
                        raise ValueError("Legacy capped gradient accumulation needs an explicit migration")
                updates = completed_epochs * ((batches_per_epoch + accumulation - 1) // accumulation)
                completed_batches = completed_epochs * batches_per_epoch
                evidence = "completed epoch/batch counters and original accumulation setting"
            if updates < 1:
                raise ValueError("Legacy checkpoint contains no completed optimizer updates")
            self.epoch = completed_epochs
            self.global_step = completed_batches
            self.completed_optimizer_steps = updates
            self.ema_state = {"optimization_step": updates}
            self.lr_scheduler_state = None
            repaired = 0
            for state in self.optimizer.state.values():
                step = state.get("step")
                if step is not None and float(step) > updates:
                    if isinstance(step, torch.Tensor):
                        step.fill_(updates)
                    else:
                        state["step"] = updates
                    repaired += 1
            migration.update(update_evidence=evidence, clamped_optimizer_counters=repaired,
                             note="Assumes historical end-of-epoch saves; RNG/worker state was not stored.")
        else:
            if not all(key in saved for key in ("epoch", "global_step", "completed_optimizer_steps", "ema_state", "lr_scheduler_state")):
                raise ValueError("Version 2 checkpoint is missing continuation state")
            if self.lr_scheduler_state is None:
                raise ValueError("Version 2 checkpoint is missing scheduler state")
        self.checkpoint_version = 2
        self.resume_migration = migration
        print(f"Resuming at epoch {self.epoch}, completed batches {self.global_step}, "
              f"optimizer updates {self.completed_optimizer_steps}; {migration}")
        return payload

    @staticmethod
    def _predict_validation_action(policy, batch):
        if isinstance(policy, Past2NextPolicy):
            return policy.predict_action(batch["obs"], past_actions=batch["past_action"])
        return policy.predict_action(batch["obs"])

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        init_checkpoint = cfg.training.get("init_checkpoint")
        if init_checkpoint and cfg.training.resume:
            raise ValueError("Fine-tuning with init_checkpoint requires training.resume=false")

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
        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # configure dataset
        dataset: BaseDataset = hydra.utils.instantiate(
            cfg.task.policy.dataset)
        train_dataloader = self._make_training_dataloader(dataset, cfg.dataloader)
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

        if init_checkpoint:
            self._initialize_policy_weights(
                init_checkpoint, weights=cfg.training.get("init_weights", "ema"),
                allow_spatial_resize=cfg.training.get("init_allow_spatial_resize", False),
            )
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
            if latest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {latest_ckpt_path}")
                self._resume_training_checkpoint(latest_ckpt_path)
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
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len_train_dataloader * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.completed_optimizer_steps-1
        )
        self._restore_training_helpers(ema, lr_scheduler)

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

        # training loop
        with JsonLogger(os.path.join(self.output_dir, 'logs.json'), filter_fn=lambda key, value:
                        isinstance(value, numbers.Number) or key == "offline_validation_reason") as json_logger:
            while self.epoch < cfg.training.num_epochs:

                if accelerator.is_main_process:
                    step_log = dict()

                train_started = time.monotonic()
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
                                lr_scheduler.step()

                                if not accelerator.optimizer_step_was_skipped:
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
                    generated_validation = (
                        cfg.training.get("validate_generated_history", False)
                        and isinstance(policy, Past2NextSelfPastPolicy)
                    )
                    loss_info = torch.zeros(3, device=device)  # expert loss, count, generated loss
                    with torch.inference_mode():
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

                                # Keep expert-history and generated-history metrics separate.
                                if isinstance(policy, Past2NextSelfPastPolicy):
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
                        step_log['val_loss'] = (loss_info[0] / loss_info[1]).item()
                        if isinstance(policy, Past2NextPolicy):
                            step_log['val_loss_expert_history'] = step_log['val_loss']
                        if generated_validation:
                            step_log['val_loss_generated_history'] = (loss_info[2] / loss_info[1]).item()

                # action prediction eval
                if offline_validation_enabled and self.epoch % cfg.training.sample_every == 0:
                    loss_info = torch.zeros(2, device=device)   # [total loss, total batch_size]
                    with torch.inference_mode():
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
                checkpoint_due = (completed_epoch % cfg.training.checkpoint_every) == 0
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
    workspace = TrainPolicyWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()