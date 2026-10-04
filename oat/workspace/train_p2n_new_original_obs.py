"""Independent original-fused training entry; existing workspaces stay unchanged.

The modern loop is retained explicitly because its unconditional normalizer fitting
cannot satisfy artifact-only resume. Successful-update scheduling, optimizer/EMA
state, RNG continuation and execution-aware validation use the modern helpers.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import numbers
import os
import pathlib
import platform
import time

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs, set_seed as accelerate_set_seed
import dill
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import tqdm

from oat.common.checkpoint_util import TopKCheckpointManager
from oat.common.json_logger import JsonLogger
from oat.common.p2n_new_capabilities import policy_capability, resolve_update_schedule, validate_history_batch
from oat.common.pytorch_util import dict_apply, maybe_to_device
from oat.dataset.base_dataset import BaseDataset
from oat.env_runner.base_runner import BaseRunner
from oat.model.common.lr_scheduler import get_scheduler
from oat.model.common.misc import detect_bf16_support
from oat.perception.original_fused_obs_adapter import normalize_original_obs_config
from oat.policy.base_policy import BasePolicy
from oat.workspace.base_workspace import _atomic_torch_save, _copy_to_cpu
from oat.workspace.train_p2n_new import TrainP2NNewWorkspace as _ModernWorkspace, _cap_dataloader


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _encoder_recipe(policy_config):
    """Resolve public/embedded recipes and reject conflicting duplicate settings."""
    public = policy_config.get("original_obs_config")
    embedded = policy_config.get("obs_encoder_config")
    inner = embedded.get("original_obs_config") if embedded is not None else None
    if public is not None and inner is not None:
        if normalize_original_obs_config(_plain(public)) != normalize_original_obs_config(_plain(inner)):
            raise ValueError("Conflicting public and embedded original observation recipes")
    return normalize_original_obs_config(_plain(public if public is not None else inner))


class TrainP2NNewOriginalObsWorkspace(_ModernWorkspace):
    def __init__(self, cfg, output_dir=None, lazy_instantiation=True):
        # Eager restore must also construct from the embedded artifact recipe,
        # never from a stale fresh initializer path in cfg.policy.
        super().__init__(cfg, output_dir=output_dir, lazy_instantiation=True)
        if not lazy_instantiation:
            self.model = hydra.utils.instantiate(self._policy_config_for_run(cfg))
            self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)

    @staticmethod
    def validate_resume_payload(payload, cfg):
        _ModernWorkspace.validate_resume_payload(payload, cfg)
        expected = {"obs_encoder_type": "original_fused", "context_layout": "original_fused_v1",
                    "context_schema_version": 2}
        saved = payload["cfg"].policy
        exported = payload["policy_config"]
        for key, value in expected.items():
            for location, configuration in (("saved policy", saved), ("requested policy", cfg.policy),
                                            ("exported policy", exported), ("metadata", payload["metadata"])):
                if configuration.get(key) != value:
                    raise ValueError(f"Resume encoder/layout mismatch: {location}.{key} must be {value!r}")
        for key in ("expected_action_tokens", "dropout", "use_rms_norm", "use_qk_norm",
                    "activation_checkpointing", "history_gate_hidden_dim", "history_gate_init",
                    "history_gate_mode", "history_dropout"):
            if _plain(saved.get(key)) != _plain(cfg.policy.get(key)):
                raise ValueError(f"Resume architecture/schema mismatch: policy.{key}")
        requested_recipe = _encoder_recipe(cfg.policy)
        for label, configuration in (("saved policy", saved), ("exported policy", exported)):
            recipe = _encoder_recipe(configuration)
            if recipe != requested_recipe:
                different = next(key for key in requested_recipe if recipe.get(key) != requested_recipe[key])
                raise ValueError(f"Resume original observation recipe mismatch: {label}.{different}")
        embedded = exported.get("obs_encoder_config")
        if embedded is None:
            raise ValueError("Resume requires self-contained original observation encoder configuration")
        for key in ("shape_meta", "n_obs_steps", "embed_dim"):
            if _plain(embedded.get(key)) != _plain(cfg.policy.get(key)):
                raise ValueError(f"Resume embedded encoder architecture mismatch: {key}")
        if exported.get("_target_") != cfg.policy.get("_target_"):
            raise ValueError("Resume exported policy target does not match selected architecture")
        def ports(configuration):
            observations = configuration.get("shape_meta", {}).get("obs", {})
            return {kind: [key for key, value in observations.items() if value.get("type") == kind]
                    for kind in ("rgb", "state")}
        expected_ports = ports(cfg.policy)
        for label, configuration in (("saved", saved), ("exported", exported), ("embedded", embedded)):
            if ports(configuration) != expected_ports:
                raise ValueError(f"Resume original observation field order mismatch: {label} shape_meta")
        contract = payload["metadata"].get("observation_encoder_contract")
        if contract is not None:
            for kind in ("rgb", "state"):
                if contract.get(kind + "_ports") != expected_ports[kind]:
                    raise ValueError(f"Resume encoder metadata disagrees with {kind} field order")
            if contract.get("frame_embedding_shape") != [cfg.policy.n_obs_steps, cfg.policy.embed_dim]:
                raise ValueError("Resume encoder metadata disagrees with frame embedding dimensions")
            if normalize_original_obs_config(contract.get("original_obs_config")) != requested_recipe:
                raise ValueError("Resume encoder metadata disagrees with original observation recipe")
        return payload

    def _initialize_normalizers(self, dataset, cfg):
        if cfg.training.resume:
            # All policy, fused-encoder, OAT and EMA normalizers are restored
            # by strict state loading below. Even a temporary refit is forbidden.
            return
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        self._assert_normalizers_frozen()

    def _assert_normalizers_frozen(self):
        for model in (self.model, self.ema_model):
            if model is None:
                continue
            for name, parameter in model.named_parameters():
                if "normalizer" in name and parameter.requires_grad:
                    raise ValueError(f"Dataset and tokenizer normalizers must remain frozen: {name}")

    def _resume_training_checkpoint(self, path):
        payload = super()._resume_training_checkpoint(path)
        self._assert_normalizers_frozen()
        return payload

    @staticmethod
    def _validation_limit(val_dataset, cfg):
        limit = cfg.training.get("validation_max_samples")
        if limit is None:
            return len(val_dataset)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("training.validation_max_samples must be a positive integer or null")
        return min(limit, len(val_dataset))

    @classmethod
    def _make_validation_dataloader(cls, val_dataset, cfg, accelerator):
        # Each held-out example is evaluated exactly once globally; Accelerate's
        # even-batch padding would otherwise duplicate examples on small tests.
        limit = cls._validation_limit(val_dataset, cfg)
        indices = range(accelerator.process_index, limit, accelerator.num_processes)
        kwargs = dict(cfg.val_dataloader)
        kwargs.update(shuffle=False, drop_last=False)
        return DataLoader(Subset(val_dataset, indices), **kwargs)

    def training_report(self, model):
        report = super().training_report(model)
        encoder = model.obs_encoder
        report.update(
            encoder_type="original_fused",
            policy_name=model.get_policy_name(),
            observation_tokens=int(model.n_obs_steps),
            context_tokens=int(model.n_obs_steps + model.past_n + 2 +
                               (model.history_summary_tokens if model.requires_state_history else 0)),
            fused_feature_dim=int(encoder.fused_feature_dim),
            observation_encoder=encoder.export_metadata(),
            environment={"python": platform.python_version(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda,
                         "world_size": int(os.environ.get("WORLD_SIZE", "1"))},
            evaluation={
                "requested_lazy_eval": bool(self.cfg.task.policy.lazy_eval),
                "mode": ("simulator rollout and offline validation" if
                         self.cfg.task.policy.get("env_runner") is not None and not self.cfg.task.policy.lazy_eval
                         else "offline validation; no physical robot rollout"),
                "validation_max_samples": self.cfg.training.get("validation_max_samples"),
                "offline_validation_enabled": bool(self.cfg.training.get("offline_validation_enabled", True)),
            },
            performance_contract={
                "measurements": "Performance is unmeasured; parameter counts do not imply speedup.",
                "cnn_trainable": True,
                "crop_shape": copy.deepcopy(encoder.original_obs_config["crop_shape"]),
                "batch_per_rank": int(self.cfg.dataloader.batch_size),
                "gradient_accumulation": int(self.cfg.training.gradient_accumulate_every),
                "self_past_probability": model.self_past_probability(),
            },
        )
        return report

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
            "workspace_source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "evaluation": {
                "requested_lazy_eval": bool(cfg.task.policy.lazy_eval),
                "physical_robot_rollout": False,
                "validation_max_samples": cfg.training.get("validation_max_samples"),
            },
            "software": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchvision", "robomimic", "transformers", "accelerate", "hydra-core")},
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


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        init_checkpoint = cfg.training.get("init_checkpoint")
        if init_checkpoint:
            raise ValueError("New architectures require fresh initialization or an explicit compatible resume; init_checkpoint is unsupported")

        # Keep Accelerate's environment-based tracker lifecycle consistent with
        # the explicit W&B mode, even if the parent shell was set to offline.
        if cfg.logging.get("mode") == "online":
            os.environ["WANDB_MODE"] = "online"

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
        full_val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        val_dataloader = self._make_validation_dataloader(val_dataset, cfg, accelerator)
        dataset_split = self._offline_validation_metadata(
            dataset, val_dataset, cfg.training, train_dataloader, full_val_dataloader)
        offline_validation_enabled = dataset_split["offline_validation_enabled"]

        # Fresh-only fitting; resume restores all statistics from the artifact.
        self._initialize_normalizers(dataset, cfg)

        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure checkpoint
        if accelerator.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **cfg.checkpoint.topk
            )

        # configure env
        lazy_eval = bool(cfg.task.policy.lazy_eval) or cfg.task.policy.get("env_runner") is None
        if cfg.task.policy.get("env_runner") is None and not cfg.task.policy.lazy_eval:
            accelerator.print(
                f"lazy_eval=false: offline validation enabled={offline_validation_enabled}; "
                "this task has no physical robot rollout runner.")
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
                    if not lazy_eval and accelerator.is_main_process:
                        env_runner.close()
                    accelerator.end_training()
                    return

        # prepare with accelerator
        (
            train_dataloader,
            self.model,
            self.optimizer,
        ) = accelerator.prepare(
            train_dataloader,
            self.model,
            self.optimizer,
        )
        # Report the batches each process actually executes after sharding.
        dataset_split = self._offline_validation_metadata(
            dataset, val_dataset, cfg.training, train_dataloader, full_val_dataloader)
        dataset_split["validation"].update(
            evaluated_windows=self._validation_limit(val_dataset, cfg),
            batches_per_rank=len(val_dataloader),
            max_loss_batches_per_rank=cfg.training.max_val_steps,
            max_reconstruction_batches_per_rank=cfg.training.max_reconst_steps,
        )
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


@hydra.main(version_base=None,
            config_path=str(pathlib.Path(__file__).parent.parent / "config"),
            config_name="experimental/train_p2n_new_original_obs_real_robot")
def main(cfg):
    TrainP2NNewOriginalObsWorkspace(cfg).run()


if __name__ == "__main__":
    main()
