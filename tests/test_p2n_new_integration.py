"""New-only integration contracts; no downloads, real cameras, or GPU training."""
from pathlib import Path
import copy
import importlib.util

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from oat.common.p2n_new_capabilities import (
    policy_capability, predict_validation_action, resolve_update_schedule, validate_history_batch)
from oat.workspace.train_p2n_new import TrainP2NNewWorkspace, _cap_dataloader

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        return compose(config_name=name)


@pytest.mark.parametrize("variant", ["p2n_new", "p2n_state_gate_new"])
@pytest.mark.parametrize("task", ["libero", "real_robot"])
def test_four_configs_preserve_architecture_and_task_contract(variant, task):
    name = "train_" + variant
    if task == "real_robot":
        name = "experimental/" + name + "_real_robot"
    cfg = config(name)
    assert cfg.variant == cfg.policy.variant == variant
    assert cfg.policy.embed_dim == 768 and cfg.policy.n_layers == 16
    assert cfg.policy.n_heads == 12 and cfg.policy.ffn_dim == 2048
    assert cfg.policy.num_visual_queries == 64
    assert cfg.policy.self_past_chunk_size == 4
    assert cfg.val_dataloader.batch_size == 4
    assert cfg.training.lr_warmup_steps is None
    assert cfg.task.policy.dataset.return_history_validity is True
    if task == "real_robot":
        assert cfg.task.policy.env_runner is None
        assert cfg.training.num_demo == 77
        assert cfg.task.policy.dataset.val_ratio == 0.05
        assert "nut_washer_v3_N77" in cfg.task.policy.dataset.zarr_path
        assert "ep-1540_mse-0.000.ckpt" in cfg.policy.tokenizer_checkpoint
        assert "robot0_eef_rot6d" in cfg.shape_meta.obs
    else:
        assert cfg.task.policy.dataset.val_ratio == 0.1
        assert cfg.policy.tokenizer_checkpoint is None
    if variant == "p2n_new":
        assert not any(k.startswith("history_") or k.startswith("state_history") for k in cfg.policy)
    else:
        assert cfg.policy.history_summary_tokens == 4
        assert cfg.policy.history_gate_init == 0.9


def test_capability_declarations_none_legacy_and_explicit_disable():
    class Past2NextPolicy:
        supports_explicit_past_actions = None
        def predict_action(self, obs, past_actions=None):
            return past_actions
        def forward(self, batch):
            return batch
    legacy = Past2NextPolicy()
    assert policy_capability(legacy, "supports_explicit_past_actions")
    assert not policy_capability(legacy, "supports_explicit_past_action_valid")
    legacy.supports_explicit_past_actions = False
    assert not policy_capability(legacy, "supports_explicit_past_actions")
    class New(Past2NextPolicy):
        supports_explicit_past_actions = True
        supports_explicit_past_action_valid = True
        def predict_action(self, obs, past_actions=None, past_action_valid=None):
            return past_actions, past_action_valid
    past = torch.zeros(2, 7, 1)
    valid = torch.ones(2, 7, dtype=torch.bool)
    result = predict_validation_action(New(), {"obs": {}, "past_action": past, "past_action_valid": valid})
    assert result == (past, valid)


def test_update_schedule_counts_tail_and_explicit_cap():
    assert resolve_update_schedule(11, 3, 4) == {
        "batches_per_rank": 11, "updates_per_epoch": 3,
        "planned_optimizer_updates": 9, "lr_warmup_steps": 1}
    assert resolve_update_schedule(11, 3, 4, max_train_steps=5)["planned_optimizer_updates"] == 6
    assert resolve_update_schedule(11, 3, 4, warmup_steps=2)["lr_warmup_steps"] == 2


def test_capped_loader_exposes_actual_end_to_accelerate():
    from accelerate import Accelerator
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=4)
    loader = torch.utils.data.DataLoader(torch.arange(40), batch_size=2)
    loader = accelerator.prepare(_cap_dataloader(loader, 5))
    model = torch.nn.Linear(1, 1)
    sync = []
    for _ in loader:
        with accelerator.accumulate(model):
            sync.append(accelerator.sync_gradients)
    assert sync == [False, False, False, True, True]


def test_gate_validation_compares_seven_aligned_transitions():
    class Policy:
        supports_explicit_past_action_valid = True
        requires_state_history = True
    state = torch.tensor([[False, False, True, True, True, True, True, True]])
    batch = {}
    for prefix, obs in (("", "obs"), ("prev_", "prev_obs")):
        batch[prefix + "past_action"] = torch.zeros(1, 7, 7)
        batch[prefix + "past_action_valid"] = state[:, :-1] & state[:, 1:]
        batch[obs] = {"state_history_valid": state}
    validate_history_batch(Policy(), batch)
    batch["prev_past_action_valid"] = torch.ones(1, 7, dtype=torch.bool)
    with pytest.raises(ValueError, match="aligned measured state"):
        validate_history_batch(Policy(), batch)


class TinyPolicy(torch.nn.Module):
    supports_explicit_past_actions = True
    supports_explicit_past_action_valid = True
    supports_generated_history_validation = True
    requires_state_history = False
    requires_execution_acknowledgement = True
    def __init__(self, variant="p2n_new", construction_mode="fresh"):
        super().__init__()
        self.variant = variant
        self.linear = torch.nn.Linear(1, 1)
        self.register_buffer("_self_past_optimizer_step", torch.zeros((), dtype=torch.int64))
        self.pending = None
    @property
    def self_past_step(self):
        return int(self._self_past_optimizer_step)
    def set_self_past_step(self, value):
        self._self_past_optimizer_step.fill_(value)
    def on_optimizer_step(self):
        self._self_past_optimizer_step.add_(1)
    def forward(self, batch, history_mode=None):
        return self.linear(batch["action"]).square().mean()
    def predict_action(self, obs, past_actions=None, past_action_valid=None):
        assert past_action_valid is not None
        return {"action_pred": torch.zeros(past_actions.shape[0], 16, 1)}
    def set_normalizer(self, normalizer):
        pass
    def get_optimizer(self, **kwargs):
        return torch.optim.AdamW(self.parameters(), lr=kwargs.get("policy_lr", 0.01))
    def reset(self):
        self.pending = None
    def export_config(self):
        return {"_target_": "test_p2n_new_integration.TinyPolicy", "variant": self.variant,
                "construction_mode": "restore"}
    def artifact_metadata(self):
        return {"variant": self.variant, "context_schema_version": 1}


class TinyReplay(dict):
    episode_ends = np.array([5, 10])


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path=None):
        self.train_mask = np.array([True, False])
        self.replay_buffer = TinyReplay(action=np.ones((10, 1), dtype=np.float32))
        self.action_key = "action"
    def __len__(self):
        return 10
    def __getitem__(self, index):
        return {"obs": {"state": torch.ones(2, 1)}, "prev_obs": {"state": torch.ones(2, 1)},
                "action": torch.full((16, 1), 1.0 + index / 10), "past_action": torch.zeros(7, 1),
                "past_action_valid": torch.ones(7, dtype=torch.bool),
                "prev_past_action": torch.zeros(7, 1),
                "prev_past_action_valid": torch.ones(7, dtype=torch.bool),
                "prev_window_valid": torch.tensor(True)}
    def get_validation_dataset(self):
        result = copy.copy(self)
        result.train_mask = ~self.train_mask
        return result
    def get_normalizer(self):
        return None


def tiny_config():
    cfg = config("train_p2n_new")
    cfg.policy = {"_target_": "test_p2n_new_integration.TinyPolicy", "variant": "p2n_new"}
    cfg.task.policy.dataset = {"_target_": "test_p2n_new_integration.TinyDataset", "zarr_path": "/unused"}
    cfg.training.num_epochs = 2
    cfg.training.gradient_accumulate_every = 2
    cfg.training.max_train_steps = 3
    cfg.training.max_val_steps = 1
    cfg.training.max_reconst_steps = 1
    cfg.training.checkpoint_every = 25
    cfg.training.snapshot_every = 0
    cfg.training.allow_bf16 = False
    cfg.logging.mode = "disabled"
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.dataloader.persistent_workers = False
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    return cfg


def test_training_resume_artifact_is_strict_and_self_contained(tmp_path):
    cfg = tiny_config()
    workspace = TrainP2NNewWorkspace(cfg, output_dir=str(tmp_path), lazy_instantiation=False)
    workspace.completed_optimizer_steps = 3
    workspace.epoch = 2
    workspace.global_step = 5
    workspace.model.on_optimizer_step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(workspace.optimizer, lambda _: 1)
    workspace._capture_training_state(None, scheduler)
    checkpoint = workspace.save_checkpoint(use_thread=False)
    restored = TrainP2NNewWorkspace.create_from_checkpoint(checkpoint, output_dir=str(tmp_path / "restored"))
    assert restored.epoch == 2 and restored.completed_optimizer_steps == 3
    assert restored.model.self_past_step == 1
    assert restored.model.pending is None
    assert not restored.model.training
    for a, b in zip(workspace.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(a, b)
    import dill
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    assert payload["policy_config"]["construction_mode"] == "restore"
    assert payload["metadata"]["variant"] == "p2n_new"
    assert len(payload["pickles"]["rng_states"]) > 0
    incompatible = copy.deepcopy(cfg)
    incompatible.variant = "p2n_state_gate_new"
    with pytest.raises(ValueError, match="variant"):
        TrainP2NNewWorkspace.validate_resume_payload(payload, incompatible)


def test_two_cpu_epochs_flush_accumulation_and_save_final_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    cfg = tiny_config()
    workspace = TrainP2NNewWorkspace(cfg, output_dir=str(tmp_path))
    workspace.run()
    assert workspace.epoch == 2
    assert workspace.global_step == 6
    assert workspace.completed_optimizer_steps == 4
    assert workspace.model.self_past_step == 4
    assert workspace.update_schedule["planned_optimizer_updates"] == 4
    import dill
    payload = torch.load(tmp_path / "checkpoints/latest.ckpt", map_location="cpu", pickle_module=dill, weights_only=False)
    assert dill.loads(payload["pickles"]["epoch"]) == 2
    assert dill.loads(payload["pickles"]["completed_optimizer_steps"]) == 4
    assert payload["metadata"]["dataset_split"]["identity"]["train_episode_ids"] == [0]


def test_resume_next_epoch_matches_uninterrupted_training_and_rng(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    cfg = tiny_config()
    cfg.training.snapshot_every = 1
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = TrainP2NNewWorkspace(cfg, output_dir=str(full_dir))
    full.run()
    full_weights = copy.deepcopy(full.model.state_dict())
    full_ema = copy.deepcopy(full.ema_model.state_dict())
    expected_draw = torch.rand(4)

    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg.training.resume = True
    resumed_cfg.training.resume_checkpoint = str(full_dir / "checkpoints/ep-0000.ckpt")
    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    resumed = TrainP2NNewWorkspace(resumed_cfg, output_dir=str(resumed_dir))
    resumed.run()
    assert (resumed.epoch, resumed.global_step, resumed.completed_optimizer_steps) == (2, 6, 4)
    for key, tensor in resumed.model.state_dict().items():
        torch.testing.assert_close(tensor, full_weights[key], rtol=0, atol=0)
    for key, tensor in resumed.ema_model.state_dict().items():
        torch.testing.assert_close(tensor, full_ema[key], rtol=0, atol=0)
    assert resumed.lr_scheduler_state == full.lr_scheduler_state
    assert resumed.ema_state == full.ema_state
    torch.testing.assert_close(torch.rand(4), expected_draw, rtol=0, atol=0)


def test_skipped_optimizer_update_does_not_advance_scheduler_ema_or_curriculum(tmp_path, monkeypatch):
    from accelerate.optimizer import AcceleratedOptimizer
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    original = AcceleratedOptimizer.step
    attempts = []
    def step(self, closure=None):
        attempts.append(1)
        self._is_overflow = len(attempts) == 2
        if self._is_overflow:
            return None
        return original(self, closure=closure)
    monkeypatch.setattr(AcceleratedOptimizer, "step", step)
    cfg = tiny_config()
    workspace = TrainP2NNewWorkspace(cfg, output_dir=str(tmp_path))
    workspace.run()
    assert len(attempts) == 4
    assert workspace.completed_optimizer_steps == 3
    assert workspace.model.self_past_step == 3
    assert workspace.ema_model.self_past_step == 3
    assert workspace.ema_state["optimization_step"] == 3
    assert workspace.lr_scheduler_state["last_epoch"] == 3
