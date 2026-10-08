"""P2N-VLA training workspace (M4) on CPU, against a stub implementing the M3 policy API.

The stub (``StubVLAPolicy``) follows the contract in docs/P2N_VLA_IMPLEMENTATION.md
("M3: policy API"): forward(batch, history_mode) -> scalar, last_loss_components,
named optimizer groups, clip_groups, the optimizer-step self-past counter, stateless
predict_action(past_actions=, past_action_valid=), trainable_named_parameters and the
artifact methods. ``StubVLADataset`` emits the prev-window batch keys. Other P2N-VLA
test files import both from here.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from oat.workspace.train_p2n_vla import (
    CHECKPOINT_FORMAT, EpochSeededRandomSampler, TrainP2NVLAWorkspace as Workspace, WarmupCosineSchedule,
    _validation_subset, clip_gradients_per_group, make_lr_scheduler, validate_parameter_partition)

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
FROZEN_KEYS = ("backbone.weight", "backbone.bias")


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)


# ----------------------------------------------------------------------------- stub policy
class StubVLAPolicy(nn.Module):
    """Tiny stand-in with the P2N-VLA M3 policy API (no pi05 weights needed)."""

    VARIANT = "p2n_vla"
    VARIANT_CODE = 10
    policy_family = "p2n_vla"
    requires_execution_acknowledgement = True
    supports_explicit_past_actions = True
    supports_explicit_past_action_valid = True
    supports_generated_history_validation = True
    requires_state_history = False
    supports_history_summary_gate = False

    def __init__(self, shape_meta=None, n_action_steps=8, n_obs_steps=1, past_n=7, horizon=16, variant=None,
                 task="libero", construction_mode="fresh", model_size="tiny", pi05_weights=None, pi05_sha256=None,
                 spm_path=None, spm_sha256=None, tokenizer_checkpoint=None, lambda_ki=1.0, hidden=8,
                 dropout=0.1, temperature=0.0, topk=10, self_past_p=0.5, self_past_warmup_steps=1,
                 self_past_ramp_steps=2, self_past_chunk_size=4, self_past_temperature=1.0, self_past_topk=10,
                 self_past_schedule="optimizer_step"):
        super().__init__()
        if variant is not None and variant != self.VARIANT:
            raise ValueError(f"{type(self).__name__} requires variant={self.VARIANT!r}")
        if construction_mode not in ("fresh", "restore"):
            raise ValueError("construction_mode must be fresh or restore")
        self.variant, self.task = self.VARIANT, task
        self.n_action_steps, self.n_obs_steps = int(n_action_steps), int(n_obs_steps)
        self.past_n, self.horizon = int(past_n), int(horizon)
        self.lambda_ki = float(lambda_ki)
        self.temperature, self.topk = temperature, topk
        self.self_past_p = float(self_past_p)
        self.self_past_warmup_steps = int(self_past_warmup_steps)
        self.self_past_ramp_steps = int(self_past_ramp_steps)
        self.shape_meta = _plain(shape_meta) or {
            "obs": {"agentview_rgb": {"shape": [4, 4, 3], "type": "rgb"},
                    "robot0_eef_pos": {"shape": [3], "type": "state"},
                    "task_uid": {"shape": [1], "type": "state"}},
            "action": {"shape": [7]}}
        # "pi05" backbone: frozen and identical in every construction (independent of the global RNG).
        generator = torch.Generator().manual_seed(1234)
        self.backbone = nn.Linear(7, hidden)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.randn(hidden, 7, generator=generator) * 0.3)
            self.backbone.bias.zero_()
        self.backbone.requires_grad_(False)
        self.expert = nn.Linear(hidden, hidden)                       # pretrained group, AR clip
        self.lora_A = nn.Parameter(torch.randn(2, hidden) * 0.1)      # pretrained group, KI clip
        self.lora_B = nn.Parameter(torch.zeros(hidden, 2))            # pretrained group, KI clip
        self.raw_proj = nn.Linear(7, hidden)                          # new group, AR clip
        self.tok_emb = nn.Parameter(torch.randn(16, hidden) * 0.1)    # new group, AR clip
        self.ki_table = nn.Parameter(torch.randn(16, hidden) * 0.1)   # new group, KI clip
        self.head = nn.Linear(hidden, self.horizon * 7)               # new group, AR clip
        self.dropout = nn.Dropout(dropout)
        if self.lambda_ki == 0:
            for parameter in (self.lora_A, self.lora_B, self.ki_table):
                parameter.requires_grad_(False)
        self.action_scale = nn.Parameter(torch.full((7,), float("nan")), requires_grad=False)
        self.register_buffer("_self_past_optimizer_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_normalizer_fitted", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_variant_code", torch.tensor(self.VARIANT_CODE))
        self._construction = dict(
            shape_meta=self.shape_meta, n_action_steps=n_action_steps, n_obs_steps=n_obs_steps, past_n=past_n,
            horizon=horizon, variant=self.VARIANT, task=task, model_size=model_size, pi05_weights=pi05_weights,
            pi05_sha256=pi05_sha256, spm_path=spm_path, spm_sha256=spm_sha256, lambda_ki=lambda_ki,
            hidden=hidden, dropout=dropout, temperature=temperature, topk=topk, self_past_p=self_past_p,
            self_past_warmup_steps=self_past_warmup_steps, self_past_ramp_steps=self_past_ramp_steps,
            self_past_chunk_size=self_past_chunk_size, self_past_temperature=self_past_temperature,
            self_past_topk=self_past_topk, self_past_schedule=self_past_schedule)
        self.last_loss_components = None
        self.validation_calls = []
        self.reset()

    # -- capabilities used by the runner / eval script
    @property
    def device(self):
        return self.expert.weight.device

    @property
    def dtype(self):
        return torch.float32

    def get_observation_ports(self):
        return ["agentview_rgb", "robot0_eef_pos", "task_uid"]

    def get_policy_name(self):
        return f"{self.variant}_stub_{self.task}"

    # -- normalizer and counters
    def set_normalizer(self, normalizer):
        scale = torch.as_tensor(normalizer["action_scale"], dtype=torch.float32)
        with torch.no_grad():
            self.action_scale.copy_(scale)
            self._normalizer_fitted.fill_(True)

    @property
    def self_past_step(self):
        return int(self._self_past_optimizer_step.item())

    def set_self_past_step(self, step):
        if step < 0:
            raise ValueError("Optimizer update count must be nonnegative")
        self._self_past_optimizer_step.fill_(int(step))

    def on_optimizer_step(self):
        if self.training:
            self._self_past_optimizer_step.add_(1)

    def self_past_probability(self):
        progress = self.self_past_step - self.self_past_warmup_steps
        if progress < 0:
            return 0.0
        ramp = min(1.0, progress / self.self_past_ramp_steps) if self.self_past_ramp_steps else 1.0
        return self.self_past_p * ramp

    # -- model
    def _encode(self, past, valid):
        x = torch.where(valid[..., None], past / self.action_scale, torch.zeros_like(past)).mean(1)
        return self.dropout(torch.tanh(self.backbone(x)) + self.raw_proj(x))

    def _decode(self, past, valid):
        hidden = self.expert(self._encode(past, valid)) + self.tok_emb.mean(0)
        return self.head(hidden).view(-1, self.horizon, 7) * self.action_scale

    def forward(self, batch, history_mode=None):
        mode = ("configured" if self.training else "expert") if history_mode is None else history_mode
        if mode not in ("expert", "generated", "configured"):
            raise ValueError("history_mode must be expert, generated or configured")
        if not bool(self._normalizer_fitted):
            raise RuntimeError("set_normalizer must run before forward")
        past, valid = batch["past_action"], batch["past_action_valid"]
        probability = {"expert": 0.0, "generated": 1.0}.get(mode)
        if probability is None:
            probability = self.self_past_probability()
        rows = 0
        if probability > 0:
            # Select rows first, then generate them in eval mode without gradients.
            selected = batch["prev_window_valid"].reshape(-1).bool() & (
                torch.rand(past.shape[0], device=past.device) < probability)
            if selected.any():
                modes = [(module, module.training) for module in self.modules()]
                self.eval()
                try:
                    with torch.no_grad():
                        previous = batch["prev_past_action"][selected]
                        generated = self._decode(previous, batch["prev_past_action_valid"][selected])
                        chunk = torch.cat((previous, generated[:, :self.n_action_steps]), 1)[:, -self.past_n:]
                finally:
                    for module, flag in modes:
                        module.training = flag
                past = past.clone()
                past[selected] = chunk
                valid = valid.clone()
                valid[selected] = True
                rows = int(selected.sum())
        prediction = self._decode(past, valid) / self.action_scale
        target = batch["action"] / self.action_scale
        loss_ar = F.mse_loss(prediction, target)
        loss, loss_ki = loss_ar, None
        if self.lambda_ki > 0:
            features = torch.tanh(self.backbone(target[:, 0]))
            lora = F.linear(F.linear(features, self.lora_A), self.lora_B)
            loss_ki = (lora + self.ki_table.mean(0)).square().mean()
            loss = loss_ar + self.lambda_ki * loss_ki
        self.last_loss_components = {
            "loss": float(loss.detach()), "loss_ar": float(loss_ar.detach()),
            "loss_ki": None if loss_ki is None else float(loss_ki.detach()),
            "ar_token_acc": float((prediction.detach() - target).abs().lt(0.5).float().mean()),
            "ki_token_acc": None, "self_past_p": float(probability), "self_past_rows": rows,
            "gate_mean": None, "gate_min": None, "gate_max": None, "hist_attention_mass": None}
        if not self.training:
            self.validation_calls.append({
                "mode": mode, "training": self.training, "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "expert_weight_sum": float(self.expert.weight.detach().double().sum())})
        return loss

    def reset(self):
        self._past = None
        self._past_valid = None
        self._pending = None

    @torch.no_grad()
    def predict_action(self, obs_dict, use_k_tokens=None, temperature=None, topk=None,
                       past_actions=None, past_action_valid=None):
        if (past_actions is None) != (past_action_valid is None):
            raise ValueError("Explicit past_actions and past_action_valid must be provided together")
        if not bool(self._normalizer_fitted):
            raise RuntimeError("set_normalizer must run before predict_action")
        stateful = past_actions is None
        if stateful:
            if self._pending is not None:
                raise RuntimeError("Execution feedback is pending")
            batch = obs_dict["robot0_eef_pos"].shape[0]
            if self._past is None:
                self._past = torch.zeros(batch, self.past_n, 7, device=self.device)
                self._past_valid = torch.zeros(batch, self.past_n, dtype=torch.bool, device=self.device)
            past_actions, past_action_valid = self._past, self._past_valid
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            prediction = self._decode(past_actions, past_action_valid.bool())
        finally:
            for module, flag in modes:
                module.training = flag
        action = prediction[:, :self.n_action_steps]
        if stateful:
            self._pending = action.shape[1]
        return {"action": action, "action_pred": prediction}

    def record_executed_actions(self, actions, executed_lengths=None):
        if self._pending is None:
            raise RuntimeError("No prediction is awaiting execution feedback")
        commands = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        self._past = torch.cat((self._past, commands), 1)[:, -self.past_n:]
        self._past_valid = torch.cat((self._past_valid, torch.ones(commands.shape[:2], dtype=torch.bool,
                                                                   device=self.device)), 1)[:, -self.past_n:]
        self._pending = None

    def get_history_gate_metrics(self):
        return {}

    # -- optimizer contract
    def trainable_named_parameters(self):
        return [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]

    def get_optimizer(self, policy_lr=5e-5, new_module_lr=1e-4, weight_decay=1e-10, betas=(0.9, 0.95),
                      eps=1e-8, fused=None):
        pretrained, new = [], []
        for name, parameter in self.trainable_named_parameters():
            (pretrained if name.startswith(("expert.", "lora_")) else new).append(parameter)
        if fused is None:
            fused = all(parameter.is_cuda for parameter in pretrained + new)
        groups = [{"name": "pretrained", "params": pretrained, "lr": policy_lr},
                  {"name": "new", "params": new, "lr": new_module_lr}]
        return torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=weight_decay,
                                 betas=tuple(betas), eps=eps, fused=bool(fused))

    def clip_groups(self):
        groups = {"ar": [], "ki": []}
        for name, parameter in self.trainable_named_parameters():
            groups["ki" if name in ("lora_A", "lora_B", "ki_table") else "ar"].append(parameter)
        return groups

    # -- artifacts
    def frozen_base_keys(self):
        return list(FROZEN_KEYS)

    def artifact_state_dict(self, trainable_override=None):
        frozen = set(self.frozen_base_keys())
        state = {key: value for key, value in self.state_dict().items() if key not in frozen}
        if trainable_override:
            trainable = dict(self.trainable_named_parameters())
            unknown = set(trainable_override) - set(trainable)
            if unknown:
                raise KeyError(f"Override names are not trainable parameters: {sorted(unknown)}")
            for key, value in trainable_override.items():
                if value.shape != state[key].shape:
                    raise ValueError(f"Override shape mismatch for {key}")
                state[key] = value
        return state

    def load_artifact_state(self, state):
        expected = set(self.state_dict()) - set(self.frozen_base_keys())
        if set(state) != expected:
            raise ValueError(f"Artifact keys differ: missing={sorted(expected - set(state))} "
                             f"unexpected={sorted(set(state) - expected)}")
        if int(state["_variant_code"]) != self.VARIANT_CODE:
            raise ValueError("Artifact variant code does not match this policy")
        missing, unexpected = super().load_state_dict(dict(state), strict=False)
        if unexpected or set(missing) != set(self.frozen_base_keys()):
            raise ValueError("Artifact load did not account for every key")
        self.reset()

    def export_config(self):
        return dict(_target_=f"{type(self).__module__}.{type(self).__name__}",
                    **copy.deepcopy(self._construction), construction_mode="restore")

    def artifact_metadata(self):
        return {"variant": self.variant, "policy_family": self.policy_family, "variant_code": self.VARIANT_CODE,
                "self_past_optimizer_updates": self.self_past_step,
                "frozen_base_keys": list(self.frozen_base_keys())}

    @classmethod
    def from_checkpoint(cls, path, *, base_weights=None, weights="ema", device="cpu", spm_path=None):
        payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
        key = {"ema": "ema_model", "model": "model"}[weights]
        config = dict(payload["policy_config"])
        if base_weights is not None:
            config["pi05_weights"] = str(base_weights)
        policy = hydra.utils.instantiate(config)
        if not isinstance(policy, cls):
            raise ValueError("Checkpoint target and class disagree")
        policy.load_artifact_state(payload["state_dicts"][key])
        return policy.to(device).eval()


class StubGatePolicy(StubVLAPolicy):
    VARIANT = "p2n_vla_state_gate"
    VARIANT_CODE = 11
    requires_state_history = True
    supports_history_summary_gate = True

    def __init__(self, *args, history_gate_mode="learned", **kwargs):
        super().__init__(*args, **kwargs)
        self.history_gate_mode = history_gate_mode
        self._construction["history_gate_mode"] = history_gate_mode

    def set_history_gate_mode(self, mode):
        if mode not in ("learned", "open", "closed"):
            raise ValueError("history gate mode must be learned, open or closed")
        self.history_gate_mode = mode

    def get_history_gate_metrics(self):
        return {"gate_mean": {"learned": 0.9, "open": 1.0, "closed": 0.0}[self.history_gate_mode]}


# ---------------------------------------------------------------------------- stub dataset
class StubReplay(dict):
    def __init__(self, actions, episode_ends):
        super().__init__(action=actions)
        self.episode_ends = episode_ends


class StubVLADataset(torch.utils.data.Dataset):
    """Synthetic prev-window samples; the last episode is the held-out split."""

    normalizer_calls = 0

    def __init__(self, zarr_path="/unused/stub.zarr", n_episodes=4, episode_length=12, past_n=7, horizon=16,
                 n_exec_steps=8, action_scale=2.0, seed=0):
        rng = np.random.default_rng(seed)
        self.zarr_path = zarr_path
        self.past_n, self.horizon, self.n_exec = int(past_n), int(horizon), int(n_exec_steps)
        self.action_scale = float(action_scale)
        ends = np.cumsum([int(episode_length)] * int(n_episodes))
        self.replay_buffer = StubReplay(rng.standard_normal((int(ends[-1]), 7)).astype(np.float32), ends)
        self.action_key = "action"
        self.train_mask = np.array([True] * (int(n_episodes) - 1) + [False])

    def _frames(self):
        ends = self.replay_buffer.episode_ends
        starts = np.r_[0, ends[:-1]]
        return [(int(start), int(end), frame) for keep, start, end in zip(self.train_mask, starts, ends) if keep
                for frame in range(int(start), int(end))]

    def __len__(self):
        return len(self._frames())

    def __getitem__(self, index):
        start, end, frame = self._frames()[index]
        actions = self.replay_buffer["action"]

        def history(anchor):
            rows = np.arange(anchor - self.past_n, anchor)
            valid = rows >= start
            values = actions[np.clip(rows, start, end - 1)] * valid[:, None]
            return torch.from_numpy(values.astype(np.float32)), torch.from_numpy(valid)

        def observation(at):
            at = max(at, start)
            return {"agentview_rgb": torch.full((1, 4, 4, 3), at % 251, dtype=torch.uint8),
                    "robot0_eef_pos": torch.from_numpy(actions[at:at + 1, :3].copy()),
                    "task_uid": torch.tensor([[30]], dtype=torch.int64)}

        past, valid = history(frame)
        previous, previous_valid = history(frame - self.n_exec)
        rows = np.clip(np.arange(frame, frame + self.horizon), start, end - 1)
        return {"obs": observation(frame), "action": torch.from_numpy(actions[rows].copy()),
                "past_action": past, "past_action_valid": valid,
                "prev_obs": observation(frame - self.n_exec), "prev_past_action": previous,
                "prev_past_action_valid": previous_valid,
                "prev_window_valid": torch.tensor(frame - start >= self.n_exec),
                "episode_step": torch.tensor(frame - start)}

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.train_mask = ~self.train_mask
        return result

    def get_normalizer(self):
        type(self).normalizer_calls += 1
        return {"action_scale": torch.full((7,), self.action_scale)}


def stub_config(**overrides):
    cfg = OmegaConf.create({
        "name": "stub", "variant": "p2n_vla", "policy_family": "p2n_vla", "task_type": "libero",
        "seed": 42, "horizon": 16, "n_action_steps": 8, "n_obs_steps": 1, "past_n": 7,
        "policy": {"_target_": "test_p2n_vla_workspace.StubVLAPolicy", "variant": "p2n_vla", "task": "libero",
                   "construction_mode": "fresh", "model_size": "tiny", "pi05_weights": None, "spm_path": None,
                   "lambda_ki": 1.0, "hidden": 8, "dropout": 0.1, "self_past_p": 0.5,
                   "self_past_warmup_steps": 1, "self_past_ramp_steps": 2},
        "task": {"policy": {"lazy_eval": True, "env_runner": None,
                            "dataset": {"_target_": "test_p2n_vla_workspace.StubVLADataset",
                                        "zarr_path": "/unused/stub.zarr", "episode_length": 12}}},
        "training": {"resume": False, "resume_checkpoint": None, "init_checkpoint": None, "seed": 42,
                     "num_epochs": 2, "max_train_steps": 5, "max_optimizer_steps": 100,
                     "gradient_accumulate_every": 2, "lr_warmup_steps": 2, "min_lr_ratio": 0.1,
                     "max_grad_norm": 1.0, "use_ema": True, "val_every": 1, "max_val_steps": 2,
                     "max_reconst_steps": 1, "validate_generated_history": True,
                     "offline_validation_enabled": True, "offline_validation_reason": "held_out_stub_episode",
                     "checkpoint_every": 1, "snapshot_every": 2, "keep_resume_checkpoint": True, "log_every": 1,
                     "max_consecutive_skipped_updates": 20, "tqdm_interval_sec": 1.0,
                     "probe": {"enabled": False, "optimizer_steps": 2, "self_past_step": None,
                               "worst_case_pass": True, "validation_batches": 1,
                               "max_reserved_gb": 22.5, "min_headroom_gb": 2.0}},
        "ema": {"decay": 0.9, "warmup_power": None},
        "optimizer": {"policy_lr": 0.01, "new_module_lr": 0.02, "weight_decay": 1.0e-10, "betas": [0.9, 0.95],
                      "eps": 1.0e-8, "fused": None},
        "dataloader": {"batch_size": 2, "num_workers": 0, "shuffle": True, "pin_memory": False,
                       "persistent_workers": False, "drop_last": True},
        "val_dataloader": {"batch_size": 2, "num_workers": 0, "shuffle": False, "pin_memory": False,
                           "persistent_workers": False, "drop_last": False},
        "logging": {"project": "p2n_vla_test", "mode": "disabled", "name": "stub", "tags": [], "id": None,
                    "group": None, "resume": False},
    })
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    return cfg


def load(path):
    return torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)


def records(path, event):
    lines = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return [line for line in lines if line.get("event") == event]


@pytest.fixture(autouse=True)
def cpu_accelerate(monkeypatch):
    from accelerate.state import AcceleratorState
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    AcceleratorState._reset_state(True)
    StubVLADataset.normalizer_calls = 0
    yield
    AcceleratorState._reset_state(True)


def run_workspace(cfg, path):
    workspace = Workspace(cfg, output_dir=str(path))
    workspace.run()
    return workspace


# ------------------------------------------------------------------------------- helpers
def test_lr_schedule_warms_up_then_cosine_decays_to_floor():
    schedule = WarmupCosineSchedule(warmup_steps=4, total_steps=12, min_lr_ratio=0.1)
    values = [schedule(step) for step in range(16)]
    assert values[0] == pytest.approx(1 / 5)          # openpi: first update at peak/(W+1), never 0
    assert values[4] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(values[:4], values[1:5]))
    assert all(a > b for a, b in zip(values[4:12], values[5:13]))
    assert values[8] == pytest.approx(0.1 + 0.9 * 0.5)  # cosine midpoint
    assert values[12] == pytest.approx(0.1) and values[15] == pytest.approx(0.1)
    assert WarmupCosineSchedule(0, 10, 0.1)(0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        WarmupCosineSchedule(11, 10, 0.1)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        WarmupCosineSchedule(1, 10, 1.5)
    parameters = [nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))]
    optimizer = torch.optim.SGD([{"params": [parameters[0]], "lr": 1.0}, {"params": [parameters[1]], "lr": 2.0}])
    scheduler = make_lr_scheduler(optimizer, 4, 12, 0.1)
    for step in range(14):
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
            [values[step], 2 * values[step]])
        optimizer.step()
        scheduler.step()


def test_each_clip_group_is_clipped_independently():
    ar = nn.Parameter(torch.zeros(3))
    ar.grad = torch.tensor([30.0, 40.0, 0.0])
    ki = nn.Parameter(torch.zeros(2))
    ki.grad = torch.tensor([0.3, 0.4])
    norms = clip_gradients_per_group({"ar": [ar], "ki": [ki]}, 1.0)
    assert float(norms["ar"]) == pytest.approx(50.0) and float(norms["ki"]) == pytest.approx(0.5)
    assert float(ar.grad.norm()) == pytest.approx(1.0, rel=1e-4)
    torch.testing.assert_close(ki.grad, torch.tensor([0.3, 0.4]))  # a joint clip would have scaled it too
    unclipped = nn.Parameter(torch.zeros(1))
    unclipped.grad = torch.tensor([5.0])
    measured = clip_gradients_per_group({"ar": [unclipped], "ki": []}, None)
    assert float(measured["ar"]) == pytest.approx(5.0) and float(measured["ki"]) == 0.0
    assert float(unclipped.grad) == pytest.approx(5.0)


def test_parameter_partition_enforces_optimizer_and_clip_ownership(monkeypatch):
    policy = StubVLAPolicy()
    optimizer = policy.get_optimizer()
    named, clip = validate_parameter_partition(policy, optimizer)
    assert [name for name, _ in named] == [name for name, p in policy.named_parameters() if p.requires_grad]
    assert {id(p) for p in clip["ki"]} == {id(policy.lora_A), id(policy.lora_B), id(policy.ki_table)}
    partial = torch.optim.AdamW([{"name": "pretrained", "params": [policy.expert.weight]}])
    with pytest.raises(ValueError, match="exactly once"):
        validate_parameter_partition(policy, partial)
    unnamed = torch.optim.AdamW([p for _, p in named])
    with pytest.raises(ValueError, match="named"):
        validate_parameter_partition(policy, unnamed)
    monkeypatch.setattr(policy, "clip_groups", lambda: {"ar": [p for _, p in named], "ki": [policy.ki_table]})
    with pytest.raises(ValueError, match="disjoint"):
        validate_parameter_partition(policy, optimizer)
    monkeypatch.setattr(policy, "clip_groups", lambda: {"ar": [policy.expert.weight], "ki": []})
    with pytest.raises(ValueError, match="cover"):
        validate_parameter_partition(policy, optimizer)
    frozen_ki = StubVLAPolicy(lambda_ki=0.0)
    _, clip = validate_parameter_partition(frozen_ki, frozen_ki.get_optimizer())
    assert clip["ki"] == []  # lambda_ki=0 freezes LoRA/KI: an empty KI clip group is legal


def test_epoch_seeded_sampler_is_rank_independent_and_epoch_dependent():
    first, second = EpochSeededRandomSampler(range(50), 7), EpochSeededRandomSampler(range(50), 7)
    torch.manual_seed(0)
    a = list(first)
    torch.manual_seed(1)  # global RNG state is irrelevant
    assert list(second) == a and sorted(a) == list(range(50))
    first.set_epoch(1)
    assert list(first) != a
    second.set_epoch(1)
    assert list(second) == list(first)


def test_validation_subset_spans_the_whole_held_out_split():
    subset = _validation_subset(list(range(100)), 2, 4, 2)
    assert len(subset) == 16 and subset.indices[0] == 0 and subset.indices[-1] == 99
    assert _validation_subset(list(range(10)), 2, 4, 2) == list(range(10))
    assert _validation_subset(list(range(10)), None, 4, 2) == list(range(10))


# -------------------------------------------------------------------------- end-to-end
def test_cpu_run_writes_checkpoint_snapshots_and_logs(tmp_path):
    workspace = run_workspace(stub_config(), tmp_path)
    # 5 micro-batches per epoch at accumulation 2 -> groups (2, 2, 1) -> 3 updates per epoch.
    assert (workspace.epoch, workspace.global_step, workspace.completed_optimizer_steps) == (2, 10, 6)
    assert workspace.model.self_past_step == 6 and workspace.ema.updates == 6
    assert workspace.lr_scheduler.last_epoch == 6
    assert workspace.update_schedule["updates_per_epoch"] == 3
    assert workspace.update_schedule["planned_optimizer_updates"] == 6
    assert workspace.update_schedule["effective_batch"] == 4
    assert StubVLADataset.normalizer_calls == 1

    payload = load(tmp_path / "checkpoints/latest.ckpt")
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["policy_config"]["construction_mode"] == "restore"
    assert payload["metadata"]["variant"] == "p2n_vla"
    assert payload["metadata"]["training_run"]["kind"] == "resume"
    assert set(payload["state_dicts"]) == {"model", "ema_model"}
    trainable = {name for name, _ in workspace.model.trainable_named_parameters()}
    for name in ("model", "ema_model"):
        state = payload["state_dicts"][name]
        assert not set(FROZEN_KEYS) & set(state)
        assert set(state) == set(workspace.model.state_dict()) - set(FROZEN_KEYS)
        assert all(state[key].dtype == torch.float32 for key in trainable)
    for key in trainable:
        torch.testing.assert_close(payload["state_dicts"]["ema_model"][key], workspace.ema.shadow[key],
                                   rtol=0, atol=0)
        torch.testing.assert_close(payload["state_dicts"]["model"][key],
                                   dict(workspace.model.named_parameters())[key].detach(), rtol=0, atol=0)
    assert int(payload["state_dicts"]["model"]["_self_past_optimizer_step"]) == 6
    torch.testing.assert_close(payload["state_dicts"]["model"]["action_scale"], torch.full((7,), 2.0))
    training = payload["training"]
    assert training["counters"]["completed_optimizer_steps"] == 6 and training["counters"]["epoch"] == 2
    assert training["ema"]["updates"] == 6 and training["world_size"] == 1 and len(training["rng_states"]) == 1
    assert training["lr_scheduler"]["last_epoch"] == 6

    names = sorted(path.name for path in (tmp_path / "snapshots").iterdir())
    assert names == ["upd-000002_ema.ckpt", "upd-000004_ema.ckpt", "upd-000006_ema.ckpt"]
    snapshot = load(tmp_path / "snapshots/upd-000006_ema.ckpt")
    assert "training" not in snapshot and set(snapshot["state_dicts"]) == {"ema_model"}
    assert snapshot["metadata"]["training_run"]["trainable_dtype"] == "bfloat16"
    state = snapshot["state_dicts"]["ema_model"]
    for key in trainable:
        assert state[key].dtype == torch.bfloat16
        torch.testing.assert_close(state[key], workspace.ema.shadow[key].to(torch.bfloat16), rtol=0, atol=0)
    assert state["action_scale"].dtype == torch.float32 and state["_self_past_optimizer_step"].dtype == torch.long

    steps = records(tmp_path / "logs.jsonl", "train_step")
    assert [record["optimizer_step"] for record in steps] == [1, 2, 3, 4, 5, 6]
    schedule = WarmupCosineSchedule(2, 100, 0.1)
    for record in steps:
        for key in ("grad_norm_ar", "grad_norm_ki", "samples_per_sec", "train/loss", "train/loss_ar",
                    "train/loss_ki", "train/ar_token_acc", "train/self_past_p", "lr/pretrained", "lr/new"):
            assert record[key] is not None, key
        # Logged after scheduler.step(): the rate the next update will use.
        assert record["lr/pretrained"] == pytest.approx(0.01 * schedule(record["optimizer_step"]))
        assert record["lr/new"] == pytest.approx(0.02 * schedule(record["optimizer_step"]))
    epochs = records(tmp_path / "logs.jsonl", "epoch")
    assert [record["epoch"] for record in epochs] == [0, 1]
    for record in epochs:
        assert record["val_loss"] is not None and record["val_loss_generated_history"] is not None
        assert record["val_reconstruction_mse"] is not None and record["val/generated/self_past_p"] == 1.0
        assert record["val/expert/self_past_p"] == 0.0 and record["train_loss"] is not None
    for name in ("training_report.json", "dataset_split.json", "training_summary.json", "resolved_config.yaml"):
        assert (tmp_path / name).is_file(), name
    report = json.loads((tmp_path / "training_report.json").read_text())
    assert [group["name"] for group in report["optimizer_groups"]] == ["pretrained", "new"]
    assert report["ddp"]["wrapper"] == "StubVLAPolicy" and report["ddp"]["mixed_precision"] == "no"


def test_validation_uses_ema_under_no_grad_and_restores_live_state(tmp_path):
    workspace = run_workspace(stub_config(**{"training.num_epochs": 1}), tmp_path)
    policy = workspace.model
    calls = policy.validation_calls
    assert {call["mode"] for call in calls} == {"expert", "generated"}
    assert not any(call["training"] or call["grad_enabled"] or call["inference_mode"] for call in calls)
    ema_sum = float(workspace.ema.shadow["expert.weight"].double().sum())
    live_sum = float(policy.expert.weight.detach().double().sum())
    assert calls[-1]["expert_weight_sum"] == pytest.approx(ema_sum, abs=1e-12)
    assert abs(live_sum - ema_sum) > 1e-6  # live weights were restored after validation
    assert policy.training and policy.dropout.training  # modes restored after swap_in
    assert policy._pending is None


def test_tail_group_and_skipped_update_never_advance_step_counters(tmp_path, monkeypatch):
    from accelerate.optimizer import AcceleratedOptimizer
    original = AcceleratedOptimizer.step
    attempts = []

    def step(self, closure=None):
        attempts.append(1)
        self._is_overflow = len(attempts) == 2
        if self._is_overflow:
            return None
        return original(self, closure=closure)

    monkeypatch.setattr(AcceleratedOptimizer, "step", step)
    workspace = run_workspace(stub_config(**{"training.num_epochs": 1, "training.snapshot_every": 0}), tmp_path)
    assert len(attempts) == 3 and workspace.global_step == 5
    assert workspace.completed_optimizer_steps == 2 and workspace.skipped_optimizer_steps == 1
    assert workspace.model.self_past_step == 2 and workspace.ema.updates == 2
    assert workspace.lr_scheduler.last_epoch == 2
    assert len(records(tmp_path / "logs.jsonl", "skipped_update")) == 1
    assert sorted(path.name for path in (tmp_path / "snapshots").iterdir()) == ["upd-000002_ema.ckpt"]


def test_resume_restores_optimizer_scheduler_ema_rng_and_counters_exactly(tmp_path):
    full = run_workspace(stub_config(), tmp_path / "full")
    expected_draw = torch.rand(5)
    expected_model = full.model.artifact_state_dict()
    expected_shadow = {key: value.clone() for key, value in full.ema.shadow.items()}
    expected_optimizer = copy.deepcopy(full.optimizer.state_dict())
    expected_scheduler = full.lr_scheduler.state_dict()

    run_workspace(stub_config(**{"training.num_epochs": 1}), tmp_path / "first")
    checkpoint = tmp_path / "first/checkpoints/latest.ckpt"
    assert load(checkpoint)["training"]["counters"]["epoch"] == 1
    StubVLADataset.normalizer_calls = 0
    resumed = run_workspace(stub_config(**{"training.resume": True, "training.resume_checkpoint": str(checkpoint)}),
                            tmp_path / "resumed")
    assert StubVLADataset.normalizer_calls == 0  # normalizers come from the artifact, never refit
    assert (resumed.epoch, resumed.global_step, resumed.completed_optimizer_steps) == (2, 10, 6)
    assert resumed.model.self_past_step == 6 and resumed.ema.updates == 6
    for key, value in resumed.model.artifact_state_dict().items():
        torch.testing.assert_close(value, expected_model[key], rtol=0, atol=0)
    for key, value in resumed.ema.shadow.items():
        torch.testing.assert_close(value, expected_shadow[key], rtol=0, atol=0)
    actual_optimizer = resumed.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for index, state in actual_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, expected_optimizer["state"][index][name], rtol=0, atol=0)
    assert resumed.lr_scheduler.state_dict() == expected_scheduler
    torch.testing.assert_close(torch.rand(5), expected_draw, rtol=0, atol=0)
    # The resumed run's logged trajectory matches the uninterrupted one for the second epoch.
    full_steps = records(tmp_path / "full/logs.jsonl", "train_step")[3:]
    resumed_steps = records(tmp_path / "resumed/logs.jsonl", "train_step")
    assert [r["train/loss"] for r in resumed_steps] == [r["train/loss"] for r in full_steps]


def test_resume_rejects_incompatible_settings(tmp_path):
    run_workspace(stub_config(**{"training.num_epochs": 1}), tmp_path / "first")
    checkpoint = tmp_path / "first/checkpoints/latest.ckpt"
    payload = load(checkpoint)
    base = stub_config(**{"training.resume": True, "training.resume_checkpoint": str(checkpoint)})
    Workspace.validate_resume_payload(payload, base, world_size=1)
    with pytest.raises(ValueError, match="world size"):
        Workspace.validate_resume_payload(payload, base, world_size=2)
    cases = {"variant": ("p2n_vla_state_gate", "variant"),
             "task.policy.dataset.episode_length": (13, "dataset.episode_length"),
             "policy.hidden": (16, "policy.hidden"), "ema.decay": (0.99, "ema"),
             "dataloader.batch_size": (4, "micro-batch"), "training.max_optimizer_steps": (50, "max_optimizer_steps"),
             "training.use_ema": (False, "EMA")}
    for key, (value, message) in cases.items():
        changed = copy.deepcopy(base)
        OmegaConf.update(changed, key, value, merge=False)
        with pytest.raises(ValueError, match=message):
            Workspace.validate_resume_payload(payload, changed, world_size=1)
    moved = copy.deepcopy(base)
    moved.policy.pi05_weights = "/elsewhere/model.safetensors"  # asset paths may move
    Workspace.validate_resume_payload(payload, moved, world_size=1)
    snapshot = load(sorted((tmp_path / "first/snapshots").iterdir())[-1])
    with pytest.raises(ValueError, match="cannot resume"):
        Workspace.validate_resume_payload(snapshot, base, world_size=1)


def test_validation_settings_never_change_the_training_trajectory(tmp_path):
    """Validation draws (self-past sampling in 'generated' mode) run on a forked RNG."""
    trajectories = []
    for index, (val_steps, reconst_steps) in enumerate(((1, 0), (2, 1))):
        cfg = stub_config(**{"training.max_val_steps": val_steps, "training.max_reconst_steps": reconst_steps,
                             "training.snapshot_every": 0})
        run_workspace(cfg, tmp_path / str(index))
        validations = records(tmp_path / str(index) / "logs.jsonl", "epoch")
        assert all(record["val_loss_generated_history"] is not None for record in validations)
        trajectories.append([record["train/loss"] for record in records(tmp_path / str(index) / "logs.jsonl",
                                                                         "train_step")])
    assert trajectories[0] == trajectories[1] and len(trajectories[0]) == 6


def test_no_ema_validation_restores_modes_and_model_snapshots_resume(tmp_path):
    """Regression: without an EMA, validation left every module in eval mode after the epoch."""
    cfg = stub_config(**{"training.use_ema": False, "training.num_epochs": 1, "training.snapshot_every": 2})
    workspace = run_workspace(cfg, tmp_path / "first")
    policy = workspace.model
    assert workspace.ema is None and {call["mode"] for call in policy.validation_calls} == {"expert", "generated"}
    assert not any(call["training"] or call["grad_enabled"] for call in policy.validation_calls)
    # The final validation is the last thing run(): live modes must be back.
    assert all(module.training for module in policy.modules())
    names = sorted(path.name for path in (tmp_path / "first/snapshots").iterdir())
    assert names == ["upd-000002_model.ckpt", "upd-000003_model.ckpt"]
    snapshot = load(tmp_path / "first/snapshots/upd-000003_model.ckpt")
    assert set(snapshot["state_dicts"]) == {"model"} and "training" not in snapshot
    assert snapshot["state_dicts"]["model"]["expert.weight"].dtype == torch.bfloat16
    payload = load(tmp_path / "first/checkpoints/latest.ckpt")
    assert payload["training"]["ema"] is None and set(payload["state_dicts"]) == {"model"}
    resumed = run_workspace(stub_config(**{"training.use_ema": False, "training.resume": True,
                                           "training.resume_checkpoint": str(tmp_path / "first/checkpoints/latest.ckpt")}),
                            tmp_path / "resumed")
    assert (resumed.epoch, resumed.completed_optimizer_steps) == (2, 6) and resumed.model.self_past_step == 6
    with pytest.raises(ValueError, match="EMA"):
        Workspace.validate_resume_payload(payload, stub_config(**{"training.resume": True}), world_size=1)


def test_resume_refuses_reordered_trainable_parameters(tmp_path):
    """Optimizer state is keyed by position: a reordered trainable set must not load silently."""
    run_workspace(stub_config(**{"training.num_epochs": 1}), tmp_path)
    training = load(tmp_path / "checkpoints/latest.ckpt")["training"]
    policy = StubVLAPolicy()
    names = [name for name, _ in policy.trainable_named_parameters()]
    assert training["trainable_names"] == names
    optimizer = policy.get_optimizer()
    Workspace.load_optimizer_state(optimizer, training, names)
    first = policy.trainable_named_parameters()[0][1]
    assert optimizer.state[first]["exp_avg"].shape == first.shape
    with pytest.raises(ValueError, match="same_set_reordered=True"):
        Workspace.load_optimizer_state(policy.get_optimizer(), {**training, "trainable_names": names[::-1]}, names)
    with pytest.raises(ValueError, match="missing=\\['head.bias'\\]"):
        Workspace.load_optimizer_state(policy.get_optimizer(), training, names[:-1])
    # A checkpoint without recorded names still cannot attach moments to wrongly shaped parameters.
    legacy = {key: value for key, value in training.items() if key != "trainable_names"}
    legacy["optimizer"] = copy.deepcopy(legacy["optimizer"])
    state = legacy["optimizer"]["state"]
    state[0], state[1] = state[1], state[0]  # expert.weight [8, 8] <-> expert.bias [8]
    with pytest.raises(ValueError, match="shape"):
        Workspace.load_optimizer_state(policy.get_optimizer(), legacy, names)


def test_max_optimizer_steps_hard_stop_and_final_snapshot(tmp_path):
    cfg = stub_config(**{"training.num_epochs": 3, "training.max_optimizer_steps": 4, "training.lr_warmup_steps": 1,
                         "training.snapshot_every": 0, "training.keep_resume_checkpoint": False})
    workspace = run_workspace(cfg, tmp_path)
    assert workspace.completed_optimizer_steps == 4 and workspace.epoch == 2
    assert workspace.global_step == 7  # epoch 1 stopped after its first update (two micro-batches)
    assert sorted(path.name for path in (tmp_path / "snapshots").iterdir()) == ["upd-000004_ema.ckpt"]
    assert not (tmp_path / "checkpoints/latest.ckpt").exists()
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["stopped_by_max_optimizer_steps"] and not summary["resume_checkpoint_kept"]
    assert records(tmp_path / "logs.jsonl", "epoch")[-1]["val_loss"] is not None
    assert workspace.lr_scheduler.get_last_lr()[0] == pytest.approx(0.01 * 0.1)  # cosine floor reached


def test_probe_reports_throughput_memory_and_writes_no_checkpoints(tmp_path):
    cfg = stub_config(**{"training.probe.enabled": True, "training.probe.optimizer_steps": 2})
    workspace = run_workspace(cfg, tmp_path)
    report = json.loads((tmp_path / "probe.json").read_text())
    assert report["self_past_step"] == 3 and report["self_past_p"] == pytest.approx(0.5)
    assert report["optimizer_steps"] == 2 and len(report["seconds_per_update"]) == 2
    assert report["samples_per_sec"] > 0 and report["world_size"] == 1
    assert report["worst_case_pass"]["components"]["self_past_p"] == 1.0
    assert report["validation"]["val_loss"] is not None
    assert workspace.completed_optimizer_steps == 2
    assert not (tmp_path / "checkpoints").exists() and not (tmp_path / "snapshots").exists()


def test_run_config_rejects_unsupported_settings(tmp_path):
    for key, value, message in (("training.init_checkpoint", "/x.ckpt", "init_checkpoint"),
                                ("task.policy.lazy_eval", False, "rollout_every"),
                                ("task.policy.lazy_eval", "yes", "lazy_eval must be a boolean"),
                                ("optimizer.obs_enc_lr", 1e-4, "Unknown optimizer"),
                                ("logging.mode", "verbose", "logging.mode"),
                                ("training.gradient_accumulate_every", 0, "gradient_accumulate_every"),
                                ("variant", "p2n_new", "variant")):
        cfg = stub_config()
        OmegaConf.update(cfg, key, value, merge=False)
        with pytest.raises(ValueError, match=message):
            Workspace(cfg, output_dir=str(tmp_path)).run()


# ------------------------------------------------------------------- in-training rollouts
class StubRolloutRunner:
    """Duck-typed LIBERO runner: drives the policy like ``LiberoRunner.run`` (reset, ``predict_action`` under
    ``torch.inference_mode``, then ``record_executed_actions``) and writes deterministic episode records."""

    events = []

    def __init__(self, output_dir, task_name="libero10", n_test=4, n_test_vis=0, test_start_seed=3000, n_obs_steps=1,
                 n_action_steps=8, n_parallel_envs=2, protocol="official", init_state_offset=0, max_episode_steps=16,
                 episode_records_path=None, **kwargs):
        self.output_dir, self.n_test, self.protocol = output_dir, int(n_test), protocol
        self.test_start_seed, self.init_state_offset = int(test_start_seed), int(init_state_offset)
        self.n_parallel_envs, self.episode_records_path = int(n_parallel_envs), episode_records_path
        self.last_episode_records = []
        StubRolloutRunner.events.append({"event": "init", "pid": os.getpid(), "n_test": self.n_test,
                                         "n_action_steps": n_action_steps, "n_obs_steps": n_obs_steps,
                                         "protocol": protocol, "output_dir": str(output_dir)})

    def run(self, policy, **kwargs):
        policy.reset()
        batch = min(self.n_parallel_envs, self.n_test)
        if callable(getattr(policy, "create_dummy_observation", None)):  # the real P2N-VLA policies
            dummy = policy.create_dummy_observation(batch_size=batch, device=policy.device)
            obs = {port: dummy[port] for port in policy.get_observation_ports()}
        else:
            obs = {"agentview_rgb": torch.zeros(batch, 1, 3, 8, 8, dtype=torch.uint8, device=policy.device),
                   "robot0_eef_pos": torch.zeros(batch, 1, 3, device=policy.device),
                   "task_uid": torch.full((batch, 1, 1), 30.0, device=policy.device)}
        for _ in range(2):  # two one-chunk episodes: the static obs stays consistent with an empty history
            policy.reset()
            with torch.inference_mode():
                action = policy.predict_action(obs)["action"]
            policy.record_executed_actions(action.cpu().numpy(), executed_lengths=np.full(batch, action.shape[1]))
        weight = getattr(policy, "expert", None)
        weight = weight.weight if weight is not None else next(p for p in policy.parameters() if p.requires_grad)
        StubRolloutRunner.events.append({"event": "run", "training": policy.training,
                                         "any_training": any(module.training for module in policy.modules()),
                                         "grad_enabled": torch.is_grad_enabled(),
                                         "weight_sum": float(weight.detach().double().sum())})
        self.last_episode_records = [
            {"episode_index": i, "task_name": f"TASK_{i % 2}", "task_trial": i // 2,
             "episode_seed": self.test_start_seed + i, "init_state_id": self.init_state_offset + i // 2,
             "protocol": self.protocol, "success": i % 3 == 0, "policy_steps": 10 + i} for i in range(self.n_test)]
        if self.episode_records_path:
            Path(self.episode_records_path).write_text(
                "".join(json.dumps(record) + "\n" for record in self.last_episode_records))
        return {"mean_success_rate": float(np.mean([r["success"] for r in self.last_episode_records]))}

    def close(self):
        StubRolloutRunner.events.append({"event": "close"})


def rollout_config(**overrides):
    values = {"task.policy.lazy_eval": False, "training.rollout_every": 2, "training.rollout_at_final": True,
              "training.rollout_seed": 44, "training.num_epochs": 5, "training.max_train_steps": 2,
              "training.snapshot_every": 0,
              "task.policy.env_runner": {"_target_": "test_p2n_vla_workspace.StubRolloutRunner",
                                         "task_name": "libero10", "protocol": "official", "n_test": 6,
                                         "n_test_vis": 0, "test_start_seed": 3000, "init_state_offset": 0,
                                         "n_parallel_envs": 2, "max_episode_steps": 16}}
    values.update(overrides)
    return stub_config(**values)


def test_rollout_due_after_every_n_completed_epochs_and_at_the_end(tmp_path):
    workspace = Workspace(rollout_config(**{"training.rollout_every": 50, "training.num_epochs": 300}),
                          output_dir=str(tmp_path))
    due = [epoch for epoch in range(300) if workspace.rollout_due(epoch, final=epoch == 299)]
    assert due == [49, 99, 149, 199, 249, 299]  # 1-indexed epochs 50, 100, ..., 300
    assert workspace.rollout_due(122, final=True)  # a run cut short still ends with a rollout
    workspace.cfg.training.rollout_at_final = False
    assert not workspace.rollout_due(122, final=True)


def test_in_training_rollouts_cadence_artifacts_and_ema_weights(tmp_path):
    StubRolloutRunner.events.clear()
    workspace = run_workspace(rollout_config(), tmp_path)
    runs = [event for event in StubRolloutRunner.events if event["event"] == "run"]
    inits = [event for event in StubRolloutRunner.events if event["event"] == "init"]
    assert len(runs) == len(inits) == 3  # after epochs 2 and 4 (1-indexed) and the final epoch 5
    assert sum(event["event"] == "close" for event in StubRolloutRunner.events) == 3  # simulators freed each time
    assert all(event["n_test"] == 6 and event["protocol"] == "official" and event["n_action_steps"] == 8
               for event in inits)
    # Eval mode, no grad, EMA weights during the rollout; live weights and train mode afterwards.
    assert not any(event["training"] or event["any_training"] or event["grad_enabled"] for event in runs)
    policy = workspace.model
    assert all(module.training for module in policy.modules())
    # The rollout scores the snapshot's weights: the EMA rounded to bf16 (the live weights are restored).
    shadow = workspace.ema.shadow["expert.weight"]
    assert runs[-1]["weight_sum"] == pytest.approx(float(shadow.to(torch.bfloat16).double().sum()), abs=1e-12)
    assert abs(runs[-1]["weight_sum"] - float(policy.expert.weight.detach().double().sum())) > 1e-6
    rollouts = records(tmp_path / "logs.jsonl", "rollout")
    assert [record["epoch"] for record in rollouts] == [1, 3, 4]  # 0-indexed, as in the epoch records
    epochs = records(tmp_path / "logs.jsonl", "epoch")
    assert not any("rollout/success_rate" in record for record in epochs)  # logged before the rollout runs
    # Each epoch record is logged before the rollout that follows it.
    lines = [json.loads(line) for line in (tmp_path / "logs.jsonl").read_text().splitlines()]
    assert [line["event"] for line in lines if line["event"] in ("epoch", "rollout")] == [
        "epoch", "epoch", "rollout", "epoch", "epoch", "rollout", "epoch", "rollout"]
    record = rollouts[0]
    assert record["rollout/success_rate"] == pytest.approx(2 / 6) and record["rollout/trials"] == 6
    assert record["mean_success_rate"] == record["rollout/success_rate"]
    assert record["rollout/task/TASK_0"] == pytest.approx(1 / 3) and record["rollout/task/TASK_1"] == pytest.approx(1 / 3)
    directory = tmp_path / "eval" / f"rollout_epoch-0002_upd-{record['optimizer_step']:06d}"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["successes"] == 2 and summary["trials"] == 6 and summary["epoch"] == 2
    assert summary["protocol"] == "official" and summary["episode_start_seed"] == 3000 and summary["weights"] == "ema"
    assert summary["precision"].startswith("bf16")
    assert summary["renderer"] is None  # CPU run: no EGL renderer to resolve
    assert len((directory / "episodes.jsonl").read_text().splitlines()) == 6
    final = json.loads((tmp_path / "training_summary.json").read_text())
    assert [entry["epoch"] for entry in final["rollouts"]] == [2, 4, 5]
    assert final["last_rollout"]["epoch"] == 5
    report = json.loads((tmp_path / "training_report.json").read_text())
    assert report["rollout"]["enabled"] and report["rollout"]["every_epochs"] == 2 and report["rollout"]["n_test"] == 6


def test_rollouts_never_change_the_training_trajectory(tmp_path):
    """Rollouts run on a forked RNG with python/numpy states restored, so training is bit-identical."""
    trajectories, weights = [], []
    for index, overrides in enumerate(({"task.policy.lazy_eval": True}, {})):
        workspace = run_workspace(rollout_config(**overrides), tmp_path / str(index))
        trajectories.append([record["train/loss"]
                             for record in records(tmp_path / str(index) / "logs.jsonl", "train_step")])
        weights.append(workspace.model.expert.weight.detach().clone())
    assert trajectories[0] == trajectories[1] and len(trajectories[0]) == 5
    assert torch.equal(weights[0], weights[1])
    assert not (tmp_path / "0" / "eval").exists() and len(list((tmp_path / "1" / "eval").iterdir())) == 3


def test_no_ema_rollouts_use_the_live_weights_and_restore_modes(tmp_path):
    StubRolloutRunner.events.clear()
    workspace = run_workspace(rollout_config(**{"training.use_ema": False, "training.num_epochs": 2}), tmp_path)
    runs = [event for event in StubRolloutRunner.events if event["event"] == "run"]
    assert len(runs) == 1 and not runs[0]["any_training"]
    live = workspace.model.expert.weight.detach()
    assert runs[0]["weight_sum"] == pytest.approx(float(live.to(torch.bfloat16).double().sum()), abs=1e-12)
    assert all(module.training for module in workspace.model.modules())
    summary = json.loads(next((tmp_path / "eval").iterdir()).joinpath("summary.json").read_text())
    assert summary["weights"] == "model"


def test_probe_rolls_out_the_first_episodes_on_rank0(tmp_path):
    StubRolloutRunner.events.clear()
    cfg = rollout_config(**{"training.probe.enabled": True, "training.probe.rollout_episodes": 4,
                            "training.max_train_steps": 4})  # the probe times 2 updates at accumulation 2
    workspace = run_workspace(cfg, tmp_path)
    report = workspace.probe_report
    assert report["rollout"]["rollout/trials"] == 4 and report["rollout"]["rollout/success_rate"] == pytest.approx(0.5)
    assert report["go_no_go"]["rollout_headroom_gb"] is None  # nvidia-smi is not sampled on CPU
    assert (tmp_path / "eval" / "rollout_probe" / "summary.json").is_file()
    assert [event["n_test"] for event in StubRolloutRunner.events if event["event"] == "init"] == [4]


class FailingRolloutRunner(StubRolloutRunner):
    """A simulator worker dies mid-rollout; close() would block forever (AsyncVectorEnv after a worker error)."""

    class _Process:
        def __init__(self):
            self.alive, self.killed = True, False

        def is_alive(self):
            return self.alive

        def kill(self):
            self.alive, self.killed = False, True

        def terminate(self):
            self.alive, self.killed = False, True

        def join(self, timeout=None):
            pass

    processes = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = type("Env", (), {})()
        self.env.processes = [self._Process(), self._Process()]
        FailingRolloutRunner.processes.extend(self.env.processes)

    def run(self, policy, **kwargs):
        raise EOFError("simulator worker 3 died")

    def close(self):
        raise AssertionError("close() must not be called after a failed rollout (it would block)")


def test_checkpoint_is_written_before_every_rollout(tmp_path):
    workspace = run_workspace(rollout_config(**{"training.checkpoint_every": 100}), tmp_path)
    saved = [record["epoch"] for record in records(tmp_path / "logs.jsonl", "epoch") if "checkpoint" in record]
    assert saved == [0, 1, 3, 4]  # epoch 0 (cadence), the rollout epochs 1 and 3, and the final epoch
    assert workspace.completed_optimizer_steps == 5


def test_failed_rollout_is_recorded_and_training_continues(tmp_path, capfd):
    FailingRolloutRunner.processes.clear()
    cfg = rollout_config(**{"task.policy.env_runner._target_": "test_p2n_vla_workspace.FailingRolloutRunner"})
    workspace = run_workspace(cfg, tmp_path)  # rollout_failure defaults to continue
    assert workspace.completed_optimizer_steps == 5  # all epochs trained despite three failed rollouts
    assert FailingRolloutRunner.processes and all(process.killed for process in FailingRolloutRunner.processes)
    assert "simulator worker 3 died" in capfd.readouterr().err  # printed before any cleanup
    rollouts = records(tmp_path / "logs.jsonl", "rollout")
    assert [record["rollout/failed"] for record in rollouts] == [1, 1, 1]
    summary = json.loads(Path(rollouts[0]["rollout_dir"], "summary.json").read_text())
    assert summary["failed"] and "worker 3 died" in summary["error"] and summary["epoch"] == 2
    assert all(module.training for module in workspace.model.modules())  # modes restored after the failure
    final = json.loads((tmp_path / "training_summary.json").read_text())
    assert len(final["rollouts"]) == 3 and final["rollouts"][0]["success_rate"] is None
    assert all(entry["failed"] for entry in final["rollouts"])


def test_failed_rollout_with_raise_kills_the_simulators_and_stops(tmp_path, capfd):
    FailingRolloutRunner.processes.clear()
    cfg = rollout_config(**{"task.policy.env_runner._target_": "test_p2n_vla_workspace.FailingRolloutRunner",
                            "training.rollout_failure": "raise"})
    with pytest.raises(EOFError, match="worker 3 died"):
        run_workspace(cfg, tmp_path)
    assert FailingRolloutRunner.processes and all(process.killed for process in FailingRolloutRunner.processes)
    assert "simulator worker 3 died" in capfd.readouterr().err
    # Training up to the failing rollout is safe: its epoch record and resume checkpoint were written first.
    assert [record["epoch"] for record in records(tmp_path / "logs.jsonl", "epoch")] == [0, 1]
    assert load(tmp_path / "checkpoints/latest.ckpt")["training"]["counters"]["epoch"] == 2


def _hang_forever(connection):
    import time
    while True:
        time.sleep(60)


class HangingRolloutRunner(StubRolloutRunner):
    """A simulator worker that never answers: run() blocks on its pipe until the worker is killed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import multiprocessing
        self.parent, child = multiprocessing.get_context("fork").Pipe()
        self.worker = multiprocessing.get_context("fork").Process(target=_hang_forever, args=(child,), daemon=True)
        self.worker.start()
        child.close()

    def run(self, policy, **kwargs):
        self.parent.recv()  # blocks until the watchdog kills the worker (EOFError)

    def close(self):
        pass


class ConstructionFailureRunner(StubRolloutRunner):
    """One simulator fails while the runner is being built, after its siblings were forked."""

    def __init__(self, *args, **kwargs):
        import multiprocessing
        ConstructionFailureRunner.workers = [multiprocessing.get_context("fork").Process(
            target=_hang_forever, args=(None,), daemon=True) for _ in range(2)]
        for worker in ConstructionFailureRunner.workers:
            worker.start()
        raise ConnectionResetError("simulator 2 failed to start")


def test_rollout_watchdog_turns_a_hung_simulator_into_a_failed_rollout(tmp_path):
    cfg = rollout_config(**{"task.policy.env_runner._target_": "test_p2n_vla_workspace.HangingRolloutRunner",
                            "training.rollout_timeout_minutes": 0.02, "training.num_epochs": 2})
    import time
    started = time.monotonic()
    workspace = run_workspace(cfg, tmp_path)
    assert time.monotonic() - started < 60 and workspace.completed_optimizer_steps == 2
    rollout = records(tmp_path / "logs.jsonl", "rollout")[0]
    assert rollout["rollout/failed"] == 1 and "exceeded" in rollout["rollout_error"]


def test_failed_runner_construction_kills_the_already_forked_simulators(tmp_path):
    cfg = rollout_config(**{"task.policy.env_runner._target_": "test_p2n_vla_workspace.ConstructionFailureRunner",
                            "training.num_epochs": 2})
    workspace = run_workspace(cfg, tmp_path)
    assert workspace.completed_optimizer_steps == 2
    assert ConstructionFailureRunner.workers and not any(w.is_alive() for w in ConstructionFailureRunner.workers)
    assert "simulator 2 failed to start" in records(tmp_path / "logs.jsonl", "rollout")[0]["rollout_error"]


def test_every_rollout_has_a_snapshot_of_the_weights_it_scored(tmp_path):
    run_workspace(rollout_config(), tmp_path)  # snapshot_every=0: only the rollout snapshots (and the final one)
    names = sorted(path.name for path in (tmp_path / "snapshots").iterdir())
    assert names == ["upd-000002_ema.ckpt", "upd-000004_ema.ckpt", "upd-000005_ema.ckpt"]
    for directory in sorted((tmp_path / "eval").iterdir()):
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["snapshot"].endswith(f"upd-{summary['optimizer_step']:06d}_ema.ckpt")


def test_interrupted_rollouts_are_reported(tmp_path, capsys):
    run_workspace(rollout_config(**{"training.num_epochs": 2}), tmp_path)
    (tmp_path / "eval" / "rollout_epoch-0004_upd-000004").mkdir()  # as if the run died during that rollout
    resumed = rollout_config(**{"training.num_epochs": 4, "training.resume": True,
                                "training.resume_checkpoint": str(tmp_path / "checkpoints/latest.ckpt")})
    run_workspace(resumed, tmp_path)
    assert "never finished" in capsys.readouterr().out
    history = json.loads((tmp_path / "training_summary.json").read_text())["rollouts"]
    assert [item.get("incomplete", False) for item in history].count(True) == 1


def test_close_runner_never_blocks_on_a_stuck_simulator():
    from oat.env_runner.p2n_vla_rollout import close_runner as _close_runner
    import threading
    release = threading.Event()

    class Stuck:
        def __init__(self):
            self.env = type("Env", (), {})()
            self.env.processes = [FailingRolloutRunner._Process()]

        def close(self):
            release.wait(30)  # a worker that never answers

    runner = Stuck()
    import time
    started = time.monotonic()
    _close_runner(runner, force=False, timeout=0.5)
    assert time.monotonic() - started < 10 and runner.env.processes[0].killed
    assert runner.env.closed  # VectorEnv.__del__ must not re-enter a blocking close()
    release.set()
    forced = Stuck()
    _close_runner(forced, force=True)
    assert forced.env.processes[0].killed


def test_snapshot_precision_rounds_like_the_snapshot_and_restores_exactly():
    from oat.workspace.train_p2n_vla import _snapshot_precision
    layer = nn.Linear(4, 3)
    with torch.no_grad():
        layer.weight.add_(1e-3 * torch.arange(12.0).reshape(3, 4) / 7)
    before = layer.weight.detach().clone()
    pointer = layer.weight.data_ptr()
    with _snapshot_precision(list(layer.named_parameters())):
        assert torch.equal(layer.weight, before.to(torch.bfloat16).float())
        assert layer.weight.dtype == torch.float32 and layer.weight.data_ptr() == pointer  # in place, no GPU copy
    assert layer.weight.data_ptr() == pointer and torch.equal(layer.weight, before)  # the original storage


def test_rollout_history_survives_a_resume(tmp_path):
    first = rollout_config(**{"training.num_epochs": 2})
    run_workspace(first, tmp_path)
    resumed = rollout_config(**{"training.num_epochs": 4, "training.resume": True,
                                "training.resume_checkpoint": str(tmp_path / "checkpoints/latest.ckpt")})
    run_workspace(resumed, tmp_path)
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert [entry["epoch"] for entry in summary["rollouts"]] == [2, 4]  # epoch 2 ran before the resume


@pytest.mark.slow
def test_two_process_gloo_ddp_rollouts_run_on_rank0_while_rank1_waits(tmp_path):
    cfg = rollout_config(**{"training.num_epochs": 3, "training.max_train_steps": 3})
    output = tmp_path / "run"
    _torchrun_worker(cfg, output, tmp_path, "rollout")
    epochs = records(output / "logs.jsonl", "epoch")
    assert [record["epoch"] for record in records(output / "logs.jsonl", "rollout")] == [1, 2]
    assert all(record["replica_fingerprint"] is not None for record in epochs)  # replicas stayed identical
    names = sorted(path.name for path in (output / "eval").iterdir())
    assert names == ["rollout_epoch-0002_upd-000004", "rollout_epoch-0003_upd-000006"]
    summary = json.loads((output / "training_summary.json").read_text())
    assert [entry["epoch"] for entry in summary["rollouts"]] == [2, 3]


@pytest.mark.slow
def test_two_process_gloo_ddp_smoke_through_the_launcher_worker(tmp_path):
    """torchrun, 2 CPU ranks, gloo: DDP(find_unused_parameters=False, gradient_as_bucket_view=True)."""
    cfg = stub_config(**{"training.max_train_steps": 3, "training.snapshot_every": 2})
    config_path = tmp_path / "worker.yaml"
    OmegaConf.save(cfg, config_path)
    output = tmp_path / "run"
    env = dict(os.environ, ACCELERATE_USE_CPU="true", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(TESTS), str(REPO), os.environ.get("PYTHONPATH", "")]))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
               str(REPO / "scripts/train_p2n_vla.py"), "--worker-config", str(config_path), "--output", str(output)]
    result = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    report = json.loads((output / "training_report.json").read_text())
    assert report["world_size"] == 2
    assert report["ddp"] == {"wrapper": "DistributedDataParallel", "find_unused_parameters": False,
                             "gradient_as_bucket_view": True, "broadcast_buffers": True, "mixed_precision": "no"}
    payload = load(output / "checkpoints/latest.ckpt")
    training = payload["training"]
    assert training["world_size"] == 2 and len(training["rng_states"]) == 2
    assert not torch.equal(training["rng_states"][0]["torch"], training["rng_states"][1]["torch"])
    # 3 micro-batches per rank per epoch at accumulation 2 -> 2 updates per epoch.
    assert training["counters"]["completed_optimizer_steps"] == 4
    assert sorted(path.name for path in (output / "snapshots").iterdir()) == ["upd-000002_ema.ckpt",
                                                                             "upd-000004_ema.ckpt"]
    epochs = records(output / "logs.jsonl", "epoch")
    assert len(epochs) == 2 and all(record["replica_fingerprint"] is not None for record in epochs)
    assert all(record["val_loss_generated_history"] is not None for record in epochs)
    steps = records(output / "logs.jsonl", "train_step")
    assert [record["optimizer_step"] for record in steps] == [1, 2, 3, 4]
    assert steps[0]["samples_per_sec"] > 0


def _torchrun_worker(cfg, output, tmp_path, name):
    config_path = tmp_path / f"{name}.yaml"
    OmegaConf.save(cfg, config_path)
    env = dict(os.environ, ACCELERATE_USE_CPU="true", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(TESTS), str(REPO), os.environ.get("PYTHONPATH", "")]))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
               str(REPO / "scripts/train_p2n_vla.py"), "--worker-config", str(config_path), "--output", str(output)]
    result = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]


@pytest.mark.slow
def test_two_rank_resume_is_bitwise_identical_to_an_uninterrupted_run(tmp_path):
    """Per-rank RNG, DDP-broadcast weights, optimizer, EMA and data order all continue exactly."""
    base = {"training.max_train_steps": 3, "training.snapshot_every": 0, "training.num_epochs": 2}
    _torchrun_worker(stub_config(**base), tmp_path / "full", tmp_path, "full")
    _torchrun_worker(stub_config(**{**base, "training.num_epochs": 1}), tmp_path / "first", tmp_path, "first")
    checkpoint = tmp_path / "first/checkpoints/latest.ckpt"
    _torchrun_worker(stub_config(**{**base, "training.resume": True, "training.resume_checkpoint": str(checkpoint)}),
                     tmp_path / "resumed", tmp_path, "resumed")
    full, resumed = load(tmp_path / "full/checkpoints/latest.ckpt"), load(tmp_path / "resumed/checkpoints/latest.ckpt")
    for kind in ("model", "ema_model"):
        for key, value in full["state_dicts"][kind].items():
            torch.testing.assert_close(resumed["state_dicts"][kind][key], value, rtol=0, atol=0, msg=f"{kind}.{key}")
    for index, state in full["training"]["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(resumed["training"]["optimizer"]["state"][index][name], value,
                                       rtol=0, atol=0)
    assert full["training"]["counters"] == resumed["training"]["counters"]
    for rank in range(2):
        assert torch.equal(full["training"]["rng_states"][rank]["torch"],
                           resumed["training"]["rng_states"][rank]["torch"])
    full_losses = [record["train/loss"] for record in records(tmp_path / "full/logs.jsonl", "train_step")][2:]
    resumed_losses = [record["train/loss"] for record in records(tmp_path / "resumed/logs.jsonl", "train_step")]
    assert resumed_losses == full_losses and len(resumed_losses) == 2


# ------------------------------------------------- integration with the real (tiny) M3 policies
OAT_CHECKPOINT = Path("/workspace/hf_upload/tokenizer_oattok_so3aug_ep4960_mse0.001.ckpt")


def _require_real_policy_stack():
    if not OAT_CHECKPOINT.is_file():
        pytest.skip(f"MISSING RESOURCE: frozen OAT checkpoint {OAT_CHECKPOINT}")
    try:
        import oat.dataset.vla_dataset  # noqa: F401
        import oat.policy.p2n_vla  # noqa: F401
        import oat.policy.p2n_vla_state_gate  # noqa: F401
        import oat.policy.pi05_ki_flow  # noqa: F401
    except ImportError as error:  # the M3 policies are written concurrently
        pytest.skip(f"P2N-VLA policy stack not importable yet: {error}")


def make_libero_like_zarr(path, n_episodes=3, episode_length=16, seed=0):
    """A tiny zarr with the LIBERO-10 schema (128x128 uint8 cameras, robosuite quaternions, task_uid)."""
    from oat.common.replay_buffer import ReplayBuffer
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer.create_from_path(str(path), mode="w")
    for episode in range(n_episodes):
        steps = episode_length
        quat = rng.standard_normal((steps, 4)).astype(np.float32)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        buffer.add_episode({
            "action": np.concatenate([rng.uniform(-0.9, 0.9, (steps, 3)), rng.uniform(-0.2, 0.3, (steps, 3)),
                                      np.where(rng.random((steps, 1)) > 0.5, 1.0, -1.0)], 1).astype(np.float32),
            "agentview_rgb": rng.integers(0, 256, (steps, 128, 128, 3), dtype=np.uint8),
            "robot0_eye_in_hand_rgb": rng.integers(0, 256, (steps, 128, 128, 3), dtype=np.uint8),
            "robot0_eef_pos": (rng.standard_normal((steps, 3)) * 0.1).astype(np.float32),
            "robot0_eef_quat": quat,
            "robot0_gripper_qpos": rng.uniform(-0.04, 0.04, (steps, 2)).astype(np.float32),
            "task_uid": np.full((steps, 1), 30 + episode % 10, dtype=np.int64),
        })
    return path


def real_tiny_config(variant, zarr_path, *overrides):
    import scripts.train_p2n_vla as launcher
    settings = [
        "policy.model_size=tiny", "policy.pi05_weights=null", "policy.activation_checkpointing=false",
        "policy.self_past_warmup_steps=0", "policy.self_past_ramp_steps=1",
        f"task.policy.dataset.zarr_path={zarr_path}", "training.num_demo=3",
        "training.num_epochs=2", "training.max_train_steps=2", "training.max_optimizer_steps=10",
        "training.lr_warmup_steps=1", "training.snapshot_every=1", "training.max_val_steps=1",
        "training.max_reconst_steps=1", "training.keep_resume_checkpoint=true",
        "dataloader.batch_size=2", "dataloader.num_workers=0", "dataloader.persistent_workers=false",
        "dataloader.pin_memory=false", "val_dataloader.batch_size=2", "val_dataloader.num_workers=0",
        "val_dataloader.persistent_workers=false", "val_dataloader.pin_memory=false", "logging.mode=disabled",
    ]
    return launcher.compose_config(variant, "libero", [*settings, *overrides])



@pytest.mark.slow
@pytest.mark.parametrize("variant", ["p2n_vla", "p2n_vla_state_gate"])
def test_real_tiny_policy_keeps_training_after_in_training_rollouts(tmp_path, variant):
    """predict_action runs under the runner's inference_mode between epochs; nothing it caches may leak into
    the next training step's autograd graph, and the EMA swap must restore the live weights exactly."""
    _require_real_policy_stack()
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    cfg = real_tiny_config(variant, zarr_path, "training.num_epochs=3", "training.snapshot_every=0",
                           "task.policy.lazy_eval=false", "training.rollout_every=1",
                           "task.policy.env_runner.n_test=10")
    OmegaConf.update(cfg, "task.policy.env_runner._target_", "test_p2n_vla_workspace.StubRolloutRunner")
    StubRolloutRunner.events.clear()
    workspace = run_workspace(cfg, tmp_path / "run")
    runs = [event for event in StubRolloutRunner.events if event["event"] == "run"]
    assert len(runs) == 3 and not any(event["any_training"] for event in runs)
    steps = records(tmp_path / "run/logs.jsonl", "train_step")
    assert [record["optimizer_step"] for record in steps] == [1, 2, 3]
    assert all(np.isfinite(record["train/loss"]) for record in steps)
    # The EMA swap restored the live weights: they differ from the EMA shadow after training.
    name = workspace.ema.names[0]
    live = dict(workspace.model.named_parameters())[name].detach()
    assert not torch.equal(live, workspace.ema.shadow[name].to(live.dtype))
    assert [record["rollout/trials"] for record in records(tmp_path / "run/logs.jsonl", "rollout")] == [10, 10, 10]

def _fixed_eval_batch(dataset, indices=(4, 9)):
    batch = torch.utils.data.default_collate([dataset[index] for index in indices])
    return batch


@pytest.mark.slow
def test_real_tiny_p2n_vla_trains_resumes_and_reloads_exactly(tmp_path):
    _require_real_policy_stack()
    from oat.policy.p2n_vla import P2NVLAPolicy
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    full = run_workspace(real_tiny_config("p2n_vla", zarr_path), tmp_path / "full")
    assert full.completed_optimizer_steps == 2 and full.model.self_past_step == 2
    steps = records(tmp_path / "full/logs.jsonl", "train_step")
    assert all(record["train/loss_ar"] is not None and record["train/loss_ki"] is not None for record in steps)
    assert steps[-1]["train/self_past_p"] == 0.5  # warmup 0, ramp 1: full probability from the second update
    epochs = records(tmp_path / "full/logs.jsonl", "epoch")
    assert all(record["val_loss_generated_history"] is not None and record["val_reconstruction_mse"] is not None
               for record in epochs)

    run_workspace(real_tiny_config("p2n_vla", zarr_path, "training.num_epochs=1"), tmp_path / "first")
    checkpoint = tmp_path / "first/checkpoints/latest.ckpt"
    resumed = run_workspace(real_tiny_config("p2n_vla", zarr_path, "training.resume=true",
                                             f"training.resume_checkpoint={checkpoint}"), tmp_path / "resumed")
    expected = full.model.artifact_state_dict()
    for key, value in resumed.model.artifact_state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0, msg=key)
    for key, value in resumed.ema.shadow.items():
        torch.testing.assert_close(value, full.ema.shadow[key], rtol=0, atol=0, msg=key)

    # The resume checkpoint's fp32 EMA artifact reloads into an identical policy (restore mode).
    dataset = hydra.utils.instantiate(real_tiny_config("p2n_vla", zarr_path).task.policy.dataset)
    batch = _fixed_eval_batch(dataset.get_validation_dataset())
    with full.ema.swap_in(full.model):
        expected_action = full.model.predict_action(batch["obs"], past_actions=batch["past_action"],
                                                    past_action_valid=batch["past_action_valid"])["action_pred"]
    reloaded = P2NVLAPolicy.from_checkpoint(str(tmp_path / "full/checkpoints/latest.ckpt"), weights="ema")
    action = reloaded.predict_action(batch["obs"], past_actions=batch["past_action"],
                                     past_action_valid=batch["past_action_valid"])["action_pred"]
    torch.testing.assert_close(action, expected_action, rtol=0, atol=0)
    assert reloaded.self_past_step == 2
    # bf16 EMA snapshots load through the same entry point.
    snapshot = sorted((tmp_path / "full/snapshots").iterdir())[-1]
    assert P2NVLAPolicy.from_checkpoint(str(snapshot), weights="ema").self_past_step == 2


FRESH_PROCESS_RELOAD = r"""
import sys
import torch
sys.path.insert(0, sys.argv[1])
torch.set_num_threads(int(sys.argv[5]))
from oat.policy.p2n_vla import P2NVLAPolicy
batch = torch.load(sys.argv[3])
policy = P2NVLAPolicy.from_checkpoint(sys.argv[2], weights="ema")
action = policy.predict_action(batch["obs"], past_actions=batch["past_action"],
                               past_action_valid=batch["past_action_valid"])["action_pred"]
torch.save({"action": action, "self_past_step": policy.self_past_step,
            "action_scale": policy.action_normalizer["action"].params_dict["scale"].detach().clone(),
            "modules": sorted(sys.modules)}, sys.argv[4])
"""


@pytest.mark.slow
def test_real_tiny_artifacts_reload_identically_in_a_fresh_process(tmp_path):
    """M4 exit criterion: a fresh process without the dataset reproduces the EMA policy bitwise."""
    _require_real_policy_stack()
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    cfg = real_tiny_config("p2n_vla", zarr_path, "training.num_epochs=2")
    workspace = run_workspace(cfg, tmp_path / "run")
    assert workspace.completed_optimizer_steps == 2
    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    batch = _fixed_eval_batch(dataset.get_validation_dataset())
    torch.save(batch, tmp_path / "batch.pt")
    with workspace.ema.swap_in(workspace.model):
        expected = workspace.model.predict_action(batch["obs"], past_actions=batch["past_action"],
                                                  past_action_valid=batch["past_action_valid"])["action_pred"]
    snapshot = sorted((tmp_path / "run/snapshots").iterdir())[-1]
    from oat.policy.p2n_vla import P2NVLAPolicy
    in_process = P2NVLAPolicy.from_checkpoint(str(snapshot), weights="ema").predict_action(
        batch["obs"], past_actions=batch["past_action"], past_action_valid=batch["past_action_valid"])["action_pred"]
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="")
    for checkpoint, reference in ((tmp_path / "run/checkpoints/latest.ckpt", expected),  # fp32 EMA artifact
                                  (snapshot, in_process)):                             # bf16 EMA snapshot
        output = tmp_path / f"{checkpoint.stem}_fresh.pt"
        result = subprocess.run([sys.executable, "-c", FRESH_PROCESS_RELOAD, str(REPO), str(checkpoint),
                                 str(tmp_path / "batch.pt"), str(output), str(torch.get_num_threads())],
                                cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
        fresh = torch.load(output)
        torch.testing.assert_close(fresh["action"], reference, rtol=0, atol=0, msg=str(checkpoint))
        assert fresh["self_past_step"] == 2
        assert not torch.equal(fresh["action_scale"], torch.ones_like(fresh["action_scale"]))  # fitted normalizer
        assert not any(name.startswith("oat.dataset") for name in fresh["modules"])  # no dataset needed


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["p2n_vla_state_gate", "pi05_ki_flow"])
def test_real_tiny_gate_and_flow_variants_train_through_the_workspace(tmp_path, variant):
    _require_real_policy_stack()
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    workspace = run_workspace(real_tiny_config(variant, zarr_path, "training.num_epochs=1"), tmp_path / "run")
    assert workspace.completed_optimizer_steps == 1
    epoch = records(tmp_path / "run/logs.jsonl", "epoch")[0]
    if variant == "pi05_ki_flow":
        assert epoch["val/expert/loss_flow"] is not None and epoch["val_loss_generated_history"] is None
    else:
        assert epoch["val_loss_generated_history"] is not None and epoch["val/expert/gate_mean"] is not None
    payload = load(tmp_path / "run/checkpoints/latest.ckpt")
    assert payload["metadata"]["variant"] == variant
    assert not set(workspace.model.frozen_base_keys()) & set(payload["state_dicts"]["model"])


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["p2n_vla", "p2n_vla_state_gate", "pi05_ki_flow"])
def test_real_tiny_policies_two_rank_gloo_ddp(tmp_path, variant):
    """Every trainable parameter of the real policies gets a gradient under DDP(find_unused_parameters=False)."""
    _require_real_policy_stack()
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    cfg = real_tiny_config(variant, zarr_path, "training.max_train_steps=2")
    config_path = tmp_path / "worker.yaml"
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), config_path)
    output = tmp_path / "run"
    env = dict(os.environ, ACCELERATE_USE_CPU="true", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join([str(TESTS), str(REPO), os.environ.get("PYTHONPATH", "")]))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
               str(REPO / "scripts/train_p2n_vla.py"), "--worker-config", str(config_path), "--output", str(output)]
    result = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1200)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    report = json.loads((output / "training_report.json").read_text())
    assert report["ddp"]["wrapper"] == "DistributedDataParallel" and report["world_size"] == 2
    assert report["ddp"]["find_unused_parameters"] is False and report["ddp"]["gradient_as_bucket_view"] is True
    epochs = records(output / "logs.jsonl", "epoch")
    assert len(epochs) == 2 and all(record["replica_fingerprint"] is not None for record in epochs)
    payload = load(output / "checkpoints/latest.ckpt")
    assert payload["training"]["counters"]["completed_optimizer_steps"] == 2
    assert len(payload["training"]["rng_states"]) == 2


@pytest.mark.gpu
@pytest.mark.slow
def test_real_tiny_policy_on_cuda_uses_fused_adamw_and_restores_cuda_rng(tmp_path, monkeypatch):
    """Single CUDA device (run with CUDA_VISIBLE_DEVICES set to a free GPU)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available (gpu marker)")
    _require_real_policy_stack()
    monkeypatch.setenv("ACCELERATE_USE_CPU", "false")
    zarr_path = make_libero_like_zarr(tmp_path / "libero_like.zarr")
    first = run_workspace(real_tiny_config("p2n_vla", zarr_path, "training.num_epochs=1"), tmp_path / "first")
    assert first.model.device.type == "cuda"
    assert first.optimizer.optimizer.defaults["fused"] is True
    assert all(shadow.is_cuda for shadow in first.ema.shadow.values())
    step = records(tmp_path / "first/logs.jsonl", "train_step")[0]
    assert step["max_memory_reserved_gib"] > 0 and step["samples_per_sec"] > 0
    payload = load(tmp_path / "first/checkpoints/latest.ckpt")
    saved_cuda = payload["training"]["rng_states"][0]["cuda"]
    assert isinstance(saved_cuda, torch.Tensor) and saved_cuda.dtype == torch.uint8
    assert all(not value.is_cuda for value in payload["state_dicts"]["model"].values())
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state(True)
    checkpoint = tmp_path / "first/checkpoints/latest.ckpt"
    resumed_cfg = real_tiny_config("p2n_vla", zarr_path, "training.resume=true",
                                   f"training.resume_checkpoint={checkpoint}")
    workspace = Workspace(resumed_cfg, output_dir=str(tmp_path / "resumed"))
    restored = {}
    original = Workspace._restore_rng

    def spy(self, rank, world_size, device):
        original(self, rank, world_size, device)
        restored["cuda"] = torch.cuda.get_rng_state(device).clone()

    monkeypatch.setattr(Workspace, "_restore_rng", spy)
    workspace.run()
    assert torch.equal(restored["cuda"], saved_cuda)
    assert workspace.completed_optimizer_steps == 2 and workspace.optimizer.optimizer.defaults["fused"] is True
