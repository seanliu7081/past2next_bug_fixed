"""P2N-VLA with 8 measured states summarized into 4 HIST tokens behind a learned log-gate.

Only AR rows read HIST keys (layout.py), so ``closed`` removes the history exactly
and ``open`` is a bitwise no-op. The learned gate adds ``logsigmoid(MLP(...))`` in
fp32 to every AR->HIST attention logit in all expert layers. Its observation input
is pooled from the *detached* final VLM hidden states (knowledge insulation).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from oat.model.state_action_history import StateActionHistoryEncoder
from oat.policy.p2n_new_common import bool_mask
from oat.policy.p2n_vla_common import ConditionBatch, P2NVLACommonPolicy
from oat.policy.past2next_state_history_gate_real_robot import Rotation6DStateActionHistoryEncoder


class ObservationPool(nn.Module):
    """Pooled VLM context and the normalized prompt state, each normalized on its own scale."""

    def __init__(self, vlm_width: int, state_dim: int, state_width: int, out_width: int):
        super().__init__()
        self.vlm_norm = nn.LayerNorm(vlm_width)
        self.state_proj = nn.Linear(state_dim, state_width)
        self.proj = nn.Linear(vlm_width + state_width, out_width)

    def forward(self, pooled: Tensor, state: Tensor) -> Tensor:
        features = torch.cat((self.vlm_norm(pooled.float()), self.state_proj(state.float())), dim=-1)
        return F.silu(self.proj(features))


class P2NVLAStateGatePolicy(P2NVLACommonPolicy):
    VARIANT = "p2n_vla_state_gate"
    VARIANT_CODE = 11
    requires_state_history = True
    supports_history_summary_gate = True

    def __init__(self, *args, state_history_steps=8, state_history_keys=None, history_embed_dim=128,
                 history_n_heads=4, history_n_layers=2, history_summary_tokens=4, history_dropout=0.1,
                 history_gate_mode="learned", history_gate_hidden_dim=128, history_gate_init=0.9,
                 rotation_6d_layout="rows", gate_state_dim=256, **kwargs):
        if history_gate_mode not in ("learned", "open", "closed"):
            raise ValueError("history_gate_mode must be learned, open, or closed")
        if not math.isfinite(history_gate_init) or not 0 < history_gate_init < 1:
            raise ValueError("history_gate_init must be finite and strictly between zero and one")
        self._history_keys_requested = state_history_keys
        self._history_steps_requested = int(state_history_steps)
        super().__init__(*args, **kwargs)
        if not self.use_past:
            raise ValueError("The state-gate variant needs past conditioning (use_past=True)")
        if state_history_steps != self.past_n + 1:
            raise ValueError("state_history_steps must equal past_n + 1")
        keys = self.history_keys
        width = self.expert_width
        encoder_class = (Rotation6DStateActionHistoryEncoder if any(k.endswith("_rot6d") for k in keys)
                         else StateActionHistoryEncoder)
        geometry = ({"rotation_6d_layout": rotation_6d_layout}
                    if encoder_class is Rotation6DStateActionHistoryEncoder else {})
        self.state_history_steps = int(state_history_steps)
        self.history_summary_tokens = int(history_summary_tokens)
        self.history_gate_mode = history_gate_mode
        self.history_encoder = encoder_class(
            state_shapes={key: self.obs_key_shapes[key] for key in keys}, action_dim=self.action_dim,
            history_steps=self.state_history_steps, output_dim=width, embed_dim=history_embed_dim,
            n_heads=history_n_heads, n_layers=history_n_layers, n_summary_tokens=self.history_summary_tokens,
            dropout=history_dropout, **geometry)
        self.summary_type_embedding = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.summary_type_embedding, std=0.02)
        self.observation_pool = ObservationPool(self.vlm_width, self.prompt_state_dim, int(gate_state_dim), width)
        self.history_gate = nn.Sequential(
            nn.LayerNorm(2 * width + 1), nn.Linear(2 * width + 1, int(history_gate_hidden_dim)), nn.GELU(),
            nn.Linear(int(history_gate_hidden_dim), 1))
        nn.init.zeros_(self.history_gate[-1].weight)
        nn.init.constant_(self.history_gate[-1].bias, math.log(history_gate_init / (1 - history_gate_init)))
        learned = history_gate_mode == "learned"
        self.history_gate.requires_grad_(learned)
        self.observation_pool.requires_grad_(learned)
        if history_gate_mode == "closed":
            self.history_encoder.requires_grad_(False)
            self.summary_type_embedding.requires_grad_(False)
        self._last_history_gate: Optional[Tensor] = None
        self.obs_ports = self.obs_ports + ["state_history__" + key for key in keys] + ["state_history_valid"]
        self._construction.update(dict(
            state_history_steps=state_history_steps, state_history_keys=list(keys),
            history_embed_dim=history_embed_dim, history_n_heads=history_n_heads,
            history_n_layers=history_n_layers, history_summary_tokens=history_summary_tokens,
            history_dropout=history_dropout, history_gate_mode=history_gate_mode,
            history_gate_hidden_dim=history_gate_hidden_dim, history_gate_init=history_gate_init,
            rotation_6d_layout=rotation_6d_layout, gate_state_dim=gate_state_dim))

    # ------------------------------------------------------------------ schema
    @property
    def history_keys(self) -> Tuple[str, ...]:
        keys = self._history_keys_requested
        if keys is None:
            rotation = "robot0_eef_rot6d" if "robot0_eef_rot6d" in self.obs_key_shapes else "robot0_eef_quat"
            keys = ("robot0_eef_pos", rotation, "robot0_gripper_qpos")
        keys = tuple(keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("state_history_keys must be nonempty and unique")
        for key in keys:
            if key not in self.obs_key_shapes or len(self.obs_key_shapes[key]) != 1:
                raise ValueError(f"History key {key!r} must be a vector observation in shape_meta")
        return keys

    def _normalizer_fields(self) -> List[str]:
        geometric = ("_quat", "_rot6d")
        return super()._normalizer_fields() + [k for k in self.history_keys if not k.endswith(geometric)]

    def create_dummy_observation(self, batch_size=1, device=None):
        device = self.device if device is None else device
        obs = super().create_dummy_observation(batch_size, device)
        for key in self.history_keys:
            value = torch.zeros(batch_size, self.state_history_steps, *self.obs_key_shapes[key], device=device)
            if key.endswith("_quat"):
                value[..., 3] = 1
            elif key.endswith("_rot6d"):
                value[...] = value.new_tensor([1, 0, 0, 0, 1, 0])
            obs["state_history__" + key] = value
        valid = torch.zeros(batch_size, self.state_history_steps, dtype=torch.bool, device=device)
        valid[:, -1] = True
        obs["state_history_valid"] = valid
        return obs

    # ------------------------------------------------------------------ conditions and gate
    def build_conditions(self, obs, past: Tensor, past_valid: Tensor) -> ConditionBatch:
        base = super().build_conditions(obs, past, past_valid)
        required = ["state_history__" + key for key in self.history_keys] + ["state_history_valid"]
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"Missing measured state-history fields: {missing}")
        batch = past.shape[0]
        state_valid = bool_mask(obs["state_history_valid"], (batch, self.state_history_steps),
                                "state_history_valid", past.device)
        normalized, valid = self._safe_past(past, past_valid)
        transition_valid = state_valid[:, :-1] & state_valid[:, 1:]
        if not torch.equal(valid, transition_valid):
            raise ValueError("past_action_valid must match aligned adjacent-state transitions")
        summaries = self.history_encoder(
            {key: obs["state_history__" + key].float() for key in self.history_keys},
            state_valid, normalized.float(), self.action_normalizer)
        base.hist = summaries.float() + self.summary_type_embedding
        base.state_valid = state_valid
        return base

    def compute_log_gate(self, obs, prefix, prefix_out, layout, cond: ConditionBatch):
        if self.history_gate_mode == "closed":
            self._last_history_gate = torch.zeros(cond.hist.shape[0], 1, device=cond.hist.device)
            return None, True
        if self.history_gate_mode == "open":
            self._last_history_gate = torch.ones(cond.hist.shape[0], 1, device=cond.hist.device)
            return None, False
        with torch.autocast(device_type=cond.hist.device.type, enabled=False):
            hidden = prefix_out.hidden[:, :layout.nonki_len].detach().float()
            mask = layout.nonki_valid.float()[..., None]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            observed = self.observation_pool(pooled, prefix.prompt_state)
            fraction = cond.state_valid.float().mean(dim=1, keepdim=True)
            features = torch.cat((observed, cond.hist.float().mean(dim=1), fraction), dim=-1)
            log_gate = F.logsigmoid(self.history_gate(features).float())
        self._last_history_gate = log_gate.detach().exp()
        return log_gate, False

    def _gate_components(self, log_gate, hist_closed):
        gate = self._last_history_gate
        if gate is None:
            return super()._gate_components(log_gate, hist_closed)
        gate = gate.float()
        return dict(gate_mean=gate.mean().item(), gate_min=gate.min().item(), gate_max=gate.max().item())

    def get_history_gate_metrics(self):
        if self._last_history_gate is None:
            return {}
        gate = self._last_history_gate.float()
        return {"history_gate/mean": gate.mean().item(), "history_gate/min": gate.min().item(),
                "history_gate/max": gate.max().item()}

    def reset(self):
        super().reset()
        self._last_history_gate = None
