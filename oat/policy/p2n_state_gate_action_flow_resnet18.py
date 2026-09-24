"""Trainable ResNet18 policy with measured state/action summaries and a summary-only gate."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from oat.model.common.context_batch import Segment
from oat.model.common.action_flow_context import ActionFlowContextBatch
from oat.model.state_action_history import StateActionHistoryEncoder
from oat.policy.p2n_action_flow_resnet_common import P2NActionFlowResNetCommonPolicy, bool_mask
from oat.model.action_flow_state_history import Rotation6DStateActionHistoryEncoder


class P2NStateGateActionFlowResNet18Policy(P2NActionFlowResNetCommonPolicy):
    VARIANT = 'p2n_state_gate_action_flow'
    requires_state_history = True
    supports_history_summary_gate = True

    def __init__(self, *args, state_history_steps=8, state_history_keys=None,
                 history_embed_dim=128, history_n_heads=4, history_n_layers=2,
                 history_summary_tokens=4, history_dropout=0.0,
                 history_gate_mode='learned', history_gate_hidden_dim=128,
                 history_gate_init=0.9, rotation_6d_layout='rows', **kwargs):
        if history_dropout != 0.0:
            raise ValueError('Flow history dropout must be zero')
        if history_gate_mode not in ('learned', 'open', 'closed'):
            raise ValueError('history_gate_mode must be learned, open, or closed')
        if not 0 < history_gate_init < 1 or not math.isfinite(history_gate_init):
            raise ValueError('history_gate_init must be finite and between zero and one')
        super().__init__(*args, **kwargs)
        if state_history_steps != self.past_n + 1:
            raise ValueError('state_history_steps must equal past_n + 1')
        if state_history_keys is None:
            rotation = 'robot0_eef_rot6d' if 'robot0_eef_rot6d' in self.obs_key_shapes else 'robot0_eef_quat'
            state_history_keys = ('robot0_eef_pos', rotation, 'robot0_gripper_qpos')
        keys = tuple(state_history_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError('state_history_keys must be nonempty and unique')
        if any(key not in self.obs_encoder.state_ports for key in keys):
            raise ValueError('State-history ports must be current low-dimensional state fields')
        self.state_history_steps, self.state_history_keys = state_history_steps, keys
        self.history_summary_tokens = int(history_summary_tokens)
        self.history_gate_mode = history_gate_mode
        self.rotation_6d_layout = rotation_6d_layout
        encoder_class = (Rotation6DStateActionHistoryEncoder if any(k.endswith('_rot6d') for k in keys)
                         else StateActionHistoryEncoder)
        geometry = {'rotation_6d_layout': rotation_6d_layout} if encoder_class is Rotation6DStateActionHistoryEncoder else {}
        self.history_encoder = encoder_class(
            state_shapes={key: self.obs_key_shapes[key] for key in keys},
            action_dim=self.action_dim, history_steps=state_history_steps,
            output_dim=self.obs_feature_dim, embed_dim=history_embed_dim,
            n_heads=history_n_heads, n_layers=history_n_layers,
            n_summary_tokens=history_summary_tokens, dropout=history_dropout, **geometry,
        )
        # Each camera/frame contributes one average, separately from proprioception.
        self.observation_pool = nn.Sequential(
            nn.LayerNorm(2 * self.obs_feature_dim),
            nn.Linear(2 * self.obs_feature_dim, self.obs_feature_dim), nn.SiLU(),
        )
        self.summary_type_embedding = nn.Parameter(torch.empty(1, 1, self.obs_feature_dim))
        nn.init.normal_(self.summary_type_embedding, std=0.02)
        self.history_gate = nn.Sequential(
            nn.LayerNorm(2 * self.obs_feature_dim + 1),
            nn.Linear(2 * self.obs_feature_dim + 1, history_gate_hidden_dim), nn.GELU(),
            nn.Linear(history_gate_hidden_dim, 1),
        )
        for parent in (self.history_encoder, self.observation_pool, self.history_gate):
            for module in parent.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.zeros_(self.history_gate[-1].weight)
        nn.init.constant_(self.history_gate[-1].bias, math.log(history_gate_init / (1 - history_gate_init)))
        self.history_gate.requires_grad_(history_gate_mode == 'learned')
        self.observation_pool.requires_grad_(history_gate_mode == 'learned')
        if history_gate_mode == 'closed':
            self.history_encoder.requires_grad_(False)
            self.summary_type_embedding.requires_grad_(False)
        self._last_history_gate = None
        self._construction.update(dict(
            state_history_steps=state_history_steps, state_history_keys=list(keys),
            history_embed_dim=history_embed_dim, history_n_heads=history_n_heads,
            history_n_layers=history_n_layers, history_summary_tokens=history_summary_tokens,
            history_dropout=history_dropout, history_gate_mode=history_gate_mode,
            history_gate_hidden_dim=history_gate_hidden_dim, history_gate_init=history_gate_init,
            rotation_6d_layout=rotation_6d_layout,
        ))

    def build_context(self, obs, past_actions, past_action_valid, prepared_visual=None):
        required = ['state_history__' + key for key in self.state_history_keys] + ['state_history_valid']
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f'Missing required measured state-history fields: {missing}')
        state_valid = bool_mask(obs['state_history_valid'],
                               (past_actions.shape[0], self.state_history_steps),
                               'state_history_valid', past_actions.device)
        past, valid = self._safe_past(past_actions, past_action_valid)
        transition_valid = state_valid[:, :-1] & state_valid[:, 1:]
        if not torch.equal(valid, transition_valid):
            raise ValueError('past_action_valid must match aligned adjacent-state transitions')
        base = super().build_context(obs, past_actions, valid, prepared_visual)
        summaries = self.history_encoder(
            {key: obs['state_history__' + key] for key in self.state_history_keys},
            state_valid, past, self.action_normalizer,
        ) + self.summary_type_embedding
        visual = base.memory[:, base.segment_ids == int(Segment.VISUAL)]
        visual = visual.reshape(visual.shape[0], self.n_obs_steps, len(self.obs_encoder.rgb_ports),
                                self.obs_encoder.num_queries, self.obs_feature_dim)
        visual_pool = visual.mean(dim=3).mean(dim=(1, 2))
        proprio_pool = base.memory[:, base.segment_ids == int(Segment.PROPRIO)].mean(dim=1)
        observed = self.observation_pool(torch.cat((visual_pool, proprio_pool), dim=-1))
        history_pool = summaries.mean(dim=1)
        fraction = state_valid.to(summaries.dtype).mean(dim=1, keepdim=True)
        if self.history_gate_mode == 'learned':
            log_gate = F.logsigmoid(self.history_gate(torch.cat((observed, history_pool, fraction), dim=-1)).float())
        else:
            log_gate = summaries.new_full((summaries.shape[0], 1),
                                          0. if self.history_gate_mode == 'open' else -float('inf'))
        self._last_history_gate = log_gate.detach().exp()
        # The current measured state is valid even when no preceding transition exists.
        # The history encoder always executes, preserving its DDP graph at episode start.
        summary_valid = state_valid[:, -1:].expand(-1, self.history_summary_tokens)
        return ActionFlowContextBatch(
            memory=torch.cat((base.memory, summaries), dim=1),
            valid_mask=torch.cat((base.valid_mask, summary_valid), dim=1),
            segment_ids=torch.cat((base.segment_ids, base.segment_ids.new_full(
                (self.history_summary_tokens,), int(Segment.HISTORY_SUMMARY)))),
            observation_summary=observed, history_summary_pool=history_pool,
            history_valid_fraction=fraction, history_log_gate=log_gate,
        ).validate()

    def get_observation_ports(self):
        return super().get_observation_ports() + ['state_history__' + key for key in self.state_history_keys] + ['state_history_valid']

    def create_dummy_observation(self, batch_size=1, device=None):
        obs = super().create_dummy_observation(batch_size, device)
        device = self.device if device is None else device
        for key in self.state_history_keys:
            value = torch.zeros(batch_size, self.state_history_steps, *self.obs_key_shapes[key],
                                device=device, dtype=self.dtype)
            if key.endswith('_quat'):
                value[..., 3] = 1
            elif key.endswith('_rot6d'):
                value[...] = value.new_tensor([1, 0, 0, 0, 1, 0])
            obs['state_history__' + key] = value
        valid = torch.zeros(batch_size, self.state_history_steps, device=device, dtype=torch.bool)
        valid[:, -1] = True
        obs['state_history_valid'] = valid
        return obs

    def get_history_gate_metrics(self):
        if self._last_history_gate is None:
            return {}
        value = self._last_history_gate.float()
        return {'history_gate/mean': value.mean().item(), 'history_gate/min': value.min().item(),
                'history_gate/max': value.max().item()}

    def reset(self):
        super().reset()
        self._last_history_gate = None
