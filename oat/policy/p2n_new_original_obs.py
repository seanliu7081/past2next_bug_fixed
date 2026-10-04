"""Modern AR with the unmodified, trainable original fused observation encoder.

The explicit constructor and context path avoid all DINO/resampler assumptions.
Training loss, generated history, cached AR generation, and execution acknowledgement
are inherited from the modern policy. Existing policy files are left untouched.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.metadata
import math
from pathlib import Path

import dill
import hydra
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from oat.common.hydra_util import register_new_resolvers
from oat.model.autoregressive.modern_transformer_cache import ModernAutoregressiveModel
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.common.original_fused_context_batch import ContextBatch, Segment, CONTEXT_LAYOUT, CONTEXT_SCHEMA_VERSION
from oat.model.state_action_history import StateActionHistoryEncoder
from oat.perception.original_fused_obs_adapter import OriginalFusedObservationAdapter, validate_normalizer_fields
from oat.policy.base_policy import BasePolicy
from oat.policy.p2n_new_common import P2NNewCommonPolicy, _plain, bool_mask, file_sha256
from oat.policy.past2next_state_history_gate_real_robot import Rotation6DStateActionHistoryEncoder


def _observation_order(shape_meta):
    return {kind: [name for name, spec in shape_meta['obs'].items() if spec.get('type') == kind]
            for kind in ('rgb', 'state')}


class P2NNewOriginalObsPolicy(P2NNewCommonPolicy):
    VARIANT = 'p2n_new'
    CONTEXT_SCHEMA_VERSION = CONTEXT_SCHEMA_VERSION

    def __init__(
        self, shape_meta, obs_encoder=None, action_tokenizer=None,
        n_action_steps=8, n_obs_steps=2, past_n=7, horizon=16,
        embed_dim=768, n_layers=16, n_heads=12, ffn_dim=2048, dropout=0.1,
        temperature=1.0, topk=10, variant=None, task='libero',
        construction_mode='fresh', tokenizer_checkpoint=None,
        tokenizer_config=None, tokenizer_metadata=None, obs_encoder_config=None,
        original_obs_config=None, obs_encoder_type='original_fused',
        context_layout=CONTEXT_LAYOUT, context_schema_version=CONTEXT_SCHEMA_VERSION,
        expected_action_tokens=8, activation_checkpointing=False,
        self_past_p=0.5, self_past_warmup_steps=1000, self_past_ramp_steps=4000,
        self_past_chunk_size=4, self_past_temperature=None, self_past_topk=None,
        self_past_schedule='optimizer_step',
    ):
        BasePolicy.__init__(self)
        if variant is not None and variant != self.VARIANT:
            raise ValueError(f'{type(self).__name__} requires variant={self.VARIANT!r}')
        if construction_mode not in ('fresh', 'restore'):
            raise ValueError('construction_mode must be fresh or restore')
        if past_n < 3 or n_obs_steps < 1 or not 1 <= n_action_steps <= horizon:
            raise ValueError('Invalid observation, action, or history horizon')
        if self_past_schedule != 'optimizer_step':
            raise ValueError('New policies advance self-past only on successful optimizer updates')
        if not 0 <= self_past_p <= 1 or min(self_past_warmup_steps, self_past_ramp_steps) < 0:
            raise ValueError('Invalid self-past probability or schedule')
        if self_past_chunk_size < 1:
            raise ValueError('self_past_chunk_size must be positive')
        self.variant, self.task = self.VARIANT, task
        self.shape_meta = _plain(shape_meta)
        self.obs_key_shapes = {k: tuple(v['shape']) for k, v in self.shape_meta['obs'].items()}
        action_shape = self.shape_meta['action']['shape']
        if len(action_shape) != 1:
            raise ValueError('Action schema must be a vector')
        self.action_dim = int(action_shape[0])
        self.horizon, self.past_n = int(horizon), int(past_n)
        self.n_obs_steps, self.n_action_steps = int(n_obs_steps), int(n_action_steps)
        self.obs_feature_dim = int(embed_dim)
        self._tokenizer_config = _plain(tokenizer_config)
        self._tokenizer_metadata = _plain(tokenizer_metadata) or {}
        register_new_resolvers()
        if action_tokenizer is None:
            if construction_mode == 'restore':
                if not self._tokenizer_config:
                    raise ValueError('Offline restore requires embedded tokenizer_config')
                action_tokenizer = hydra.utils.instantiate(self._tokenizer_config)
            else:
                if not tokenizer_checkpoint or not Path(tokenizer_checkpoint).is_file():
                    raise FileNotFoundError(f'Frozen OAT checkpoint not found: {tokenizer_checkpoint}')
                with open(tokenizer_checkpoint, 'rb') as stream:
                    payload = torch.load(stream, map_location='cpu', pickle_module=dill, weights_only=False)
                self._tokenizer_config = _plain(payload['cfg'].tokenizer)
                action_tokenizer = hydra.utils.instantiate(self._tokenizer_config)
                if 'ema_model' not in payload['state_dicts']:
                    raise ValueError('The selected frozen OAT must contain EMA weights')
                action_tokenizer.load_state_dict(payload['state_dicts']['ema_model'], strict=True)
                self._tokenizer_metadata = {
                    'source': str(tokenizer_checkpoint), 'sha256': file_sha256(tokenizer_checkpoint),
                    'weights': 'ema',
                    'training_task': _plain(payload['cfg'].get('task', {})),
                    'normalizer_source': 'frozen_tokenizer_checkpoint',
                }
        self.action_tokenizer = action_tokenizer.requires_grad_(False).eval()
        decoder = action_tokenizer.decoder
        if decoder.sample_dim != self.action_dim or decoder.sample_horizon != self.horizon:
            raise ValueError('Tokenizer action dimension/horizon does not match the task schema')
        self.max_seq_len = int(action_tokenizer.latent_horizon)
        self.bos_id = int(action_tokenizer.quantizer.codebook_size)
        if self.max_seq_len < 1 or self.bos_id < 2:
            raise ValueError('Invalid tokenizer latent horizon or codebook size')
        if obs_encoder_type != 'original_fused' or context_layout != CONTEXT_LAYOUT:
            raise ValueError('This policy requires original_fused / original_fused_v1')
        if context_schema_version != CONTEXT_SCHEMA_VERSION:
            raise ValueError('Original fused observations require context_schema_version=2')
        if self.max_seq_len != expected_action_tokens:
            raise ValueError(f'OAT latent horizon must equal expected_action_tokens={expected_action_tokens}')
        if obs_encoder is None:
            if obs_encoder_config is not None:
                enc_cfg = _plain(obs_encoder_config)
                if enc_cfg.pop('_target_', None) != 'oat.perception.original_fused_obs_adapter.OriginalFusedObservationAdapter':
                    raise ValueError('Embedded observation encoder must be OriginalFusedObservationAdapter')
                obs_encoder = OriginalFusedObservationAdapter(**enc_cfg)
            else:
                obs_encoder = OriginalFusedObservationAdapter(
                    shape_meta=self.shape_meta, n_obs_steps=n_obs_steps,
                    embed_dim=embed_dim, original_obs_config=original_obs_config)
        if not isinstance(obs_encoder, OriginalFusedObservationAdapter):
            raise TypeError('Original fused policy requires OriginalFusedObservationAdapter')
        resolved_obs = obs_encoder.export_config()['original_obs_config']
        if original_obs_config is not None:
            for key, value in _plain(original_obs_config).items():
                if resolved_obs.get(key) != value:
                    raise ValueError(f'Conflicting original_obs_config.{key}')
        self.original_obs_config = copy.deepcopy(resolved_obs)
        self.obs_encoder_type = obs_encoder_type
        self.context_layout = context_layout
        self.context_schema_version = context_schema_version
        self.expected_action_tokens = expected_action_tokens
        if obs_encoder.output_feature_dim() != embed_dim:
            raise ValueError('Observation encoder width must equal action decoder width')
        if obs_encoder.n_obs_steps != self.n_obs_steps:
            raise ValueError('Observation encoder and policy observation windows must match')
        if (_plain(obs_encoder.shape_meta) != self.shape_meta
                or _observation_order(obs_encoder.shape_meta) != _observation_order(self.shape_meta)):
            raise ValueError('Observation encoder and policy input schemas must match')
        self.obs_encoder = obs_encoder
        self.modalities = obs_encoder.modalities()
        self.obs_ports = list(obs_encoder.rgb_ports) + list(obs_encoder.state_ports)
        self.action_normalizer = LinearNormalizer()
        for key in ['action', *obs_encoder.state_ports]:
            self.action_normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        self.action_normalizer.requires_grad_(False)
        self.raw_proj = nn.Linear(self.action_dim, embed_dim)
        self.acc_proj = nn.Linear(self.action_dim, embed_dim)
        self.jerk_proj = nn.Linear(self.action_dim, embed_dim)
        # Separate difference identities; no unused gate/type row in the base policy.
        self.type_embedding = nn.Parameter(torch.empty(4, embed_dim))
        self.action_time_embedding = nn.Parameter(torch.empty(past_n, embed_dim))
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.action_time_embedding, std=0.02)
        self.model = ModernAutoregressiveModel(
            vocab_size=self.bos_id + 1, max_seq_len=self.max_seq_len,
            n_layer=n_layers, n_emb=embed_dim, n_head=n_heads, ffn_dim=ffn_dim,
            dropout=dropout, activation_checkpointing=activation_checkpointing,
        )
        self.temperature, self.topk = temperature, topk
        self.self_past_p = float(self_past_p)
        self.self_past_warmup_steps = int(self_past_warmup_steps)
        self.self_past_ramp_steps = int(self_past_ramp_steps)
        self.self_past_chunk_size = int(self_past_chunk_size)
        self.self_past_temperature = temperature if self_past_temperature is None else self_past_temperature
        self.self_past_topk = topk if self_past_topk is None else self_past_topk
        self.self_past_schedule = self_past_schedule
        self.register_buffer('_self_past_optimizer_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('_context_schema', torch.tensor(self.CONTEXT_SCHEMA_VERSION))
        self.register_buffer('_variant_code', torch.tensor(int(self.requires_state_history)))
        self._construction = dict(
            shape_meta=self.shape_meta, variant=self.variant, task=task,
            n_action_steps=n_action_steps, n_obs_steps=n_obs_steps, past_n=past_n, horizon=horizon,
            embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout,
            temperature=temperature, topk=topk, activation_checkpointing=activation_checkpointing,
            self_past_p=self_past_p, self_past_warmup_steps=self_past_warmup_steps,
            self_past_ramp_steps=self_past_ramp_steps, self_past_chunk_size=self_past_chunk_size,
            self_past_temperature=self.self_past_temperature, self_past_topk=self.self_past_topk,
            self_past_schedule=self_past_schedule,
        )
        self._construction.update(
            original_obs_config=copy.deepcopy(self.original_obs_config),
            obs_encoder_type=self.obs_encoder_type, context_layout=self.context_layout,
            context_schema_version=self.context_schema_version,
            expected_action_tokens=self.expected_action_tokens)
        self.reset()

    def _freeze_normalizers(self):
        for module in self.modules():
            if isinstance(module, (LinearNormalizer, SingleFieldLinearNormalizer)):
                module.requires_grad_(False)
        self.action_tokenizer.requires_grad_(False).eval()

    def set_normalizer(self, normalizer):
        if isinstance(normalizer, (list, tuple)):
            if len(normalizer) != 1:
                raise ValueError('Expected one task normalizer')
            normalizer = normalizer[0]
        required = ['action', *self.obs_encoder.rgb_ports, *self.obs_encoder.state_ports]
        missing = [key for key in required if key not in normalizer.params_dict]
        if missing:
            raise KeyError(f'Missing training-set normalizer fields: {missing}')
        self.action_normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)
        self._freeze_normalizers()

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        self._freeze_normalizers()
        # Dynamic normalizer fields bypass ordinary Module strict-key checks.
        validate_normalizer_fields(self.action_normalizer, {
            'action': self.action_dim,
            **{key: self.obs_key_shapes[key][0] for key in self.obs_encoder.state_ports}})
        validate_normalizer_fields(self.action_tokenizer.normalizer, {'action': self.action_dim})
        return result

    def build_context(self, obs, past_actions, past_action_valid):
        past, valid = self._safe_past(past_actions, past_action_valid)
        # RGB and state normalization happens exactly once, inside the original encoder.
        fused = self.obs_encoder(obs) + self.type_embedding[0]
        if fused.shape[0] != past.shape[0]:
            raise ValueError('Observation and command-history batch sizes differ')
        raw = self.raw_proj(past) + self.action_time_embedding + self.type_embedding[1]
        diff_valid = torch.stack((valid[:, -2:].all(1), valid[:, -3:].all(1)), dim=1)
        first = past[:, -1] - past[:, -2]
        second = past[:, -1] - 2 * past[:, -2] + past[:, -3]
        first = torch.where(diff_valid[:, :1], first, torch.zeros_like(first))
        second = torch.where(diff_valid[:, 1:], second, torch.zeros_like(second))
        diff = torch.stack((self.acc_proj(first) + self.type_embedding[2],
                            self.jerk_proj(second) + self.type_embedding[3]), dim=1)
        memory = torch.cat((fused, raw, diff), dim=1)
        # Dataset observation padding repeats the anchor frame, just as in the original.
        observed = torch.ones(past.shape[0], self.n_obs_steps, device=past.device, dtype=torch.bool)
        visible = torch.cat((observed, valid, diff_valid), dim=1)
        segments = torch.cat([torch.full((count,), int(kind), dtype=torch.long, device=memory.device)
                              for count, kind in ((self.n_obs_steps, Segment.FUSED_OBSERVATION),
                                                  (self.past_n, Segment.RAW_ACTION),
                                                  (2, Segment.ACTION_DIFF))])
        memory = torch.where(visible[..., None], memory, torch.zeros_like(memory))
        return ContextBatch(memory, visible, segments).validate()

    @contextlib.contextmanager
    def _rollout_mode(self):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            yield
        finally:
            for module, mode in modes:
                module.training = mode
            self.action_tokenizer.eval()

    def get_optimizer(self, policy_lr=5e-5, obs_enc_lr=1e-5, weight_decay=0.01, betas=(0.9, 0.95)):
        self._freeze_normalizers()
        original_ids = {id(p) for p in self.obs_encoder.fused_encoder.parameters()}
        groups, seen = {}, set()
        for parameter in self.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            lr = obs_enc_lr if id(parameter) in original_ids else policy_lr
            decay = weight_decay if parameter.ndim >= 2 else 0.
            groups.setdefault((lr, decay), []).append(parameter)
        return torch.optim.AdamW([{'params': params, 'lr': lr, 'weight_decay': decay}
                                  for (lr, decay), params in groups.items()], betas=tuple(betas))

    def get_policy_name(self):
        return f'{self.variant}_original_fused_{self.task}'

    def parameter_counts(self):
        result = super().parameter_counts()
        components = {'original_observation_encoder': self.obs_encoder.fused_encoder,
                      'observation_projection': self.obs_encoder.obs_projection,
                      'action_model': self.model, 'tokenizer': self.action_tokenizer}
        if self.requires_state_history:
            components.update(history=self.history_encoder, gate=self.history_gate)
        result['components'] = {name: {'total': sum(p.numel() for p in module.parameters()),
                                      'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)}
                                for name, module in components.items()}
        return result

    def artifact_metadata(self):
        root = Path(__file__).resolve().parents[2]
        files = ['oat/policy/p2n_new_original_obs.py', 'oat/policy/p2n_new_common.py',
                 'oat/policy/past2next_state_history_gate_real_robot.py',
                 'oat/model/state_action_history.py', 'oat/model/common/normalizer.py',
                 'oat/model/common/dict_of_tensor_mixin.py',
                 'oat/model/common/context_batch.py', 'oat/model/common/original_fused_context_batch.py',
                 'oat/model/autoregressive/modern_transformer_cache.py',
                 'oat/perception/original_fused_obs_adapter.py', 'oat/perception/fused_obs_encoder.py',
                 'oat/perception/robomimic_vision_encoder.py', 'oat/perception/state_encoder.py',
                 'oat/perception/crop_randomizer.py', 'oat/workspace/train_p2n_new_original_obs.py']
        versions = {}
        for package in ('torch', 'torchvision', 'robomimic', 'accelerate'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        encoder = self.obs_encoder.export_config()
        return dict(variant=self.variant, policy_target=f'{type(self).__module__}.{type(self).__name__}',
                    task=self.task, context_schema=self.CONTEXT_SCHEMA_VERSION,
                    context_schema_version=self.CONTEXT_SCHEMA_VERSION, context_layout=self.context_layout,
                    obs_encoder_type=self.obs_encoder_type, encoder_type=self.obs_encoder_type,
                    execution_protocol='acknowledged_commands_v1',
                    segments={kind.name: int(kind) for kind in Segment},
                    shape_meta=copy.deepcopy(self.shape_meta), tokenizer=copy.deepcopy(self._tokenizer_metadata),
                    vocab_size=self.bos_id + 1, latent_horizon=self.max_seq_len,
                    horizon=self.horizon, n_action_steps=self.n_action_steps,
                    parameters=self.parameter_counts(), software=versions,
                    self_past_optimizer_updates=self.self_past_step,
                    architecture=self.export_config(), observation_encoder=encoder,
                    observation_encoder_contract=self.obs_encoder.export_metadata(),
                    observation_tokens=self.n_obs_steps,
                    context_tokens=self.n_obs_steps + self.past_n + 2 + getattr(self, 'history_summary_tokens', 0),
                    fused_feature_dim=self.obs_encoder.fused_feature_dim,
                    observation_pool='mean_fused' if self.requires_state_history else None,
                    gate_bias='history_summary_only' if self.requires_state_history else None,
                    initialization='Original ResNet18 random initialization; modern AR normal(0,.02); frozen OAT EMA',
                    source_sha256={name: file_sha256(root / name) for name in files})

    @classmethod
    def from_checkpoint(cls, checkpoint, output_dir=None, return_configuration=False,
                        weights=None, policy_overrides=None):
        payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill, weights_only=False)
        register_new_resolvers()
        cfg = OmegaConf.create(_plain(payload['cfg']))
        config = OmegaConf.create(payload.get('policy_config', cfg.policy))
        if config.get('variant') != cls.VARIANT:
            raise ValueError(f'Checkpoint variant must be {cls.VARIANT}')
        target = f'{cls.__module__}.{cls.__name__}'
        if (config.get('_target_') != target or config.get('obs_encoder_type') != 'original_fused'
                or config.get('context_layout') != CONTEXT_LAYOUT or config.get('context_schema_version') != 2):
            raise ValueError('Checkpoint architecture/schema is not this original-fused policy')
        if policy_overrides:
            # Only generation controls and unused initialization paths may change at deployment.
            mutable = {'temperature', 'topk', 'self_past_chunk_size', 'tokenizer_checkpoint'}
            for key, value in policy_overrides.items():
                if key not in mutable and _plain(config.get(key)) != _plain(value):
                    raise ValueError(f'Checkpoint architecture/schema cannot be overridden: {key}')
                if key in ('shape_meta', 'obs_encoder_config'):
                    saved_meta = config[key] if key == 'shape_meta' else config[key]['shape_meta']
                    requested_meta = value if key == 'shape_meta' else value['shape_meta']
                    if _observation_order(saved_meta) != _observation_order(requested_meta):
                        raise ValueError(f'Checkpoint observation field order cannot be overridden: {key}')
            config = OmegaConf.merge(config, policy_overrides)
        config.construction_mode = 'restore'
        if weights is None:
            weights = 'ema' if cfg.get('training', {}).get('use_ema', False) else 'model'
        if weights not in ('ema', 'model'):
            raise ValueError('weights must be ema or model')
        key = 'ema_model' if weights == 'ema' else 'model'
        if key not in payload['state_dicts']:
            raise ValueError(f'Checkpoint does not contain {key}')
        policy = hydra.utils.instantiate(config)
        policy.load_state_dict(payload['state_dicts'][key], strict=True)
        policy.eval()
        policy.reset()
        cfg.policy = config
        return (policy, cfg) if return_configuration else policy


class P2NStateGateNewOriginalObsPolicy(P2NNewOriginalObsPolicy):
    VARIANT = 'p2n_state_gate_new'
    requires_state_history = True
    supports_history_summary_gate = True
    def __init__(self, *args, state_history_steps=8, state_history_keys=None,
                 history_embed_dim=128, history_n_heads=4, history_n_layers=2,
                 history_summary_tokens=4, history_dropout=0.1,
                 history_gate_mode='learned', history_gate_hidden_dim=128,
                 history_gate_init=0.9, rotation_6d_layout='rows', **kwargs):
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
        self.observation_pool = nn.Identity()
        self.summary_type_embedding = nn.Parameter(torch.empty(1, 1, self.obs_feature_dim))
        nn.init.normal_(self.summary_type_embedding, std=0.02)
        self.history_gate = nn.Sequential(
            nn.LayerNorm(2 * self.obs_feature_dim + 1),
            nn.Linear(2 * self.obs_feature_dim + 1, history_gate_hidden_dim), nn.GELU(),
            nn.Linear(history_gate_hidden_dim, 1),
        )
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

    def build_context(self, obs, past_actions, past_action_valid):
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
        base = super().build_context(obs, past_actions, valid)
        summaries = self.history_encoder(
            {key: obs['state_history__' + key] for key in self.state_history_keys},
            state_valid, past, self.action_normalizer,
        ) + self.summary_type_embedding
        observed = self.observation_pool(
            base.memory[:, base.segment_ids == int(Segment.FUSED_OBSERVATION)].mean(dim=1))
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
        summaries = torch.where(summary_valid[..., None], summaries, torch.zeros_like(summaries))
        return ContextBatch(
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
