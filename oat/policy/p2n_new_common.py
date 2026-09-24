"""Shared DINOv3 Past2Next policy; old policies and their contracts are untouched."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
from pathlib import Path

import dill
import hydra
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from oat.common.hydra_util import register_new_resolvers
from oat.model.common.context_batch import ContextBatch, Segment
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.autoregressive.modern_transformer_cache import ModernAutoregressiveModel
from oat.perception.token_obs_encoder import TokenObservationEncoder
from oat.policy.base_policy import BasePolicy


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bool_mask(value, shape, name, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}')
    if value.device != device:
        raise ValueError(f'{name} must be on {device}')
    if value.dtype != torch.bool:
        if not ((value == 0) | (value == 1)).all():
            raise ValueError(f'{name} must contain boolean or exact zero/one flags')
        value = value.bool()
    return value


class P2NNewCommonPolicy(BasePolicy):
    """One context path for training, self-past, offline and acknowledged rollout."""

    supports_explicit_past_actions = True
    supports_explicit_past_action_valid = True
    supports_generated_history_validation = True
    requires_execution_acknowledgement = True
    requires_state_history = False
    supports_history_summary_gate = False
    VARIANT = 'p2n_new'
    CONTEXT_SCHEMA_VERSION = 1

    def __init__(
        self, shape_meta, obs_encoder=None, action_tokenizer=None,
        n_action_steps=8, n_obs_steps=2, past_n=7, horizon=16,
        embed_dim=768, n_layers=16, n_heads=12, ffn_dim=2048, dropout=0.1,
        temperature=1.0, topk=10, variant=None, task='libero',
        construction_mode='fresh', dino_path=None, dino_revision=None,
        dino_config=None, processor_config=None, tokenizer_checkpoint=None,
        tokenizer_config=None, tokenizer_metadata=None, obs_encoder_config=None,
        rgb_range='uint8', image_brightness=0.1, image_contrast=0.1,
        num_visual_queries=64, resampler_depth=2, activation_checkpointing=False,
        self_past_p=0.5, self_past_warmup_steps=1000, self_past_ramp_steps=4000,
        self_past_chunk_size=4, self_past_temperature=None, self_past_topk=None,
        self_past_schedule='optimizer_step',
    ):
        super().__init__()
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
                    payload = torch.load(stream, map_location='cpu', pickle_module=dill)
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
        if obs_encoder is None:
            if obs_encoder_config is not None:
                enc_cfg = _plain(obs_encoder_config)
                enc_cfg.pop('_target_', None)
                if construction_mode == 'restore':
                    enc_cfg['dino']['load_mode'] = 'restore'
                    enc_cfg['dino'].pop('pretrained_path', None)
                obs_encoder = TokenObservationEncoder(**enc_cfg)
            else:
                obs_encoder = TokenObservationEncoder(
                    shape_meta=self.shape_meta, n_obs_steps=n_obs_steps, n_emb=embed_dim,
                    n_head=n_heads, ffn_dim=ffn_dim, dropout=dropout,
                    num_queries=num_visual_queries, resampler_depth=resampler_depth,
                    activation_checkpointing=activation_checkpointing,
                    dino=dict(pretrained_path=dino_path, load_mode=construction_mode,
                              config=_plain(dino_config), processor_config=_plain(processor_config),
                              revision=dino_revision, rgb_range=rgb_range,
                              brightness=image_brightness, contrast=image_contrast),
                )
        if obs_encoder.output_feature_dim() != embed_dim:
            raise ValueError('Observation encoder width must equal action decoder width')
        if obs_encoder.n_obs_steps != self.n_obs_steps:
            raise ValueError('Observation encoder and policy observation windows must match')
        if _plain(obs_encoder.shape_meta) != self.shape_meta:
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
        self.type_embedding = nn.Parameter(torch.empty(5, embed_dim))
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
        self.reset()

    def train(self, mode=True):
        super().train(mode)
        self.action_tokenizer.eval()
        return self

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if not strict:
            raise ValueError('New policy checkpoints require strict loading')
        if '_variant_code' not in state_dict or int(state_dict['_variant_code']) != int(self.requires_state_history):
            raise ValueError('Checkpoint variant does not match this policy')
        if int(state_dict.get('_context_schema', -1)) != self.CONTEXT_SCHEMA_VERSION:
            raise ValueError('Checkpoint ContextBatch schema does not match')
        result = super().load_state_dict(state_dict, strict=True, **kwargs)
        self.action_tokenizer.requires_grad_(False).eval()
        self.reset()
        return result

    def set_normalizer(self, normalizer):
        if isinstance(normalizer, (list, tuple)):
            if len(normalizer) != 1:
                raise ValueError('Expected one task normalizer')
            normalizer = normalizer[0]
        required = ['action', *self.obs_encoder.state_ports]
        missing = [k for k in required if k not in normalizer.params_dict]
        if missing:
            raise KeyError(f'Missing training-set normalizer fields: {missing}')
        self.action_normalizer.load_state_dict(normalizer.state_dict())
        self.action_normalizer.requires_grad_(False)
        # The tokenizer retains its checkpoint's own action normalizer.

    def _safe_past(self, past_actions, past_action_valid):
        if (not isinstance(past_actions, torch.Tensor) or past_actions.ndim != 3
                or tuple(past_actions.shape[1:]) != (self.past_n, self.action_dim)
                or not past_actions.is_floating_point()):
            raise ValueError(f'past_actions must be floating [B,{self.past_n},{self.action_dim}]')
        valid = bool_mask(past_action_valid, past_actions.shape[:2], 'past_action_valid', past_actions.device)
        if not torch.isfinite(past_actions[valid]).all():
            raise ValueError('Valid historical commands must be finite')
        safe = torch.where(valid[..., None], past_actions, torch.zeros_like(past_actions))
        normalized = self.action_normalizer['action'].normalize(safe)
        return torch.where(valid[..., None], normalized, torch.zeros_like(normalized)), valid

    def build_context(self, obs, past_actions, past_action_valid):
        past, valid = self._safe_past(past_actions, past_action_valid)
        normalized_obs = dict(obs)
        for key in self.obs_encoder.state_ports:
            if key not in obs:
                raise KeyError(f'Missing observation port {key!r}')
            if not torch.isfinite(obs[key]).all():
                raise ValueError(f'Current observation {key!r} must be finite')
            normalized_obs[key] = self.action_normalizer[key].normalize(obs[key])
        visual, proprio = self.obs_encoder(normalized_obs)
        if visual.shape[0] != past.shape[0]:
            raise ValueError('Observation and command-history batch sizes differ')
        raw = self.raw_proj(past) + self.action_time_embedding + self.type_embedding[2]
        diff_valid = torch.stack((valid[:, -2:].all(1), valid[:, -3:].all(1)), dim=1)
        first = past[:, -1] - past[:, -2]
        second = past[:, -1] - 2 * past[:, -2] + past[:, -3]
        first = torch.where(diff_valid[:, :1], first, torch.zeros_like(first))
        second = torch.where(diff_valid[:, 1:], second, torch.zeros_like(second))
        diff = torch.stack((self.acc_proj(first) + self.type_embedding[3],
                            self.jerk_proj(second) + self.type_embedding[4]), dim=1)
        visual = visual + self.type_embedding[0]
        proprio = proprio + self.type_embedding[1]
        memory = torch.cat((visual, proprio, raw, diff), dim=1)
        observed = torch.ones(memory.shape[0], visual.shape[1] + proprio.shape[1],
                              dtype=torch.bool, device=memory.device)
        visible = torch.cat((observed, valid, diff_valid), dim=1)
        segments = torch.cat([torch.full((length,), int(kind), dtype=torch.long, device=memory.device)
                              for length, kind in ((visual.shape[1], Segment.VISUAL),
                                                   (proprio.shape[1], Segment.PROPRIO),
                                                   (self.past_n, Segment.RAW_ACTION),
                                                   (2, Segment.ACTION_DIFF))])
        memory = torch.where(visible[..., None], memory, torch.zeros_like(memory))
        return ContextBatch(memory, visible, segments).validate()

    @property
    def self_past_step(self):
        return int(self._self_past_optimizer_step.item())

    def set_self_past_step(self, step):
        if step < 0:
            raise ValueError('Optimizer update count must be nonnegative')
        self._self_past_optimizer_step.fill_(step)

    def on_optimizer_step(self):
        if self.training:
            self._self_past_optimizer_step.add_(1)

    def self_past_probability(self):
        progress = self.self_past_step - self.self_past_warmup_steps
        if progress < 0:
            return 0.0
        return self.self_past_p * (min(1., progress / self.self_past_ramp_steps)
                                   if self.self_past_ramp_steps else 1.)

    @contextlib.contextmanager
    def _rollout_mode(self):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            yield
        finally:
            # Direct restoration preserves intentionally mixed train/eval modes.
            for module, mode in modes:
                module.training = mode
            self.action_tokenizer.eval()
            self.obs_encoder.dino_encoder.backbone.eval()

    @contextlib.contextmanager
    def _clean_autocast_cache(self):
        torch.clear_autocast_cache()
        try:
            yield
        finally:
            torch.clear_autocast_cache()

    def _generate_actions(self, obs, past, valid, n_tokens, temperature, topk):
        context = self.build_context(obs, past, valid)
        tokens = self.model.generate(context, max_new_tokens=n_tokens,
                                     temperature=temperature, top_k=topk)
        return self.action_tokenizer.detokenize(tokens=tokens)

    def _maybe_self_past(self, batch, past, probability=None):
        probability = self.self_past_probability() if probability is None else probability
        if probability <= 0:
            return past
        required = ['prev_obs', 'prev_past_action', 'prev_past_action_valid', 'prev_window_valid']
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f'Self-past needs previous-window metadata: {missing}')
        valid_windows = bool_mask(batch['prev_window_valid'].reshape(-1), (past.shape[0],),
                                  'prev_window_valid', past.device)
        indices = valid_windows.nonzero(as_tuple=True)[0]
        if not len(indices):
            return past
        generated = past.detach().clone()
        # Previous-window generation finishes before the current training graph exists.
        with self._rollout_mode(), self._clean_autocast_cache():
            for index in indices.split(self.self_past_chunk_size):
                with torch.inference_mode():
                    previous = batch['prev_past_action'][index]
                    pred = self._generate_actions(
                        {key: value[index] for key, value in batch['prev_obs'].items()},
                        previous, batch['prev_past_action_valid'][index], self.max_seq_len,
                        self.self_past_temperature, self.self_past_topk,
                    )
                    chunk = torch.cat((previous, pred[:, :self.n_action_steps]), dim=1)[:, -self.past_n:]
                with torch.inference_mode(False):
                    chunk = chunk.detach().clone().to(dtype=past.dtype)
                generated[index] = chunk
        mask = bool_mask(batch['past_action_valid'], past.shape[:2], 'past_action_valid', past.device)
        use_self = valid_windows[:, None] & mask
        if probability < 1:
            use_self = use_self & (torch.rand(past.shape[0], 1, device=past.device) < probability)
        return torch.where(use_self[..., None], generated, past)

    def forward(self, batch, history_mode=None):
        history_mode = ('configured' if self.training else 'expert') if history_mode is None else history_mode
        if history_mode not in ('expert', 'generated', 'configured'):
            raise ValueError('history_mode must be expert, generated, or configured')
        if 'past_action_valid' not in batch:
            raise KeyError('New policies require top-level past_action_valid')
        past = batch['past_action']
        if history_mode != 'expert':
            past = self._maybe_self_past(batch, past, 1. if history_mode == 'generated' else None)
        with torch.no_grad():
            targets = self.action_tokenizer.tokenize(batch['action']).long()
        if targets.shape != (past.shape[0], self.max_seq_len):
            raise ValueError('Tokenizer returned an unexpected latent shape')
        context = self.build_context(batch['obs'], past, batch['past_action_valid'])
        bos = torch.full((past.shape[0], 1), self.bos_id, dtype=torch.long, device=past.device)
        logits = self.model(torch.cat((bos, targets[:, :-1]), dim=1), context)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    def reset(self):
        self._past_buffer = None
        self._past_valid_buffer = None
        self._pending_execution_steps = None

    @torch.no_grad()
    def predict_action(self, obs_dict, use_k_tokens=None, temperature=None, topk=None,
                       past_actions=None, past_action_valid=None):
        if (past_actions is None) != (past_action_valid is None):
            raise ValueError('Explicit past_actions and past_action_valid must be provided together')
        stateful = past_actions is None
        if stateful:
            if self._pending_execution_steps is not None:
                raise RuntimeError('Execution feedback is pending; call record_executed_actions or reset')
            batch = obs_dict[self.obs_ports[0]].shape[0]
            expected = (batch, self.past_n, self.action_dim)
            if self._past_buffer is None:
                self._past_buffer = torch.zeros(expected, device=self.device, dtype=self.dtype)
                self._past_valid_buffer = torch.zeros(expected[:2], device=self.device, dtype=torch.bool)
            elif tuple(self._past_buffer.shape) != expected:
                raise RuntimeError('Execution-history batch size changed; call reset first')
            self._past_buffer = self._past_buffer.to(device=self.device, dtype=self.dtype)
            self._past_valid_buffer = self._past_valid_buffer.to(device=self.device)
            past_actions, past_action_valid = self._past_buffer, self._past_valid_buffer
        n_tokens = self.max_seq_len if use_k_tokens is None else use_k_tokens
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or not 1 <= n_tokens <= self.max_seq_len:
            raise ValueError(f'use_k_tokens must be in [1,{self.max_seq_len}]')
        with self._rollout_mode(), self._clean_autocast_cache():
            prediction = self._generate_actions(
                obs_dict, past_actions, past_action_valid, n_tokens,
                self.temperature if temperature is None else temperature,
                self.topk if topk is None else topk,
            )
        action = prediction[:, :self.n_action_steps]
        if stateful:
            self._pending_execution_steps = action.shape[1]
        return {'action': action, 'action_pred': prediction}

    def record_executed_actions(self, actions, executed_lengths=None):
        if self._pending_execution_steps is None or self._past_buffer is None:
            raise RuntimeError('No prediction is awaiting execution feedback')
        commands = torch.as_tensor(actions, device=self._past_buffer.device)
        batch = self._past_buffer.shape[0]
        if commands.ndim != 3 or commands.shape[0] != batch or commands.shape[2] != self.action_dim:
            raise ValueError('actions must have shape (batch,steps,action_dim)')
        if commands.dtype == torch.bool or commands.is_complex():
            raise ValueError('Executed commands must be real numbers')
        steps = commands.shape[1]
        if steps > self._pending_execution_steps:
            raise ValueError('Acknowledged chunk exceeds the predicted execution horizon')
        lengths = torch.full((batch,), steps, device=commands.device, dtype=torch.long) if executed_lengths is None else torch.as_tensor(executed_lengths, device=commands.device)
        if lengths.shape != (batch,) or lengths.dtype == torch.bool or lengths.is_complex():
            raise ValueError('executed_lengths must be an integer vector of shape (batch,)')
        if lengths.is_floating_point() and (not torch.isfinite(lengths).all() or not torch.equal(lengths, lengths.floor())):
            raise ValueError('executed_lengths must contain finite integer counts')
        if (lengths < 0).any() or (lengths > steps).any():
            raise ValueError('executed_lengths must lie within the supplied chunk')
        executed = torch.arange(steps, device=commands.device)[None, :] < lengths[:, None]
        if not torch.isfinite(commands[executed]).all():
            raise ValueError('Executed commands must be finite')
        with torch.inference_mode(False), torch.no_grad():
            history = self._past_buffer.detach().clone()
            valid = self._past_valid_buffer.detach().clone()
            for i, count in enumerate(lengths.long().cpu().tolist()):
                if count:
                    history[i] = torch.cat((history[i], commands[i, :count].to(history.dtype)), dim=0)[-self.past_n:]
                    valid[i] = torch.cat((valid[i], valid.new_ones(count)))[-self.past_n:]
        self._past_buffer, self._past_valid_buffer = history, valid
        self._pending_execution_steps = None

    def get_optimizer(self, policy_lr=5e-5, obs_enc_lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95)):
        groups, seen = {}, set()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            visual = name.startswith('obs_encoder.') and not name.startswith('obs_encoder.state_projection.')
            lr = obs_enc_lr if visual else policy_lr
            decay = weight_decay if parameter.ndim >= 2 else 0.
            groups.setdefault((lr, decay), []).append(parameter)
        return torch.optim.AdamW([{'params': params, 'lr': lr, 'weight_decay': decay}
                                  for (lr, decay), params in groups.items()], betas=tuple(betas))

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities

    def get_observation_ports(self):
        return list(self.obs_ports)

    def get_policy_name(self):
        return f'{self.variant}_dinov3_s16_{self.task}'

    def create_dummy_observation(self, batch_size=1, device=None):
        device = self.device if device is None else device
        obs = {}
        for key in self.obs_ports:
            shape = (batch_size, self.n_obs_steps, *self.obs_key_shapes[key])
            obs[key] = torch.zeros(shape, device=device,
                                   dtype=torch.uint8 if key in self.obs_encoder.rgb_ports else self.dtype)
            if key.endswith('_quat'):
                obs[key][..., 3] = 1
            elif key.endswith('_rot6d'):
                obs[key][...] = obs[key].new_tensor([1, 0, 0, 0, 1, 0])
        return obs

    def parameter_counts(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'total': trainable + frozen, 'trainable': trainable, 'frozen': frozen}

    def export_config(self):
        if not self._tokenizer_config:
            raise ValueError('Export requires tokenizer_config for injected tokenizers')
        return dict(_target_=f'{type(self).__module__}.{type(self).__name__}', _recursive_=False,
                    **copy.deepcopy(self._construction), construction_mode='restore',
                    obs_encoder_config=self.obs_encoder.export_config(),
                    tokenizer_config=copy.deepcopy(self._tokenizer_config),
                    tokenizer_metadata=copy.deepcopy(self._tokenizer_metadata))

    def artifact_metadata(self):
        versions = {}
        for package in ('torch', 'torchvision', 'transformers', 'accelerate'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return dict(variant=self.variant, policy_target=f'{type(self).__module__}.{type(self).__name__}',
                    task=self.task, context_schema=self.CONTEXT_SCHEMA_VERSION,
                    execution_protocol='acknowledged_commands_v1',
                    segments={kind.name: int(kind) for kind in Segment},
                    shape_meta=copy.deepcopy(self.shape_meta), tokenizer=copy.deepcopy(self._tokenizer_metadata),
                    vocab_size=self.bos_id + 1, latent_horizon=self.max_seq_len,
                    horizon=self.horizon, n_action_steps=self.n_action_steps,
                    parameters=self.parameter_counts(), software=versions,
                    self_past_optimizer_updates=self.self_past_step,
                    architecture=self.export_config(),
                    initialization='AR normal(0,.02), residual outputs .02/sqrt(3*depth)',
                    source_sha256={str(path.relative_to(Path(__file__).resolve().parents[2])): file_sha256(path)
                                   for path in [Path(__file__).resolve(),
                                                Path(__file__).with_name('p2n_new.py'),
                                                Path(__file__).with_name('p2n_state_gate_new.py'),
                                                Path(__file__).resolve().parents[1] / 'model/autoregressive/modern_transformer_cache.py',
                                                Path(__file__).resolve().parents[1] / 'model/common/context_batch.py',
                                                Path(__file__).resolve().parents[1] / 'perception/token_obs_encoder.py',
                                                Path(__file__).resolve().parents[1] / 'perception/dinov3_patch_encoder.py',
                                                Path(__file__).resolve().parents[1] / 'perception/visual_resampler.py']})

    @classmethod
    def from_checkpoint(cls, checkpoint, output_dir=None, return_configuration=False,
                        weights=None, policy_overrides=None):
        with open(checkpoint, 'rb') as stream:
            payload = torch.load(stream, map_location='cpu', pickle_module=dill)
        register_new_resolvers()
        cfg = OmegaConf.create(_plain(payload['cfg']))
        config = OmegaConf.create(payload.get('policy_config', cfg.policy))
        if config.get('variant') != cls.VARIANT:
            raise ValueError(f'Checkpoint variant must be {cls.VARIANT}')
        if policy_overrides:
            protected = {'_target_', 'variant', 'task', 'shape_meta', 'n_action_steps', 'n_obs_steps',
                         'past_n', 'horizon', 'embed_dim', 'n_layers', 'n_heads', 'ffn_dim',
                         'obs_encoder_config', 'tokenizer_config', 'state_history_steps',
                         'state_history_keys', 'history_summary_tokens', 'history_embed_dim',
                         'history_n_heads', 'history_n_layers', 'rotation_6d_layout'}
            for key in protected & set(policy_overrides):
                if _plain(config.get(key)) != _plain(policy_overrides[key]):
                    raise ValueError(f'Checkpoint architecture/schema cannot be overridden: {key}')
            config = OmegaConf.merge(config, policy_overrides)
        if config.get('variant') != cls.VARIANT:
            raise ValueError('Overrides cannot change checkpoint variant')
        config.construction_mode = 'restore'
        if weights is None:
            weights = 'ema' if cfg.get('training', {}).get('use_ema', False) else 'model'
        if weights not in ('ema', 'model'):
            raise ValueError('weights must be ema or model')
        key = 'ema_model' if weights == 'ema' else 'model'
        if key not in payload['state_dicts']:
            raise ValueError(f'Checkpoint does not contain {key}')
        policy = hydra.utils.instantiate(config)
        if not isinstance(policy, cls) or policy.variant != cls.VARIANT:
            raise ValueError('Checkpoint target and variant disagree')
        policy.load_state_dict(payload['state_dicts'][key], strict=True)
        policy.eval()
        policy.reset()
        cfg.policy = config
        return (policy, cfg) if return_configuration else policy
