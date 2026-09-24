"""OAT code-space flow policies. No AR modules are constructed.

State, normalization, optimizer grouping and execution acknowledgement reuse the
existing tested helpers. All flow-specific paths live in new files.
"""
from __future__ import annotations
import copy
import contextlib
import importlib.metadata
from numbers import Integral
from pathlib import Path
import dill
import hydra
import torch
from torch import nn
from omegaconf import OmegaConf
from oat.common.hydra_util import register_new_resolvers
from oat.common.latent_flow_batch import PreparedFlowBatch
from oat.model.common.latent_flow_context import FlowContextBatch
from oat.model.common.context_batch import Segment
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.flow.ditx_latent import DiTXLatentFlow
from oat.model.flow.consistency_flow import (sample_flow_schedule, interpolate_latents,
    consistency_velocity_target, packed_flow_loss)
from oat.model.flow.euler_sampler import euler_sample
from oat.perception.latent_flow_token_obs_encoder import FlowTokenObservationEncoder
from oat.policy.base_policy import BasePolicy
from oat.policy.p2n_new_common import P2NNewCommonPolicy, _plain, file_sha256, bool_mask
from oat.tokenizer.oat.latent_adapter import FrozenOATLatentAdapter

FLOW_DEFAULTS = dict(latent_space='fsq_normalized_codes', num_slots=8, code_dim=5,
    levels=[8, 5, 5, 5, 5], fm_fraction=0.75, ct_weight=1.0, fm_beta=[1.0, 1.5],
    fm_time_scale=0.999, ct_time_bins=10, teacher_dt_mode='same_relative_dt',
    solver='euler', inference_steps=8, self_past_steps=8,
    endpoint_projection='fsq_nearest_grid')

def validate_flow_config(value):
    value = _plain(value) or {}
    unknown = set(value) - set(FLOW_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown flow configuration fields: {sorted(unknown)}')
    config = {**copy.deepcopy(FLOW_DEFAULTS), **value}
    for key in ('num_slots','code_dim','levels','latent_space','fm_fraction','ct_weight','fm_beta','fm_time_scale',
                'ct_time_bins','teacher_dt_mode','solver','endpoint_projection'):
        if config[key] != FLOW_DEFAULTS[key]:
            raise ValueError(f'This recipe requires flow.{key}={FLOW_DEFAULTS[key]!r}')
    for key in ('num_slots','code_dim','inference_steps','self_past_steps'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'flow.{key} must be a positive integer')
    if len(config['levels']) != config['code_dim']:
        raise ValueError('FSQ levels must match code_dim')
    return config

class P2NLatentFlowCommonPolicy(P2NNewCommonPolicy):
    VARIANT = 'p2n_latent_flow'
    policy_family = 'oat_latent_flow'
    supports_flow_loss_preparation = True
    supports_flow_sampling = True
    ARTIFACT_SCHEMA_VERSION = 1

    def __init__(
        self, shape_meta, obs_encoder=None, action_tokenizer=None,
        n_action_steps=8, n_obs_steps=2, past_n=7, horizon=16,
        embed_dim=768, n_layers=16, n_heads=12, ffn_dim=2048, dropout=0.0,
        variant=None, task='libero', flow=None,
        obs_encoder_type='dinov3', resnet_config=None,
        initialization='xavier_uniform_slot_normal_0.02',
        construction_mode='fresh', dino_path=None, dino_revision=None,
        dino_config=None, processor_config=None, tokenizer_checkpoint=None,
        tokenizer_config=None, tokenizer_metadata=None, obs_encoder_config=None,
        rgb_range='uint8', image_brightness=0.1, image_contrast=0.1,
        num_visual_queries=64, resampler_depth=2, activation_checkpointing=True,
        self_past_p=0.5, self_past_warmup_steps=1000, self_past_ramp_steps=4000,
        self_past_chunk_size=2,
        self_past_schedule='optimizer_step',
    ):
        BasePolicy.__init__(self)
        if dropout != 0.0:
            raise ValueError("Flow consistency training requires dropout=0.0")
        if initialization != 'xavier_uniform_slot_normal_0.02':
            raise ValueError('Unsupported flow initialization')
        if obs_encoder_type not in ('dinov3', 'resnet18'):
            raise ValueError('obs_encoder_type must be dinov3 or resnet18')
        if obs_encoder_type == 'dinov3' and resnet_config is not None:
            raise ValueError('ResNet configuration cannot be used with DINO')
        if obs_encoder_type == 'resnet18' and any(value is not None for value in
                (dino_path, dino_revision, dino_config, processor_config)):
            raise ValueError('ResNet flow does not use DINO sources or preprocessing')
        self.obs_encoder_type = obs_encoder_type
        self.flow = validate_flow_config(flow)
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
        self.latent_adapter = FrozenOATLatentAdapter(
            action_tokenizer, levels=self.flow['levels'], num_slots=self.flow['num_slots'],
            action_horizon=self.horizon, action_dim=self.action_dim)
        decoder = action_tokenizer.decoder
        if decoder.sample_dim != self.action_dim or decoder.sample_horizon != self.horizon:
            raise ValueError('Tokenizer action dimension/horizon does not match the task schema')
        self.max_seq_len = int(action_tokenizer.latent_horizon)
        if obs_encoder_type == 'resnet18':
            from oat.perception.latent_flow_resnet_obs_encoder import FlowResNetObservationEncoder
            encoder_class = FlowResNetObservationEncoder
            if obs_encoder is None:
                if obs_encoder_config is not None:
                    enc_cfg = _plain(obs_encoder_config)
                    enc_cfg.pop('_target_', None)
                    obs_encoder = FlowResNetObservationEncoder(**enc_cfg)
                else:
                    obs_encoder = FlowResNetObservationEncoder(
                        shape_meta=self.shape_meta, n_obs_steps=n_obs_steps, n_emb=embed_dim,
                        **(_plain(resnet_config) or {}))
        else:
            encoder_class = FlowTokenObservationEncoder
            if obs_encoder is None:
                if obs_encoder_config is not None:
                    enc_cfg = _plain(obs_encoder_config)
                    enc_cfg.pop('_target_', None)
                    if construction_mode == 'restore':
                        enc_cfg['dino']['load_mode'] = 'restore'
                        enc_cfg['dino'].pop('pretrained_path', None)
                    obs_encoder = FlowTokenObservationEncoder(**enc_cfg)
                else:
                    obs_encoder = FlowTokenObservationEncoder(
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
        if not isinstance(obs_encoder, encoder_class):
            raise TypeError(f'obs_encoder must match obs_encoder_type={obs_encoder_type}')
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
        for projection in (self.raw_proj, self.acc_proj, self.jerk_proj):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        # Separate difference identities; no unused gate/type row in the base policy.
        self.type_embedding = nn.Parameter(torch.empty(5, embed_dim))
        self.action_time_embedding = nn.Parameter(torch.empty(past_n, embed_dim))
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.action_time_embedding, std=0.02)
        self.model = DiTXLatentFlow(
            code_dim=self.flow['code_dim'], num_slots=self.flow['num_slots'],
            current_state_dim=n_obs_steps * sum(obs_encoder.state_shapes.values()),
            embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads,
            ffn_dim=ffn_dim, dropout=dropout, activation_checkpointing=activation_checkpointing,
            initialization=initialization)
        self.self_past_p = float(self_past_p)
        self.self_past_warmup_steps = int(self_past_warmup_steps)
        self.self_past_ramp_steps = int(self_past_ramp_steps)
        self.self_past_chunk_size = int(self_past_chunk_size)
        self.self_past_schedule = self_past_schedule
        self.register_buffer('_self_past_optimizer_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('_context_schema', torch.tensor(self.CONTEXT_SCHEMA_VERSION))
        self.register_buffer('_variant_code', torch.tensor(int(self.requires_state_history)))
        self.register_buffer('_flow_artifact_schema', torch.tensor(self.ARTIFACT_SCHEMA_VERSION))
        # Existing DINO state dictionaries remain unchanged and load strictly.
        # The ResNet-only marker rejects cross-encoder loads before any tensors mutate.
        if obs_encoder_type == 'resnet18':
            self.register_buffer('_resnet18_obs_schema', torch.tensor(1))
        self._construction = dict(
            shape_meta=self.shape_meta, variant=self.variant, task=task,
            obs_encoder_type=obs_encoder_type,
            n_action_steps=n_action_steps, n_obs_steps=n_obs_steps, past_n=past_n, horizon=horizon,
            embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout,
            activation_checkpointing=activation_checkpointing, flow=copy.deepcopy(self.flow),
            initialization=initialization,
            self_past_p=self_past_p, self_past_warmup_steps=self_past_warmup_steps,
            self_past_ramp_steps=self_past_ramp_steps, self_past_chunk_size=self_past_chunk_size,
            self_past_schedule=self_past_schedule)
        self.reset()

    @property
    def action_tokenizer(self):
        # One registered owner; no duplicated codec keys or optimizer parameters.
        return self.latent_adapter.tokenizer

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        saved_encoder = 'resnet18' if '_resnet18_obs_schema' in state_dict else 'dinov3'
        if saved_encoder != self.obs_encoder_type:
            raise ValueError('Checkpoint observation encoder does not match this policy')
        if saved_encoder == 'resnet18' and int(state_dict['_resnet18_obs_schema']) != 1:
            raise ValueError('Unsupported ResNet observation schema')
        if int(state_dict.get('_flow_artifact_schema', -1)) != self.ARTIFACT_SCHEMA_VERSION:
            raise ValueError('Checkpoint is not a compatible oat_latent_flow artifact')
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def set_normalizer(self, normalizer):
        super().set_normalizer(normalizer)
        if self.obs_encoder_type == 'resnet18':
            # Policy normalizes current state once; the ResNet adapter owns RGB normalization.
            self.obs_encoder.set_normalizer(normalizer[0] if isinstance(normalizer, (list, tuple)) else normalizer)

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
            if self.obs_encoder_type == 'dinov3':
                self.obs_encoder.dino_encoder.backbone.eval()

    @torch.no_grad()
    def prepare_visual_conditioning(self, obs):
        if self.obs_encoder_type == 'dinov3':
            return {'frozen_patches': self.obs_encoder.extract_frozen_patches(obs).detach()}
        return {'prepared_visual': self.obs_encoder.prepare_conditioning(obs).detach()}

    def get_optimizer(self, policy_lr=5e-5, obs_enc_lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95)):
        # ResNet keeps the original state head inside ProjectionStateEncoder;
        # parameter identity handles both encoders without renaming checkpoints.
        state_parameters = {id(value) for value in self.obs_encoder.state_projection.parameters()}
        groups, seen = {}, set()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            visual = name.startswith('obs_encoder.') and id(parameter) not in state_parameters
            lr = obs_enc_lr if visual else policy_lr
            decay = weight_decay if parameter.ndim >= 2 else 0.
            groups.setdefault((lr, decay), []).append(parameter)
        return torch.optim.AdamW([{'params': params, 'lr': lr, 'weight_decay': decay}
                                  for (lr, decay), params in groups.items()], betas=tuple(betas))

    def get_policy_name(self):
        if self.obs_encoder_type == 'dinov3':
            return super().get_policy_name()
        return f'{self.variant}_resnet18_{self.task}'

    def normalized_observation(self, obs):
        normalized = dict(obs)
        for key in self.obs_encoder.state_ports:
            if not torch.isfinite(obs[key]).all():
                raise ValueError(f'Current observation {key!r} must be finite')
            normalized[key] = self.action_normalizer[key].normalize(obs[key].float())
        return normalized

    def current_state_features(self, obs):
        return self.obs_encoder.state_features(self.normalized_observation(obs)).flatten(1)

    def build_context(self, obs, past_actions, past_action_valid, frozen_patches=None, prepared_visual=None):
        past, valid = self._safe_past(past_actions, past_action_valid)
        normalized_obs = dict(obs)
        for key in self.obs_encoder.state_ports:
            if key not in obs:
                raise KeyError(f'Missing observation port {key!r}')
            if not torch.isfinite(obs[key]).all():
                raise ValueError(f'Current observation {key!r} must be finite')
            normalized_obs[key] = self.action_normalizer[key].normalize(obs[key])
        if self.obs_encoder_type == 'dinov3':
            if prepared_visual is not None:
                raise ValueError('DINO conditioning requires frozen patches, not prepared ResNet images')
            visual, proprio = self.obs_encoder(normalized_obs, frozen_patches=frozen_patches)
        else:
            if frozen_patches is not None:
                raise ValueError('ResNet conditioning requires images, not frozen feature tensors')
            visual, proprio = self.obs_encoder(normalized_obs, prepared_visual=prepared_visual)
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
        return FlowContextBatch(memory, visible, segments).validate()

    def forward(self, prepared):
        if not isinstance(prepared, PreparedFlowBatch):
            raise TypeError('Flow forward requires prepare_flow_training_batch output, including EMA targets')
        prepared.validate()
        if prepared.obs_encoder_type != self.obs_encoder_type:
            raise ValueError('Prepared batch observation encoder does not match student')
        context = self.build_context(prepared.obs, prepared.past_actions,
                                     prepared.past_action_valid, prepared.frozen_patches, prepared.prepared_visual)
        context.validate_variant(self.variant)
        velocity = self.model(prepared.noisy_latents, time=prepared.time,
            step_size=prepared.step_size, context=context,
            current_state=self.current_state_features(prepared.obs))
        losses = packed_flow_loss(velocity, prepared.velocity_targets,
                                 prepared.fm_indices, prepared.ct_indices)
        self._last_flow_losses = {k: v.detach() for k, v in losses.items()}
        return losses['loss']

    def prepare_training_batch(self, batch, teacher, generator, self_past_generator=None):
        return prepare_flow_training_batch(batch, student=self, teacher=teacher,
            generator=generator, self_past_generator=self_past_generator)

    @torch.no_grad()
    def _generate_latents(self, obs, past, valid, *, num_flow_steps=None, generator=None,
                          initial_noise=None, frozen_patches=None, prepared_visual=None):
        context = self.build_context(obs, past, valid, frozen_patches, prepared_visual)
        context.validate_variant(self.variant)
        z = euler_sample(self.model, context=context,
            current_state=self.current_state_features(obs),
            num_steps=self.flow['inference_steps'] if num_flow_steps is None else num_flow_steps,
            generator=generator, initial_noise=initial_noise)
        return z, self.latent_adapter.snap_codes(z)

    @torch.no_grad()
    def _generate_actions(self, obs, past, valid, *, num_flow_steps=None, generator=None,
                          initial_noise=None):
        _, grid = self._generate_latents(obs, past, valid,
            num_flow_steps=num_flow_steps, generator=generator, initial_noise=initial_noise)
        return self.latent_adapter.decode_grid_codes(grid)

    @torch.no_grad()
    def _maybe_self_past(self, batch, past, probability=None, generator=None):
        probability = self.self_past_probability() if probability is None else probability
        if not 0 <= probability <= 1:
            raise ValueError('Self-past probability must be in [0,1]')
        if probability == 0:
            return past.detach()
        required = ('prev_obs','prev_past_action','prev_past_action_valid','prev_window_valid')
        if any(key not in batch for key in required):
            raise KeyError('Self-past requires previous-window observations and validity metadata')
        windows = bool_mask(batch['prev_window_valid'].reshape(-1), (past.shape[0],),
                            'prev_window_valid', past.device)
        selected = windows.clone()
        if probability < 1:
            selected &= torch.rand(past.shape[0], device=past.device, generator=generator) < probability
        indices = selected.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return past.detach()
        generated = past.detach().clone()
        # Exclude invalid windows before gate encoding (their states can all be padding).
        with self._rollout_mode(), self._clean_autocast_cache():
            for index in indices.split(self.self_past_chunk_size):
                previous = batch['prev_past_action'][index]
                prediction = self._generate_actions(
                    {key: value[index] for key, value in batch['prev_obs'].items()},
                    previous, batch['prev_past_action_valid'][index],
                    num_flow_steps=self.flow['self_past_steps'], generator=generator)
                chunk = torch.cat((previous, prediction[:, :self.n_action_steps]), dim=1)[:, -self.past_n:]
                # no_grad rather than inference_mode keeps tensors safe for later autograd.
                generated[index] = chunk.detach().clone().to(past.dtype)
        valid = bool_mask(batch['past_action_valid'], past.shape[:2], 'past_action_valid', past.device)
        return torch.where((selected[:, None] & valid)[..., None], generated, past).detach()

    @torch.no_grad()
    def predict_action(self, obs_dict, past_actions=None, past_action_valid=None,
                       num_flow_steps=None, generator=None):
        if (past_actions is None) != (past_action_valid is None):
            raise ValueError('Explicit past_actions and bool past_action_valid must be provided together')
        stateful = past_actions is None
        if stateful:
            if self._pending_execution_steps is not None:
                raise RuntimeError('Execution feedback is pending; call record_executed_actions or reset')
            batch = obs_dict[self.obs_ports[0]].shape[0]
            expected = (batch, self.past_n, self.action_dim)
            if self._past_buffer is None:
                self._past_buffer = torch.zeros(expected, device=self.device, dtype=torch.float32)
                self._past_valid_buffer = torch.zeros(expected[:2], device=self.device, dtype=torch.bool)
            elif tuple(self._past_buffer.shape) != expected:
                raise RuntimeError('Execution-history batch size changed; call reset first')
            self._past_buffer = self._past_buffer.to(self.device)
            self._past_valid_buffer = self._past_valid_buffer.to(self.device)
            past_actions, past_action_valid = self._past_buffer, self._past_valid_buffer
        if past_action_valid.dtype != torch.bool:
            raise ValueError('past_action_valid must be bool')
        with self._rollout_mode(), self._clean_autocast_cache():
            prediction = self._generate_actions(obs_dict, past_actions, past_action_valid,
                num_flow_steps=num_flow_steps, generator=generator)
        action = prediction[:, :self.n_action_steps]
        if stateful:
            self._pending_execution_steps = action.shape[1]
        return {'action': action, 'action_pred': prediction}

    @torch.no_grad()
    def _validation_generated_past(self, batch, past, row_generators):
        """Generate every valid history in one batch with independent sample noise.

        Training keeps its bounded self-past chunks. Validation is bounded by the
        validation loader batch instead, including the previous-window forwards.
        """
        required = ('prev_obs', 'prev_past_action', 'prev_past_action_valid', 'prev_window_valid')
        if any(key not in batch for key in required):
            raise KeyError('Self-past requires previous-window observations and validity metadata')
        selected = bool_mask(batch['prev_window_valid'].reshape(-1), (past.shape[0],),
                             'prev_window_valid', past.device)
        indices = selected.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return past.detach()
        # Match each old batch-one Euler draw after its FM/time/current-Euler draws.
        # No Bernoulli draw occurs: generated-history validation has probability 1.
        noise_shape = (1, self.flow['num_slots'], self.flow['code_dim'])
        initial_noise = torch.cat([
            torch.randn(noise_shape, device=past.device, dtype=torch.float32,
                        generator=row_generators[row])
            for row in indices.tolist()
        ], dim=0)
        previous = batch['prev_past_action'][indices]
        # Exclude invalid previous windows before the gated encoder sees padding.
        prediction = self._generate_actions(
            {key: value[indices] for key, value in batch['prev_obs'].items()},
            previous, batch['prev_past_action_valid'][indices],
            num_flow_steps=self.flow['self_past_steps'], initial_noise=initial_noise)
        chunk = torch.cat((previous, prediction[:, :self.n_action_steps]), dim=1)[:, -self.past_n:]
        generated = past.detach().clone()
        generated[indices] = chunk.detach().to(past.dtype)
        valid = bool_mask(batch['past_action_valid'], past.shape[:2], 'past_action_valid', past.device)
        return torch.where((selected[:, None] & valid)[..., None], generated, past).detach()

    @torch.no_grad()
    def validation_metrics(self, batch, generator=None, history_mode='expert', compute_decoded=True,
                           *, sample_seeds=None):
        """Return metric sums/counts; sample seeds make loader batching reproducible.

        The generator-only API retains its existing batched RNG stream. With
        sample_seeds, random tensors match independent batch-one evaluations,
        while all model, observation-encoder and tokenizer calls stay batched.
        """
        if history_mode not in ('expert','generated'):
            raise ValueError('Validation history must be expert or generated')
        b = len(batch['action'])
        if b == 0:
            raise ValueError('Validation requires at least one sample')
        device = batch['action'].device
        shape = (b, self.flow['num_slots'], self.flow['code_dim'])
        row_generators = None
        if sample_seeds is not None:
            if generator is not None:
                raise ValueError('Provide either generator or sample_seeds for validation, not both')
            seeds = list(sample_seeds)
            if len(seeds) != b:
                raise ValueError('sample_seeds must contain one integer seed per validation sample')
            if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds):
                raise ValueError('sample_seeds must contain integer seeds')
            row_generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
            row_fm_noise, row_times, row_sampling_noise = [], [], []
            for row_generator in row_generators:
                row_shape = (1, self.flow['num_slots'], self.flow['code_dim'])
                row_fm_noise.append(torch.randn(row_shape, device=device,
                    generator=row_generator, dtype=torch.float32))
                row_times.append(0.999 * (1 - torch.rand(1, device=device,
                    generator=row_generator).pow(2 / 3)))
                row_sampling_noise.append(torch.randn(row_shape, device=device,
                    generator=row_generator, dtype=torch.float32))
            fm_noise = torch.cat(row_fm_noise, dim=0)
            t = torch.cat(row_times, dim=0)
            sampling_noise = torch.cat(row_sampling_noise, dim=0)
        else:
            # Draw current-window randomness before optional history generation so both
            # history evaluations receive identical FM times/noise and Euler noise.
            fm_noise = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
            t = 0.999 * (1 - torch.rand(b, device=device, generator=generator).pow(2 / 3))
            sampling_noise = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        with self._rollout_mode(), self._clean_autocast_cache():
            past = batch['past_action']
            if history_mode == 'generated':
                if row_generators is None:
                    past = self._maybe_self_past(batch, past, probability=1., generator=generator)
                else:
                    past = self._validation_generated_past(batch, past, row_generators)
            visual_inputs = self.prepare_visual_conditioning(batch['obs'])
            context = self.build_context(batch['obs'], past, batch['past_action_valid'], **visual_inputs)
            state = self.current_state_features(batch['obs'])
            target = self.latent_adapter.encode_actions(batch['action']).codes
            velocity = self.model(interpolate_latents(target, fm_noise, t), time=t,
                step_size=torch.zeros_like(t), context=context, current_state=state).float()
            fm_error = (velocity - (target - fm_noise)).square()
            if not compute_decoded:
                return {'fm_loss': {'sum': fm_error.double().sum(),
                    'count': torch.tensor(fm_error.numel(), device=device, dtype=torch.float64)}}
            continuous = euler_sample(self.model, context=context, current_state=state,
                num_steps=self.flow['inference_steps'], initial_noise=sampling_noise)
            grid = self.latent_adapter.snap_codes(continuous)
            prediction = self.latent_adapter.decode_grid_codes(grid)
            reconstruction = self.latent_adapter.decode_grid_codes(target)
        valid = batch['future_action_valid']
        if valid.dtype != torch.bool or valid.shape != batch['action'].shape[:2]:
            raise ValueError('future_action_valid must be bool [B,horizon]')
        def ratio(values, mask=None):
            if mask is not None:
                mask = mask.expand_as(values)
                values = torch.where(mask, values, torch.zeros_like(values))
                count = mask.sum().double()
            else:
                count = torch.tensor(values.numel(), device=device, dtype=torch.float64)
            return {'sum': values.double().sum(), 'count': count}
        error = (prediction.float() - batch['action'].float()).square()
        ae = (reconstruction.float() - batch['action'].float()).square()
        levels = torch.tensor(self.flow['levels'], device=device, dtype=torch.float32)
        half = (levels // 2)
        upper = (levels - 1 - half) / half
        bounds = (continuous < -1) | (continuous > upper)
        legal = torch.isfinite(grid) & (grid >= -1) & (grid <= upper) & ((grid * half) == (grid * half).round())
        mask = valid[..., None]
        return {'fm_loss': ratio(fm_error), 'decoded_action_mse': ratio(error, mask),
            'translation_mse': ratio(error[..., :3], mask),
            'rotation_mse': ratio(error[..., 3:6], mask),
            'gripper_mse': ratio(error[..., 6:], mask),
            'oat_autoencoding_mse': ratio(ae, mask),
            'projection_distance': ratio((continuous - grid).square()),
            'projection_out_of_bounds': ratio(bounds.float()),
            'projection_legal': ratio(legal.float())}

    def artifact_metadata(self):
        versions = {}
        for package in ('torch','torchvision','transformers','accelerate','robomimic'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        directory = Path(__file__).resolve().parents[1]
        sources = [Path(__file__), directory/'policy/p2n_state_gate_latent_flow.py',
                   directory/'perception/latent_flow_token_obs_encoder.py',
                   directory/'tokenizer/oat/latent_adapter.py',
                   directory/'model/flow/ditx_latent.py', directory/'model/flow/consistency_flow.py',
                   directory/'model/flow/euler_sampler.py',
                   directory/'policy/p2n_new_common.py', directory/'model/state_action_history.py',
                   directory/'policy/past2next_state_history_gate_real_robot.py',
                   directory/'perception/visual_resampler.py', directory/'perception/dinov3_patch_encoder.py',
                   directory/'model/common/latent_flow_context.py', directory/'common/latent_flow_batch.py',
                   directory/'tokenizer/oat/tokenizer.py', directory/'tokenizer/oat/quantizer/fsq.py']
        if self.obs_encoder_type == 'resnet18':
            sources.extend([directory/'perception/latent_flow_resnet_obs_encoder.py',
                            directory/'perception/robomimic_vision_encoder.py',
                            directory/'perception/state_encoder.py', directory/'perception/crop_randomizer.py'])
        return dict(policy_family=self.policy_family, artifact_schema_version=self.ARTIFACT_SCHEMA_VERSION,
            obs_encoder_type=self.obs_encoder_type,
            observation_conditioning=('frozen_dino_patches_shared' if self.obs_encoder_type == 'dinov3'
                                      else 'shared_normalized_crops_independent_trainable_resnet18'),
            variant=self.variant, task=self.task, task_type=self.task,
            context_schema=self.CONTEXT_SCHEMA_VERSION, shape_meta=copy.deepcopy(self.shape_meta),
            execution_protocol='acknowledged_commands_v1',
            tokenizer={**copy.deepcopy(self._tokenizer_metadata), **self.latent_adapter.metadata()},
            latent_horizon=self.max_seq_len, horizon=self.horizon, n_action_steps=self.n_action_steps,
            flow=copy.deepcopy(self.flow), architecture=self.export_config(),
            initialization=self.model.initialization,
            time_embedding=dict(sinusoidal_dim=256, hidden_dim=1024, independent_t_dt=True),
            upstream_reference=dict(repository='https://github.com/geyan21/ManiFlow_Policy',
                commit='ef2f116f1f90163ed36e657b8c5503740bb468af'),
            parameters=self.parameter_counts(), software=versions,
            self_past_optimizer_updates=self.self_past_step,
            noise_seed_strategy='sha256(dataset_identity,sample_id,validation_seed); advancing_online_generator',
            source_sha256={str(p.relative_to(directory.parent)): file_sha256(p) for p in sources})

    @classmethod
    def from_checkpoint(cls, checkpoint, output_dir=None, return_configuration=False,
                        weights=None, policy_overrides=None):
        with open(checkpoint, 'rb') as stream:
            payload = torch.load(stream, map_location='cpu', pickle_module=dill)
        metadata = payload.get('metadata', {})
        if metadata.get('policy_family') != cls.policy_family or metadata.get('variant') != cls.VARIANT:
            raise ValueError('Checkpoint family/variant does not match this flow policy')
        if metadata.get('artifact_schema_version') != cls.ARTIFACT_SCHEMA_VERSION:
            raise ValueError('Unsupported flow artifact schema')
        register_new_resolvers()
        cfg = OmegaConf.create(_plain(payload['cfg']))
        config = OmegaConf.create(payload['policy_config'])
        if metadata.get('obs_encoder_type', 'dinov3') != config.get('obs_encoder_type', 'dinov3'):
            raise ValueError('Artifact observation encoder metadata and architecture disagree')
        if policy_overrides:
            raise ValueError('Flow artifacts restore their exact architecture; set num_flow_steps in predict_action')
        config.construction_mode = 'restore'
        if config.get('variant') != cls.VARIANT:
            raise ValueError('Policy configuration and artifact variant disagree')
        weights = 'ema' if weights is None else weights
        if weights not in ('ema','model'):
            raise ValueError('weights must be ema or model')
        key = 'ema_model' if weights == 'ema' else 'model'
        policy = hydra.utils.instantiate(config)
        if not isinstance(policy, cls):
            raise ValueError('Artifact policy target does not match requested class')
        policy.load_state_dict(payload['state_dicts'][key], strict=True)
        policy.eval().reset()
        cfg.policy = config
        return (policy, cfg) if return_configuration else policy


@torch.no_grad()
def prepare_flow_training_batch(batch, *, student, teacher, generator, self_past_generator=None):
    if teacher is student or teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Consistency requires an independent eval/no-grad full EMA policy')
    if teacher.variant != student.variant or teacher.obs_encoder_type != student.obs_encoder_type:
        raise ValueError('Teacher and student variants and observation encoders must match')
    past = batch['past_action']
    if batch['past_action_valid'].dtype != torch.bool:
        raise ValueError('Training past_action_valid must be bool')
    schedule = sample_flow_schedule(len(past), device=past.device, generator=generator)
    past = student._maybe_self_past(batch, past,
        generator=generator if self_past_generator is None else self_past_generator)
    targets = student.latent_adapter.encode_actions(batch['action']).codes
    noise = torch.randn(targets.shape, device=targets.device, dtype=torch.float32, generator=generator)
    visual_inputs = student.prepare_visual_conditioning(batch['obs'])
    zt = interpolate_latents(targets, noise, schedule.time)
    velocity_targets = targets - noise
    ct = schedule.ct_indices
    def teacher_velocity(z_next, time_next, original_dt, selected_rows):
        rows = ct[selected_rows]
        obs = {key: value[rows] for key, value in batch['obs'].items()}
        context = teacher.build_context(obs, past[rows], batch['past_action_valid'][rows],
            **{key: value[rows] for key, value in visual_inputs.items()})
        context.validate_variant(teacher.variant)
        return teacher.model(z_next, time=time_next, step_size=original_dt,
                             context=context, current_state=teacher.current_state_features(obs)).float()
    velocity_targets[ct] = consistency_velocity_target(
        targets[ct], noise[ct], schedule.time[ct], schedule.step_size[ct], teacher_velocity)
    return PreparedFlowBatch(zt.detach(), schedule.time, schedule.step_size,
        velocity_targets.detach(), schedule.fm_indices, schedule.ct_indices,
        {key: value.detach() for key, value in batch['obs'].items()}, past.detach(),
        batch['past_action_valid'], obs_encoder_type=student.obs_encoder_type, **visual_inputs).validate()
