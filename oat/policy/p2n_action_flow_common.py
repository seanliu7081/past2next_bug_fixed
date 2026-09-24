"""Direct continuous-action flow with acknowledged Past2Next history.

No tokenizer, quantizer, or autoregressive decoder is constructed.
"""
from __future__ import annotations
import copy
import contextlib
import importlib.metadata
from pathlib import Path
import dill
import hydra
import torch
from torch import nn
from omegaconf import OmegaConf
from oat.common.hydra_util import register_new_resolvers
from oat.common.action_flow_batch import PreparedActionFlowBatch
from oat.model.common.action_flow_context import ActionFlowContextBatch
from oat.model.common.context_batch import Segment
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.flow.ditx_action import DiTXActionVectorField
from oat.model.flow.consistency_flow import (sample_flow_schedule, interpolate_latents,
    consistency_velocity_target, packed_flow_loss)
from oat.model.flow.action_euler_sampler import euler_sample
from oat.perception.latent_flow_token_obs_encoder import FlowTokenObservationEncoder
from oat.policy.base_policy import BasePolicy
from oat.policy.p2n_new_common import P2NNewCommonPolicy, _plain, file_sha256, bool_mask

FLOW_DEFAULTS = dict(space='normalized_actions', ffn_type='gelu', ffn_hidden_dim=3072,
    self_qk_norm=False, cross_qk_norm='layernorm', qkv_bias=True, time_embed_dim=128,
    global_condition='time_and_step_only', pre_norm_modality=False,
    fm_fraction=0.75, ct_weight=1.0, fm_beta=[1.0, 1.5], fm_time_scale=0.999,
    ct_time_bins=10, teacher_dt_mode='same_relative_dt',
    loss_padding='supervise_edge_repeated_targets', solver='euler',
    inference_steps=8, self_past_steps=8, output_transform='action_unnormalize')


def validate_flow_config(value):
    value = _plain(value) or {}
    unknown = set(value) - set(FLOW_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown continuous_action_flow configuration fields: {sorted(unknown)}')
    config = {**copy.deepcopy(FLOW_DEFAULTS), **value}
    flexible = {'ffn_hidden_dim', 'time_embed_dim', 'inference_steps', 'self_past_steps'}
    for key in set(FLOW_DEFAULTS) - flexible:
        if config[key] != FLOW_DEFAULTS[key]:
            raise ValueError(f'This recipe requires flow.{key}={FLOW_DEFAULTS[key]!r}')
    for key in flexible:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'flow.{key} must be a positive integer')
    return config

class P2NActionFlowCommonPolicy(BasePolicy):
    VARIANT = 'p2n_action_flow'
    policy_family = 'continuous_action_flow'
    supports_flow_loss_preparation = True
    supports_flow_sampling = True
    ARTIFACT_SCHEMA_VERSION = 1
    CONTEXT_SCHEMA_VERSION = 1
    supports_explicit_past_actions = True
    supports_explicit_past_action_valid = True
    supports_generated_history_validation = True
    requires_execution_acknowledgement = True
    requires_state_history = False
    supports_history_summary_gate = False

    # Reuse only codec-independent contracts, without invoking an AR constructor.
    _safe_past = P2NNewCommonPolicy._safe_past
    self_past_step = P2NNewCommonPolicy.self_past_step
    set_self_past_step = P2NNewCommonPolicy.set_self_past_step
    on_optimizer_step = P2NNewCommonPolicy.on_optimizer_step
    self_past_probability = P2NNewCommonPolicy.self_past_probability
    _clean_autocast_cache = P2NNewCommonPolicy._clean_autocast_cache
    record_executed_actions = P2NNewCommonPolicy.record_executed_actions
    get_optimizer = P2NNewCommonPolicy.get_optimizer
    get_observation_encoder = P2NNewCommonPolicy.get_observation_encoder
    get_observation_modalities = P2NNewCommonPolicy.get_observation_modalities
    get_observation_ports = P2NNewCommonPolicy.get_observation_ports
    get_policy_name = P2NNewCommonPolicy.get_policy_name
    create_dummy_observation = P2NNewCommonPolicy.create_dummy_observation
    parameter_counts = P2NNewCommonPolicy.parameter_counts

    def __init__(
        self, shape_meta, obs_encoder=None, action_dim=7,
        n_action_steps=8, n_obs_steps=2, past_n=7, horizon=16,
        embed_dim=768, n_layers=16, n_heads=12, resampler_ffn_dim=2048,
        resampler_ffn_type="swiglu", dropout=0.0,
        variant=None, task='libero', flow=None,
        initialization='xavier_uniform_position_normal_0.02',
        construction_mode='fresh', dino_path=None, dino_revision=None,
        dino_config=None, processor_config=None, obs_encoder_config=None,
        action_schema=None, normalizer_metadata=None, online_seed=42,
        rgb_range='uint8', image_brightness=0.1, image_contrast=0.1,
        num_visual_queries=64, resampler_depth=2, activation_checkpointing=True,
        self_past_p=0.5, self_past_warmup_steps=1000, self_past_ramp_steps=4000,
        self_past_chunk_size=2,
        self_past_schedule='optimizer_step',
    ):
        BasePolicy.__init__(self)
        if dropout != 0.0:
            raise ValueError("Flow consistency training requires dropout=0.0")
        if initialization != 'xavier_uniform_position_normal_0.02':
            raise ValueError('Unsupported flow initialization')
        if resampler_ffn_type != 'swiglu':
            raise ValueError('The visual Resampler requires swiglu')
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
        if action_dim != self.action_dim or self.action_dim != 7:
            raise ValueError('Direct action schema must specify seven matching action dimensions')
        self.action_schema = _plain(action_schema) or dict(
            translation='position_delta', rotation='rotation_vector_delta',
            gripper='absolute_command', execution_anchor='current_observation_t',
            units='task_schema', control_frame='task_schema', control_frequency_hz=None)
        self.normalizer_metadata = _plain(normalizer_metadata) or {}
        self.online_seed = int(online_seed)
        self.horizon, self.past_n = int(horizon), int(past_n)
        self.n_obs_steps, self.n_action_steps = int(n_obs_steps), int(n_action_steps)
        self.obs_feature_dim = int(embed_dim)
        register_new_resolvers()
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
                    n_head=n_heads, ffn_dim=resampler_ffn_dim, dropout=dropout,
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
        if not isinstance(obs_encoder, FlowTokenObservationEncoder):
            raise TypeError('Flow policies require FlowTokenObservationEncoder with frozen-patch inputs')
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
        self.model = DiTXActionVectorField(
            action_dim=self.action_dim, horizon=self.horizon, embed_dim=embed_dim,
            n_layers=n_layers, n_heads=n_heads, ffn_hidden_dim=self.flow['ffn_hidden_dim'],
            time_embed_dim=self.flow['time_embed_dim'], dropout=dropout,
            activation_checkpointing=activation_checkpointing, variant=self.variant)
        self.self_past_p = float(self_past_p)
        self.self_past_warmup_steps = int(self_past_warmup_steps)
        self.self_past_ramp_steps = int(self_past_ramp_steps)
        self.self_past_chunk_size = int(self_past_chunk_size)
        self.self_past_schedule = self_past_schedule
        self.register_buffer('_self_past_optimizer_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('_context_schema', torch.tensor(self.CONTEXT_SCHEMA_VERSION))
        self.register_buffer('_variant_code', torch.tensor(int(self.requires_state_history)))
        self.register_buffer('_action_flow_artifact_schema', torch.tensor(self.ARTIFACT_SCHEMA_VERSION))
        self.register_buffer('_normalizer_ready', torch.tensor(False))
        self._construction = dict(
            shape_meta=self.shape_meta, variant=self.variant, task=task,
            n_action_steps=n_action_steps, n_obs_steps=n_obs_steps, past_n=past_n, horizon=horizon,
            embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads,
            resampler_ffn_dim=resampler_ffn_dim, resampler_ffn_type=resampler_ffn_type, dropout=dropout,
            action_dim=action_dim, action_schema=self.action_schema, online_seed=self.online_seed,
            activation_checkpointing=activation_checkpointing, flow=copy.deepcopy(self.flow),
            initialization=initialization,
            self_past_p=self_past_p, self_past_warmup_steps=self_past_warmup_steps,
            self_past_ramp_steps=self_past_ramp_steps, self_past_chunk_size=self_past_chunk_size,
            self_past_schedule=self_past_schedule)
        self.reset()

    def train(self, mode=True):
        super().train(mode)
        self.obs_encoder.dino_encoder.backbone.eval()
        return self

    def reset(self):
        self._past_buffer = None
        self._past_valid_buffer = None
        self._pending_execution_steps = None
        self._online_generator = None

    @contextlib.contextmanager
    def _rollout_mode(self):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            yield
        finally:
            for module, mode in modes:
                module.training = mode
            self.obs_encoder.dino_encoder.backbone.eval()

    def set_normalizer(self, normalizer):
        if isinstance(normalizer, (list, tuple)):
            if len(normalizer) != 1:
                raise ValueError('Expected exactly one task normalizer')
            normalizer = normalizer[0]
        selected = LinearNormalizer()
        for key in ['action', *self.obs_encoder.state_ports]:
            if key not in normalizer.params_dict:
                raise KeyError(f'Missing training normalizer field: {key}')
            selected[key] = copy.deepcopy(normalizer[key])
        if any(not torch.isfinite(value).all() for value in selected.state_dict().values()):
            raise ValueError('Training normalizer must contain finite statistics')
        self.action_normalizer.load_state_dict(selected.state_dict(), strict=True)
        self.action_normalizer.to(self.device).requires_grad_(False)
        self._normalizer_ready.fill_(True)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if not strict:
            raise ValueError('Direct-action artifacts require strict loading')
        if int(state_dict.get('_action_flow_artifact_schema', -1)) != self.ARTIFACT_SCHEMA_VERSION:
            raise ValueError('Checkpoint is not a compatible continuous_action_flow artifact')
        if int(state_dict.get('_variant_code', -1)) != int(self.requires_state_history):
            raise ValueError('Checkpoint variant does not match this policy')
        if int(state_dict.get('_context_schema', -1)) != self.CONTEXT_SCHEMA_VERSION:
            raise ValueError('Checkpoint context schema does not match this policy')
        result = super().load_state_dict(state_dict, strict=True, **kwargs)
        self.action_normalizer.to(self.device).requires_grad_(False)
        self.obs_encoder.dino_encoder.backbone.requires_grad_(False).eval()
        self.reset()
        return result

    def _encode_target(self, actions):
        if not bool(self._normalizer_ready):
            raise RuntimeError('Set the training-set normalizer before using this policy')
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError('Action target must be [B,horizon,action_dim]')
        if not torch.isfinite(actions).all():
            raise ValueError('Action targets must be finite')
        with torch.autocast(device_type=actions.device.type, enabled=False):
            return self.action_normalizer['action'].normalize(actions.float()).float()

    def export_config(self):
        return dict(_target_=f'{type(self).__module__}.{type(self).__name__}', _recursive_=False,
            **copy.deepcopy(self._construction), construction_mode='restore',
            obs_encoder_config=self.obs_encoder.export_config(),
            normalizer_metadata=copy.deepcopy(self.normalizer_metadata))

    def normalized_observation(self, obs):
        normalized = dict(obs)
        for key in self.obs_encoder.state_ports:
            if not torch.isfinite(obs[key]).all():
                raise ValueError(f'Current observation {key!r} must be finite')
            normalized[key] = self.action_normalizer[key].normalize(obs[key].float())
        return normalized

    def build_context(self, obs, past_actions, past_action_valid, frozen_patches=None):
        past, valid = self._safe_past(past_actions, past_action_valid)
        normalized_obs = dict(obs)
        for key in self.obs_encoder.state_ports:
            if key not in obs:
                raise KeyError(f'Missing observation port {key!r}')
            if not torch.isfinite(obs[key]).all():
                raise ValueError(f'Current observation {key!r} must be finite')
            normalized_obs[key] = self.action_normalizer[key].normalize(obs[key].float())
        visual, proprio = self.obs_encoder(normalized_obs, frozen_patches=frozen_patches)
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
        return ActionFlowContextBatch(memory, visible, segments).validate()

    def forward(self, prepared):
        if not isinstance(prepared, PreparedActionFlowBatch):
            raise TypeError('Flow forward requires prepare_flow_training_batch output, including EMA targets')
        prepared.validate()
        context = self.build_context(prepared.obs, prepared.past_actions,
                                     prepared.past_action_valid, prepared.frozen_patches)
        context.validate_variant(self.variant)
        velocity = self.model(prepared.noisy_actions, time=prepared.time,
            step_size=prepared.step_size, context=context)
        losses = packed_flow_loss(velocity, prepared.velocity_targets,
                                 prepared.fm_indices, prepared.ct_indices)
        self._last_flow_losses = {k: v.detach() for k, v in losses.items()}
        return losses['loss']

    def prepare_training_batch(self, batch, teacher, generator, self_past_generator=None):
        return prepare_action_flow_training_batch(batch, student=self, teacher=teacher,
            generator=generator, self_past_generator=self_past_generator)

    @torch.no_grad()
    def _generate_normalized_actions(self, obs, past, valid, *, num_flow_steps=None,
                                     generator=None, initial_noise=None, frozen_patches=None):
        if not bool(self._normalizer_ready):
            raise RuntimeError('Set the training-set normalizer before using this policy')
        context = self.build_context(obs, past, valid, frozen_patches)
        context.validate_variant(self.variant)
        return euler_sample(self.model, context=context,
            num_steps=self.flow['inference_steps'] if num_flow_steps is None else num_flow_steps,
            generator=generator, initial_noise=initial_noise)

    @torch.no_grad()
    def _generate_actions(self, obs, past, valid, *, num_flow_steps=None, generator=None):
        normalized = self._generate_normalized_actions(obs, past, valid,
            num_flow_steps=num_flow_steps, generator=generator)
        with torch.autocast(device_type=normalized.device.type, enabled=False):
            actions = self.action_normalizer['action'].unnormalize(normalized.float())
        if not torch.isfinite(actions).all():
            raise ValueError('Generated commands must be finite')
        return actions

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
        if stateful and generator is None:
            if self._online_generator is None:
                self._online_generator = torch.Generator(device=self.device).manual_seed(self.online_seed)
            generator = self._online_generator
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
    def validation_metrics(self, batch, generator, history_mode='expert', compute_actions=True):
        if history_mode not in ('expert', 'generated'):
            raise ValueError('Validation history must be expert or generated')
        b, device = len(batch['action']), batch['action'].device
        shape = (b, self.horizon, self.action_dim)
        noise = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        t = 0.999 * (1 - torch.rand(b, device=device, generator=generator).pow(2 / 3))
        initial = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        def ratio(values, mask=None):
            if mask is None:
                count = torch.tensor(values.numel(), device=device, dtype=torch.float64)
            else:
                mask = mask.expand_as(values)
                values = torch.where(mask, values, torch.zeros_like(values))
                count = mask.sum().double()
            return {'sum': values.double().sum(), 'count': count}
        with self._rollout_mode(), self._clean_autocast_cache():
            past = batch['past_action']
            if history_mode == 'generated':
                past = self._maybe_self_past(batch, past, probability=1., generator=generator)
            patches = self.obs_encoder.extract_frozen_patches(batch['obs'])
            context = self.build_context(batch['obs'], past, batch['past_action_valid'], patches)
            context.validate_variant(self.variant)
            target = self._encode_target(batch['action'])
            velocity = self.model(interpolate_latents(target, noise, t), time=t,
                step_size=torch.zeros_like(t), context=context).float()
            fm_error = (velocity - (target - noise)).square()
            if not compute_actions:
                return {'fm_loss': ratio(fm_error)}
            normalized = euler_sample(self.model, context=context,
                num_steps=self.flow['inference_steps'], initial_noise=initial)
            prediction = self.action_normalizer['action'].unnormalize(normalized.float())
        valid = batch['future_action_valid']
        if valid.dtype != torch.bool or valid.shape != batch['action'].shape[:2]:
            raise ValueError('future_action_valid must be bool [B,horizon]')
        mask = valid[..., None]
        error = (prediction.float() - batch['action'].float()).square()
        return {'fm_loss': ratio(fm_error), 'action_mse': ratio(error, mask),
            'normalized_action_mse': ratio((normalized - target).square(), mask),
            'translation_mse': ratio(error[..., :3], mask),
            'rotation_mse': ratio(error[..., 3:6], mask),
            'gripper_mse': ratio(error[..., 6:], mask),
            'normalized_out_of_bounds': ratio(((normalized < -1) | (normalized > 1)).float(), mask)}

    def artifact_metadata(self):
        versions = {}
        for package in ('torch','torchvision','transformers','accelerate'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        directory = Path(__file__).resolve().parents[1]
        sources = [Path(__file__), directory/'policy/p2n_state_gate_action_flow.py',
            directory/'model/flow/ditx_action.py', directory/'model/flow/action_euler_sampler.py',
            directory/'model/flow/consistency_flow.py', directory/'common/action_flow_batch.py',
            directory/'model/common/action_flow_context.py', directory/'model/common/normalizer.py',
            directory/'perception/latent_flow_token_obs_encoder.py',
            directory/'perception/visual_resampler.py', directory/'perception/dinov3_patch_encoder.py',
            directory/'policy/p2n_new_common.py', directory/'model/state_action_history.py',
            directory/'model/action_flow_state_history.py']
        return dict(policy_family=self.policy_family, artifact_schema_version=self.ARTIFACT_SCHEMA_VERSION,
            variant=self.variant, task=self.task, task_type=self.task,
            context_schema=self.CONTEXT_SCHEMA_VERSION, shape_meta=copy.deepcopy(self.shape_meta),
            execution_protocol='acknowledged_commands_v1', action_schema=copy.deepcopy(self.action_schema),
            normalizer=copy.deepcopy(self.normalizer_metadata),
            horizon=self.horizon, n_action_steps=self.n_action_steps,
            flow=copy.deepcopy(self.flow), architecture=self.export_config(),
            initialization=self._construction['initialization'],
            time_embedding=dict(sinusoidal_dim=self.flow['time_embed_dim'], hidden_dim=512, independent_t_dt=True),
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
def prepare_action_flow_training_batch(batch, *, student, teacher, generator, self_past_generator=None):
    if teacher is student or teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Consistency requires an independent eval/no-grad full EMA policy')
    if teacher.variant != student.variant:
        raise ValueError('Teacher and student variants must match')
    past = batch['past_action']
    if batch['past_action_valid'].dtype != torch.bool:
        raise ValueError('Training past_action_valid must be bool')
    schedule = sample_flow_schedule(len(past), device=past.device, generator=generator)
    past = student._maybe_self_past(batch, past,
        generator=generator if self_past_generator is None else self_past_generator)
    targets = student._encode_target(batch['action'])
    noise = torch.randn(targets.shape, device=targets.device, dtype=torch.float32, generator=generator)
    patches = student.obs_encoder.extract_frozen_patches(batch['obs']).detach()
    zt = interpolate_latents(targets, noise, schedule.time)
    velocity_targets = targets - noise
    ct = schedule.ct_indices
    def teacher_velocity(z_next, time_next, original_dt, selected_rows):
        rows = ct[selected_rows]
        obs = {key: value[rows] for key, value in batch['obs'].items()}
        context = teacher.build_context(obs, past[rows], batch['past_action_valid'][rows], patches[rows])
        context.validate_variant(teacher.variant)
        return teacher.model(z_next, time=time_next, step_size=original_dt,
                             context=context).float()
    velocity_targets[ct] = consistency_velocity_target(
        targets[ct], noise[ct], schedule.time[ct], schedule.step_size[ct], teacher_velocity)
    return PreparedActionFlowBatch(zt.detach(), schedule.time, schedule.step_size,
        velocity_targets.detach(), schedule.fm_indices, schedule.ct_indices,
        {key: value.detach() for key, value in batch['obs'].items()}, past.detach(),
        batch['past_action_valid'], patches).validate()
