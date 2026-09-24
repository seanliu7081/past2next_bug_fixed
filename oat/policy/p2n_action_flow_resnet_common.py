"""Trainable ResNet-18 conditioning for direct continuous-action flow.

The DINO policy and launch path remain unchanged. Action-space math, history,
execution acknowledgement and strict artifact contracts are reused; visual
preparation shares normalized crops, never student backbone activations.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.metadata
from pathlib import Path

import dill
import hydra
import torch
from torch import nn
from omegaconf import OmegaConf

from oat.common.hydra_util import register_new_resolvers
from oat.common.action_flow_resnet_batch import PreparedActionFlowResNetBatch
from oat.model.common.action_flow_context import ActionFlowContextBatch
from oat.model.common.context_batch import Segment
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.flow.ditx_action import DiTXActionVectorField
from oat.model.flow.consistency_flow import (
    sample_flow_schedule, interpolate_latents, consistency_velocity_target, packed_flow_loss)
from oat.model.flow.action_euler_sampler import euler_sample
from oat.perception.action_flow_resnet_obs_encoder import ActionFlowResNetObservationEncoder
from oat.policy.base_policy import BasePolicy
from oat.policy.p2n_action_flow_common import P2NActionFlowCommonPolicy, validate_flow_config
from oat.policy.p2n_new_common import _plain, bool_mask, file_sha256


class P2NActionFlowResNetCommonPolicy(P2NActionFlowCommonPolicy):
    # Artifact envelope remains compatible with the action-flow workspace;
    # tensor state has a distinct schema, rejected early by DINO policies.
    STATE_SCHEMA_VERSION = 2
    obs_encoder_type = 'resnet18'

    def __init__(
        self, shape_meta, obs_encoder=None, action_dim=7,
        n_action_steps=8, n_obs_steps=2, past_n=7, horizon=16,
        embed_dim=768, n_layers=16, n_heads=12, dropout=0.0,
        variant=None, task='libero', flow=None,
        initialization='xavier_uniform_position_normal_0.02',
        construction_mode='fresh', obs_encoder_type='resnet18',
        resnet_config=None, obs_encoder_config=None,
        action_schema=None, normalizer_metadata=None, online_seed=42,
        activation_checkpointing=True,
        self_past_p=0.5, self_past_warmup_steps=1000, self_past_ramp_steps=4000,
        self_past_chunk_size=2,
        self_past_schedule='optimizer_step',
    ):
        BasePolicy.__init__(self)
        if dropout != 0.0:
            raise ValueError("Flow consistency training requires dropout=0.0")
        if initialization != 'xavier_uniform_position_normal_0.02':
            raise ValueError('Unsupported flow initialization')
        if obs_encoder_type != 'resnet18':
            raise ValueError('This policy requires obs_encoder_type=resnet18')
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
                obs_encoder = ActionFlowResNetObservationEncoder(**enc_cfg)
            else:
                obs_encoder = ActionFlowResNetObservationEncoder(
                    shape_meta=self.shape_meta, n_obs_steps=n_obs_steps, n_emb=embed_dim,
                    **(_plain(resnet_config) or {}))
        if obs_encoder.output_feature_dim() != embed_dim:
            raise ValueError('Observation encoder width must equal action decoder width')
        if obs_encoder.n_obs_steps != self.n_obs_steps:
            raise ValueError('Observation encoder and policy observation windows must match')
        if _plain(obs_encoder.shape_meta) != self.shape_meta:
            raise ValueError('Observation encoder and policy input schemas must match')
        if not isinstance(obs_encoder, ActionFlowResNetObservationEncoder):
            raise TypeError('ResNet action flow requires ActionFlowResNetObservationEncoder with shared crops')
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
        self.register_buffer('_action_flow_artifact_schema', torch.tensor(self.STATE_SCHEMA_VERSION))
        self.register_buffer('_resnet18_obs_schema', torch.tensor(1))
        self.register_buffer('_normalizer_ready', torch.tensor(False))
        self._construction = dict(
            shape_meta=self.shape_meta, variant=self.variant, task=task,
            n_action_steps=n_action_steps, n_obs_steps=n_obs_steps, past_n=past_n, horizon=horizon,
            embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads,
            obs_encoder_type=obs_encoder_type, resnet_config=_plain(resnet_config), dropout=dropout,
            action_dim=action_dim, action_schema=self.action_schema, online_seed=self.online_seed,
            activation_checkpointing=activation_checkpointing, flow=copy.deepcopy(self.flow),
            initialization=initialization,
            self_past_p=self_past_p, self_past_warmup_steps=self_past_warmup_steps,
            self_past_ramp_steps=self_past_ramp_steps, self_past_chunk_size=self_past_chunk_size,
            self_past_schedule=self_past_schedule)
        self.reset()

    def train(self, mode=True):
        # Do not invoke the DINO parent's train hook: these backbones train.
        nn.Module.train(self, mode)
        self.action_normalizer.requires_grad_(False)
        self.obs_encoder.vision_encoder.normalizer.requires_grad_(False)
        return self

    @contextlib.contextmanager
    def _rollout_mode(self):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            yield
        finally:
            for module, mode in modes:
                module.training = mode

    def set_normalizer(self, normalizer):
        actual = normalizer[0] if isinstance(normalizer, (tuple, list)) and len(normalizer) == 1 else normalizer
        if not isinstance(actual, LinearNormalizer):
            raise TypeError('Expected one complete training dataset normalizer')
        for port in self.obs_encoder.rgb_ports:
            if port not in actual.params_dict:
                raise KeyError(f'Missing training RGB normalizer field: {port}')
            values = actual[port].state_dict().values()
            if any(not torch.isfinite(value).all() for value in values):
                raise ValueError('RGB normalizer must contain finite statistics')
        super().set_normalizer(actual)
        self.obs_encoder.set_normalizer(actual)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if not strict:
            raise ValueError('ResNet action-flow artifacts require strict loading')
        if int(state_dict.get('_resnet18_obs_schema', -1)) != 1:
            raise ValueError('Checkpoint observation encoder must be resnet18')
        if int(state_dict.get('_action_flow_artifact_schema', -1)) != self.STATE_SCHEMA_VERSION:
            raise ValueError('Checkpoint is not a compatible ResNet continuous_action_flow state')
        if int(state_dict.get('_variant_code', -1)) != int(self.requires_state_history):
            raise ValueError('Checkpoint variant does not match this policy')
        if int(state_dict.get('_context_schema', -1)) != self.CONTEXT_SCHEMA_VERSION:
            raise ValueError('Checkpoint context schema does not match this policy')
        # The parent load hook enforces frozen DINO; bypass only that hook.
        result = nn.Module.load_state_dict(self, state_dict, strict=True, **kwargs)
        self.action_normalizer.to(self.device).requires_grad_(False)
        self.obs_encoder.vision_encoder.normalizer.to(self.device).requires_grad_(False)
        self.reset()
        return result

    @torch.no_grad()
    def prepare_visual_conditioning(self, obs):
        return self.obs_encoder.prepare_conditioning(obs).detach()

    def get_policy_name(self):
        return f'{self.variant}_resnet18_{self.task}'

    def get_optimizer(self, policy_lr=5e-5, obs_enc_lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95)):
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

    def build_context(self, obs, past_actions, past_action_valid, prepared_visual=None):
        past, valid = self._safe_past(past_actions, past_action_valid)
        normalized_obs = dict(obs)
        for key in self.obs_encoder.state_ports:
            if key not in obs:
                raise KeyError(f'Missing observation port {key!r}')
            if not torch.isfinite(obs[key]).all():
                raise ValueError(f'Current observation {key!r} must be finite')
            normalized_obs[key] = self.action_normalizer[key].normalize(obs[key].float())
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
        return ActionFlowContextBatch(memory, visible, segments).validate()

    def forward(self, prepared):
        if not isinstance(prepared, PreparedActionFlowResNetBatch):
            raise TypeError('ResNet flow forward requires prepare_action_flow_resnet_training_batch output')
        prepared.validate()
        context = self.build_context(prepared.obs, prepared.past_actions,
                                     prepared.past_action_valid, prepared.prepared_visual)
        context.validate_variant(self.variant)
        velocity = self.model(prepared.noisy_actions, time=prepared.time,
            step_size=prepared.step_size, context=context)
        losses = packed_flow_loss(velocity, prepared.velocity_targets,
                                 prepared.fm_indices, prepared.ct_indices)
        self._last_flow_losses = {k: v.detach() for k, v in losses.items()}
        return losses['loss']

    def prepare_training_batch(self, batch, teacher, generator, self_past_generator=None):
        return prepare_action_flow_resnet_training_batch(batch, student=self, teacher=teacher,
            generator=generator, self_past_generator=self_past_generator)

    @torch.no_grad()
    def _generate_normalized_actions(self, obs, past, valid, *, num_flow_steps=None,
                                     generator=None, initial_noise=None, prepared_visual=None):
        if not bool(self._normalizer_ready):
            raise RuntimeError('Set the training-set normalizer before using this policy')
        context = self.build_context(obs, past, valid, prepared_visual)
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
            crops = self.prepare_visual_conditioning(batch['obs'])
            context = self.build_context(batch['obs'], past, batch['past_action_valid'], crops)
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
        sources = [Path(__file__), directory/'policy/p2n_action_flow_resnet18.py',
            directory/'policy/p2n_state_gate_action_flow_resnet18.py',
            directory/'policy/p2n_action_flow_common.py', directory/'policy/p2n_new_common.py',
            directory/'common/action_flow_resnet_batch.py',
            directory/'perception/action_flow_resnet_obs_encoder.py',
            directory/'perception/latent_flow_resnet_obs_encoder.py',
            directory/'perception/robomimic_vision_encoder.py', directory/'perception/crop_randomizer.py',
            directory/'perception/state_encoder.py', directory/'model/common/normalizer.py',
            directory/'model/flow/ditx_action.py', directory/'model/flow/action_euler_sampler.py',
            directory/'model/flow/consistency_flow.py', directory/'model/common/action_flow_context.py',
            directory/'model/state_action_history.py', directory/'model/action_flow_state_history.py',
            directory/'workspace/train_p2n_action_flow_resnet18.py',
            directory.parent/'scripts/train_p2n_action_flow_resnet18.py']
        return dict(policy_family=self.policy_family, artifact_schema_version=self.ARTIFACT_SCHEMA_VERSION,
            variant=self.variant, task=self.task, task_type=self.task,
            obs_encoder_type=self.obs_encoder_type, tensor_state_schema=self.STATE_SCHEMA_VERSION,
            observation_conditioning='shared_normalized_crops_independent_trainable_resnet18',
            vision_pretraining='none', visual_tokens=self.obs_encoder.num_visual_tokens,
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
        if metadata.get('obs_encoder_type') != 'resnet18':
            raise ValueError('Checkpoint observation encoder must be resnet18')
        register_new_resolvers()
        cfg = OmegaConf.create(_plain(payload['cfg']))
        config = OmegaConf.create(payload['policy_config'])
        if policy_overrides:
            raise ValueError('Flow artifacts restore their exact architecture; set num_flow_steps in predict_action')
        config.construction_mode = 'restore'
        if config.get('variant') != cls.VARIANT:
            raise ValueError('Policy configuration and artifact variant disagree')
        if config.get('obs_encoder_type') != 'resnet18':
            raise ValueError('Artifact observation encoder configuration must be resnet18')
        if config.get('_target_') != f'{cls.__module__}.{cls.__name__}':
            raise ValueError('Artifact policy target does not match requested class')
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
def prepare_action_flow_resnet_training_batch(batch, *, student, teacher, generator, self_past_generator=None):
    if teacher is student or teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Consistency requires an independent eval/no-grad full EMA policy')
    if teacher.obs_encoder_type != student.obs_encoder_type:
        raise ValueError('Teacher and student observation encoders must match')
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
    crops = student.prepare_visual_conditioning(batch['obs'])
    zt = interpolate_latents(targets, noise, schedule.time)
    velocity_targets = targets - noise
    ct = schedule.ct_indices
    def teacher_velocity(z_next, time_next, original_dt, selected_rows):
        rows = ct[selected_rows]
        obs = {key: value[rows] for key, value in batch['obs'].items()}
        context = teacher.build_context(obs, past[rows], batch['past_action_valid'][rows], crops[rows])
        context.validate_variant(teacher.variant)
        return teacher.model(z_next, time=time_next, step_size=original_dt,
                             context=context).float()
    velocity_targets[ct] = consistency_velocity_target(
        targets[ct], noise[ct], schedule.time[ct], schedule.step_size[ct], teacher_velocity)
    return PreparedActionFlowResNetBatch(zt.detach(), schedule.time, schedule.step_size,
        velocity_targets.detach(), schedule.fm_indices, schedule.ct_indices,
        {key: value.detach() for key, value in batch['obs'].items()}, past.detach(),
        batch['past_action_valid'], crops).validate()
