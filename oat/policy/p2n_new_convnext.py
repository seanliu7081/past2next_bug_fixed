"""Additive visual-backend integration for the existing modern Past2Next policies.

The action model, OAT, history/gate equations, optimizer grouping and execution
protocol are inherited unchanged. Only encoder construction and its lifecycle
are dispatched here, so existing DINO entry points remain untouched.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.metadata
from pathlib import Path

import dill
import hydra
import torch
from omegaconf import OmegaConf

from oat.common.hydra_util import register_new_resolvers
from oat.perception.obs_encoder_factory import (
    build_observation_encoder,
    maintain_frozen_backbone_mode,
    normalize_observation_encoder_config,
)
from oat.policy.p2n_new import P2NNewPolicy as _DINOBasePolicy
from oat.policy.p2n_state_gate_new import P2NStateGateNewPolicy as _DINOGatePolicy
from oat.policy.p2n_new_common import _plain, file_sha256


ENCODER_POLICY_FIELDS = frozenset({
    'obs_encoder', 'obs_encoder_config', 'obs_encoder_type',
    'convnext_model_name', 'convnext_path', 'convnext_revision', 'convnext_frozen',
    'vision_image_size', 'vision_feature_stages', 'visual_resampler_dim',
    'visual_resampler_heads', 'visual_resampler_ffn_dim', 'num_visual_queries',
    'resampler_depth', 'rgb_range', 'image_brightness', 'image_contrast',
    'dino_path', 'dino_revision', 'dino_config', 'processor_config',
})


def _reject_conflicting_sources(kind, cfg):
    forbidden = (('dino_path', 'dino_revision', 'dino_config', 'processor_config')
                 if kind == 'convnextv2_tokens' else ('convnext_path', 'convnext_revision'))
    if any(cfg.get(key) is not None for key in forbidden):
        raise ValueError(f'{kind} configuration conflicts with the other visual backend sources')


def policy_observation_encoder_config(policy_config):
    """Resolve public policy fields without loading weights or touching files."""
    cfg = _plain(policy_config)
    if cfg.get('obs_encoder_config') is not None:
        encoder = normalize_observation_encoder_config(cfg['obs_encoder_config'])
        _reject_conflicting_sources(encoder['encoder_type'], cfg)
        if cfg.get('obs_encoder_type') not in (None, encoder['encoder_type']):
            raise ValueError('obs_encoder_type disagrees with obs_encoder_config.encoder_type')
        return encoder
    kind = cfg.get('obs_encoder_type') or 'dinov3_tokens'
    common = dict(
        encoder_type=kind, shape_meta=cfg['shape_meta'],
        n_obs_steps=cfg.get('n_obs_steps', 2), n_emb=cfg.get('embed_dim', 768),
        num_queries=cfg.get('num_visual_queries', 64),
        resampler_depth=cfg.get('resampler_depth', 2), dropout=cfg.get('dropout', 0.1),
        activation_checkpointing=cfg.get('activation_checkpointing', False),
    )
    if kind == 'convnextv2_tokens':
        if any(cfg.get(key) is not None for key in ('dino_path', 'dino_revision', 'dino_config', 'processor_config')):
            raise ValueError('ConvNeXt configuration conflicts with nonempty DINO weight/config fields')
        ranges = {'uint8': 'uint8_0_255', '0_1': 'float_0_1', '0_255': 'float_0_255'}
        rgb_range = cfg.get('rgb_range', 'uint8')
        if rgb_range not in ranges:
            raise ValueError('rgb_range must explicitly be uint8, 0_1, or 0_255')
        common.update(
            visual_resampler_dim=cfg.get('visual_resampler_dim', 256),
            visual_resampler_heads=cfg.get('visual_resampler_heads', 4),
            visual_resampler_ffn_dim=cfg.get('visual_resampler_ffn_dim', 768),
            convnext=dict(
                model_name=cfg.get('convnext_model_name', 'convnextv2_nano.fcmae_ft_in22k_in1k'),
                model_path=cfg.get('convnext_path'), revision=cfg.get('convnext_revision'),
                construction_mode=cfg.get('construction_mode', 'fresh'),
                frozen=cfg.get('convnext_frozen', True),
                image_size=cfg.get('vision_image_size', 224),
                feature_stages=cfg.get('vision_feature_stages', [2, 3]),
                input_range=ranges[rgb_range], brightness=cfg.get('image_brightness', 0.1),
                contrast=cfg.get('image_contrast', 0.1),
            ),
        )
    elif kind == 'dinov3_tokens':
        if cfg.get('convnext_path') is not None or cfg.get('convnext_revision') is not None:
            raise ValueError('DINO configuration conflicts with ConvNeXt weight fields')
        common.update(
            n_head=cfg.get('n_heads', 12), ffn_dim=cfg.get('ffn_dim', 2048),
            dino=dict(
                pretrained_path=cfg.get('dino_path'), revision=cfg.get('dino_revision'),
                load_mode=cfg.get('construction_mode', 'fresh'),
                config=cfg.get('dino_config'), processor_config=cfg.get('processor_config'),
                rgb_range=cfg.get('rgb_range', 'uint8'),
                brightness=cfg.get('image_brightness', 0.1), contrast=cfg.get('image_contrast', 0.1),
            ),
        )
    return normalize_observation_encoder_config(common)


def observation_encoder_contract(config):
    """Compare structure/preprocessing while allowing relocated source files.

    Source identity stays in artifact provenance. Resuming embedded weights does
    not use the original path, revision, package version, or construction mode.
    """
    config = normalize_observation_encoder_config(config)
    source_only = {'_target_', 'model_path', 'pretrained_path', 'revision',
                   'construction_mode', 'load_mode', 'metadata', 'weight_sha256',
                   'timm_version', 'source_sha256', 'source_path'}

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in source_only}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    return clean(config)


def first_config_difference(left, right, path='obs_encoder'):
    """Name the first incompatible field for a useful pre-construction error."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f'{path}.{key}'
            different = first_config_difference(left[key], right[key], f'{path}.{key}')
            if different:
                return different
        return None
    return None if left == right else path


class _ObservationBackendMixin:
    def __init__(
        self, shape_meta, obs_encoder=None, obs_encoder_config=None,
        obs_encoder_type=None, convnext_model_name='convnextv2_nano.fcmae_ft_in22k_in1k',
        convnext_path=None, convnext_revision=None, convnext_frozen=True,
        vision_image_size=224, vision_feature_stages=(2, 3),
        visual_resampler_dim=256, visual_resampler_heads=4, visual_resampler_ffn_dim=768,
        expected_action_tokens=None, **kwargs,
    ):
        mode = kwargs.get('construction_mode', 'fresh')
        if mode not in ('fresh', 'restore'):
            raise ValueError('construction_mode must be fresh or restore')
        if obs_encoder is not None and obs_encoder_config is not None:
            raise ValueError('Supply an observation encoder or its configuration, not both')
        if obs_encoder is None:
            if obs_encoder_config is None and obs_encoder_type is None:
                obs_encoder_type = 'convnextv2_tokens'
            config = policy_observation_encoder_config(dict(
                **kwargs, shape_meta=_plain(shape_meta), obs_encoder_config=obs_encoder_config,
                obs_encoder_type=obs_encoder_type, convnext_model_name=convnext_model_name,
                convnext_path=convnext_path, convnext_revision=convnext_revision,
                convnext_frozen=convnext_frozen, vision_image_size=vision_image_size,
                vision_feature_stages=list(vision_feature_stages),
                visual_resampler_dim=visual_resampler_dim,
                visual_resampler_heads=visual_resampler_heads,
                visual_resampler_ffn_dim=visual_resampler_ffn_dim,
            ))
            obs_encoder = build_observation_encoder(config, construction_mode=mode)
        actual = normalize_observation_encoder_config(obs_encoder.export_config())
        _reject_conflicting_sources(actual['encoder_type'], dict(kwargs, convnext_path=convnext_path,
                                                               convnext_revision=convnext_revision))
        if obs_encoder_type is not None and actual['encoder_type'] != obs_encoder_type:
            raise ValueError('Injected observation encoder does not match obs_encoder_type')
        # Injection bypasses only the DINO-specific construction branch; every
        # schema, tokenizer, AR and gate check still runs in the original classes.
        super().__init__(shape_meta=shape_meta, obs_encoder=obs_encoder, **kwargs)
        self.obs_encoder_type = actual['encoder_type']
        self._construction['obs_encoder_type'] = self.obs_encoder_type
        if expected_action_tokens is None:
            expected_action_tokens = 8 if self.obs_encoder_type == 'convnextv2_tokens' else self.max_seq_len
        if isinstance(expected_action_tokens, bool) or int(expected_action_tokens) != expected_action_tokens or expected_action_tokens < 1:
            raise ValueError('expected_action_tokens must be a positive integer')
        if self.max_seq_len != expected_action_tokens:
            raise ValueError(f'Tokenizer latent horizon {self.max_seq_len} does not match expected_action_tokens={expected_action_tokens}')
        self._construction['expected_action_tokens'] = int(expected_action_tokens)
        maintain_frozen_backbone_mode(self.obs_encoder)

    def train(self, mode=True):
        super().train(mode)
        maintain_frozen_backbone_mode(self.obs_encoder)
        return self

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        maintain_frozen_backbone_mode(self.obs_encoder)
        return result

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
            maintain_frozen_backbone_mode(self.obs_encoder)

    def get_policy_name(self):
        name = 'convnextv2_nano' if self.obs_encoder_type == 'convnextv2_tokens' else 'dinov3_s16'
        return f'{self.variant}_{name}_{self.task}'

    def parameter_counts(self):
        result = super().parameter_counts()

        def count(parameters):
            unique = {id(value): value for value in parameters}.values()
            trainable, frozen = 0, 0
            for value in unique:
                if value.requires_grad:
                    trainable += value.numel()
                else:
                    frozen += value.numel()
            return dict(total=trainable + frozen, trainable=trainable, frozen=frozen)

        encoder = self.obs_encoder
        backend = getattr(encoder, 'convnext_encoder', getattr(encoder, 'dino_encoder', None))
        backbone_ids = {id(value) for value in backend.parameters()} if backend is not None else set()
        state_ids = {id(value) for value in encoder.state_projection.parameters()}
        visual = [value for value in encoder.parameters() if id(value) not in state_ids | backbone_ids]
        obs_ids = {id(value) for value in encoder.parameters()}
        oat_ids = {id(value) for value in self.action_tokenizer.parameters()}
        result['components'] = {
            'visual_backbone': count(backend.parameters()) if backend is not None else count([]),
            'visual_adapter': count(visual),
            'visual_resampler': count(encoder.resampler.parameters()),
            'state_projection': count(encoder.state_projection.parameters()),
            'action_model': count(self.model.parameters()),
            'oat': count(self.action_tokenizer.parameters()),
            'policy_outside_observation_and_oat': count(
                value for value in self.parameters() if id(value) not in obs_ids | oat_ids),
        }
        return result

    def artifact_metadata(self):
        result = super().artifact_metadata()
        config = normalize_observation_encoder_config(self.obs_encoder.export_config())
        result.update(encoder_type=self.obs_encoder_type, encoder_schema_version=config['schema_version'],
                      observation_encoder=config,
                      observation_encoder_contract=observation_encoder_contract(config),
                      parameter_component_note='visual_resampler is included in visual_adapter; component groups overlap')
        try:
            result['software']['timm'] = importlib.metadata.version('timm')
        except importlib.metadata.PackageNotFoundError:
            result['software']['timm'] = None
        root = Path(__file__).resolve().parents[2]
        for relative in (
            'oat/policy/p2n_new_convnext.py', 'oat/policy/p2n_state_gate_new.py',
            'oat/perception/obs_encoder_factory.py', 'oat/perception/convnext_feature_encoder.py',
            'oat/perception/convnext_token_obs_encoder.py', 'oat/workspace/train_p2n_new_convnext.py',
        ):
            path = root / relative
            if path.is_file():
                result['source_sha256'][relative] = file_sha256(path)
        return result

    @classmethod
    def from_checkpoint(cls, checkpoint, output_dir=None, return_configuration=False,
                        weights=None, policy_overrides=None):
        with open(checkpoint, 'rb') as stream:
            payload = torch.load(stream, map_location='cpu', pickle_module=dill, weights_only=False)
        register_new_resolvers()
        cfg = OmegaConf.create(_plain(payload['cfg']))
        config = OmegaConf.create(payload.get('policy_config', cfg.policy))
        if config.get('variant') != cls.VARIANT:
            raise ValueError(f'Checkpoint variant must be {cls.VARIANT}')
        old_target = ('oat.policy.p2n_new.P2NNewPolicy' if cls.VARIANT == 'p2n_new'
                      else 'oat.policy.p2n_state_gate_new.P2NStateGateNewPolicy')
        target = f'{cls.__module__}.{cls.__name__}'
        if config.get('_target_') not in (old_target, target):
            raise ValueError('Checkpoint target and variant disagree')
        if policy_overrides:
            overrides = _plain(policy_overrides)
            protected = ENCODER_POLICY_FIELDS | {
                '_target_', 'variant', 'task', 'shape_meta', 'n_action_steps', 'n_obs_steps',
                'past_n', 'horizon', 'embed_dim', 'n_layers', 'n_heads', 'ffn_dim', 'dropout',
                'tokenizer_config', 'tokenizer_metadata', 'expected_action_tokens', 'state_history_steps',
                'state_history_keys', 'history_summary_tokens', 'history_embed_dim',
                'history_n_heads', 'history_n_layers', 'history_gate_mode',
                'history_gate_hidden_dim', 'history_gate_init', 'history_dropout', 'rotation_6d_layout',
            }
            for key in sorted(protected & set(overrides)):
                saved, proposed = _plain(config.get(key)), overrides[key]
                if key == 'obs_encoder_config' and saved is not None and proposed is not None:
                    merged_encoder = OmegaConf.to_container(OmegaConf.merge(saved, proposed), resolve=True)
                    difference = first_config_difference(observation_encoder_contract(saved),
                                                         observation_encoder_contract(merged_encoder))
                    if difference:
                        raise ValueError(f'Checkpoint architecture/schema cannot be overridden: {difference}')
                elif saved != proposed:
                    raise ValueError(f'Checkpoint architecture/schema cannot be overridden: {key}')
            config = OmegaConf.merge(config, overrides)
        # A legacy DINO artifact retains the exact encoder/state_dict names, but
        # gains the generic frozen-mode lifecycle when loaded through this API.
        config._target_ = target
        if old_target == payload.get('policy_config', cfg.policy).get('_target_'):
            config.obs_encoder_type = 'dinov3_tokens'
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


class P2NNewPolicy(_ObservationBackendMixin, _DINOBasePolicy):
    """Modern AR base policy with factory-selected observation tokens."""


class P2NStateGateNewPolicy(_ObservationBackendMixin, _DINOGatePolicy):
    """The existing measured-state summary gate with factory-selected vision."""
