"""CPU integration contracts with a deliberately tiny mocked Nano backbone.

Actual OAT/FSQ, modern AR, fusion, resampler and gate modules execute. These tests
exercise wiring and continuation; they do not claim pretrained-policy quality.
"""
import copy
from types import SimpleNamespace

import dill
import hydra
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from oat.model.common.normalizer import SingleFieldLinearNormalizer
from oat.perception.convnext_feature_encoder import (
    ConvNeXtFeatureEncoder, CONVNEXT_MODEL_ID, IMAGE_MEAN, IMAGE_STD,
    TIMM_VERSION, _timm_source_fingerprints,
)
from oat.perception.convnext_token_obs_encoder import ConvNeXtTokenObservationEncoder
from oat.policy.p2n_new_convnext import (
    P2NNewPolicy, P2NStateGateNewPolicy, policy_observation_encoder_config,
)
from oat.workspace.train_p2n_new_convnext import TrainP2NNewWorkspace
from test_p2n_new_policy import (
    META, tokenizer_config, batch_for, local_mock_dino, make_policy as make_dino_policy,
)


class TinyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection16 = nn.Conv2d(3, 320, 1)
        self.projection32 = nn.Conv2d(3, 640, 1)
        self.feature_info = SimpleNamespace(channels=lambda: [320, 640], reduction=lambda: [16, 32])
        self.pretrained_cfg = dict(mean=IMAGE_MEAN, std=IMAGE_STD)

    def forward(self, pixels):
        return [self.projection16(F.avg_pool2d(pixels, 16)),
                self.projection32(F.avg_pool2d(pixels, 32))]


def tiny_timm_model(name, pretrained, **kwargs):
    assert name == 'convnextv2_nano.fcmae_ft_in22k_in1k'
    assert pretrained is False
    assert kwargs['features_only'] and kwargs['out_indices'] == [2, 3]
    return TinyFeatures()


@pytest.fixture(autouse=True)
def local_mock_nano(monkeypatch):
    timm = pytest.importorskip('timm')
    monkeypatch.setattr(timm, 'create_model', tiny_timm_model)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def make_policy(gate=False, *, queries=2, past_n=3, **overrides):
    torch.manual_seed(122)
    feature = ConvNeXtFeatureEncoder(
        construction_mode='restore', metadata=dict(
            model_id=CONVNEXT_MODEL_ID, revision='a' * 40, weight_sha256='b' * 64,
            timm_version=TIMM_VERSION, source_sha256=_timm_source_fingerprints()))
    # The mocked timm factory is the sole substitution; real restore guards run.
    feature.load_state_dict(feature.state_dict(), strict=True)
    encoder = ConvNeXtTokenObservationEncoder(
        META, convnext_encoder=feature, n_emb=16, visual_resampler_dim=16,
        visual_resampler_heads=2, visual_resampler_ffn_dim=32, num_queries=queries,
        resampler_depth=1, dropout=0.)
    tok_cfg = tokenizer_config()
    tokenizer = hydra.utils.instantiate(tok_cfg)
    tokenizer.normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
    kwargs = dict(
        shape_meta=META, obs_encoder=encoder, action_tokenizer=tokenizer,
        tokenizer_config=tok_cfg, horizon=4, n_action_steps=2, n_obs_steps=2,
        past_n=past_n, embed_dim=16, n_layers=2, n_heads=2, ffn_dim=32,
        dropout=0., temperature=0., expected_action_tokens=2,
        self_past_p=1., self_past_warmup_steps=0, self_past_ramp_steps=0,
        self_past_chunk_size=2)
    if gate:
        kwargs.update(state_history_steps=past_n + 1, history_embed_dim=16,
                      history_n_heads=2, history_n_layers=1, history_dropout=0.,
                      history_summary_tokens=4 if queries == 64 else 2,
                      history_gate_hidden_dim=16)
    kwargs.update(overrides)
    return (P2NStateGateNewPolicy if gate else P2NNewPolicy)(**kwargs)


@pytest.mark.parametrize('gate', [False, True])
def test_context_gate_and_visual_optimizer_contract(gate):
    policy = make_policy(gate, queries=64, past_n=7).eval()
    batch = batch_for(policy)
    context = policy.build_context(batch['obs'], batch['past_action'], batch['past_action_valid'])
    assert context.memory.shape == (2, 271 if gate else 267, 16)
    context.validate_variant(policy.variant, num_summary_tokens=4 if gate else 0)
    if gate:
        torch.testing.assert_close(context.history_log_gate.exp(), torch.full((2, 1), .9))
    assert 'convnextv2_nano' in policy.get_policy_name()
    assert not hasattr(policy.obs_encoder, 'dino_encoder')
    optimizer = policy.get_optimizer(policy_lr=5e-5, obs_enc_lr=1e-4)
    rates = {id(parameter): group['lr'] for group in optimizer.param_groups for parameter in group['params']}
    actual = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    assert len(actual) == len(set(actual))
    assert set(actual) == {id(p) for p in policy.parameters() if p.requires_grad}
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            visual = name.startswith('obs_encoder.') and not name.startswith('obs_encoder.state_projection.')
            assert rates[id(parameter)] == (1e-4 if visual else 5e-5), name
    counts = policy.parameter_counts()
    assert counts['components']['visual_backbone']['trainable'] == 0
    assert counts['components']['visual_adapter']['trainable'] > 0
    assert counts['components']['action_model']['trainable'] > 0


@pytest.mark.parametrize('gate', [False, True])
def test_self_past_bf16_backward_and_frozen_student_ema_modes(gate):
    policy = make_policy(gate).train()
    ema = copy.deepcopy(policy).train()
    policy.obs_encoder.output_projection.eval()
    batch = batch_for(policy, batch_size=3, previous=True)
    batch['prev_window_valid'][1] = False
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = policy(batch, history_mode='generated')
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    assert not policy.obs_encoder.output_projection.training
    assert not policy.obs_encoder.convnext_encoder.backbone.training
    assert not ema.obs_encoder.convnext_encoder.backbone.training
    assert policy._pending_execution_steps is None
    assert policy.self_past_step == 0
    policy.on_optimizer_step()
    assert policy.self_past_step == 1
    assert policy.obs_encoder.fusion.projection16.weight.grad.abs().sum() > 0
    assert policy.obs_encoder.fusion.projection32.weight.grad.abs().sum() > 0
    assert policy.obs_encoder.output_projection.weight.grad.abs().sum() > 0
    assert policy.model.tok_emb.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('gate', [False, True])
def test_offline_checkpoint_prediction_override_guards_and_strict_weights(gate, tmp_path):
    policy = make_policy(gate).eval()
    batch = batch_for(policy)
    expected = policy.predict_action(batch['obs'], past_actions=batch['past_action'],
                                     past_action_valid=batch['past_action_valid'])['action_pred']
    exported = policy.export_config()
    exported['obs_encoder_config']['convnext']['model_path'] = '/deleted/convnext/model.safetensors'
    exported['tokenizer_checkpoint'] = '/deleted/oat.ckpt'
    payload = dict(cfg=OmegaConf.create(dict(policy=exported, training=dict(use_ema=True))),
                   policy_config=exported, state_dicts=dict(model=policy.state_dict(), ema_model=policy.state_dict()))
    path = tmp_path / 'nano.ckpt'
    torch.save(payload, path, pickle_module=dill)
    cls = type(policy)
    restored = cls.from_checkpoint(path)
    actual = restored.predict_action(batch['obs'], past_actions=batch['past_action'],
                                     past_action_valid=batch['past_action_valid'])['action_pred']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored._pending_execution_steps is None
    assert set(restored.state_dict()) == set(policy.state_dict())
    for field, value in [('vision_image_size', 256), ('obs_encoder_type', 'dinov3_tokens'),
                         ('visual_resampler_dim', 512), ('expected_action_tokens', 8)]:
        with pytest.raises(ValueError, match='cannot be overridden'):
            cls.from_checkpoint(path, policy_overrides={field: value})
    altered = copy.deepcopy(exported['obs_encoder_config'])
    altered['convnext']['antialias'] = False
    with pytest.raises(ValueError, match='antialias'):
        cls.from_checkpoint(path, policy_overrides={'obs_encoder_config': altered})
    missing = dict(policy.state_dict())
    del missing['obs_encoder.convnext_encoder.backbone.projection16.weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(missing, strict=True)
    with pytest.raises(ValueError, match='strict'):
        restored.load_state_dict(policy.state_dict(), strict=False)
    with pytest.raises(ValueError, match='antialias'):
        cls.from_checkpoint(path, policy_overrides={'obs_encoder_config': {'convnext': {'antialias': False}}})
    partial = cls.from_checkpoint(path, policy_overrides={'obs_encoder_config': {'convnext': {'antialias': True}}})
    assert partial.obs_encoder.convnext_encoder.antialias is True
    metadata = policy.artifact_metadata()
    assert metadata['encoder_type'] == 'convnextv2_tokens'
    assert 'oat/perception/convnext_feature_encoder.py' in metadata['source_sha256']


@pytest.mark.parametrize('gate', [False, True])
def test_legacy_dino_artifact_preserves_state_keys_and_prediction(gate, tmp_path):
    policy = make_dino_policy(gate).eval()
    batch = batch_for(policy)
    expected = policy.predict_action(batch['obs'], past_actions=batch['past_action'],
                                     past_action_valid=batch['past_action_valid'])['action_pred']
    exported = policy.export_config()
    payload = dict(cfg=OmegaConf.create(dict(policy=exported, training=dict(use_ema=False))),
                   policy_config=exported, state_dicts=dict(model=policy.state_dict()))
    path = tmp_path / 'old_dino.ckpt'
    torch.save(payload, path, pickle_module=dill)
    cls = P2NStateGateNewPolicy if gate else P2NNewPolicy
    restored = cls.from_checkpoint(path)
    assert restored.obs_encoder_type == 'dinov3_tokens'
    assert set(restored.state_dict()) == set(policy.state_dict())
    actual = restored.predict_action(batch['obs'], past_actions=batch['past_action'],
                                     past_action_valid=batch['past_action_valid'])['action_pred']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_latent_horizon_and_conflicting_sources_are_rejected():
    with pytest.raises(ValueError, match='latent horizon'):
        make_policy(expected_action_tokens=8)
    with pytest.raises(ValueError, match='conflicts'):
        make_policy(dino_path='/other/backend')
    policy = make_policy()
    config = policy.export_config()
    config['dino_path'] = '/other/backend'
    with pytest.raises(ValueError, match='conflicts'):
        policy_observation_encoder_config(config)


def test_resume_compares_normalized_embedded_preprocessing_and_fusion():
    policy = make_policy()
    exported = policy.export_config()
    cfg = OmegaConf.create(dict(
        variant=policy.variant, policy=exported, training=dict(use_ema=False),
        task=dict(policy=dict(dataset=dict(zarr_path='/same/data')))))
    payload = dict(cfg=copy.deepcopy(cfg), policy_config=copy.deepcopy(exported),
                   metadata=policy.artifact_metadata(), state_dicts=dict(model={}, optimizer={}),
                   pickles={key: b'' for key in ('epoch', 'global_step', 'completed_optimizer_steps',
                                                 'ema_state', 'lr_scheduler_state', 'rng_states')})
    TrainP2NNewWorkspace.validate_resume_payload(payload, cfg)
    moved = copy.deepcopy(cfg)
    moved.policy.obs_encoder_config.convnext.model_path = '/relocated/source'
    TrainP2NNewWorkspace.validate_resume_payload(payload, moved)
    for field, value in [('input_range', 'float_0_1'), ('image_size', 256), ('feature_stages', [1, 3]),
                         ('antialias', False)]:
        changed = copy.deepcopy(cfg)
        changed.policy.obs_encoder_config.convnext[field] = value
        with pytest.raises(ValueError, match=field):
            TrainP2NNewWorkspace.validate_resume_payload(payload, changed)
    changed = copy.deepcopy(cfg)
    changed.policy.obs_encoder_config.fusion.scale = '1'
    with pytest.raises(ValueError, match='fusion.scale'):
        TrainP2NNewWorkspace.validate_resume_payload(payload, changed)
    broken = copy.deepcopy(payload)
    broken['policy_config']['obs_encoder_config']['visual_resampler_heads'] = 4
    with pytest.raises(ValueError, match='visual_resampler_heads'):
        TrainP2NNewWorkspace.validate_resume_payload(broken, cfg)


@pytest.mark.parametrize('gate', [False, True])
def test_fresh_production_config_matches_normalized_exported_artifact(gate):
    from scripts.train_p2n_new_convnext import compose_config
    from oat.policy.p2n_new_convnext import observation_encoder_contract
    variant = 'p2n_state_gate_new' if gate else 'p2n_new'
    cfg = compose_config(variant, 'real_robot')
    # No model, tokenizer, dataset, or external weights are opened here.
    exported_encoder = policy_observation_encoder_config(cfg.policy)
    exported_encoder['convnext'].pop('model_path', None)
    exported_encoder['convnext'].update(construction_mode='restore', revision='a' * 40, metadata=dict(
        model_id=CONVNEXT_MODEL_ID, revision='a' * 40, weight_sha256='b' * 64,
        timm_version=TIMM_VERSION, source_sha256={'test_fixture': 'c' * 64}))
    artifact_config = dict(obs_encoder_config=exported_encoder)
    payload = dict(cfg=copy.deepcopy(cfg), policy_config=artifact_config,
                   metadata=dict(variant=variant, observation_encoder_contract=observation_encoder_contract(exported_encoder)),
                   state_dicts=dict(model={}, ema_model={}, optimizer={}),
                   pickles={key: b'' for key in ('epoch', 'global_step', 'completed_optimizer_steps',
                                                 'ema_state', 'lr_scheduler_state', 'rng_states')})
    TrainP2NNewWorkspace.validate_resume_payload(payload, cfg)
    moved = copy.deepcopy(cfg)
    moved.policy.convnext_path = '/deleted/source/is/not/opened'
    moved.policy.tokenizer_checkpoint = '/deleted/tokenizer/is/not/opened'
    TrainP2NNewWorkspace.validate_resume_payload(payload, moved)
    changed = copy.deepcopy(cfg)
    changed.policy.visual_resampler_dim = 512
    with pytest.raises(ValueError, match='visual_resampler_dim'):
        TrainP2NNewWorkspace.validate_resume_payload(payload, changed)
    corrupted = copy.deepcopy(payload)
    corrupted['policy_config']['obs_encoder_config']['convnext']['antialias'] = False
    with pytest.raises(ValueError, match='antialias'):
        TrainP2NNewWorkspace.validate_resume_payload(corrupted, cfg)


def test_resume_rejects_top_level_visual_override_hidden_by_embedded_encoder():
    policy = make_policy()
    cfg = OmegaConf.create(dict(variant=policy.variant, policy=policy.export_config(),
        training=dict(use_ema=False), task=dict(policy=dict(dataset=dict(zarr_path='/same')))))
    payload = dict(cfg=copy.deepcopy(cfg), policy_config=policy.export_config(),
        metadata=policy.artifact_metadata(), state_dicts=dict(model={}, optimizer={}),
        pickles={key: b'' for key in ('epoch', 'global_step', 'completed_optimizer_steps',
                                     'ema_state', 'lr_scheduler_state', 'rng_states')})
    cfg.policy.visual_resampler_dim = 512
    with pytest.raises(ValueError, match='visual_resampler_dim'):
        TrainP2NNewWorkspace.validate_resume_payload(payload, cfg)
