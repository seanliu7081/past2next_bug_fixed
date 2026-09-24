"""Small CPU policy contracts with actual OAT/FSQ and a mocked DINO backbone.

The backbone mock preserves production patch dimensions and artifact structure;
these tests do not claim pretrained DINO or complete 16x768 acceptance.
"""
import contextlib
import copy
from types import SimpleNamespace

import dill
import hydra
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from oat.model.common.context_batch import Segment
from oat.model.common.normalizer import SingleFieldLinearNormalizer
from oat.perception.dinov3_patch_encoder import DINOv3PatchEncoder
from oat.perception.token_obs_encoder import TokenObservationEncoder
from oat.policy.p2n_new import P2NNewPolicy
from oat.policy.p2n_state_gate_new import P2NStateGateNewPolicy


META = {
    'obs': {
        'camera_a': {'shape': [8, 8, 3], 'type': 'rgb'},
        'camera_b': {'shape': [8, 8, 3], 'type': 'rgb'},
        'robot0_eef_pos': {'shape': [3], 'type': 'state'},
        'robot0_eef_quat': {'shape': [4], 'type': 'state'},
        'robot0_gripper_qpos': {'shape': [2], 'type': 'state'},
        'task_uid': {'shape': [1], 'type': 'state'},
    },
    'action': {'shape': [7]},
}
DINO_CONFIG = dict(model_type='dinov3_vit', patch_size=16, hidden_size=384,
                   num_hidden_layers=12, num_attention_heads=6, num_register_tokens=4)
PROCESSOR = dict(do_normalize=True, do_rescale=True, rescale_factor=1 / 255.,
                 image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])


class MockDINOBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(3, config.hidden_size)
        self.special = nn.Parameter(torch.zeros(1, 1 + config.num_register_tokens, config.hidden_size))

    def forward(self, pixel_values):
        patches = F.avg_pool2d(pixel_values, 16).flatten(2).transpose(1, 2)
        patches = self.projection(patches)
        return SimpleNamespace(last_hidden_state=torch.cat(
            (self.special.expand(patches.shape[0], -1, -1), patches), dim=1))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError('Tests must never fetch external pretrained DINO files')


@pytest.fixture(autouse=True)
def local_mock_dino(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers, 'DINOv3ViTModel', MockDINOBackbone)


def tokenizer_config():
    common = dict(sample_dim=7, sample_horizon=4, emb_dim=16, head_dim=8,
                  depth=1, pdropout=0., latent_dim=2)
    return {
        '_target_': 'oat.tokenizer.oat.tokenizer.OATTok',
        'encoder': dict(_target_='oat.tokenizer.oat.encoder.register_encoder.RegisterEncoder',
                        **common, num_registers=2),
        'decoder': dict(_target_='oat.tokenizer.oat.decoder.single_pass_decoder.SinglePassDecoder',
                        **common, token_dropout_mode='pow2', use_causal_decoder=True, latent_horizon=2),
        'quantizer': {'_target_': 'oat.tokenizer.oat.quantizer.fsq.FSQ', 'levels': [3, 3]},
    }


def make_policy(gate=False, *, queries=2, past_n=3, shape_meta=None, **overrides):
    torch.manual_seed(122)
    shape_meta = META if shape_meta is None else shape_meta
    dino = DINOv3PatchEncoder(load_mode='restore', config=DINO_CONFIG,
                            processor_config=PROCESSOR, revision='a' * 40,
                            weight_sha256='b' * 64, brightness=0.1)
    # A real restore is forbidden to execute before all backbone tensors load.
    # Only its backbone class is mocked; source/config/load guards remain active.
    dino.load_state_dict(dino.state_dict(), strict=True)
    observation = TokenObservationEncoder(shape_meta, n_obs_steps=2, n_emb=16,
                                          n_head=2, ffn_dim=32, num_queries=queries,
                                          resampler_depth=1, dino_encoder=dino)
    tok_cfg = tokenizer_config()
    oat = hydra.utils.instantiate(tok_cfg)
    oat.normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
    kwargs = dict(shape_meta=shape_meta, obs_encoder=observation, action_tokenizer=oat,
                  tokenizer_config=tok_cfg, horizon=4, n_action_steps=2,
                  n_obs_steps=2, past_n=past_n, embed_dim=16, n_layers=2,
                  n_heads=2, ffn_dim=32, dropout=0., temperature=0.,
                  self_past_p=1., self_past_warmup_steps=0,
                  self_past_ramp_steps=0, self_past_chunk_size=2)
    if gate:
        kwargs.update(state_history_steps=past_n + 1, history_embed_dim=16,
                      history_n_heads=2, history_n_layers=1,
                      history_summary_tokens=4 if queries == 64 else 2,
                      history_dropout=0., history_gate_hidden_dim=16)
    kwargs.update(overrides)
    return (P2NStateGateNewPolicy if gate else P2NNewPolicy)(**kwargs)


def observation(policy, batch_size=2, valid=None):
    obs = policy.create_dummy_observation(batch_size)
    for key in policy.obs_encoder.rgb_ports:
        obs[key] = torch.randint(0, 256, obs[key].shape, dtype=torch.uint8)
    for key in policy.obs_encoder.state_ports:
        if not key.endswith(('_quat', '_rot6d')):
            obs[key] = torch.randn_like(obs[key]) * 0.1
    if policy.requires_state_history:
        if valid is None:
            valid = torch.ones(batch_size, policy.past_n, dtype=torch.bool)
        obs['state_history_valid'] = torch.cat((valid, torch.ones(batch_size, 1, dtype=torch.bool)), dim=1)
        for key in policy.state_history_keys:
            if not key.endswith(('_quat', '_rot6d')):
                obs['state_history__' + key] = torch.randn_like(obs['state_history__' + key]) * 0.1
            obs['state_history__' + key][:, -2:] = obs[key]
    return obs


def batch_for(policy, batch_size=2, *, padded=False, previous=False):
    valid = torch.ones(batch_size, policy.past_n, dtype=torch.bool)
    if padded:
        valid[:, 0] = False
    batch = dict(obs=observation(policy, batch_size, valid),
                 action=torch.randn(batch_size, 4, 7) * 0.1,
                 past_action=torch.randn(batch_size, policy.past_n, 7) * 0.1,
                 past_action_valid=valid)
    if previous:
        batch.update(prev_obs=observation(policy, batch_size, valid),
                     prev_past_action=torch.randn(batch_size, policy.past_n, 7) * 0.1,
                     prev_past_action_valid=valid.clone(),
                     prev_window_valid=torch.ones(batch_size, dtype=torch.bool))
    return batch


@pytest.mark.parametrize('gate', [False, True])
def test_default_condition_layout_and_no_unrequested_base_modules(gate):
    policy = make_policy(gate, queries=64, past_n=7).eval()
    batch = batch_for(policy)
    context = policy.build_context(batch['obs'], batch['past_action'], batch['past_action_valid'])
    assert context.memory.shape == (2, 271 if gate else 267, 16)
    assert int(context.segment_mask(Segment.VISUAL).sum()) == 256
    context.validate_variant(policy.variant, num_summary_tokens=4 if gate else 0)
    if gate:
        assert torch.allclose(context.history_log_gate.exp(), torch.full((2, 1), 0.9))
        with pytest.raises(KeyError, match='measured state-history'):
            policy.build_context({key: value for key, value in batch['obs'].items()
                                  if not key.startswith('state_history')},
                                 batch['past_action'], batch['past_action_valid'])
    else:
        assert not any('history_encoder' in key or 'history_gate' in key or 'observation_pool' in key
                       for key in policy.state_dict())
        assert not any(port.startswith('state_history') for port in policy.get_observation_ports())


@pytest.mark.parametrize('gate', [False, True])
@pytest.mark.parametrize('shortest', [False, True])
def test_actual_oat_training_freezes_backbones_and_reaches_every_trainable_parameter(gate, shortest):
    policy = make_policy(gate).train()
    batch = batch_for(policy)
    if shortest:
        batch['past_action_valid'].fill_(False)
        batch['past_action'].fill_(float('nan'))
        if gate:
            batch['obs']['state_history_valid'][:, :-1] = False
            for key in policy.state_history_keys:
                batch['obs']['state_history__' + key][:, :-1] = float('nan')
    loss = policy(batch, history_mode='expert')
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    assert not policy.action_tokenizer.training
    assert not policy.obs_encoder.dino_encoder.backbone.training
    assert policy.obs_encoder.dino_encoder.training
    assert policy.obs_encoder.patch_projection.weight.grad.abs().sum() > 0
    assert policy.model.tok_emb.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('gate', [False, True])
def test_nan_padding_isolation_and_difference_validity(gate):
    policy = make_policy(gate).eval()
    batch = batch_for(policy, padded=True)
    before = policy.build_context(batch['obs'], batch['past_action'], batch['past_action_valid'])
    altered = copy.deepcopy(batch)
    altered['past_action'][:, 0] = float('nan')
    if gate:
        for key in policy.state_history_keys:
            altered['obs']['state_history__' + key][:, 0] = float('nan')
    after = policy.build_context(altered['obs'], altered['past_action'], altered['past_action_valid'])
    torch.testing.assert_close(after.memory, before.memory)
    _, diff_valid = after.segment_memory(Segment.ACTION_DIFF)
    assert diff_valid.tolist() == [[True, False], [True, False]]
    torch.testing.assert_close(policy(batch, history_mode='expert'), policy(altered, history_mode='expert'))


def test_closed_gate_isolates_additional_state_history_but_preserves_raw_commands():
    policy = make_policy(True, history_gate_mode='closed').eval()
    batch = batch_for(policy)
    before = policy.build_context(batch['obs'], batch['past_action'], batch['past_action_valid'])
    changed = copy.deepcopy(batch)
    changed['obs']['state_history__robot0_eef_pos'][:, :2] += 50
    after = policy.build_context(changed['obs'], changed['past_action'], changed['past_action_valid'])
    tokens = torch.full((2, 1), policy.bos_id)
    torch.testing.assert_close(policy.model(tokens, before), policy.model(tokens, after))
    changed['past_action'][:, -1] += 10
    after_actions = policy.build_context(changed['obs'], changed['past_action'], changed['past_action_valid'])
    assert not torch.allclose(policy.model(tokens, before), policy.model(tokens, after_actions))
    assert not any(p.requires_grad for p in policy.history_gate.parameters())
    assert not any(p.requires_grad for p in policy.history_encoder.parameters())


@pytest.mark.parametrize('gate', [False, True])
def test_current_and_previous_action_validity_are_required(gate):
    policy = make_policy(gate)
    batch = batch_for(policy, previous=True)
    no_current = dict(batch)
    del no_current['past_action_valid']
    with pytest.raises(KeyError, match='past_action_valid'):
        policy(no_current)
    no_previous = dict(batch)
    del no_previous['prev_past_action_valid']
    with pytest.raises(KeyError, match='prev_past_action_valid'):
        policy(no_previous, history_mode='generated')
    with pytest.raises(ValueError, match='provided together'):
        policy.predict_action(batch['obs'], past_actions=batch['past_action'])
    if gate:
        mismatch = batch['past_action_valid'].clone()
        mismatch[:, 0] = False
        with pytest.raises(ValueError, match='aligned'):
            policy.build_context(batch['obs'], batch['past_action'], mismatch)


@pytest.mark.parametrize('gate', [False, True])
@pytest.mark.parametrize('mixed_precision', [False, True])
def test_self_past_filters_invalid_windows_chunks_restores_modes_and_keeps_gradients(gate, mixed_precision):
    policy = make_policy(gate).train()
    # Deliberately preserve one preexisting mixed module mode across rollout.
    policy.obs_encoder.patch_projection.eval()
    batch = batch_for(policy, batch_size=5, previous=True)
    batch['prev_window_valid'] = torch.tensor([True, False, True, True, False])
    invalid = ~batch['prev_window_valid']
    batch['prev_past_action'][invalid] = float('nan')
    for key in policy.obs_encoder.state_ports:
        batch['prev_obs'][key][invalid] = float('nan')
    if gate:
        batch['prev_obs']['state_history_valid'][invalid] = False
        for key in policy.state_history_keys:
            batch['prev_obs']['state_history__' + key][invalid] = float('nan')
    before_modes = [(module, module.training) for module in policy.modules()]
    calls = []
    hook = policy.obs_encoder.register_forward_pre_hook(
        lambda module, args: calls.append((args[0]['camera_a'].shape[0], module.training,
                                          torch.is_inference_mode_enabled())))
    autocast = torch.autocast('cpu', dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()
    try:
        with autocast:
            loss = policy(batch, history_mode='generated')
            assert torch.isfinite(loss)
            loss.backward()
    finally:
        hook.remove()
    assert calls == [(2, False, True), (1, False, True), (5, True, False)]
    assert all(module.training == mode for module, mode in before_modes)
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert policy._past_buffer is None and policy._pending_execution_steps is None
    assert policy.self_past_step == 0
    policy.on_optimizer_step()
    assert policy.self_past_step == 1
    policy.eval()
    policy.on_optimizer_step()
    assert policy.self_past_step == 1


@pytest.mark.parametrize('gate', [False, True])
def test_self_past_replacement_preserves_batch_indices_masks_and_original_input(gate, monkeypatch):
    policy = make_policy(gate).train()
    batch = batch_for(policy, batch_size=5, padded=True, previous=True)
    batch['prev_window_valid'] = torch.tensor([True, False, True, True, False])
    sample_ids = torch.arange(5, dtype=torch.float32)
    batch['prev_obs']['task_uid'] = sample_ids[:, None, None].expand(-1, 2, 1).clone()
    original = batch['past_action'].clone()
    seen = []

    def generate(obs, past, valid, *args):
        ids = obs['task_uid'][:, 0, 0]
        seen.append(ids.tolist())
        assert not policy.training and torch.is_inference_mode_enabled()
        return (ids[:, None, None] * 100 + torch.arange(4)[None, :, None]).expand(-1, -1, 7).clone()

    monkeypatch.setattr(policy, '_generate_actions', generate)
    generated = policy._maybe_self_past(batch, batch['past_action'], probability=1.)
    assert seen == [[0., 2.], [3.]]
    torch.testing.assert_close(batch['past_action'], original)
    assert not generated.is_inference()
    for index in range(5):
        torch.testing.assert_close(generated[index, 0], original[index, 0])
        if batch['prev_window_valid'][index]:
            torch.testing.assert_close(generated[index, -2], torch.full((7,), index * 100.))
            torch.testing.assert_close(generated[index, -1], torch.full((7,), index * 100. + 1))
        else:
            torch.testing.assert_close(generated[index], original[index])
    batch['prev_window_valid'].fill_(False)
    seen.clear()
    torch.testing.assert_close(policy._maybe_self_past(batch, original, probability=1.), original)
    assert not seen


@pytest.mark.parametrize('gate', [False, True])
def test_execution_feedback_partial_zero_stateless_and_reset(gate):
    policy = make_policy(gate).eval()
    valid = torch.zeros(2, 3, dtype=torch.bool)
    obs = observation(policy, valid=valid)
    first = policy.predict_action(obs)
    assert not policy._past_valid_buffer.any()
    assert not policy._past_buffer.any()
    assert policy._pending_execution_steps == 2
    with pytest.raises(RuntimeError, match='pending'):
        policy.predict_action(obs)
    pending = policy._pending_execution_steps
    policy.predict_action(obs, past_actions=torch.zeros(2, 3, 7), past_action_valid=valid)
    assert policy._pending_execution_steps == pending
    assert not policy._past_valid_buffer.any()
    commands = torch.full((2, 2, 7), float('nan'))
    commands[0, 0] = 0.25
    policy.record_executed_actions(commands, executed_lengths=torch.tensor([1, 0]))
    assert policy._past_valid_buffer.tolist() == [[False, False, True], [False, False, False]]
    torch.testing.assert_close(policy._past_buffer[0, -1], torch.full((7,), 0.25))
    history = policy._past_buffer.clone()
    obs = observation(policy, valid=policy._past_valid_buffer)
    policy.predict_action(obs)
    policy.record_executed_actions(torch.empty(2, 0, 7))
    torch.testing.assert_close(policy._past_buffer, history)
    assert policy._pending_execution_steps is None
    with pytest.raises(RuntimeError, match='No prediction'):
        policy.record_executed_actions(first['action'])
    with pytest.raises(RuntimeError, match='batch size changed'):
        policy.predict_action(observation(policy, batch_size=1, valid=torch.zeros(1, 3, dtype=torch.bool)))
    policy.reset()
    assert policy._past_buffer is None and policy._past_valid_buffer is None
    assert policy._pending_execution_steps is None
    policy.predict_action(observation(policy, batch_size=1, valid=torch.zeros(1, 3, dtype=torch.bool)))
    assert policy._past_buffer.shape[0] == 1


@pytest.mark.parametrize('gate', [False, True])
def test_optimizer_has_each_trainable_parameter_exactly_once_and_preserves_oat_normalizer(gate):
    policy = make_policy(gate)
    optimizer = policy.get_optimizer()
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in policy.parameters() if p.requires_grad}
    assert set(actual) == expected and len(actual) == len(expected)
    assert policy.model.tok_emb.weight is policy.model.head.weight
    tokenizer_normalizer = copy.deepcopy(policy.action_tokenizer.normalizer.state_dict())
    replacement = copy.deepcopy(policy.action_normalizer)
    replacement['action'] = SingleFieldLinearNormalizer.create_manual(
        scale=torch.full((7,), 2.), offset=torch.full((7,), 3.),
        input_stats_dict={'min': torch.zeros(7), 'max': torch.ones(7), 'mean': torch.zeros(7), 'std': torch.ones(7)})
    policy.set_normalizer(replacement)
    for key, value in tokenizer_normalizer.items():
        torch.testing.assert_close(policy.action_tokenizer.normalizer.state_dict()[key], value)


@pytest.mark.parametrize('gate', [False, True])
def test_self_contained_offline_checkpoint_strict_load_and_cross_variant_rejection(gate, tmp_path, monkeypatch):
    policy = make_policy(gate).train()
    policy.on_optimizer_step()
    policy.eval()
    batch = batch_for(policy)
    expected = policy.predict_action(batch['obs'], past_actions=batch['past_action'],
                                     past_action_valid=batch['past_action_valid'])['action_pred']
    exported = policy.export_config()
    exported['tokenizer_checkpoint'] = '/deleted/frozen_tokenizer.ckpt'
    exported['obs_encoder_config']['dino']['pretrained_path'] = '/deleted/dino_snapshot'
    config = OmegaConf.create({'policy': exported, 'training': {'use_ema': True}})
    path = tmp_path / 'policy.ckpt'
    payload = {'cfg': config, 'policy_config': exported,
               'state_dicts': {'model': policy.state_dict(), 'ema_model': policy.state_dict()}}
    torch.save(payload, path, pickle_module=dill)
    # Source constructors are forbidden in the mocked backbone as well as offline mode.
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    restored = type(policy).from_checkpoint(path)
    assert not restored.training and restored.self_past_step == 1
    assert restored._past_buffer is None and restored._pending_execution_steps is None
    actual = restored.predict_action(batch['obs'], past_actions=batch['past_action'],
                                      past_action_valid=batch['past_action_valid'])['action_pred']
    torch.testing.assert_close(actual, expected)
    assert restored.model.head.weight is restored.model.tok_emb.weight
    opposite = P2NNewPolicy if gate else P2NStateGateNewPolicy
    with pytest.raises(ValueError, match='variant'):
        opposite.from_checkpoint(path)
    incompatible_schema = copy.deepcopy(policy.shape_meta)
    incompatible_schema['obs']['task_uid']['shape'] = [2]
    with pytest.raises(ValueError, match='architecture/schema'):
        type(policy).from_checkpoint(path, policy_overrides={'shape_meta': incompatible_schema})
    with pytest.raises(ValueError, match='strict'):
        restored.load_state_dict(policy.state_dict(), strict=False)
    damaged = dict(policy.state_dict())
    del damaged['raw_proj.weight']
    with pytest.raises(RuntimeError, match='raw_proj.weight'):
        restored.load_state_dict(damaged)


@pytest.mark.parametrize('layout', ['rows', 'columns'])
@pytest.mark.parametrize('shortest', [False, True])
def test_real_robot_rot6d_dispatch_padding_and_shortest_history_backward(layout, shortest):
    from oat.policy.past2next_state_history_gate_real_robot import Rotation6DStateActionHistoryEncoder

    meta = copy.deepcopy(META)
    del meta['obs']['robot0_eef_quat']
    meta['obs']['robot0_eef_rot6d'] = {'shape': [6], 'type': 'state'}
    policy = make_policy(True, shape_meta=meta, task='real_robot', rotation_6d_layout=layout).train()
    assert isinstance(policy.history_encoder, Rotation6DStateActionHistoryEncoder)
    assert policy.rotation_6d_layout == layout
    batch = batch_for(policy, padded=True)
    if shortest:
        batch['past_action_valid'].fill_(False)
        batch['obs']['state_history_valid'][:, :-1] = False
    invalid_states = ~batch['obs']['state_history_valid']
    for key in policy.state_history_keys:
        batch['obs']['state_history__' + key][invalid_states] = float('nan')
    batch['past_action'][~batch['past_action_valid']] = float('nan')
    loss = policy(batch, history_mode='expert')
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert policy.history_encoder.input_projection[0].weight.grad.abs().sum() > 0
    assert policy.export_config()['rotation_6d_layout'] == layout
