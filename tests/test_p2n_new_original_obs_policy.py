"""Real ResNet18/GN + real OAT/FSQ + small modern AR integration checks."""
import copy

import dill
import hydra
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.common.original_fused_context_batch import Segment
from oat.policy.p2n_new_original_obs import P2NNewOriginalObsPolicy, P2NStateGateNewOriginalObsPolicy
from test_p2n_new_policy import tokenizer_config, batch_for


META = {
    'obs': {
        'camera_a': {'shape': [128, 128, 3], 'type': 'rgb'},
        'camera_b': {'shape': [128, 128, 3], 'type': 'rgb'},
        'robot0_eef_pos': {'shape': [3], 'type': 'state'},
        'robot0_eef_rot6d': {'shape': [6], 'type': 'state'},
        'robot0_gripper_qpos': {'shape': [1], 'type': 'state'},
        'task_uid': {'shape': [1], 'type': 'state'},
    }, 'action': {'shape': [7]},
}


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_policy(gate=False, **overrides):
    torch.manual_seed(122)
    tok_cfg = tokenizer_config()
    tokenizer = hydra.utils.instantiate(tok_cfg)
    tokenizer.normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
    kwargs = dict(shape_meta=copy.deepcopy(META), action_tokenizer=tokenizer,
                  tokenizer_config=tok_cfg, horizon=4, n_action_steps=2,
                  n_obs_steps=2, past_n=7, embed_dim=16, n_layers=2,
                  n_heads=2, ffn_dim=32, dropout=0., temperature=0., task='real_robot',
                  expected_action_tokens=2, self_past_p=.5, self_past_warmup_steps=0,
                  self_past_ramp_steps=0, self_past_chunk_size=2)
    if gate:
        kwargs.update(state_history_steps=8, history_embed_dim=16, history_n_heads=2,
                      history_n_layers=1, history_summary_tokens=4, history_dropout=0.,
                      history_gate_hidden_dim=16)
    kwargs.update(overrides)
    policy = (P2NStateGateNewOriginalObsPolicy if gate else P2NNewOriginalObsPolicy)(**kwargs)
    normalizer = LinearNormalizer()
    for name in ['action', *policy.obs_ports]:
        # Deliberately nonidentity state stats catch double normalization.
        endpoints = torch.tensor([[0.], [255.]]) if name.startswith('camera_') else torch.tensor([[-2.], [6.]])
        normalizer[name] = SingleFieldLinearNormalizer.create_fit(endpoints)
    policy.set_normalizer(normalizer)
    return policy


@pytest.mark.parametrize('gate', [False, True])
def test_context_order_normalization_and_optimizer(gate):
    policy = make_policy(gate).eval()
    batch = batch_for(policy, padded=True)
    batch['past_action'][~batch['past_action_valid']] = float('nan')
    context = policy.build_context(batch['obs'], batch['past_action'], batch['past_action_valid'])
    assert context.memory.shape == (2, 15 if gate else 11, 16)
    context.validate_variant(policy.variant, 4 if gate else 0)
    assert context.segment_ids.tolist() == [5] * 2 + [2] * 7 + [3] * 2 + ([4] * 4 if gate else [])
    assert policy.type_embedding.shape == (4, 16)
    assert policy.obs_encoder.fused_feature_dim == 139
    assert not hasattr(policy.obs_encoder, 'resampler')
    assert not hasattr(policy, 'history_encoder') if not gate else isinstance(policy.observation_pool, nn.Identity)
    fused = policy.obs_encoder.encode_fused(batch['obs'])
    expected_state = torch.cat([policy.action_normalizer[key].normalize(batch['obs'][key])
                                for key in policy.obs_encoder.state_ports], dim=-1)
    torch.testing.assert_close(fused[..., 128:], expected_state)
    torch.testing.assert_close(context.memory[:, :2], policy.obs_encoder(batch['obs']) + policy.type_embedding[0])
    assert torch.isfinite(context.memory).all()
    assert not context.memory[~context.valid_mask].any()
    if gate:
        torch.testing.assert_close(context.observation_summary, context.memory[:, :2].mean(1))
        torch.testing.assert_close(context.history_log_gate.exp(), torch.full((2, 1), .9))
    optimizer = policy.get_optimizer(policy_lr=5e-5, obs_enc_lr=1e-5)
    rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    assert len(rates) == sum(len(group['params']) for group in optimizer.param_groups)
    assert set(rates) == {id(p) for p in policy.parameters() if p.requires_grad}
    original = {id(p) for p in policy.obs_encoder.fused_encoder.parameters()}
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert rates[id(parameter)] == (1e-5 if id(parameter) in original else 5e-5), name
    assert rates[id(policy.obs_encoder.obs_projection.weight)] == 5e-5
    assert not any(p.requires_grad for p in policy.action_tokenizer.parameters())


@pytest.mark.parametrize('gate', [False, True])
def test_generated_history_bf16_backward_and_mode_restore(gate):
    policy = make_policy(gate).train()
    policy.obs_encoder.obs_projection.eval()
    modes = [(module, module.training) for module in policy.modules()]
    batch = batch_for(policy, previous=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = policy(batch, history_mode='generated')
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    for module, mode in modes:
        assert module.training == mode
    assert policy.obs_encoder.fused_encoder.training
    assert not policy.action_tokenizer.training
    assert policy.obs_encoder.obs_projection.weight.grad.abs().sum() > 0
    rgb_encoder = policy.obs_encoder.fused_encoder.vision_encoder.encoder
    for key in policy.obs_encoder.rgb_ports:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in rgb_encoder.obs_nets[key].parameters())
    assert policy.self_past_step == 0
    policy.on_optimizer_step()
    assert policy.self_past_step == 1


@pytest.mark.parametrize('mode', ['learned', 'open', 'closed'])
def test_gate_bias_metadata_and_closed_history_isolation(mode):
    policy = make_policy(True, history_gate_mode=mode).eval()
    batch = batch_for(policy, padded=True)
    context = policy.build_context(batch['obs'], batch['past_action'], batch['past_action_valid'])
    context.validate_variant(policy.variant, 4)
    bias = context.attention_bias(dtype=torch.float32)
    assert torch.isfinite(bias[..., :2]).all()
    if mode == 'closed':
        assert torch.isneginf(bias[..., -4:]).all()
        tokens = torch.full((2, 2), policy.bos_id, dtype=torch.long)
        before = policy.model(tokens, context)
        changed = copy.deepcopy(batch['obs'])
        changed['state_history__robot0_eef_pos'] += 100
        after_context = policy.build_context(changed, batch['past_action'], batch['past_action_valid'])
        after = policy.model(tokens, after_context)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    if mode == 'open':
        assert not context.history_log_gate.any()


@pytest.mark.parametrize('gate', [False, True])
def test_checkpoint_offline_strict_and_ema_equivalence(gate, tmp_path):
    policy = make_policy(gate).eval()
    batch = batch_for(policy)
    prediction = policy.predict_action(batch['obs'], past_actions=batch['past_action'],
                                       past_action_valid=batch['past_action_valid'])['action_pred']
    exported = policy.export_config()
    exported['tokenizer_checkpoint'] = '/deleted/tokenizer.ckpt'
    path = tmp_path / 'original.ckpt'
    torch.save(dict(cfg=OmegaConf.create(dict(policy=exported, training=dict(use_ema=True))),
                    policy_config=exported, state_dicts=dict(model=policy.state_dict(), ema_model=policy.state_dict())),
               path, pickle_module=dill)
    restored = type(policy).from_checkpoint(path)
    actual = restored.predict_action(batch['obs'], past_actions=batch['past_action'],
                                     past_action_valid=batch['past_action_valid'])['action_pred']
    torch.testing.assert_close(actual, prediction, rtol=0, atol=0)
    assert set(restored.state_dict()) == set(policy.state_dict())
    assert restored.model.head.weight is restored.model.tok_emb.weight
    assert all(not p.requires_grad for module in restored.modules() if isinstance(module, LinearNormalizer)
               for p in module.parameters())
    for field, value in [('obs_encoder_type', 'dinov3_tokens'), ('context_schema_version', 1),
                         ('original_obs_config', {'crop_shape': [76, 76]})]:
        with pytest.raises(ValueError, match='cannot be overridden'):
            type(policy).from_checkpoint(path, policy_overrides={field: value})
    reordered = copy.deepcopy(policy.shape_meta)
    reordered['obs'] = dict(reversed(list(reordered['obs'].items())))
    with pytest.raises(ValueError, match='field order'):
        type(policy).from_checkpoint(path, policy_overrides={'shape_meta': reordered})
    for prefix in ('action_normalizer', 'action_tokenizer.normalizer'):
        damaged_stats = dict(policy.state_dict())
        del damaged_stats[f'{prefix}.params_dict.action.scale']
        with pytest.raises(ValueError, match='normalizer'):
            restored.load_state_dict(damaged_stats)
    broken = dict(policy.state_dict())
    del broken['obs_encoder.obs_projection.weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(broken)
    with pytest.raises(ValueError, match='strict'):
        restored.load_state_dict(policy.state_dict(), strict=False)
    broken = dict(policy.state_dict())
    broken['_context_schema'] = torch.tensor(1)
    with pytest.raises(ValueError, match='schema'):
        restored.load_state_dict(broken)


def test_normalizer_requires_rgb_and_preserves_tokenizer():
    policy = make_policy()
    original = copy.deepcopy(policy.action_tokenizer.normalizer.state_dict())
    missing = LinearNormalizer()
    for key in ['action', *policy.obs_encoder.state_ports]:
        missing[key] = SingleFieldLinearNormalizer.create_identity()
    with pytest.raises(KeyError, match='camera_a'):
        policy.set_normalizer(missing)
    for key, value in original.items():
        torch.testing.assert_close(value, policy.action_tokenizer.normalizer.state_dict()[key])
