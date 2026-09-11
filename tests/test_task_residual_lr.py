"""Optional task-table LR: unique coverage, parity, adaptation and continuation."""
import copy
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import pytest
import torch

from oat.common.hydra_util import register_new_resolvers
from oat.model.diffusion.ema_model import EMAModel
from oat.perception.fused_obs_encoder import FusedObservationEncoder
from oat.perception.task_residual_fused_obs_encoder import TaskResidualFusedObservationEncoder
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy
from oat.policy.past2next_task_lr import Past2NextSelfPastTaskLRPolicy
from test_task_residual import SHAPE_META, FakeTokenizer, encoder, normalizer, observation, sink3_components

KWARGS = dict(policy_lr=1e-5, obs_enc_lr=2e-6, weight_decay=1e-4, betas=(.9, .95))


def make_policy(cls=Past2NextSelfPastTaskLRPolicy, encoder_cls=TaskResidualFusedObservationEncoder):
    policy = cls(SHAPE_META, encoder(encoder_cls), FakeTokenizer(), n_action_steps=8,
        n_obs_steps=2, past_n=7, embed_dim=16, n_layers=2, n_heads=2, dropout=0,
        temperature=0, self_past_schedule='optimizer_step', self_past_p=.5,
        self_past_warmup_steps=1000, self_past_ramp_steps=4000, self_past_temperature=1.0)
    policy.set_normalizer(normalizer())
    return policy


def parameter_ids(optimizer):
    return [id(parameter) for group in optimizer.param_groups for parameter in group['params']]


def test_default_returns_exact_parent_optimizer_without_any_rewrite():
    policy = make_policy()
    sentinel = object()
    with patch.object(Past2NextSelfPastPolicy, 'get_optimizer', return_value=sentinel) as parent:
        assert policy.get_optimizer(**KWARGS) is sentinel
        parent.assert_called_once_with(**KWARGS)
    assert len(policy.get_optimizer(**KWARGS).param_groups) == 4


def test_only_table_moves_and_every_parameter_keeps_unique_coverage():
    policy = make_policy()
    original = policy.get_optimizer(**KWARGS)
    candidate = policy.get_optimizer(**KWARGS, task_residual_lr=1e-3)
    table = policy.obs_encoder.task_residual.weight
    assert len(candidate.param_groups) == 5
    assert set(parameter_ids(candidate)) == set(parameter_ids(original))
    assert len(parameter_ids(candidate)) == len(set(parameter_ids(candidate)))
    assert parameter_ids(candidate).count(id(table)) == 1
    assert all(not parameter.requires_grad for parameter in policy.action_tokenizer.parameters())
    assert not any(id(parameter) in set(parameter_ids(candidate)) for parameter in policy.action_tokenizer.parameters())
    for before, after in zip(original.param_groups, candidate.param_groups[:4]):
        assert [id(parameter) for parameter in after['params']] == [id(parameter) for parameter in before['params'] if parameter is not table]
        assert {key:value for key,value in after.items() if key != 'params'} == {key:value for key,value in before.items() if key != 'params'}
    group = candidate.param_groups[4]
    assert len(group['params']) == 1 and group['params'][0] is table
    assert group['lr'] == 1e-3 and group['weight_decay'] == 1e-4
    assert not candidate.state


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '0.001'])
def test_invalid_learning_rates_rejected(value):
    with pytest.raises(ValueError, match='finite positive'):
        make_policy().get_optimizer(**KWARGS, task_residual_lr=value)


def test_missing_frozen_or_duplicated_table_cannot_silently_use_wrong_group():
    plain = make_policy(encoder_cls=FusedObservationEncoder)
    with pytest.raises(ValueError, match='TaskResidualFusedObservationEncoder'):
        plain.get_optimizer(**KWARGS, task_residual_lr=1e-3)
    policy = make_policy()
    policy.obs_encoder.task_residual.weight.requires_grad_(False)
    with pytest.raises(ValueError, match='trainable'):
        policy.get_optimizer(**KWARGS, task_residual_lr=1e-3)
    policy.obs_encoder.task_residual.weight.requires_grad_(True)
    duplicated = policy.get_optimizer(**KWARGS)
    duplicated.param_groups[0]['params'].append(policy.obs_encoder.task_residual.weight)
    with patch.object(Past2NextSelfPastPolicy, 'get_optimizer', return_value=duplicated):
        with pytest.raises(ValueError, match='duplicate'):
            policy.get_optimizer(**KWARGS, task_residual_lr=1e-3)


def test_fresh_strict_initialization_and_predictions_match_original():
    torch.manual_seed(819)
    original = make_policy(Past2NextSelfPastPolicy, FusedObservationEncoder).eval()
    original_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(819)
    candidate = make_policy().eval()
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    candidate.load_state_dict(original.state_dict(), strict=True)
    assert candidate.obs_encoder.task_residual.weight.count_nonzero() == 0
    assert set(candidate.state_dict()) - set(original.state_dict()) == {'obs_encoder.task_residual.weight'}
    for temperature in (0, .7):
        torch.manual_seed(246)
        expected = original.predict_action(observation(), temperature=temperature)
        torch.manual_seed(246)
        actual = candidate.predict_action(observation(), temperature=temperature)
        assert all(torch.equal(actual[key], value) for key, value in expected.items())
        assert torch.equal(candidate._past_buffer, original._past_buffer)
    state = original.state_dict()
    del state['model.tok_emb.weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        candidate.load_state_dict(state, strict=True)


def test_actual_loss_gradient_update_ema_and_five_group_roundtrip():
    torch.manual_seed(425)
    candidate = make_policy().train()
    slow = copy.deepcopy(candidate)
    ema_policy = copy.deepcopy(candidate)
    fast_optimizer = candidate.get_optimizer(**KWARGS, task_residual_lr=1e-3)
    slow_optimizer = slow.get_optimizer(**KWARGS)
    batch = {'obs':observation(), 'action':torch.zeros(2,16,7), 'past_action':torch.zeros(2,7,7)}
    loss = candidate(batch, history_mode='expert')
    assert torch.isfinite(loss)
    loss.backward()
    table = candidate.obs_encoder.task_residual.weight
    assert table.grad is not None and torch.isfinite(table.grad).all()
    assert table.grad[[0,9]].abs().sum() > 0 and table.grad[1:9].count_nonzero() == 0
    # Identical actual-loss gradients isolate the effect of the one LR change.
    for (_, target), (_, source) in zip(slow.named_parameters(), candidate.named_parameters()):
        target.grad = None if source.grad is None else source.grad.clone()
    fast_optimizer.step()
    slow_optimizer.step()
    torch.testing.assert_close(table, slow.obs_encoder.task_residual.weight * 500, rtol=1e-5, atol=1e-9)
    for name, value in candidate.state_dict().items():
        if name != 'obs_encoder.task_residual.weight':
            assert torch.equal(value, slow.state_dict()[name])
    assert table.count_nonzero() > 0 and table[1:9].count_nonzero() == 0
    ema = EMAModel(ema_policy)
    ema.step(candidate)
    assert torch.equal(ema_policy.obs_encoder.task_residual.weight, table)
    restored = make_policy()
    restored.load_state_dict(candidate.state_dict(), strict=True)
    assert not restored.obs_encoder.legacy_task_residual_initialized
    restored_optimizer = restored.get_optimizer(**KWARGS, task_residual_lr=1e-3)
    restored_optimizer.load_state_dict(fast_optimizer.state_dict())
    restored_table = restored.obs_encoder.task_residual.weight
    assert torch.equal(restored_table, table)
    assert restored_optimizer.state[restored_table]['step'].item() == 1
    assert torch.equal(restored_optimizer.state[restored_table]['exp_avg'], fast_optimizer.state[table]['exp_avg'])
    with pytest.raises(ValueError, match='parameter groups'):
        restored_optimizer.load_state_dict(slow_optimizer.state_dict())


def test_config_preserves_task_residual_variant_with_scratch_defaults():
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'oat/config'), version_base=None):
        cfg=compose(config_name='train_past2next_scratch_tasklr')
    assert cfg.policy._target_.endswith('.Past2NextSelfPastTaskLRPolicy')
    assert cfg.policy.obs_encoder._target_.endswith('.TaskResidualFusedObservationEncoder')
    assert cfg.training.init_checkpoint is None
    assert not cfg.training.resume and not cfg.logging.resume
    assert 'init_weights' not in cfg.training and 'init_allow_spatial_resize' not in cfg.training
    assert cfg.training.num_epochs == 251
    assert list(cfg.policy.obs_encoder.vision_encoder.crop_shape) == [112,112]
    assert cfg.optimizer.task_residual_lr == 1e-3 and cfg.optimizer.policy_lr == 1e-5 and cfg.optimizer.obs_enc_lr == 2e-6
    assert cfg.policy.self_past_temperature == 1 and cfg.policy.temperature == 0
    assert cfg.policy.self_past_warmup_steps == 1000 and cfg.policy.self_past_ramp_steps == 4000
    assert cfg.task.policy.dataset.val_ratio == .1 and cfg.task.policy.dataset.seed == 42
    assert cfg.training.max_val_steps is None and not cfg.val_dataloader.drop_last
    assert cfg.training.checkpoint_every == cfg.training.snapshot_every == 25
    assert cfg.policy.past_n == 7


def test_sink3_table_gets_independent_lr_and_gradients_from_12d_actions():
    meta, obs_encoder, norm, obs = sink3_components()
    candidate = Past2NextSelfPastTaskLRPolicy(
        meta, obs_encoder, FakeTokenizer(action_dim=12), n_action_steps=8,
        n_obs_steps=2, past_n=7, embed_dim=16, n_layers=2, n_heads=2, dropout=0,
        temperature=0, self_past_schedule='optimizer_step')
    candidate.set_normalizer(norm)
    optimizer = candidate.get_optimizer(**KWARGS, task_residual_lr=1e-3)
    table = candidate.obs_encoder.task_residual.weight
    assert table.shape == (3, 209)
    assert parameter_ids(optimizer).count(id(table)) == 1
    assert optimizer.param_groups[-1]['params'] == [table]
    assert optimizer.param_groups[-1]['lr'] == 1e-3
    batch = {'obs': obs, 'action': torch.zeros(3, 16, 12),
             'past_action': torch.zeros(3, 7, 12)}
    loss = candidate(batch, history_mode='expert')
    assert torch.isfinite(loss)
    loss.backward()
    assert table.grad is not None and torch.isfinite(table.grad).all()
    assert (table.grad.abs().sum(dim=-1) > 0).all()
    optimizer.step()
    assert (table.abs().sum(dim=-1) > 0).all()
