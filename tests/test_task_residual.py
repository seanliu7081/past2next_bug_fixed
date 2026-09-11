"""CPU tests for optional zero-initialized categorical task conditioning."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from oat.common.hydra_util import register_new_resolvers
from oat.model.common.normalizer import LinearNormalizer
from oat.model.diffusion.ema_model import EMAModel
from oat.perception.fused_obs_encoder import FusedObservationEncoder
from oat.perception.state_encoder import ProjectionStateEncoder
from oat.perception.task_residual_fused_obs_encoder import TaskResidualFusedObservationEncoder
from oat.policy.past2next import Past2NextPolicy


SHAPE_META = {
    'obs': {
        'agentview_rgb': {'shape': [128, 128, 3], 'type': 'rgb'},
        'robot0_eye_in_hand_rgb': {'shape': [128, 128, 3], 'type': 'rgb'},
        'robot0_eef_pos': {'shape': [3], 'type': 'state'},
        'robot0_eef_quat': {'shape': [4], 'type': 'state'},
        'robot0_gripper_qpos': {'shape': [2], 'type': 'state'},
        'task_uid': {'shape': [1], 'type': 'state'},
    },
    'action': {'shape': [7]},
}


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-1, 1, 128))

    def output_feature_dim(self):
        return 128

    def forward(self, obs):
        return obs['image_features'] + self.bias

    def set_normalizer(self, normalizer):
        pass


def normalizer():
    data = {'action': torch.linspace(-1, 1, 14).reshape(2, 7)}
    for key, meta in SHAPE_META['obs'].items():
        if meta['type'] == 'state':
            size = meta['shape'][0]
            data[key] = torch.linspace(-1, 1, 2 * size).reshape(2, size)
    data['task_uid'] = torch.tensor([[30], [39]])
    result = LinearNormalizer()
    result.fit(data)
    return result


def encoder(cls=TaskResidualFusedObservationEncoder):
    # Substitute only the expensive visual backbone; use the actual state
    # encoder, UID normalizer, fused forward, and residual implementation.
    with patch('oat.perception.fused_obs_encoder.hydra.utils.instantiate',
               side_effect=lambda specification, **kwargs: specification):
        result = cls(SHAPE_META, vision_encoder=FakeVision(),
                     state_encoder=ProjectionStateEncoder(SHAPE_META, out_dim=None))
    result.set_normalizer(normalizer())
    return result


def observation(uids=(30, 39)):
    batch = len(uids)
    obs = {'image_features': torch.linspace(-.25, .25, batch * 2 * 128).reshape(batch, 2, 128)}
    for key, meta in SHAPE_META['obs'].items():
        if meta['type'] == 'state':
            obs[key] = torch.zeros(batch, 2, *meta['shape'])
    obs['task_uid'] = torch.tensor(uids)[:, None, None].expand(-1, 2, 1).clone()
    return obs


class FakeTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.quantizer = SimpleNamespace(codebook_size=11)
        self.latent_horizon = 8

    def tokenize(self, actions):
        return (actions[:, :8, 0] * 5).round().long().remainder(11)

    def detokenize(self, tokens):
        values = tokens.float().repeat_interleave(2, dim=1)
        return values[..., None].expand(-1, -1, 7).clone() / 10


def policy(cls=TaskResidualFusedObservationEncoder):
    result = Past2NextPolicy(SHAPE_META, encoder(cls), FakeTokenizer(),
                            n_action_steps=8, n_obs_steps=2, past_n=7,
                            embed_dim=16, n_layers=2, n_heads=2, dropout=0,
                            temperature=0)
    result.set_normalizer(normalizer())
    return result


def test_constructor_rng_feature_and_existing_state_parity():
    torch.manual_seed(517)
    control = encoder(FusedObservationEncoder)
    control_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(517)
    candidate = encoder()
    assert torch.equal(torch.random.get_rng_state(), control_rng)
    candidate.load_state_dict(control.state_dict(), strict=True)
    assert candidate.output_feature_dim() == control.output_feature_dim() == 138
    assert candidate.task_residual.weight.shape == (10, 138)
    assert candidate.task_residual.weight.requires_grad
    assert candidate.task_residual.weight.count_nonzero() == 0
    assert set(candidate.state_dict()) - set(control.state_dict()) == {'task_residual.weight'}
    for key, value in control.state_dict().items():
        assert torch.equal(candidate.state_dict()[key], value)
    obs = observation(tuple(range(30, 40)))
    expected, actual = control(obs), candidate(obs)
    assert torch.equal(actual, expected)
    # The normalized scalar UID is still the final original feature.
    assert torch.equal(actual[..., -1], control.state_encoder(obs)[..., -1])
    torch.testing.assert_close(actual[:, 0, -1], torch.linspace(-1, 1, 10), atol=1e-6, rtol=0)


def test_legacy_load_resets_only_absent_table_even_after_training():
    control, candidate = encoder(FusedObservationEncoder), encoder()
    with torch.no_grad():
        candidate.task_residual.weight.fill_(5)
    source = control.state_dict()
    candidate.load_state_dict(source, strict=True)
    assert candidate.legacy_task_residual_initialized
    assert candidate.task_residual.weight.count_nonzero() == 0
    assert 'task_residual.weight' not in source  # Caller state dict is untouched.
    assert torch.equal(candidate(observation()), control(observation()))


@pytest.mark.parametrize('corruption', ['missing_existing', 'unexpected', 'wrong_existing_shape', 'wrong_table_shape'])
def test_strict_loading_rejects_every_other_mismatch(corruption):
    candidate = encoder()
    state = encoder(FusedObservationEncoder).state_dict()
    if corruption == 'missing_existing':
        del state['vision_encoder.bias']
    elif corruption == 'unexpected':
        state['unrelated.weight'] = torch.zeros(1)
    elif corruption == 'wrong_existing_shape':
        state['vision_encoder.bias'] = torch.zeros(127)
    else:
        state['task_residual.weight'] = torch.zeros(10, 137)
    with pytest.raises(RuntimeError):
        candidate.load_state_dict(state, strict=True)


def test_learned_table_roundtrip_and_raw_uid_row_selection():
    candidate = encoder()
    obs = observation((30, 31, 39))
    baseline = candidate(obs).detach()
    with torch.no_grad():
        candidate.task_residual.weight.copy_(torch.arange(10)[:, None].expand(-1, 138) / 10)
    expected = baseline + torch.tensor([0, .1, .9])[:, None, None]
    torch.testing.assert_close(candidate(obs), expected)
    restored = encoder()
    restored.load_state_dict(candidate.state_dict(), strict=True)
    assert not restored.legacy_task_residual_initialized
    assert torch.equal(restored.task_residual.weight, candidate.task_residual.weight)
    assert torch.equal(restored(obs), candidate(obs))


@pytest.mark.parametrize('bad_uid', [29, 40, 30.5, float('nan'), float('inf'), True, 30+0j])
def test_raw_uid_validation(bad_uid):
    candidate = encoder()
    obs = observation((30,))
    obs['task_uid'] = torch.full((1, 2, 1), bad_uid)
    with pytest.raises(ValueError, match='task_uid'):
        candidate(obs)


def test_uid_shape_and_missing_port_validation():
    candidate = encoder()
    obs = observation()
    del obs['task_uid']
    with pytest.raises(ValueError, match='task_uid'):
        candidate(obs)
    obs['task_uid'] = torch.full((2, 2), 30)
    with pytest.raises(ValueError, match='task_uid'):
        candidate(obs)
    meta = copy.deepcopy(SHAPE_META)
    del meta['obs']['task_uid']
    with pytest.raises(ValueError, match='scalar task_uid'):
        TaskResidualFusedObservationEncoder(meta)


def test_residual_dtype_follows_existing_features():
    candidate = encoder().double()
    obs = {key: value.double() for key, value in observation().items()}
    assert candidate(obs).dtype == torch.float64
    candidate = encoder()
    with patch.object(FusedObservationEncoder, 'forward', return_value=torch.zeros(2, 2, 138, dtype=torch.bfloat16)):
        assert candidate(observation()).dtype == torch.bfloat16


def test_full_policy_initial_inference_and_history_parity():
    torch.manual_seed(88)
    control = policy(FusedObservationEncoder).eval()
    torch.manual_seed(88)
    candidate = policy().eval()
    candidate.load_state_dict(control.state_dict(), strict=True)
    for temperature in (0, .8):
        torch.manual_seed(314)
        expected = control.predict_action(observation(), temperature=temperature)
        torch.manual_seed(314)
        actual = candidate.predict_action(observation(), temperature=temperature)
        assert torch.equal(actual['action_pred'], expected['action_pred'])
        assert torch.equal(actual['action'], expected['action'])
        assert torch.equal(candidate._past_buffer, control._past_buffer)
    saved = candidate._past_buffer
    past = torch.randn(2, 7, 7)
    candidate.predict_action(observation(), past_actions=past)
    assert candidate._past_buffer is saved
    features = candidate.obs_encoder(observation())
    condition = candidate._build_condition(features, past)
    assert condition.shape == (2, 11, 138)
    assert candidate.past_n == 7 and candidate.max_seq_len == 8


def test_table_receives_finite_gradients_optimizer_updates_and_ema():
    candidate = policy().train()
    optimizer = candidate.get_optimizer(policy_lr=1e-5, obs_enc_lr=2e-6,
                                       weight_decay=1e-4, betas=(.9, .95))
    table = candidate.obs_encoder.task_residual.weight
    groups = [group for group in optimizer.param_groups if any(param is table for param in group['params'])]
    assert len(groups) == 1
    assert groups[0]['lr'] == 2e-6 and groups[0]['weight_decay'] == 1e-4
    assert optimizer.state == {}
    ema = EMAModel(copy.deepcopy(candidate))
    batch = {'obs': observation(), 'action': torch.randn(2, 16, 7), 'past_action': torch.randn(2, 7, 7)}
    loss = candidate(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert table.grad is not None and torch.isfinite(table.grad).all()
    assert table.grad[[0, 9]].abs().sum() > 0
    assert table.grad[1:9].count_nonzero() == 0
    optimizer.step()
    assert table.count_nonzero() > 0
    ema.step(candidate)
    assert torch.equal(ema.averaged_model.obs_encoder.task_residual.weight, table)
    assert not ema.averaged_model.obs_encoder.task_residual.weight.requires_grad




def test_retained_branches_share_policy_shape_and_initialization():
    register_new_resolvers()
    directory = str(Path(__file__).resolve().parents[1] / 'oat/config')
    with initialize_config_dir(config_dir=directory, version_base=None):
        all500 = compose(config_name='train_past2next_finetune_all500')
        tasklr = compose(config_name='train_past2next_finetune_tasklr')
    for cfg in (all500, tasklr):
        assert cfg.policy.n_layers == cfg.policy.n_heads == 8
        assert cfg.policy.past_n == 7
        assert list(cfg.policy.obs_encoder.vision_encoder.crop_shape) == [112, 112]
        assert cfg.training.init_weights == 'ema' and not cfg.training.resume
        assert cfg.policy.self_past_temperature == 1
    assert all500.training.init_checkpoint == tasklr.training.init_checkpoint
    assert all500.task.policy.dataset.val_ratio == 0
    assert tasklr.task.policy.dataset.val_ratio == .1
    assert all500.policy.obs_encoder._target_.endswith('.FusedObservationEncoder')
    assert tasklr.policy.obs_encoder._target_.endswith('.TaskResidualFusedObservationEncoder')
