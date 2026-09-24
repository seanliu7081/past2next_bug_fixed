"""Exercise the bounded acceptance script on explicit tiny offline test modules."""
import copy

import numpy as np
from omegaconf import OmegaConf
import pytest

from scripts import smoke_p2n_new as smoke
from test_p2n_new_policy import local_mock_dino, make_policy, batch_for


@pytest.mark.parametrize('gate', [False, True])
def test_complete_smoke_control_flow_with_explicit_mock_backbone(gate, monkeypatch, tmp_path):
    policy = make_policy(gate)
    batch = batch_for(policy, 3, previous=True)

    def item(value, index):
        return {key: item(entry, index) for key, entry in value.items()} if isinstance(value, dict) else value[index]

    class Dataset:
        train_mask = np.array([True, False])

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return item(batch, index)

        def get_normalizer(self):
            return copy.deepcopy(policy.action_normalizer)

        def get_validation_dataset(self):
            result = copy.copy(self)
            result.train_mask = ~self.train_mask
            return result

    variant = 'p2n_state_gate_new' if gate else 'p2n_new'
    cfg = OmegaConf.create(dict(seed=42, policy={'_target_': 'test.policy'},
                               task={'policy': {'dataset': {'_target_': 'test.dataset', 'zarr_path': 'mock'}}},
                               training={'use_ema': True, 'max_grad_norm': 1.},
                               optimizer={'policy_lr': 1e-4, 'obs_enc_lr': 1e-4},
                               ema={'_target_': 'oat.model.diffusion.ema_model.EMAModel', 'power': 0.75}))
    monkeypatch.setattr(smoke, 'compose_config', lambda args: cfg)
    monkeypatch.setattr(smoke, 'validate_config', lambda *args: None)
    monkeypatch.setattr(smoke, 'validate_tokenizer_source', lambda config: None)
    monkeypatch.setattr(smoke, 'inspect_dataset', lambda config: {'test_only': True})
    instantiate = smoke.hydra.utils.instantiate
    def instantiate_test(config, **kwargs):
        if config._target_ == 'test.policy':
            return policy
        if config._target_ == 'test.dataset':
            return Dataset()
        return instantiate(config, **kwargs)
    monkeypatch.setattr(smoke.hydra.utils, 'instantiate', instantiate_test)
    report = smoke.main(['--variant', variant, '--task', 'libero', '--dino', '/explicit-mock-only',
                         '--batch-size', '2', '--val-batch-size', '2', '--warmup', '0',
                         '--iterations', '1', '--precision', 'fp32', '--output', str(tmp_path / 'smoke.json')])
    assert report['status'] == 'smoke_passed'
    assert report['successful_optimizer_updates'] == 3
    assert report['adam_parameter_states'] > 0
    assert report['ema_resident']
    assert report['ema_optimizer_updates'] == 3
    assert {entry['batch_size'] for entry in report['measurements']['predict_action']} == {1, 2}
    for name in ('train_expert', 'train_generated', 'validation_expert', 'validation_generated'):
        assert np.isfinite(report['measurements'][name]['loss'])


def test_real_weight_smoke_rejects_missing_local_dino_and_zero_iterations():
    with pytest.raises(SystemExit):
        smoke.parse_args(['--variant', 'p2n_new', '--task', 'libero'])
    with pytest.raises(SystemExit):
        smoke.parse_args(['--variant', 'p2n_new', '--task', 'libero', '--count-only', '--iterations', '0'])
