import numpy as np
import torch

from scripts.replay_pi05_weight_edit import dimension_metrics, replay_prediction, reproduction_metrics


class ToyPolicy:
    device = 'cpu'
    dtype = torch.float32

    def reset(self):
        pass

    def get_observation_ports(self):
        return ['sensor']

    def predict_action(self, obs, past_actions, past_action_valid):
        assert not past_action_valid.any()
        result = torch.randn(1, 2, 7) + obs['sensor'].sum() + past_actions.sum()
        return {'action': result[:, :1], 'action_pred': result}


def test_replay_restores_rng_and_preserves_context():
    torch.manual_seed(7)
    state = torch.get_rng_state().numpy().copy()
    archive = {'obs__sensor': np.array([[[1., 2.]]], dtype=np.float32),
               'history_actions': np.zeros((1,7,7), dtype=np.float32),
               'history_valid': np.zeros((1,7), dtype=bool), 'torch_rng_cpu': state}
    a = replay_prediction(ToyPolicy(), archive)
    torch.randn(1000)
    b = replay_prediction(ToyPolicy(), archive)
    assert np.array_equal(a['action_pred'], b['action_pred'])
    archive['obs__sensor'] += 1
    c = replay_prediction(ToyPolicy(), archive)
    assert not np.array_equal(a['action_pred'], c['action_pred'])


def test_reproduction_tolerance_and_dimension_metrics():
    ref = np.zeros((1,2,7), dtype=np.float32)
    changed = ref.copy(); changed[:,:,2] = 0.1
    assert not reproduction_metrics(ref, changed, atol=1e-6, rtol=1e-5)['passed']
    assert reproduction_metrics(ref, ref, atol=0, rtol=0)['bitwise_equal']
    metric = dimension_metrics(changed-ref)
    assert metric['z']['mean'] > 0.099
    assert metric['x']['max_abs'] == 0
