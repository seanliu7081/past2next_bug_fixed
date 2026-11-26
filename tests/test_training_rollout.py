"""Training waits for same-GPU simulation and reclaims its resources afterwards."""
import os
import random
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from oat.env_runner.p2n_vla_rollout import resolve_renderer
from oat.workspace.train_policy import TrainPolicyWorkspace
import oat.workspace.train_policy as training


@pytest.mark.parametrize('fail', [False, True])
def test_rollout_scopes_rendering_and_restores_training_state(tmp_path, monkeypatch, fail):
    monkeypatch.setenv('MUJOCO_GL', 'egl')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-training')
    # A stale launcher selection must not override the training GPU's identity.
    monkeypatch.setenv('P2N_LIBERO_EGL_DEVICE_ID', '4')
    monkeypatch.delenv('MUJOCO_EGL_DEVICE_ID', raising=False)
    ws = TrainPolicyWorkspace(OmegaConf.create({}), output_dir=str(tmp_path))
    events = []
    accelerator = SimpleNamespace(
        device=torch.device('cpu'), is_main_process=True,
        wait_for_everyone=lambda: events.append('barrier'))
    policy = object()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    class Runner:
        def run(self, received):
            assert received is policy
            assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-training'
            assert 'MUJOCO_EGL_DEVICE_ID' not in os.environ
            events.append('evaluate')
            random.random()
            np.random.random()
            torch.rand(2)
            if fail:
                raise RuntimeError('simulator failed')
            return {'mean_success_rate': 0.75}

        def close(self):
            events.append('close')
            torch.rand(2)

    def construct(config, **kwargs):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'
        assert os.environ['MUJOCO_EGL_DEVICE_ID'] == '7'
        events.append('create')
        return Runner()

    def close(runner, *, force):
        assert force is fail
        runner.close()

    monkeypatch.setattr(training.hydra.utils, 'instantiate', construct)
    monkeypatch.setattr(training, 'close_runner', close)
    if fail:
        with pytest.raises(RuntimeError, match='simulator failed'):
            ws._run_policy_rollout(accelerator, policy, {}, {'egl_device_id': 7})
    else:
        assert ws._run_policy_rollout(
            accelerator, policy, {}, {'egl_device_id': 7}) == {'mean_success_rate': 0.75}

    expected = ['barrier', 'create', 'evaluate', 'close']
    assert events == expected + ([] if fail else ['barrier'])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-training'
    assert 'MUJOCO_EGL_DEVICE_ID' not in os.environ
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_non_main_rank_waits_without_creating_simulators(tmp_path, monkeypatch):
    ws = TrainPolicyWorkspace(OmegaConf.create({}), output_dir=str(tmp_path))
    events = []
    accelerator = SimpleNamespace(
        device=torch.device('cpu'), is_main_process=False,
        wait_for_everyone=lambda: events.append('barrier'))

    def unexpected(*args, **kwargs):
        pytest.fail('A non-main training rank started evaluation')

    monkeypatch.setattr(training.hydra.utils, 'instantiate', unexpected)
    assert ws._run_policy_rollout(accelerator, object(), {}) == {}
    assert events == ['barrier', 'barrier']


def test_renderer_follows_uuid_instead_of_cuda_or_egl_ordinal(monkeypatch):
    monkeypatch.setenv('MUJOCO_GL', 'egl')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: SimpleNamespace(uuid='training-uuid'))
    records = [dict(uuid='GPU-other-uuid', cuda_ordinal=7, egl_device_id=4),
               dict(uuid='GPU-training-uuid', cuda_ordinal=4, egl_device_id=7)]
    assert resolve_renderer('cuda:0', probe=lambda: records) == records[1]
