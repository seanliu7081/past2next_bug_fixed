"""Batched validation preserves each sample's original seeded evaluation.

CPU only: use actual small DiTX/OAT models, a local DINO test double and the
original full ResNet18 backbones. Exercise nonzero transformer outputs so seeded
noise, history generation and masks affect the checked metrics.
"""
from collections import defaultdict

import pytest
import torch

from test_p2n_new_policy import local_mock_dino
from test_p2n_latent_flow_policy import make_flow, make_batch
from test_p2n_latent_flow_resnet_policy import make_resnet_flow, make_resnet_batch


@pytest.fixture(autouse=True)
def cpu_threads_and_no_downloads(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise AssertionError('Validation tests must not download model weights')
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    import torchvision.models._api
    monkeypatch.setattr(torchvision.models._api, 'load_state_dict_from_url', forbidden)
    yield
    torch.set_num_threads(previous)


def _select(tree, indices):
    if isinstance(tree, dict):
        return {key: _select(value, indices) for key, value in tree.items()}
    return tree[indices]


def _make_case(encoder, gate, size=5):
    torch.manual_seed(473)
    policy = (make_resnet_flow if encoder == 'resnet18' else make_flow)(gate).eval()
    # Remove the exact-zero initialization that would hide conditioning bugs.
    with torch.no_grad():
        for name, parameter in policy.model.named_parameters():
            if 'modulation' in name or 'output' in name:
                torch.nn.init.normal_(parameter, std=.035)
    batch = (make_resnet_batch if encoder == 'resnet18' else make_batch)(policy, size)
    batch['prev_window_valid'][-1] = False
    batch['past_action_valid'][0, :3] = False
    batch['past_action'][0, :3] = 91.
    batch['future_action_valid'][1, 10:] = False
    if gate:
        batch['obs']['state_history_valid'][0, :3] = False
        # These invalid windows must be removed before gated-history encoding.
        batch['prev_obs']['state_history_valid'][-1].zero_()
        batch['prev_past_action_valid'][-1].zero_()
    seeds = [193, 991, 7127, 8819, 22433, 34757][:size]
    return policy, batch, seeds


def _sum_metrics(measurements):
    totals = defaultdict(lambda: {'sum': torch.zeros((), dtype=torch.float64),
                                  'count': torch.zeros((), dtype=torch.float64)})
    for measured in measurements:
        for name, value in measured.items():
            totals[name]['sum'] += value['sum'].detach().double().cpu()
            totals[name]['count'] += value['count'].detach().double().cpu()
    return dict(totals)


def _assert_metrics_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name]['count'], expected[name]['count'], rtol=0, atol=0)
        # Transformer/ResNet kernels may accumulate floating point differently
        # at different batch sizes; counts and all discrete masks remain exact.
        torch.testing.assert_close(actual[name]['sum'], expected[name]['sum'],
                                   rtol=2e-5, atol=2e-6, msg=lambda text: f'{name}: {text}')


@pytest.mark.parametrize('encoder', ['dinov3', 'resnet18'])
@pytest.mark.parametrize('gate', [False, True])
@pytest.mark.parametrize('decode', [False, True])
def test_batched_metrics_match_seeded_single_rows_with_padding(encoder, gate, decode):
    policy, batch, seeds = _make_case(encoder, gate)
    for mode in ('expert', 'generated'):
        reference = _sum_metrics(policy.validation_metrics(
            _select(batch, [index]), generator=torch.Generator().manual_seed(seed),
            history_mode=mode, compute_decoded=decode)
            for index, seed in enumerate(seeds))
        sizes = []
        hook = policy.model.register_forward_pre_hook(
            lambda _, args: sizes.append(args[0].shape[0]))
        try:
            measured = policy.validation_metrics(batch, history_mode=mode,
                compute_decoded=decode, sample_seeds=seeds)
        finally:
            hook.remove()
        _assert_metrics_equal(measured, reference)
        current_forwards = 1 + (policy.flow['inference_steps'] if decode else 0)
        assert sizes.count(5) == current_forwards
        if mode == 'generated':
            assert sizes.count(4) == policy.flow['self_past_steps']
            assert len(sizes) == current_forwards + policy.flow['self_past_steps']
        else:
            assert len(sizes) == current_forwards
        assert set(sizes) <= {4, 5}, 'Validation must use loader batches, not training history chunks'
        assert measured['fm_loss']['count'] == 5 * 8 * 5
        if decode:
            assert measured['decoded_action_mse']['count'] == int(batch['future_action_valid'].sum()) * 7
            assert measured['translation_mse']['count'] == int(batch['future_action_valid'].sum()) * 3
            assert measured['rotation_mse']['count'] == int(batch['future_action_valid'].sum()) * 3
            assert measured['gripper_mse']['count'] == int(batch['future_action_valid'].sum())
            assert measured['projection_legal']['sum'] == measured['projection_legal']['count']
        else:
            assert measured.keys() == {'fm_loss'}
    assert policy._past_buffer is None
    assert policy._past_valid_buffer is None
    assert policy._pending_execution_steps is None


def test_seeded_validation_is_invariant_to_order_partition_and_global_rng():
    policy, batch, seeds = _make_case('dinov3', True, size=6)
    observed = []
    def capture_fm(_, args, kwargs):
        if bool((kwargs['step_size'] == 0).all()):
            observed.append((args[0].detach().clone(), kwargs['time'].detach().clone()))
    hook = policy.model.register_forward_pre_hook(capture_fm, with_kwargs=True)
    try:
        full = policy.validation_metrics(batch, history_mode='generated', sample_seeds=seeds)
        expected_noise, expected_times = observed.pop()
        pieces = []
        order = [[4, 1], [5, 0, 3], [2]]
        for indices in order:
            torch.randn(37)  # Deliberately perturb the process-wide RNG stream.
            pieces.append(policy.validation_metrics(_select(batch, indices),
                history_mode='generated', sample_seeds=[seeds[index] for index in indices]))
            actual_noise, actual_times = observed.pop()
            torch.testing.assert_close(actual_times, expected_times[indices], rtol=0, atol=0)
            torch.testing.assert_close(actual_noise, expected_noise[indices], rtol=0, atol=0)
        _assert_metrics_equal(_sum_metrics(pieces), full)
        expert = policy.validation_metrics(batch, history_mode='expert', sample_seeds=seeds)
        expert_noise, expert_times = observed.pop()
        torch.testing.assert_close(expert_times, expected_times, rtol=0, atol=0)
        torch.testing.assert_close(expert_noise, expected_noise, rtol=0, atol=0)
        changed = policy.validation_metrics(batch, history_mode='expert',
            sample_seeds=[seed + 1 for seed in seeds], compute_decoded=False)
        assert not torch.isclose(changed['fm_loss']['sum'], expert['fm_loss']['sum'])
    finally:
        hook.remove()


def test_validation_preserves_pending_execution_and_original_training_modes():
    policy, batch, seeds = _make_case('dinov3', True)
    rollout_obs = policy.create_dummy_observation(1)
    first = policy.predict_action(rollout_obs, generator=torch.Generator().manual_seed(48))
    policy.record_executed_actions(first['action'], executed_lengths=[3])
    rollout_obs['state_history_valid'][:, -4:] = True
    policy.predict_action(rollout_obs, generator=torch.Generator().manual_seed(49))
    buffers = policy._past_buffer.clone(), policy._past_valid_buffer.clone()
    pending = policy._pending_execution_steps
    policy.train()
    modes = {name: module.training for name, module in policy.named_modules()}
    policy.validation_metrics(batch, history_mode='generated', sample_seeds=seeds)
    assert {name: module.training for name, module in policy.named_modules()} == modes
    assert torch.equal(policy._past_buffer, buffers[0])
    assert torch.equal(policy._past_valid_buffer, buffers[1])
    assert policy._pending_execution_steps == pending


def test_validation_seed_contract_rejects_ambiguous_or_misaligned_streams():
    policy, batch, seeds = _make_case('dinov3', False)
    with pytest.raises((TypeError, ValueError)):
        policy.validation_metrics(batch, generator=torch.Generator().manual_seed(1), sample_seeds=seeds)
    for invalid in (seeds[:-1], [True] * len(seeds), [1.5] * len(seeds)):
        with pytest.raises((TypeError, ValueError)):
            policy.validation_metrics(batch, sample_seeds=invalid)


class ValidationDataset(torch.utils.data.Dataset):
    """In-memory fixture with the real workspace's sample coverage metadata."""
    def __init__(self, batch):
        from types import SimpleNamespace
        self.batch = batch
        self.dataset_identity = 'batched-validation-regression-v1'
        self.pad_before = 0
        self.seq_sampler = SimpleNamespace(indices=[
            [int(sample_id), 0, 0, 0] for sample_id in batch['sample_id']])

    def __len__(self):
        return len(self.batch['action'])

    def __getitem__(self, index):
        return _select(self.batch, index)


class CPUValidationAccelerator:
    """Minimal CPU accelerator surface; distributed reductions remain real."""
    device = torch.device('cpu')

    def __init__(self, rank=0, world_size=1):
        self.process_index = rank
        self.num_processes = world_size
        self.is_main_process = rank == 0

    @staticmethod
    def autocast():
        from contextlib import nullcontext
        return nullcontext()

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def _workspace_validation_reference(policy, batch, dataset, seed, decode):
    from oat.workspace.train_p2n_latent_flow import stable_validation_seed
    expected = {}
    for mode in ('expert', 'generated'):
        totals = _sum_metrics(policy.validation_metrics(_select(batch, [index]),
            generator=torch.Generator().manual_seed(stable_validation_seed(
                seed, dataset.dataset_identity, int(sample_id))),
            history_mode=mode, compute_decoded=decode)
            for index, sample_id in enumerate(batch['sample_id']))
        expected.update({f'val_{mode}_{name}': float(pair['sum'] / pair['count'])
                         for name, pair in totals.items()})
    return expected


@pytest.mark.parametrize('decode', [False, True])
def test_workspace_evaluates_full_loader_batches_and_tail(decode, monkeypatch, tmp_path):
    import json
    from omegaconf import OmegaConf
    from oat.workspace.train_p2n_latent_flow import (
        TrainP2NLatentFlowWorkspace, stable_validation_seed)
    policy, batch, _ = _make_case('dinov3', True)
    dataset = ValidationDataset(batch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)
    cfg = OmegaConf.create({'training': {'seed': 327, 'tqdm_interval_sec': 0},
                            'policy': {'flow': {'inference_steps': 8}}})
    workspace = TrainP2NLatentFlowWorkspace(cfg, output_dir=str(tmp_path))
    workspace.ema_model = policy
    expected = _workspace_validation_reference(policy, batch, dataset, 327, decode)
    calls = []
    original = policy.validation_metrics
    def measured(value, *args, **kwargs):
        calls.append((kwargs['history_mode'], len(value['action']), kwargs['sample_seeds']))
        return original(value, *args, **kwargs)
    monkeypatch.setattr(policy, 'validation_metrics', measured)
    result = workspace._validate(CPUValidationAccelerator(), loader, dataset,
                                 generated=True, decode=decode)
    assert [(mode, size) for mode, size, _ in calls] == [
        ('expert', 4), ('generated', 4), ('expert', 1), ('generated', 1)]
    seeds = [stable_validation_seed(327, dataset.dataset_identity, index) for index in range(5)]
    assert [values for _, _, values in calls] == [seeds[:4], seeds[:4], seeds[4:], seeds[4:]]
    for name, value in expected.items():
        assert result[name] == pytest.approx(value, rel=2e-5, abs=2e-6), name
    assert result['validation_samples'] == 5
    assert result['validation_solver_steps'] == 8
    assert result['val_loss'] == result['val_expert_fm_loss']
    assert result['val_action_mse'] == result['val_expert_decoded_action_mse']
    if not decode:
        assert result['val_action_mse'] is None
    events = [json.loads(line) for line in (tmp_path / 'progress.jsonl').read_text().splitlines()]
    assert [event['completed'] for event in events] == [0, 4, 5]
    assert all(event['phase'] == 'validation' and event['total'] == 5 for event in events)
    assert events[-1]['eta_seconds'] == 0
