"""New vision/action policy integration with the unchanged VLA workspace.

Runs tiny actual SigLIP/Gemma/expert models on CPU, with a synthetic held-out
zarr. The original workspace/evaluator are reused without monkeypatching their
policy dispatch, checkpoint formats, optimizer ownership, or EMA lifecycle.
"""
import copy
from types import SimpleNamespace
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace as Workspace
from scripts.evaluate_p2n_vla import load_policy, policy_class
from p2n_vla_vision_action_fixtures import make_test_tokenizer_checkpoint
from test_p2n_vla_workspace import (
    load, make_libero_like_zarr, real_tiny_config, run_workspace,
)


TARGETS = {
    'p2n_vla': 'oat.policy.p2n_vla_vision_action.P2NVLAVisionActionPolicy',
    'p2n_vla_state_gate': 'oat.policy.p2n_vla_vision_action.P2NVLAStateGateVisionActionPolicy',
}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    from accelerate.state import AcceleratorState
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    AcceleratorState._reset_state(True)
    yield
    AcceleratorState._reset_state(True)
    torch.set_num_threads(previous_threads)


def config(variant, data, *overrides):
    cfg = real_tiny_config(variant, data, 'training.num_epochs=1',
                           'task.policy.lazy_eval=true', *overrides)
    tokenizer = make_test_tokenizer_checkpoint(data.parent / 'test_only_oat.ckpt')
    cfg.policy.tokenizer_checkpoint = str(tokenizer)
    cfg.policy.spm_path = str(Path(__file__).resolve().parents[1] /
                              'data/pretrained/p2n_vla/paligemma_tokenizer.model')
    cfg.policy._target_ = TARGETS[variant]
    OmegaConf.update(cfg, 'policy.training_scope', 'vision_action', force_add=True)
    OmegaConf.update(cfg, 'policy.vision_activation_checkpointing', True, force_add=True)
    return cfg


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            assert_nested_equal(actual[key], value)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_nested_equal(first, second)
    else:
        assert actual == expected


@pytest.mark.parametrize('variant', tuple(TARGETS))
def test_unchanged_workspace_saves_vision_optimizer_ema_and_loads_eval_target(tmp_path, variant):
    data = make_libero_like_zarr(tmp_path / 'data.zarr')
    cfg = config(variant, data)
    workspace = run_workspace(cfg, tmp_path / 'run')
    assert workspace.completed_optimizer_steps == workspace.ema.updates == 1
    checkpoint = tmp_path / 'run/checkpoints/latest.ckpt'
    payload = load(checkpoint)
    names = payload['training']['trainable_names']
    assert any(name.startswith('siglip.') for name in names)
    assert not any(name.startswith('joint.vlm.') for name in names)
    assert any(name.startswith('siglip.') for name in payload['training']['ema']['names'])
    assert payload['metadata']['training_scope'] == 'vision_action'
    assert payload['policy_config']['_target_'] == TARGETS[variant]
    assert payload['policy_config']['training_scope'] == 'vision_action'
    assert payload['policy_config']['vision_activation_checkpointing'] is True
    assert payload['metadata']['language_trainable'] is False
    assert all(not parameter.requires_grad for parameter in workspace.model.joint.vlm.parameters())
    assert all(parameter.requires_grad for parameter in workspace.model.siglip.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in workspace.model.siglip.parameters())

    vision = dict(workspace.model.siglip.named_parameters())
    optimizer = workspace.optimizer.optimizer
    for name, parameter in vision.items():
        full_name = 'siglip.' + name
        assert full_name in payload['state_dicts']['model']
        assert full_name in payload['state_dicts']['ema_model']
        assert full_name in workspace.ema.shadow
        assert 'exp_avg' in optimizer.state[parameter]
        assert torch.isfinite(optimizer.state[parameter]['exp_avg']).all()
        assert optimizer.state[parameter]['step'].item() == 1
    assert any(optimizer.state[p]['exp_avg'].abs().sum() > 0 for p in vision.values())

    Workspace.validate_resume_payload(payload, cfg, world_size=1)
    selected = policy_class(payload)
    assert f'{selected.__module__}.{selected.__name__}' == TARGETS[variant]
    restored = load_policy(SimpleNamespace(snapshot=checkpoint, weights='ema', pi05=None,
                                            spm=None, device='cpu'), payload)
    assert not restored.training
    assert all(not parameter.requires_grad for parameter in restored.joint.vlm.parameters())
    for name, parameter in restored.siglip.named_parameters():
        torch.testing.assert_close(parameter, workspace.ema.shadow['siglip.' + name], rtol=0, atol=0)

    snapshot = next((tmp_path / 'run/snapshots').glob('upd-000001_ema.ckpt'))
    snapshot_payload = load(snapshot)
    snapshot_policy = load_policy(SimpleNamespace(snapshot=snapshot, weights='ema', pi05=None,
                                                  spm=None, device='cpu'), snapshot_payload)
    for name, parameter in snapshot_policy.siglip.named_parameters():
        saved = snapshot_payload['state_dicts']['ema_model']['siglip.' + name]
        assert saved.dtype == torch.bfloat16
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, saved.float(), rtol=0, atol=0)

    incompatible = copy.deepcopy(cfg)
    incompatible.policy.vision_activation_checkpointing = False
    with pytest.raises(ValueError, match='vision_activation_checkpointing'):
        Workspace.validate_resume_payload(payload, incompatible, world_size=1)
    incompatible = copy.deepcopy(cfg)
    incompatible.policy._target_ = 'oat.policy.p2n_vla.P2NVLAPolicy'
    with pytest.raises(ValueError, match='policy target'):
        Workspace.validate_resume_payload(payload, incompatible, world_size=1)


def test_vision_action_resume_matches_uninterrupted_weights_moments_and_ema(tmp_path, monkeypatch):
    data = make_libero_like_zarr(tmp_path / 'data.zarr')
    cfg = config('p2n_vla', data, 'training.num_epochs=2')
    full = run_workspace(cfg, tmp_path / 'full')
    first_cfg = config('p2n_vla', data, 'training.num_epochs=1')
    run_workspace(first_cfg, tmp_path / 'first')
    checkpoint = tmp_path / 'first/checkpoints/latest.ckpt'
    resumed_cfg = config('p2n_vla', data, 'training.num_epochs=2', 'training.resume=true',
                         f'training.resume_checkpoint={checkpoint}')
    # The inherited workspace already branches fresh/resume before fitting.
    import hydra
    dataset_class = hydra.utils.get_class(resumed_cfg.task.policy.dataset._target_)

    def forbidden_refit(self, *args, **kwargs):
        raise AssertionError('Resume must restore normalizers without dataset refitting')

    monkeypatch.setattr(dataset_class, 'get_normalizer', forbidden_refit)
    resumed = run_workspace(resumed_cfg, tmp_path / 'resumed')
    assert resumed.completed_optimizer_steps == full.completed_optimizer_steps == 2
    assert_nested_equal(resumed.model.artifact_state_dict(), full.model.artifact_state_dict())
    assert_nested_equal(resumed.optimizer.state_dict(), full.optimizer.state_dict())
    assert_nested_equal(resumed.ema.shadow, full.ema.shadow)
    assert_nested_equal(resumed.lr_scheduler.state_dict(), full.lr_scheduler.state_dict())
    assert resumed.ema.updates == full.ema.updates
    full_payload = load(tmp_path / 'full/checkpoints/latest.ckpt')
    resumed_payload = load(tmp_path / 'resumed/checkpoints/latest.ckpt')
    assert_nested_equal(resumed_payload['training']['counters'], full_payload['training']['counters'])
    torch.testing.assert_close(resumed_payload['training']['rng_states'][0]['torch'],
                               full_payload['training']['rng_states'][0]['torch'], rtol=0, atol=0)


def test_existing_rollout_schedule_supports_every_100_completed_epochs(tmp_path):
    cfg = config('p2n_vla', tmp_path / 'unused.zarr')
    cfg.task.policy.lazy_eval = False
    cfg.training.rollout_every = 100
    workspace = Workspace(cfg, output_dir=str(tmp_path))
    Workspace.check_run_config(cfg)
    assert not workspace.rollout_due(98, False)
    assert workspace.rollout_due(99, False)
    assert workspace.rollout_due(199, False)
    assert workspace.rollout_due(299, True)


def test_no_ema_vision_action_saves_resumes_and_evaluates_model_weights(tmp_path):
    """The memory-saving launch mode keeps current vision weights in both artifact kinds."""
    data = make_libero_like_zarr(tmp_path / 'data.zarr')
    cfg = config('p2n_vla', data, 'training.use_ema=false')
    first = run_workspace(cfg, tmp_path / 'first')
    assert first.ema is None
    checkpoint = tmp_path / 'first/checkpoints/latest.ckpt'
    snapshot = tmp_path / 'first/snapshots/upd-000001_model.ckpt'
    assert snapshot.is_file()
    assert not list(snapshot.parent.glob('*_ema.ckpt'))
    payload = load(checkpoint)
    Workspace.validate_resume_payload(payload, cfg, world_size=1)
    assert payload['training']['ema'] is None
    assert payload['metadata']['training_run']['ema'] is None
    assert set(payload['state_dicts']) == {'model'}
    for path in (checkpoint, snapshot):
        artifact = load(path)
        assert set(artifact['state_dicts']) == {'model'}
        assert artifact['metadata']['training_run']['ema'] is None
        assert artifact['metadata']['format'] == 'p2n_vla_artifact_v2'
        if path == snapshot:
            assert 'training' not in artifact
        restored = load_policy(SimpleNamespace(snapshot=path, weights='model', pi05=None,
                                                spm=None, device='cpu'), artifact)
        for name, parameter in restored.siglip.named_parameters():
            saved = artifact['state_dicts']['model']['siglip.' + name]
            assert saved.dtype == (torch.bfloat16 if path == snapshot else torch.float32)
            torch.testing.assert_close(parameter, saved.float(), rtol=0, atol=0)
        with pytest.raises(ValueError, match='ema_model'):
            load_policy(SimpleNamespace(snapshot=path, weights='ema', pi05=None,
                                        spm=None, device='cpu'), artifact)
    resumed_cfg = config('p2n_vla', data, 'training.use_ema=false', 'training.num_epochs=2',
                         'training.resume=true', f'training.resume_checkpoint={checkpoint}')
    resumed = run_workspace(resumed_cfg, tmp_path / 'resumed')
    assert resumed.ema is None
    assert resumed.completed_optimizer_steps == resumed.model.self_past_step == 2
    assert all(parameter.requires_grad for parameter in resumed.model.siglip.parameters())
    assert all(not parameter.requires_grad for parameter in resumed.model.joint.vlm.parameters())
    assert (tmp_path / 'resumed/snapshots/upd-000002_model.ckpt').is_file()
