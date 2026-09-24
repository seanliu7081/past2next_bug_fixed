"""Named tasks choose real datasets while retaining the action-flow contract."""
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from scripts import train_p2n_action_flow_resnet18 as launch
from test_p2n_action_flow_resnet18_launch import _bash_capture, VARIANTS

RECIPES = {
    'nut_washer': ('nut_washer_v3_N77', 'nut_washer_v3_N77.zarr', 77, .05),
    'pen_cabinet': ('pen_cabinet_N67', 'pen_cabinet_N67.zarr', 67, .1),
    'fruits': ('fruits', 'fruits_N51.zarr', 51, .1),
    'fruits_v2': ('fruits_v2', 'fruits_v2_N49.zarr', 49, .1),
}


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('task', RECIPES)
def test_named_dataset_recipe_keeps_domain_and_history(variant, task):
    name, path, count, val_ratio = RECIPES[task]
    cfg = launch.compose_config(variant, task)
    assert cfg.task_type == cfg.policy.task == 'real_robot'
    assert cfg.task.policy.task_name == name
    assert Path(cfg.task.policy.dataset.zarr_path).name == path
    assert cfg.training.num_demo == count
    assert cfg.task.policy.dataset.val_ratio == val_ratio
    assert cfg.training.num_epochs == 2001
    assert cfg.policy.obs_encoder_type == 'resnet18'
    assert cfg.task.policy.env_runner is None
    assert cfg.action_schema.control_frequency_hz == 30
    assert cfg.policy.action_dim == 7 and cfg.policy.horizon == 16
    assert 'StateHistory' in cfg.task.policy.dataset._target_ if 'state_gate' in variant else 'PrevWindow' in cfg.task.policy.dataset._target_
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    assert 'tokenizer' not in resolved and 'dino' not in resolved.lower()
    if task != 'nut_washer':
        assert 'nut_washer' not in resolved


@pytest.mark.parametrize('variant', VARIANTS)
def test_real_robot_alias_retains_n77_default(variant):
    named = launch.compose_config(variant, 'nut_washer')
    legacy = launch.compose_config(variant, 'real_robot')
    assert OmegaConf.to_container(named, resolve=True) == OmegaConf.to_container(legacy, resolve=True)


def test_task_and_user_overrides_are_preserved():
    cfg = launch.compose_config(VARIANTS[1], 'pen_cabinet', [
        'training.num_epochs=9', 'dataloader.batch_size=8',
        'task.policy.dataset.zarr_path=/relocated/pen_cabinet_N67.zarr'])
    assert cfg.training.num_epochs == 9 and cfg.dataloader.batch_size == 8
    assert cfg.task.policy.dataset.zarr_path == '/relocated/pen_cabinet_N67.zarr'
    assert cfg.task.policy.task_name == 'pen_cabinet_N67'
    with pytest.raises(ValueError, match='Selected task'):
        launch.compose_config(VARIANTS[0], 'pen_cabinet', ['task.policy.task_name=nut_washer_v3_N77'])
    with pytest.raises(ValueError, match='Unknown task'):
        launch.compose_config(VARIANTS[0], 'missing_task')


@pytest.mark.parametrize('source,target', [('nut_washer','pen_cabinet'), ('pen_cabinet','nut_washer'), ('fruits','fruits_v2')])
def test_resume_cannot_change_named_task_even_by_overriding_saved_name(source, target):
    cfg = launch.compose_config(VARIANTS[0], source)
    with pytest.raises(ValueError, match='Selected task'):
        launch.compose_resume_config({'cfg': cfg}, VARIANTS[0], target,
            [f'task.policy.task_name={RECIPES[target][0]}'])
    resumed = launch.compose_resume_config({'cfg':cfg}, VARIANTS[0], source,
        ['training.resume=true','training.resume_checkpoint=/tmp/saved.ckpt'])
    assert resumed.task.policy.task_name == RECIPES[source][0]


@pytest.mark.parametrize('task', RECIPES)
def test_shell_passes_named_task_to_both_variants(tmp_path, task):
    result, calls = _bash_capture(tmp_path, '--task', task, '--gpus', '2,3',
        '--batch-size', '4', '--val-batch-size', '4', '--grad-accum', '8', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    for args in calls:
        assert args[args.index('--task')+1] == task
        assert task in args[args.index('--output')+1]


def test_python_named_dry_run_resolves_task_without_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry-run must not query GPUs or launch workers')
    monkeypatch.setattr(launch,'check_gpu_selection',forbidden)
    monkeypatch.setattr(launch.subprocess,'run',forbidden)
    monkeypatch.setattr(launch,'preflight',lambda cfg,output,world_size: {
        'task':cfg.task.policy.task_name,'data':cfg.task.policy.dataset.zarr_path,
        'domain':cfg.policy.task,'world':world_size})
    report=launch.main(['--variant',VARIANTS[0],'--task','pen_cabinet','--gpus','2,3','--dry-run'])
    assert report == {'task':'pen_cabinet_N67','data':'/workspace/ysk/zarr/pen_cabinet_N67.zarr',
        'domain':'real_robot','world':2}
