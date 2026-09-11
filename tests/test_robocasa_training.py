"""Regression coverage for selecting and freezing the Sink3 Stage1 handoff."""
import importlib.util
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from oat.common.hydra_util import register_new_resolvers

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'train_robocasa_sink3', ROOT / 'scripts/train_robocasa_sink3.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def write_stage(stage_dir, records, retained):
    (stage_dir / 'checkpoints').mkdir()
    (stage_dir / 'logs.json').write_text(
        ''.join(json.dumps(record) + '\n' for record in records))
    for filename in retained:
        (stage_dir / 'checkpoints' / filename).write_bytes(b'retained checkpoint')


def test_handoff_uses_unrounded_metric_and_only_retained_checkpoints(tmp_path):
    # Both retained losses round to the same filename metric. The older
    # checkpoint is better; a still better logged checkpoint was pruned.
    write_stage(tmp_path, [
        {'epoch': 10, 'test_reconst_mse': 0.02010},
        {'epoch': 20, 'test_reconst_mse': 0.02040},
        {'epoch': 30, 'test_reconst_mse': 0.00001},
        {'epoch': 40, 'train_loss': 0.000001},
    ], ['ep-0010_mse-0.020.ckpt', 'ep-0020_mse-0.020.ckpt'])
    selected, metric = launcher.best_tokenizer_checkpoint(tmp_path)
    assert selected == tmp_path / 'checkpoints/ep-0010_mse-0.020.ckpt'
    assert metric == 0.02010


def test_equal_unrounded_metrics_choose_latest_retained_checkpoint(tmp_path):
    write_stage(tmp_path, [
        {'epoch': 10, 'test_reconst_mse': 0.025},
        {'epoch': 20, 'test_reconst_mse': 0.025},
    ], ['ep-0010_mse-0.025.ckpt', 'ep-0020_mse-0.025.ckpt'])
    selected, metric = launcher.best_tokenizer_checkpoint(tmp_path)
    assert selected.name == 'ep-0020_mse-0.025.ckpt'
    assert metric == 0.025


def test_nonfinite_reconstruction_metrics_are_ineligible(tmp_path):
    write_stage(tmp_path, [
        {'epoch': 0, 'test_reconst_mse': float('-inf')},
        {'epoch': 10, 'test_reconst_mse': float('inf')},
        {'epoch': 20, 'test_reconst_mse': float('nan')},
        {'epoch': 30, 'test_reconst_mse': None},
        {'epoch': 40, 'test_reconst_mse': 0.05},
    ], ['ep-0000_mse--inf.ckpt', 'ep-0010_mse-inf.ckpt',
        'ep-0020_mse-nan.ckpt', 'ep-0040_mse-0.050.ckpt'])
    selected, metric = launcher.best_tokenizer_checkpoint(tmp_path)
    assert selected.name == 'ep-0040_mse-0.050.ckpt'
    assert metric == 0.05
    selected.unlink()
    with pytest.raises(RuntimeError, match='no checkpoint with a finite'):
        launcher.best_tokenizer_checkpoint(tmp_path)


def compose_command(command):
    config_arg = next(arg for arg in command if arg.startswith('--config-name='))
    overrides = command[command.index(config_arg) + 1:]
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / 'oat/config'), version_base=None):
        cfg = compose(config_name=config_arg.split('=', 1)[1], overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize('num_gpus', [1, 2])
@pytest.mark.parametrize('stage,global_batch', [('tokenizer', 256), ('policy', 64)])
def test_commands_compose_correct_sink3_task_and_fresh_training(
        tmp_path, num_gpus, stage, global_batch):
    checkpoint = tmp_path / 'frozen_tokenizer.ckpt'
    command = launcher.stage_command(stage, tmp_path / stage, num_gpus, checkpoint)
    cfg = compose_command(command)
    assert f'--nproc_per_node={num_gpus}' in command
    assert cfg.training.num_demo == 600
    assert cfg.training.resume is False
    assert cfg.logging.resume is False
    assert cfg.dataloader.batch_size * num_gpus == global_batch
    assert cfg.val_dataloader.batch_size * num_gpus == global_batch
    assert cfg.task[stage].shape_meta.action.shape == [12]
    assert cfg.task[stage].dataset.val_ratio == 0.1
    if stage == 'policy':
        assert cfg.name == 'train_past2next_scratch_tasklr'
        assert cfg.training.init_checkpoint is None
        assert cfg.policy.action_tokenizer.checkpoint == str(checkpoint)
        assert cfg.policy.obs_encoder.task_uids == [2, 4, 5]
        assert cfg.policy.obs_encoder.vision_encoder.eval_fixed_crop is True
        assert cfg.task.policy.dataset.history_padding == 'zero'
        assert cfg.task.policy.dataset.return_history_validity is True
        assert cfg.task.policy.dataset.n_exec_steps == cfg.n_action_steps == 8
        assert cfg.checkpoint.topk.monitor_key == 'val_loss'
        assert cfg.checkpoint.topk.mode == 'min'
    else:
        assert cfg.tokenizer.action_aug.mode == 'left_noise'
        assert cfg.tokenizer.action_aug.augment_position is False
        assert cfg.task.tokenizer.dataset.obs_keys == []


@pytest.mark.parametrize('stage,epochs', [('tokenizer', 5001), ('policy', 251)])
def test_smoke_overrides_do_not_shorten_full_training(tmp_path, stage, epochs):
    checkpoint = tmp_path / 'frozen_tokenizer.ckpt'
    full = compose_command(launcher.stage_command(stage, tmp_path / stage, 2, checkpoint))
    smoke = compose_command(launcher.stage_command(
        stage, tmp_path / stage, 2, checkpoint, smoke=True))
    assert full.training.num_epochs == epochs
    assert full.training.max_train_steps is None
    assert full.training.max_val_steps is None
    assert full.dataloader.num_workers > 0
    assert full.val_dataloader.num_workers > 0
    assert smoke.training.num_epochs == 1
    assert smoke.training.max_train_steps == 2
    assert smoke.training.max_val_steps == smoke.training.max_reconst_steps == 1
    assert smoke.dataloader.num_workers == smoke.val_dataloader.num_workers == 0
    assert not smoke.dataloader.persistent_workers
    assert not smoke.val_dataloader.persistent_workers
    if stage == 'policy':
        assert full.policy.self_past_warmup_steps == 1000
        assert full.policy.self_past_ramp_steps == 4000
        assert full.policy.self_past_p == 0.5
        assert smoke.policy.self_past_warmup_steps == 0
        assert smoke.policy.self_past_ramp_steps == 0
        assert smoke.policy.self_past_p == 1.0
