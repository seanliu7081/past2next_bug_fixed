"""CPU training-loop regressions for scratch runs and explicit no-holdout training.

Historical dataset-only fixtures with val_ratio=0 remain valid. Any diagnostic
that actually calls TrainPolicyWorkspace.run with zero validation windows must
now set training.offline_validation_enabled=false explicitly.
"""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

from accelerate import Accelerator
import dill
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from oat.dataset.zarr_dataset_with_prev_window import ZarrDatasetWithPrevWindow
from oat.workspace.train_policy import TrainPolicyWorkspace
import oat.workspace.train_policy as training_module


class TinyDataset(Dataset):
    def __init__(self, length=4, validation_length=2, forbid_reads=False):
        self.length = length
        self.validation_length = validation_length
        self.forbid_reads = forbid_reads
        self.train_mask = np.ones(length // 2, dtype=bool)
        self.reads = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert not self.forbid_reads, 'Disabled validation dataset was iterated'
        self.reads += 1
        value = torch.tensor([float(index) / 10])
        return {'obs': {'value': value}, 'action': value}

    def get_normalizer(self):
        return None

    def get_validation_dataset(self):
        self.validation = TinyDataset(self.validation_length, forbid_reads=self.forbid_reads)
        return self.validation


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.5]))
        self.training_calls = self.validation_calls = self.reconstruction_calls = 0

    def set_normalizer(self, normalizer):
        pass

    def get_optimizer(self, **kwargs):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

    def forward(self, batch):
        if self.training:
            self.training_calls += 1
        else:
            self.validation_calls += 1
        return (self.weight - batch['action']).square().mean()

    def predict_action(self, obs):
        self.reconstruction_calls += 1
        return {'action_pred': self.weight.expand_as(obs['value'])}


def config(config_name='train_past2next_scratch'):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'oat/config'), version_base=None):
        cfg = compose(config_name=config_name)
    cfg.policy = {'_target_': 'test.TinyPolicy'}
    cfg.task.policy.dataset = {'_target_': 'test.TinyDataset'}
    cfg.training.use_ema = False
    cfg.training.allow_bf16 = False
    cfg.training.num_epochs = 1
    cfg.training.lr_warmup_steps = 0
    cfg.training.max_train_steps = 2
    cfg.checkpoint.topk.k = 0
    for key in ('dataloader', 'val_dataloader'):
        cfg[key].batch_size = 2
        cfg[key].num_workers = 0
        cfg[key].persistent_workers = False
        cfg[key].pin_memory = False
        cfg[key].drop_last = False
    return cfg


def run_tiny(tmp_path, monkeypatch, cfg, validation_length=2, forbid_validation=False):
    policy = TinyPolicy()
    dataset = TinyDataset(validation_length=validation_length)
    original_get_validation = dataset.get_validation_dataset
    def validation():
        result = original_get_validation()
        result.forbid_reads = forbid_validation
        return result
    dataset.get_validation_dataset = validation
    def instantiate(node, **kwargs):
        if node._target_ == 'test.TinyPolicy':
            return policy
        if node._target_ == 'test.TinyDataset':
            return dataset
        raise AssertionError(f'Unexpected instantiation: {node}')
    def cpu_accelerator(**kwargs):
        kwargs.pop('log_with')
        result = Accelerator(cpu=True, **kwargs)
        tracker = SimpleNamespace(run=SimpleNamespace(config={}))
        result.get_tracker = lambda *args, **kwargs: tracker
        return result
    monkeypatch.setattr(training_module.hydra.utils, 'instantiate', instantiate)
    monkeypatch.setattr(training_module, 'Accelerator', cpu_accelerator)
    workspace = TrainPolicyWorkspace(cfg, output_dir=str(tmp_path))
    workspace.run()
    if workspace._saving_thread is not None:
        workspace._saving_thread.join()
    rows = [json.loads(line) for line in (tmp_path / 'logs.json').read_text().splitlines()]
    assert all(math.isfinite(float(value)) for row in rows for value in row.values()
               if isinstance(value, (int, float)))
    return workspace, policy, dataset, rows[-1]


@pytest.mark.parametrize('config_name, validation_length, validation_enabled', [
    ('train_past2next_scratch', 2, True),
    ('train_past2next_scratch_all500', 0, False),
    ('train_past2next_scratch_tasklr', 2, True),
])
def test_scratch_defaults_train_with_fresh_progress_despite_existing_checkpoint(
        tmp_path, monkeypatch, config_name, validation_length, validation_enabled):
    cfg = config(config_name)
    assert cfg.training.init_checkpoint is None
    assert cfg.training.resume is False and cfg.logging.resume is False
    stale_checkpoint = tmp_path / 'checkpoints/latest.ckpt'
    stale_checkpoint.parent.mkdir()
    stale_checkpoint.write_bytes(b'Invalid checkpoint from an earlier run')

    def forbid_checkpoint_load(*args, **kwargs):
        pytest.fail('Scratch defaults must not initialize or resume policy weights')

    monkeypatch.setattr(TrainPolicyWorkspace, '_initialize_policy_weights', forbid_checkpoint_load)
    monkeypatch.setattr(TrainPolicyWorkspace, '_resume_training_checkpoint', forbid_checkpoint_load)
    workspace, policy, dataset, row = run_tiny(
        tmp_path, monkeypatch, cfg, validation_length,
        forbid_validation=not validation_enabled)
    assert policy.training_calls == 2
    assert policy.weight.item() < 0.5
    assert (workspace.epoch, workspace.global_step, workspace.completed_optimizer_steps) == (1, 2, 2)
    assert workspace.resume_migration is None
    assert len(workspace.optimizer.state) == 1
    assert all(int(state['step']) == 2 for state in workspace.optimizer.state.values())
    assert workspace.lr_scheduler_state['last_epoch'] == 2
    assert row['offline_validation_enabled'] == int(validation_enabled)
    assert policy.validation_calls == policy.reconstruction_calls == int(validation_enabled)
    assert dataset.validation.reads == (4 if validation_enabled else 0)
    payload = torch.load(tmp_path / 'checkpoints/ep-0000.ckpt', map_location='cpu',
                         pickle_module=dill, weights_only=False)
    assert dill.loads(payload['pickles']['epoch']) == 1
    assert dill.loads(payload['pickles']['completed_optimizer_steps']) == 2
    assert dill.loads(payload['pickles']['global_step']) == 2
    assert all(int(state['step']) == 2 for state in payload['state_dicts']['optimizer']['state'].values())


def test_default_nonempty_validation_and_reconstruction_still_run(tmp_path, monkeypatch):
    cfg = config()
    assert 'offline_validation_enabled' not in cfg.training
    workspace, policy, dataset, row = run_tiny(tmp_path, monkeypatch, cfg)
    assert policy.training_calls == 2
    assert policy.validation_calls == policy.reconstruction_calls == 1
    assert row['val_loss'] >= 0 and row['test_reconst_mse'] >= 0
    assert row['offline_validation_enabled'] == 1
    assert row['offline_validation_reason'] == 'enabled'
    assert row['train_dataset_episodes'] == 2 and row['validation_dataset_episodes'] == 1
    assert dataset.validation.reads == 4


@pytest.mark.parametrize('validation_length', [0, 2])
def test_explicit_disabled_skips_both_paths_without_nan_and_saves_progress(tmp_path, monkeypatch, validation_length):
    cfg = config()
    OmegaConf.set_struct(cfg, False)
    cfg.training.offline_validation_enabled = False
    cfg.training.offline_validation_reason = 'all supplied demonstrations used for training'
    # A disabled block must not even evaluate scheduling or touch its data.
    cfg.training.val_every = cfg.training.sample_every = 0
    workspace, policy, dataset, row = run_tiny(
        tmp_path, monkeypatch, cfg, validation_length, forbid_validation=True)
    assert policy.training_calls == 2
    assert policy.validation_calls == policy.reconstruction_calls == dataset.validation.reads == 0
    assert not any(key.startswith('val_loss') or key == 'test_reconst_mse' for key in row)
    assert row['offline_validation_enabled'] == 0
    assert row['offline_validation_reason'] == cfg.training.offline_validation_reason
    split = json.loads((tmp_path / 'dataset_split.json').read_text())
    assert split['train'] == {'windows': 4, 'episodes': 2, 'batches_per_epoch': 2}
    assert split['validation']['windows'] == validation_length
    assert not split['offline_validation_enabled']
    assert split['offline_validation_reason'] == row['offline_validation_reason']
    checkpoint = tmp_path / 'checkpoints/ep-0000.ckpt'
    payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill, weights_only=False)
    assert dill.loads(payload['pickles']['epoch']) == 1
    assert dill.loads(payload['pickles']['completed_optimizer_steps']) == 2
    assert dill.loads(payload['pickles']['global_step']) == 2


def test_default_empty_validation_is_rejected_before_training(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='validation dataset/loader is empty'):
        run_tiny(tmp_path, monkeypatch, config(), validation_length=0)
    assert not (tmp_path / 'logs.json').exists()


def test_nonempty_validation_with_all_batches_dropped_is_rejected():
    train, validation = TinyDataset(), TinyDataset(length=2)
    with pytest.raises(ValueError, match='usable batches'):
        TrainPolicyWorkspace._offline_validation_metadata(
            train, validation, {}, DataLoader(train, batch_size=2),
            DataLoader(validation, batch_size=64, drop_last=True))




def test_all500_config_and_real_lightweight_dataset_preserve_temporal_contract():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'oat/config'), version_base=None):
        cfg = compose(config_name='train_past2next_scratch_all500')
    assert cfg.policy._target_ == 'oat.policy.past2next_self_past.Past2NextSelfPastPolicy'
    assert cfg.policy.n_layers == cfg.policy.n_heads == 8
    assert cfg.policy.past_n == 7 and cfg.n_obs_steps == 2 and cfg.n_action_steps == 8 and cfg.horizon == 16
    assert cfg.policy.self_past_temperature == 1 and cfg.policy.temperature == 0
    assert list(cfg.policy.obs_encoder.vision_encoder.crop_shape) == [112, 112]
    assert OmegaConf.is_missing(cfg.policy.action_tokenizer, 'checkpoint')
    assert cfg.training.init_checkpoint is None
    assert not cfg.training.resume and not cfg.logging.resume
    assert cfg.training.num_epochs == 251
    assert cfg.training.checkpoint_every == cfg.training.snapshot_every == 25
    assert 'init_weights' not in cfg.training and 'init_allow_spatial_resize' not in cfg.training
    assert not cfg.training.offline_validation_enabled
    assert cfg.training.offline_validation_reason and cfg.task.policy.lazy_eval
    assert cfg.optimizer.policy_lr == cfg.optimizer.obs_enc_lr == 1e-5
    assert cfg.task.policy.dataset.val_ratio == 0 and cfg.task.policy.dataset.max_train_episodes is None
    assert cfg.task.policy.dataset.history_padding == 'zero' and cfg.task.policy.dataset.return_history_validity
    assert cfg.checkpoint.topk.k == 0
    path = Path('/workspace/past_action/data/libero/libero10_N500.zarr')
    if not path.is_dir():
        pytest.skip('The optional real LIBERO N500 store is not present')
    kwargs = OmegaConf.to_container(cfg.task.policy.dataset, resolve=True)
    kwargs.pop('_target_')
    kwargs.update(zarr_path=str(path), obs_keys=['task_uid'])  # No camera arrays or normalizer fitting.
    dataset = ZarrDatasetWithPrevWindow(**kwargs)
    validation = dataset.get_validation_dataset()
    train_loader = DataLoader(dataset, batch_size=64, drop_last=True)
    result = TrainPolicyWorkspace._offline_validation_metadata(
        dataset, validation, cfg.training, train_loader, DataLoader(validation, batch_size=64))
    assert result['train'] == {'windows': 138090, 'episodes': 500, 'batches_per_epoch': 2157}
    assert result['validation'] == {'windows': 0, 'episodes': 0, 'batches': 0}
    assert list(DataLoader(validation, batch_size=64)) == []
    assert np.all(dataset.train_mask)
    starts = np.r_[0, dataset.replay_buffer.episode_ends[:-1]]
    uids = dataset.replay_buffer['task_uid'][starts].reshape(-1)
    assert dict(zip(*np.unique(uids, return_counts=True))) == {uid: 50 for uid in range(30, 40)}
    for start in (int(starts[0]), int(starts[-1])):
        first, previous_ready = dataset[start], dataset[start + 8]
        assert not first['past_action_valid'].any() and not first['prev_window_valid']
        assert torch.equal(first['past_action'], torch.zeros(7, 7))
        assert previous_ready['past_action_valid'].all() and previous_ready['prev_window_valid']
    # Existing original-data split remains unchanged when the opt-in is absent.
    old = ZarrDatasetWithPrevWindow(**{**kwargs, 'val_ratio': .1})
    assert int(old.train_mask.sum()) == 450 and len(old) == 124600
    assert len(old.get_validation_dataset()) == 13490
