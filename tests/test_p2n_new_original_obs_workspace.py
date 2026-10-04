"""Offline workspace continuation and validation contracts, using a tiny policy."""
import copy
from types import SimpleNamespace

import dill
from omegaconf import OmegaConf, open_dict
import pytest
import torch

from test_p2n_new_integration import TinyDataset, TinyPolicy, tiny_config
from oat.perception.original_fused_obs_adapter import normalize_original_obs_config
from oat.workspace.train_p2n_new_original_obs import TrainP2NNewOriginalObsWorkspace as Workspace


class TinyOriginalEncoder(torch.nn.Module):
    fused_feature_dim = 139

    def __init__(self, original_obs_config):
        super().__init__()
        self.original_obs_config = normalize_original_obs_config(original_obs_config)
        self.register_parameter('normalizer', torch.nn.Parameter(torch.tensor(0.), requires_grad=False))

    def export_metadata(self):
        return {'encoder_type': 'original_fused', 'fused_feature_dim': self.fused_feature_dim,
                'observation_tokens': 2, 'original_obs_config': self.original_obs_config}


class TinyOriginalPolicy(TinyPolicy):
    def __init__(self, variant='p2n_new', construction_mode='fresh', obs_encoder_type='original_fused',
                 context_layout='original_fused_v1', context_schema_version=2,
                 shape_meta=None, n_obs_steps=2, embed_dim=8, original_obs_config=None,
                 obs_encoder_config=None, expected_action_tokens=8):
        super().__init__(variant, construction_mode)
        self.obs_encoder = TinyOriginalEncoder(original_obs_config)
        self.obs_encoder_type = obs_encoder_type
        self.context_layout = context_layout
        self.context_schema_version = context_schema_version
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.embed_dim = embed_dim
        self.past_n = 7
        self.expected_action_tokens = expected_action_tokens

    def get_policy_name(self):
        return 'tiny_original_fused_test'

    def self_past_probability(self):
        return 0.

    def forward(self, batch, history_mode=None):
        return self.linear(batch['action'] + self.obs_encoder.normalizer).square().mean()

    def set_normalizer(self, normalizer):
        self.obs_encoder.normalizer.data.fill_(normalizer)

    def export_config(self):
        encoder = {'_target_': 'oat.perception.original_fused_obs_adapter.OriginalFusedObservationAdapter',
                   'shape_meta': OmegaConf.to_container(self.shape_meta) if OmegaConf.is_config(self.shape_meta) else self.shape_meta,
                   'n_obs_steps': self.n_obs_steps, 'embed_dim': self.embed_dim,
                   'original_obs_config': self.obs_encoder.original_obs_config}
        return {'_target_': 'test_p2n_new_original_obs_workspace.TinyOriginalPolicy', '_recursive_': False,
                'variant': self.variant, 'construction_mode': 'restore',
                'obs_encoder_type': self.obs_encoder_type, 'context_layout': self.context_layout,
                'context_schema_version': self.context_schema_version,
                'shape_meta': encoder['shape_meta'], 'n_obs_steps': self.n_obs_steps, 'embed_dim': self.embed_dim,
                'original_obs_config': self.obs_encoder.original_obs_config,
                'obs_encoder_config': encoder, 'expected_action_tokens': self.expected_action_tokens}

    def artifact_metadata(self):
        return {'variant': self.variant, 'obs_encoder_type': self.obs_encoder_type,
                'context_layout': self.context_layout, 'context_schema_version': self.context_schema_version}


class CountingDataset(TinyDataset):
    normalizer_calls = 0
    refuse_fit = False

    def get_normalizer(self):
        type(self).normalizer_calls += 1
        if type(self).refuse_fit:
            raise AssertionError('Resume must never fit a dataset normalizer')
        return 3.125


def config():
    cfg = tiny_config()
    cfg.policy = {'_target_': 'test_p2n_new_original_obs_workspace.TinyOriginalPolicy',
                  'variant': 'p2n_new', 'obs_encoder_type': 'original_fused',
                  'context_layout': 'original_fused_v1', 'context_schema_version': 2,
                  'shape_meta': {'obs': {}, 'action': {'shape': [1]}},
                  'n_obs_steps': 2, 'embed_dim': 8, 'expected_action_tokens': 8,
                  'original_obs_config': normalize_original_obs_config()}
    cfg.task.policy.dataset = {'_target_': 'test_p2n_new_original_obs_workspace.CountingDataset',
                               'zarr_path': '/unused'}
    cfg.task.policy.env_runner = None
    cfg.task.policy.lazy_eval = False
    cfg.training.snapshot_every = 1
    with open_dict(cfg.training):
        cfg.training.validation_max_samples = 3
    return cfg


@pytest.fixture(autouse=True)
def reset_dataset_counters(monkeypatch):
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    CountingDataset.normalizer_calls = 0
    CountingDataset.refuse_fit = False


def test_resume_never_refits_and_matches_full_optimizer_ema_rng(tmp_path):
    cfg = config()
    complete = Workspace(cfg, output_dir=str(tmp_path / 'full'))
    complete.run()
    assert CountingDataset.normalizer_calls == 1
    expected_weights = copy.deepcopy(complete.model.state_dict())
    expected_ema = copy.deepcopy(complete.ema_model.state_dict())
    expected_optimizer = copy.deepcopy(complete.optimizer.state_dict())
    expected_draw = torch.rand(5)
    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg.training.resume = True
    resumed_cfg.training.resume_checkpoint = str(tmp_path / 'full/checkpoints/ep-0000.ckpt')
    CountingDataset.refuse_fit = True
    resumed = Workspace(resumed_cfg, output_dir=str(tmp_path / 'resumed'))
    resumed.run()
    assert CountingDataset.normalizer_calls == 1
    assert (resumed.epoch, resumed.global_step, resumed.completed_optimizer_steps) == (2, 6, 4)
    assert resumed.model.obs_encoder.normalizer.item() == 3.125
    assert not resumed.model.obs_encoder.normalizer.requires_grad
    for actual, expected in ((resumed.model.state_dict(), expected_weights),
                             (resumed.ema_model.state_dict(), expected_ema)):
        for key, tensor in actual.items():
            torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
    actual_optimizer = resumed.optimizer.state_dict()
    assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
    for key, state in actual_optimizer['state'].items():
        for item, tensor in state.items():
            torch.testing.assert_close(tensor, expected_optimizer['state'][key][item], rtol=0, atol=0)
    assert resumed.ema_state == complete.ema_state
    assert resumed.lr_scheduler_state == complete.lr_scheduler_state
    torch.testing.assert_close(torch.rand(5), expected_draw, rtol=0, atol=0)
    assert resumed.dataset_split['validation']['windows'] == 10
    assert resumed.dataset_split['validation']['evaluated_windows'] == 3


def saved_payload(tmp_path):
    cfg = config()
    workspace = Workspace(cfg, output_dir=str(tmp_path), lazy_instantiation=False)
    workspace._initialize_normalizers(CountingDataset(), cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(workspace.optimizer, lambda _: 1)
    workspace._capture_training_state(None, scheduler)
    path = workspace.save_checkpoint()
    return cfg, torch.load(path, map_location='cpu', pickle_module=dill, weights_only=False)


@pytest.mark.parametrize('key,value', [('obs_encoder_type', 'dinov3_tokens'),
                                      ('context_layout', 'visual_proprio_v1'),
                                      ('context_schema_version', 1)])
def test_cross_layout_resume_is_rejected(tmp_path, key, value):
    cfg, payload = saved_payload(tmp_path)
    cfg.policy[key] = value
    with pytest.raises(ValueError, match='encoder/layout'):
        Workspace.validate_resume_payload(payload, cfg)


def test_changed_crop_and_corrupted_embedded_bridge_are_rejected(tmp_path):
    cfg, payload = saved_payload(tmp_path)
    cfg.policy.original_obs_config.crop_shape = [76, 76]
    with pytest.raises(ValueError, match='crop_shape'):
        Workspace.validate_resume_payload(payload, cfg)
    cfg.policy.original_obs_config.crop_shape = [112, 112]
    payload['policy_config']['obs_encoder_config']['embed_dim'] = 9
    with pytest.raises(ValueError, match='embed_dim'):
        Workspace.validate_resume_payload(payload, cfg)


def test_exact_global_validation_cap_without_duplicate_padded_samples():
    cfg = config()
    cfg.val_dataloader.batch_size = 4
    cfg.val_dataloader.drop_last = True
    loaders = [Workspace._make_validation_dataloader(
        CountingDataset(), cfg, SimpleNamespace(process_index=rank, num_processes=2))
        for rank in range(2)]
    indices = [index for loader in loaders for index in loader.dataset.indices]
    assert sorted(indices) == [0, 1, 2]
    assert sum(sum(len(batch['action']) for batch in loader) for loader in loaders) == 3
    cfg.training.validation_max_samples = 1
    loaders = [Workspace._make_validation_dataloader(
        CountingDataset(), cfg, SimpleNamespace(process_index=rank, num_processes=2))
        for rank in range(2)]
    assert [len(loader.dataset) for loader in loaders] == [1, 0]


def test_report_uses_real_fused_dimensions_and_offline_robot_semantics(tmp_path):
    cfg = config()
    workspace = Workspace(cfg, output_dir=str(tmp_path), lazy_instantiation=False)
    report = workspace.training_report(workspace.model)
    assert report['observation_tokens'] == 2
    assert report['context_tokens'] == 11
    assert report['fused_feature_dim'] == 139
    assert report['performance_contract']['cnn_trainable']
    assert report['evaluation']['requested_lazy_eval'] is False
    assert 'no physical robot rollout' in report['evaluation']['mode']
    assert 'visual_queries_per_image' not in report['performance_contract']


def test_skipped_optimizer_update_does_not_advance_ema_or_curriculum(tmp_path, monkeypatch):
    from accelerate.optimizer import AcceleratedOptimizer
    original_step = AcceleratedOptimizer.step
    attempts = []

    def step(self, closure=None):
        attempts.append(1)
        self._is_overflow = len(attempts) == 2
        if self._is_overflow:
            return None
        return original_step(self, closure=closure)

    monkeypatch.setattr(AcceleratedOptimizer, 'step', step)
    workspace = Workspace(config(), output_dir=str(tmp_path))
    workspace.run()
    assert len(attempts) == 4
    assert workspace.completed_optimizer_steps == 3
    assert workspace.model.self_past_step == workspace.ema_model.self_past_step == 3
    assert workspace.ema_state['optimization_step'] == 3
    assert workspace.lr_scheduler_state['last_epoch'] == 3


def test_resume_rejects_observation_field_order_changes(tmp_path):
    cfg, payload = saved_payload(tmp_path)
    original = {'first': {'shape': [1], 'type': 'state'},
                'second': {'shape': [1], 'type': 'state'}}
    # Mapping equality alone ignores order, but fusion content depends on it.
    payload['cfg'].policy.shape_meta.obs = original
    payload['policy_config']['shape_meta']['obs'] = original
    payload['policy_config']['obs_encoder_config']['shape_meta']['obs'] = original
    cfg.policy.shape_meta.obs = dict(reversed(list(original.items())))
    with pytest.raises(ValueError, match='field order'):
        Workspace.validate_resume_payload(payload, cfg)


def test_resume_rejects_metadata_bridge_disagreement(tmp_path):
    cfg, payload = saved_payload(tmp_path)
    payload['metadata']['observation_encoder_contract'] = {
        'rgb_ports': [], 'state_ports': [], 'frame_embedding_shape': [2, 9],
        'original_obs_config': normalize_original_obs_config(),
    }
    with pytest.raises(ValueError, match='frame embedding'):
        Workspace.validate_resume_payload(payload, cfg)
