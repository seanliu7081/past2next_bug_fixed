"""CPU checks of reduction, EMA ownership, resume, and RNG continuation."""
import copy

import dill
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from oat.model.diffusion.ema_model import EMAModel
from oat.workspace.train_p2n_action_flow import (
    MetricSums, NonPaddingDistributedSampler, TrainP2NActionFlowWorkspace,
    assert_teacher_synchronized, make_fresh_ema, stable_validation_seed,
    successful_update, validate_optimizer_ownership, assert_frozen_normalizer,
    assert_equal_normalizers,
)


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioner = torch.nn.Linear(2, 2)
        self.velocity = torch.nn.Linear(2, 1)
        self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        self.register_buffer("curriculum", torch.tensor(0))
        from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
        self.action_normalizer = LinearNormalizer()
        self.action_normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
        self.action_normalizer.requires_grad_(False)

    @property
    def self_past_step(self):
        return int(self.curriculum)

    def on_optimizer_step(self):
        self.curriculum.add_(1)

    def set_self_past_step(self, step):
        self.curriculum.fill_(step)

    def forward(self, x):
        return self.velocity(self.conditioner(x))


def test_nonpadding_validation_shards_cover_uneven_and_empty_ranks():
    for size, world in [(11, 3), (1, 4), (0, 3), (7, 2)]:
        shards = [list(NonPaddingDistributedSampler(range(size), rank, world)) for rank in range(world)]
        flattened = [index for shard in shards for index in shard]
        assert sorted(flattened) == list(range(size))
        assert len(flattened) == len(set(flattened))
        for rank in range(world):
            assert len(shards[rank]) == len(NonPaddingDistributedSampler(range(size), rank, world))


def test_masked_metric_reduces_sums_not_local_means():
    accumulator = MetricSums(["mse", "empty"])
    # One partition has one valid element, the other has nine.
    accumulator.add({"mse": {"sum": torch.tensor(100.), "count": torch.tensor(1.)}})
    accumulator.add({"mse": {"sum": torch.tensor(9.), "count": torch.tensor(9.)}})
    result = accumulator.reduce()
    assert result["mse"] == pytest.approx(10.9)
    assert result["empty"] is None
    with pytest.raises(ValueError, match="Nonfinite"):
        accumulator.add({"mse": {"sum": torch.tensor(float("nan")), "count": 1}})


def test_sample_noise_is_partition_and_order_independent():
    def noises(ids):
        return {index: torch.randn(16, 7, generator=torch.Generator().manual_seed(
            stable_validation_seed(42, "dataset-identity", index))) for index in ids}
    together = noises([2, 41, 700, 10])
    partitioned = {**noises([41, 10]), **noises([700, 2])}
    assert all(torch.equal(together[key], partitioned[key]) for key in together)
    assert stable_validation_seed(42, "dataset-identity", 2) != stable_validation_seed(42, "other", 2)


def test_ema_covers_conditioner_and_curriculum_and_stays_frozen():
    student = TinyStudent()
    teacher = make_fresh_ema(student)
    assert_teacher_synchronized(teacher)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=0.01)
    validate_optimizer_ownership(student, teacher, optimizer)
    ema = EMAModel(teacher, power=0.75)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    original = teacher.conditioner.weight.clone()
    for _ in range(3):
        loss = student(torch.ones(4, 2)).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        successful_update(student, ema, scheduler)
    assert not torch.equal(original, teacher.conditioner.weight)
    assert student.self_past_step == teacher.self_past_step == ema.optimization_step == 3
    assert not teacher.training
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    # A skipped update invokes no helper and advances none of these counts.
    assert scheduler.last_epoch == 3
    assert not set(map(id, student.parameters())) & set(map(id, teacher.parameters()))


def test_optimizer_rejects_frozen_or_teacher_parameters():
    student = TinyStudent()
    teacher = make_fresh_ema(student)
    invalid = torch.optim.AdamW(list(student.parameters()))
    with pytest.raises(ValueError, match="exactly once"):
        validate_optimizer_ownership(student, teacher, invalid)


def _cfg():
    return OmegaConf.create(dict(
        policy_family="continuous_action_flow", variant="p2n_action_flow", task_type="real_robot",
        policy=dict(variant="p2n_action_flow", flow=dict(inference_steps=8)),
        training=dict(use_ema=True, seed=42, gradient_accumulate_every=8,
                      lr_scheduler="cosine", lr_warmup_steps=None, lr_warmup_ratio=.05),
        normalization=dict(source="training_replay_frames", mode="limits", refit_on_resume=False),
        ema=dict(power=.75), optimizer=dict(policy_lr=.00005), dataloader=dict(batch_size=4),
        task=dict(policy=dict(dataset=dict(zarr_path="old/data.zarr", seed=42, val_ratio=.05))))
    )


def _payload(cfg):
    return dict(cfg=copy.deepcopy(cfg), policy_config=dict(construction_mode="restore"),
                metadata=dict(policy_family="continuous_action_flow", artifact_schema_version=1,
                              variant="p2n_action_flow", task_type="real_robot"),
                state_dicts=dict(model={}, ema_model={}, optimizer={}),
                pickles={key: dill.dumps(None) for key in TrainP2NActionFlowWorkspace.include_keys})


def test_resume_rejects_wrong_family_variant_loss_and_incomplete_state():
    cfg = _cfg()
    payload = _payload(cfg)
    assert TrainP2NActionFlowWorkspace.validate_resume_payload(payload, cfg) is payload
    changed = copy.deepcopy(payload)
    changed["metadata"]["policy_family"] = "oat_autoregressive"
    with pytest.raises(ValueError, match="policy_family"):
        TrainP2NActionFlowWorkspace.validate_resume_payload(changed, cfg)
    changed = copy.deepcopy(payload)
    changed["metadata"]["variant"] = "p2n_state_gate_action_flow"
    with pytest.raises(ValueError, match="variant"):
        TrainP2NActionFlowWorkspace.validate_resume_payload(changed, cfg)
    changed_cfg = copy.deepcopy(cfg)
    changed_cfg.policy.flow.inference_steps = 2
    with pytest.raises(ValueError, match="policy.flow"):
        TrainP2NActionFlowWorkspace.validate_resume_payload(payload, changed_cfg)
    changed = copy.deepcopy(payload)
    del changed["state_dicts"]["ema_model"]
    with pytest.raises(ValueError, match="EMA"):
        TrainP2NActionFlowWorkspace.validate_resume_payload(changed, cfg)


def test_dedicated_rng_state_resumes_every_stream_exactly():
    workspace = TrainP2NActionFlowWorkspace(_cfg(), output_dir="unused")
    workspace.generators = {name: torch.Generator().manual_seed(seed) for seed, name in
                            enumerate(("train", "self_past", "dataloader"), 17)}
    for generator in workspace.generators.values():
        torch.randn(10, generator=generator)
    workspace.rng_states = [workspace._capture_rng()]
    expected = {name: torch.randn(7, generator=generator) for name, generator in workspace.generators.items()}
    expected_global = torch.randn(5)
    expected_numpy = np.random.randn(3)
    workspace._restore_rng()
    for name, generator in workspace.generators.items():
        assert torch.equal(expected[name], torch.randn(7, generator=generator))
    assert torch.equal(expected_global, torch.randn(5))
    assert np.array_equal(expected_numpy, np.random.randn(3))
    with pytest.raises(ValueError, match="world size"):
        workspace._restore_rng(world_size=2)


def test_complete_artifact_restores_distinct_ema_optimizer_and_counters(monkeypatch, tmp_path):
    import transformers
    from test_p2n_new_policy import MockDINOBackbone
    from test_p2n_action_flow_policy import make_flow
    monkeypatch.setattr(transformers, "DINOv3ViTModel", MockDINOBackbone)
    torch.set_num_threads(2)
    policy = make_flow(False)
    cfg = _cfg()
    cfg.task_type = "libero"
    cfg.policy = OmegaConf.create(policy.export_config())
    cfg.optimizer = OmegaConf.create(dict(policy_lr=.005, obs_enc_lr=.005, weight_decay=.01, betas=[.9, .95]))
    workspace = TrainP2NActionFlowWorkspace(cfg, output_dir=str(tmp_path))
    workspace.model = policy
    workspace.ema_model = make_fresh_ema(policy)
    workspace.optimizer = policy.get_optimizer(**cfg.optimizer)
    # Populate Adam moments and retain deliberately different EMA weights.
    loss = sum(p.square().sum() for p in policy.parameters() if p.requires_grad)
    loss.backward()
    workspace.optimizer.step()
    workspace.optimizer.zero_grad(set_to_none=True)
    policy.set_self_past_step(4)
    workspace.ema_model.set_self_past_step(4)
    workspace.completed_optimizer_steps = 4
    workspace.epoch, workspace.global_step = 2, 12
    workspace.ema_state = {"optimization_step": 4, "decay": .37}
    workspace.lr_scheduler_state = {"last_epoch": 4, "_last_lr": [.005] * len(workspace.optimizer.param_groups)}
    workspace.dataset_split = {"identity": "immutable-test-dataset"}
    workspace.normalizer_provenance = {
        "source": "training_replay_frames", "mode": "limits",
        "dataset_identity": "immutable-test-dataset", "train_episode_ids": [0],
        "training_frames": 32, "state_sha256": assert_frozen_normalizer(policy)}
    policy.normalizer_metadata = copy.deepcopy(workspace.normalizer_provenance)
    workspace.update_schedule = {"planned_optimizer_updates": 10}
    workspace.generators = {name: torch.Generator().manual_seed(seed) for seed, name in
                            enumerate(("train", "self_past", "dataloader"), 71)}
    path = workspace.save_checkpoint()
    restored = TrainP2NActionFlowWorkspace.create_from_checkpoint(path, output_dir=str(tmp_path))
    assert restored.epoch == 2 and restored.global_step == 12
    assert restored.completed_optimizer_steps == restored.ema_state["optimization_step"] == 4
    assert restored.ema_state["decay"] == .37
    assert restored.model.self_past_step == restored.ema_model.self_past_step == 4
    assert not restored.ema_model.training and all(not p.requires_grad for p in restored.ema_model.parameters())
    for name, tensor in workspace.ema_model.state_dict().items():
        assert torch.equal(tensor, restored.ema_model.state_dict()[name]), name
    for name, tensor in workspace.model.state_dict().items():
        assert torch.equal(tensor, restored.model.state_dict()[name]), name
    assert any(not torch.equal(p, dict(restored.ema_model.named_parameters())[name])
               for name, p in restored.model.named_parameters() if p.requires_grad)
    assert restored.optimizer.state_dict()["state"].keys() == workspace.optimizer.state_dict()["state"].keys()
    restored.generators = {name: torch.Generator() for name in workspace.generators}
    expected = {name: torch.rand(6, generator=value) for name, value in workspace.generators.items()}
    restored._restore_rng()
    assert all(torch.equal(expected[name], torch.rand(6, generator=value)) for name, value in restored.generators.items())


def test_workspace_flushes_tail_and_advances_only_optimizer_updates(monkeypatch, tmp_path):
    from accelerate import Accelerator
    import oat.workspace.train_p2n_action_flow as module

    class LossStudent(TinyStudent):
        def forward(self, packed):
            return super().forward(packed).square().mean()

        def get_optimizer(self, **_):
            return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=.001)

        def set_normalizer(self, normalizer):
            self.action_normalizer.load_state_dict(normalizer.state_dict())
            self.action_normalizer.requires_grad_(False)

        def prepare_training_batch(self, batch, **_):
            return batch["action"]

    class Dataset(torch.utils.data.Dataset):
        dataset_identity = "workspace-tail-test"

        def __init__(self, validation=False):
            self.validation = validation
            from types import SimpleNamespace
            self.train_mask = np.asarray([not validation, validation])
            self.normalization_train_mask = np.asarray([True, False])
            self.replay_buffer = SimpleNamespace(episode_ends=np.array([12, 13]))

        def __len__(self):
            return 0 if self.validation else 12

        def __getitem__(self, index):
            return {"action": torch.tensor([1., index / 10.])}

        def get_validation_dataset(self):
            return Dataset(True)

        def get_normalizer(self, mode):
            assert mode == "limits"
            return TinyStudent().action_normalizer

    cfg = _cfg()
    cfg.policy._target_ = "fake.Student"
    cfg.task.policy.dataset._target_ = "fake.Dataset"
    cfg.task.policy.lazy_eval = True
    cfg.training.update(dict(resume=False, allow_bf16=False, num_epochs=1,
                             gradient_accumulate_every=2, offline_validation_enabled=False,
                             max_grad_norm=1., checkpoint_every=100, snapshot_every=0))
    cfg.dataloader = dict(batch_size=4, drop_last=True, num_workers=0, shuffle=False)
    cfg.val_dataloader = dict(batch_size=4, drop_last=False, num_workers=0, shuffle=False)
    cfg.logging = dict(mode="disabled")
    cfg.checkpoint = dict(save_last_ckpt=False, save_last_snapshot=False)
    student = LossStudent()
    original_instantiate = module.hydra.utils.instantiate

    def instantiate(config, *args, **kwargs):
        target = config.get("_target_")
        if target == "fake.Student":
            return student
        if target == "fake.Dataset":
            return Dataset()
        return original_instantiate(config, *args, **kwargs)

    monkeypatch.setattr(module.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(module, "Accelerator", lambda **kwargs: Accelerator(cpu=True, **kwargs))
    workspace = TrainP2NActionFlowWorkspace(cfg, output_dir=str(tmp_path))
    workspace.run()
    assert workspace.global_step == 3
    assert workspace.completed_optimizer_steps == 2
    assert workspace.ema_state["optimization_step"] == 2
    assert workspace.model.self_past_step == workspace.ema_model.self_past_step == 2
    assert workspace.lr_scheduler_state["last_epoch"] == 2
    assert workspace.epoch == 1


def test_fresh_fit_uses_dataset_training_frames_once_and_records_provenance():
    from types import SimpleNamespace
    from oat.model.common.normalizer import LinearNormalizer
    calls = []
    class Dataset:
        train_mask = np.array([True, False, True])
        normalization_train_mask = train_mask.copy()
        replay_buffer = SimpleNamespace(episode_ends=np.array([2, 5, 9]))
        def get_normalizer(self, mode):
            calls.append(mode)
            normalizer = LinearNormalizer()
            # Includes a constant dimension and deliberately asymmetric scale.
            normalizer.fit({'action': torch.tensor([[2., 3.], [6., 3.]])}, mode=mode)
            return normalizer
    student = TinyStudent()
    def set_normalizer(normalizer):
        student.action_normalizer.load_state_dict(normalizer.state_dict())
        student.action_normalizer.requires_grad_(False)
    student.set_normalizer = set_normalizer
    workspace = TrainP2NActionFlowWorkspace(_cfg(), output_dir='unused')
    workspace.model = student
    workspace._fit_fresh_normalizer(Dataset(), {'identity': 'real-frames', 'train_episode_ids': [0, 2]})
    assert calls == ['limits']
    assert workspace.normalizer_provenance['training_frames'] == 6
    assert workspace.normalizer_provenance['train_episode_ids'] == [0, 2]
    assert workspace.normalizer_provenance['state_sha256'] == assert_frozen_normalizer(student)
    value = torch.tensor([[2., 3.], [4., 3.], [6., 3.]])
    field = student.action_normalizer['action']
    assert torch.isfinite(field.normalize(value)).all()
    torch.testing.assert_close(field.unnormalize(field.normalize(value)), value)
    invalid = Dataset()
    invalid.normalization_train_mask = np.ones(3, dtype=bool)
    with pytest.raises(ValueError, match='selected training episodes'):
        workspace._fit_fresh_normalizer(invalid, {'identity': 'real-frames', 'train_episode_ids': [0, 2]})
    assert calls == ['limits']


def test_ema_normalizer_requires_exact_frozen_statistics():
    student = TinyStudent()
    teacher = make_fresh_ema(student)
    assert_equal_normalizers(student, teacher)
    with torch.no_grad():
        teacher.action_normalizer['action'].params_dict['offset'].add_(.25)
    with pytest.raises(ValueError, match='match exactly'):
        assert_equal_normalizers(student, teacher)
    student.action_normalizer.requires_grad_(True)
    with pytest.raises(ValueError, match='remain frozen'):
        assert_frozen_normalizer(student)


def test_validation_covers_uneven_batches_and_rejects_duplicate_sample_ids():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    class Dataset(torch.utils.data.Dataset):
        dataset_identity = 'three-variable-length-actions'
        pad_before = 1
        seq_sampler = SimpleNamespace(indices=np.array([[0, 4, 1, 5], [1, 5, 1, 5], [2, 6, 1, 5]]))
        def __len__(self):
            return 3
        def __getitem__(self, index):
            return {'action': torch.ones(16, 7) * (index + 1),
                    'future_action_valid': torch.arange(16) < index + 1,
                    'sample_id': torch.tensor(index)}
    class MeasuredPolicy(torch.nn.Module):
        def validation_metrics(self, sample, generator, history_mode, compute_actions):
            error = sample['action'].square()
            valid = sample['future_action_valid'][..., None].expand_as(error)
            error = error * (2 if history_mode == 'generated' else 1)
            pair = {'sum': torch.where(valid, error, 0.).sum(), 'count': valid.sum()}
            return {'fm_loss': {'sum': torch.tensor(1.), 'count': torch.tensor(1.)},
                    'action_mse': pair, 'normalized_action_mse': pair}
    accelerator = SimpleNamespace(device=torch.device('cpu'), num_processes=1, autocast=nullcontext)
    workspace = TrainP2NActionFlowWorkspace(_cfg(), output_dir='unused')
    workspace.ema_model = MeasuredPolicy()
    dataset = Dataset()
    first = workspace._validate(accelerator, DataLoader(dataset, batch_size=2), dataset,
                                generated=True, compute_actions=True)
    second = workspace._validate(accelerator, DataLoader(dataset, batch_size=1), dataset,
                                 generated=True, compute_actions=True)
    assert first == second
    assert first['validation_samples'] == 3
    assert first['val_action_mse'] == pytest.approx((1 + 2*4 + 3*9) / 6)
    assert first['val_generated_action_mse'] == pytest.approx(2 * first['val_action_mse'])
    duplicate_loader = DataLoader(dataset, batch_size=2, sampler=[0, 0, 2])
    with pytest.raises(RuntimeError, match='duplicated, missing'):
        workspace._validate(accelerator, duplicate_loader, dataset, generated=True, compute_actions=True)


def test_restoring_into_non_cpu_policy_keeps_normalizer_on_its_device(monkeypatch):
    """Meta exercises the ParameterDict CPU-rebuild hazard without using a GPU."""
    import warnings
    import transformers
    from test_p2n_new_policy import MockDINOBackbone
    from test_p2n_action_flow_policy import make_flow
    monkeypatch.setattr(transformers, 'DINOv3ViTModel', MockDINOBackbone)
    torch.set_num_threads(2)
    policy = make_flow(False)
    cpu_checkpoint = copy.deepcopy(policy.state_dict())
    policy.to('meta')
    with warnings.catch_warnings():
        # CPU->meta ordinary tensors intentionally have no data copy; the
        # normalizer loader actually recreates ParameterDict from CPU tensors.
        warnings.filterwarnings('ignore', message='.*copying from a non-meta parameter.*')
        policy.load_state_dict(cpu_checkpoint, strict=True)
    assert all(parameter.device.type == 'meta' for parameter in policy.action_normalizer.parameters())
    assert all(not parameter.requires_grad for parameter in policy.action_normalizer.parameters())
