"""CPU regressions for coherent snapshots and completed-epoch continuation."""
import copy
from collections import Counter

import dill
import pytest
import torch
from omegaconf import OmegaConf

from oat.model.common.lr_scheduler import get_scheduler
from oat.model.diffusion.ema_model import EMAModel
from oat.workspace.base_workspace import _copy_to_cpu
from oat.workspace.train_policy import TrainPolicyWorkspace


class TinyPolicy(torch.nn.Module):
    def __init__(self, history_counter=True):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.2, -0.1]]))
            self.linear.bias.fill_(0.05)
        if history_counter:
            self.register_buffer('_self_past_optimizer_step', torch.zeros((), dtype=torch.long))

    @property
    def self_past_step(self):
        return int(self._self_past_optimizer_step)


def workspace(tmp_path, history_counter=True):
    cfg = OmegaConf.create({
        '_target_': 'oat.workspace.train_policy.TrainPolicyWorkspace',
        'training': {'num_epochs': 2, 'gradient_accumulate_every': 1, 'max_train_steps': None},
    })
    result = TrainPolicyWorkspace(cfg, output_dir=str(tmp_path))
    result.model = TinyPolicy(history_counter)
    result.ema_model = copy.deepcopy(result.model)
    result.optimizer = torch.optim.AdamW(result.model.parameters(), lr=0.01, weight_decay=0)
    return result


def helpers(ws):
    ema = EMAModel(ws.ema_model, power=0.75)
    scheduler = get_scheduler('cosine', ws.optimizer, num_warmup_steps=0,
                              num_training_steps=6,
                              last_epoch=ws.completed_optimizer_steps - 1)
    ws._restore_training_helpers(ema, scheduler)
    return ema, scheduler


def update(ws, ema, scheduler):
    ws.optimizer.zero_grad(set_to_none=True)
    loss = (ws.model.linear(torch.tensor([[0.4, -0.7]])) - 0.8).square().mean()
    loss.backward()
    ws.optimizer.step()
    scheduler.step()
    ws.completed_optimizer_steps += 1
    ws.global_step += 1
    if hasattr(ws.model, '_self_past_optimizer_step'):
        ws.model._self_past_optimizer_step.add_(1)
    ema.step(ws.model)
    if hasattr(ws.ema_model, '_self_past_optimizer_step'):
        ws.ema_model._self_past_optimizer_step.copy_(ws.model._self_past_optimizer_step)


def load(path):
    return torch.load(path, map_location='cpu', pickle_module=dill, weights_only=False)


def legacy_checkpoint(ws, path, optimizer_steps=None, drop_history=False):
    state_dicts = {'model': _copy_to_cpu(ws.model.state_dict()),
                  'ema_model': _copy_to_cpu(ws.ema_model.state_dict()),
                  'optimizer': _copy_to_cpu(ws.optimizer.state_dict())}
    if drop_history:
        for name in ('model', 'ema_model'):
            state_dicts[name].pop('_self_past_optimizer_step', None)
    if optimizer_steps is not None:
        for state, step in zip(state_dicts['optimizer']['state'].values(), optimizer_steps):
            state['step'].fill_(step)
    torch.save({'cfg': ws.cfg, 'state_dicts': state_dicts,
                'pickles': {key: dill.dumps(value) for key, value in
                            {'epoch': 0, 'global_step': ws.global_step - 1, '_output_dir': None}.items()}},
               path, pickle_module=dill)


def test_cpu_snapshot_owns_optimizer_counters_and_nested_tensors():
    source = {'state': {0: {'step': torch.tensor(1946.), 'exp_avg': torch.ones(2)}}}
    captured = _copy_to_cpu(source)
    source['state'][0]['step'].add_(2)
    source['state'][0]['exp_avg'].zero_()
    assert captured['state'][0]['step'].item() == 1946
    torch.testing.assert_close(captured['state'][0]['exp_avg'], torch.ones(2))
    assert source['state'][0]['step'].data_ptr() != captured['state'][0]['step'].data_ptr()


def test_new_checkpoint_resume_matches_uninterrupted_optimizer_ema_and_lr(tmp_path):
    continuous = workspace(tmp_path / 'continuous')
    uninterrupted_ema, uninterrupted_scheduler = helpers(continuous)
    for _ in range(6):
        update(continuous, uninterrupted_ema, uninterrupted_scheduler)

    staged = workspace(tmp_path)
    ema, scheduler = helpers(staged)
    for _ in range(3):
        update(staged, ema, scheduler)
    assert staged._complete_epoch(ema, scheduler) == 0
    checkpoint = staged.save_checkpoint(tag='ep-0000')
    staged._saving_thread.join()
    payload = load(checkpoint)
    assert dill.loads(payload['pickles']['epoch']) == 1
    assert dill.loads(payload['pickles']['global_step']) == 3
    assert dill.loads(payload['pickles']['completed_optimizer_steps']) == 3
    assert dill.loads(payload['pickles']['checkpoint_version']) == 2

    resumed = workspace(tmp_path)
    resumed._resume_training_checkpoint(checkpoint)
    assert resumed.epoch == 1  # epoch 0 is complete, never repeated
    restored_ema, restored_scheduler = helpers(resumed)
    assert restored_ema.optimization_step == 3
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()
    assert [group['lr'] for group in resumed.optimizer.param_groups] == scheduler.get_last_lr()
    for _ in range(3):
        update(resumed, restored_ema, restored_scheduler)
    assert restored_ema.decay > 0
    for key, value in continuous.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
    for key, value in continuous.ema_model.state_dict().items():
        torch.testing.assert_close(resumed.ema_model.state_dict()[key], value, rtol=0, atol=0)
    assert restored_scheduler.state_dict() == uninterrupted_scheduler.state_dict()
    assert restored_ema.optimization_step == uninterrupted_ema.optimization_step == 6
    assert resumed._complete_epoch(restored_ema, restored_scheduler) == 1
    assert resumed.epoch == resumed.cfg.training.num_epochs  # final save resumes at exit


def test_async_checkpoint_captures_optimizer_counters_before_next_update(tmp_path):
    ws = workspace(tmp_path)
    ema, scheduler = helpers(ws)
    for _ in range(3):
        update(ws, ema, scheduler)
    ws._complete_epoch(ema, scheduler)
    checkpoint = ws.save_checkpoint(tag='ep-0000')
    update(ws, ema, scheduler)
    ws._saving_thread.join()
    payload = load(checkpoint)
    assert Counter(int(s['step']) for s in payload['state_dicts']['optimizer']['state'].values()) == {3: 2}
    assert int(payload['state_dicts']['model']['_self_past_optimizer_step']) == 3


def test_legacy_resume_uses_history_counter_and_repairs_only_excess_steps(tmp_path):
    original = workspace(tmp_path)
    ema, scheduler = helpers(original)
    for _ in range(3):
        update(original, ema, scheduler)
    path = tmp_path / 'legacy.ckpt'
    legacy_checkpoint(original, path, optimizer_steps=[4, 2])
    resumed = workspace(tmp_path)
    resumed._resume_training_checkpoint(path)
    assert (resumed.epoch, resumed.global_step, resumed.completed_optimizer_steps) == (1, 3, 3)
    assert [int(state['step']) for state in resumed.optimizer.state.values()] == [3, 2]
    assert resumed.resume_migration['clamped_optimizer_counters'] == 1
    restored_ema, restored_scheduler = helpers(resumed)
    assert restored_ema.optimization_step == 3
    assert restored_ema.get_decay(restored_ema.optimization_step) > 0
    assert restored_scheduler.last_epoch == 3
    for key, value in original.ema_model.state_dict().items():
        torch.testing.assert_close(resumed.ema_model.state_dict()[key], value)


def test_legacy_without_history_uses_original_completed_batch_count(tmp_path):
    original = workspace(tmp_path, history_counter=False)
    ema, scheduler = helpers(original)
    for _ in range(3):
        update(original, ema, scheduler)
    path = tmp_path / 'legacy_no_history.ckpt'
    legacy_checkpoint(original, path, optimizer_steps=[4, 4])
    resumed = workspace(tmp_path, history_counter=False)
    resumed._resume_training_checkpoint(path)
    assert (resumed.epoch, resumed.global_step, resumed.completed_optimizer_steps) == (1, 3, 3)
    assert all(int(state['step']) == 3 for state in resumed.optimizer.state.values())


def test_legacy_ambiguous_accumulation_migration_is_rejected(tmp_path):
    original = workspace(tmp_path, history_counter=False)
    ema, scheduler = helpers(original)
    for _ in range(3):
        update(original, ema, scheduler)
    original.cfg.training.gradient_accumulate_every = 2
    original.cfg.training.max_train_steps = 2
    path = tmp_path / 'ambiguous.ckpt'
    legacy_checkpoint(original, path)
    with pytest.raises(ValueError, match='explicit migration'):
        workspace(tmp_path, history_counter=False)._resume_training_checkpoint(path)


def test_new_checkpoint_missing_scheduler_state_is_rejected(tmp_path):
    ws = workspace(tmp_path)
    ema, scheduler = helpers(ws)
    update(ws, ema, scheduler)
    ws._complete_epoch(ema, scheduler)
    path = ws.save_checkpoint(tag='valid', use_thread=False)
    payload = load(path)
    payload['pickles']['lr_scheduler_state'] = dill.dumps(None)
    torch.save(payload, path, pickle_module=dill)
    with pytest.raises(ValueError, match='missing scheduler state'):
        workspace(tmp_path)._resume_training_checkpoint(path)


def test_legacy_capped_global_step_is_normalized_even_with_history_counter(tmp_path):
    original = workspace(tmp_path)
    ema, scheduler = helpers(original)
    for _ in range(3):
        update(original, ema, scheduler)
    original.cfg.training.max_train_steps = 3
    original.global_step = 4  # Historical cap path included one extra count.
    path = tmp_path / 'capped_history.ckpt'
    legacy_checkpoint(original, path)
    resumed = workspace(tmp_path)
    resumed._resume_training_checkpoint(path)
    assert (resumed.epoch, resumed.global_step, resumed.completed_optimizer_steps) == (1, 3, 3)
