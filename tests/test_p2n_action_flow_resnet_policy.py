"""CPU contracts for direct action flow with actual trainable ResNet18 cameras.

The DiTX is deliberately small. Vision integration tests use the full original
ResNet18/GroupNorm/SpatialSoftmax modules, with no downloads or GPU allocation.
The explicit tiny_backbones option is reserved for the two-rank CPU smoke.
"""
import copy

import dill
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from test_p2n_new_policy import META, observation
from oat.model.common.context_batch import Segment
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.perception.action_flow_resnet_obs_encoder import ActionFlowResNetObservationEncoder
from oat.perception.dinov3_patch_encoder import DINOv3PatchEncoder
from oat.perception.robomimic_vision_encoder import RobomimicRgbEncoder
from oat.policy.p2n_action_flow_resnet18 import P2NActionFlowResNet18Policy
from oat.policy.p2n_state_gate_action_flow_resnet18 import P2NStateGateActionFlowResNet18Policy
from oat.policy.p2n_action_flow_resnet_common import prepare_action_flow_resnet_training_batch


def make_resnet_action_flow(gate=False, *, history_gate_mode='learned',
                            self_past_p=0., tiny_backbones=False):
    shape_meta = copy.deepcopy(META)
    for info in shape_meta['obs'].values():
        if info['type'] == 'rgb':
            # Keep a spatial map larger than 1x1 for meaningful SpatialSoftmax gradients.
            info['shape'] = [80, 80, 3]
    resnet_config = dict(crop_shape=[64, 64], use_group_norm=True,
                         share_rgb_model=False, eval_fixed_crop=True)
    kwargs = dict(shape_meta=shape_meta, resnet_config=resnet_config,
        embed_dim=16, n_layers=2, n_heads=2, flow=dict(ffn_hidden_dim=64),
        dropout=0., activation_checkpointing=True, self_past_p=self_past_p,
        self_past_warmup_steps=0, self_past_ramp_steps=0, self_past_chunk_size=2)
    if tiny_backbones:
        encoder = ActionFlowResNetObservationEncoder(shape_meta, n_obs_steps=2,
                                                     n_emb=16, **resnet_config)
        # Preserve the production adapter/crop/normalizer contract while keeping
        # distributed optimizer + resume coverage inexpensive on CPU.
        for camera in encoder.rgb_ports:
            encoder.vision_encoder.encoder.obs_nets[camera] = nn.Sequential(
                nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.GroupNorm(2, 8),
                nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 64))
        kwargs['obs_encoder'] = encoder
    if gate:
        kwargs.update(history_embed_dim=16, history_n_heads=2, history_n_layers=1,
            history_dropout=0., history_gate_hidden_dim=16,
            history_gate_mode=history_gate_mode)
    cls = P2NStateGateActionFlowResNet18Policy if gate else P2NActionFlowResNet18Policy
    policy = cls(**kwargs)
    normalizer = LinearNormalizer()
    for key in ['action', *policy.obs_encoder.state_ports]:
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    for key in policy.obs_encoder.rgb_ports:
        normalizer[key] = SingleFieldLinearNormalizer.create_fit(
            torch.tensor([[0., 0., 0.], [255., 255., 255.]]), mode='limits')
    policy.set_normalizer(normalizer)
    return policy


def make_resnet_action_batch(policy, size=4):
    valid = torch.ones(size, 7, dtype=torch.bool)
    result = dict(obs=observation(policy, size, valid),
        action=torch.randn(size, 16, 7) * .1,
        past_action=torch.randn(size, 7, 7) * .1, past_action_valid=valid,
        prev_obs=observation(policy, size, valid),
        prev_past_action=torch.randn(size, 7, 7) * .1,
        prev_past_action_valid=valid.clone(),
        prev_window_valid=torch.ones(size, dtype=torch.bool),
        future_action_valid=torch.ones(size, 16, dtype=torch.bool),
        sample_id=torch.arange(size))
    result['future_action_valid'][-1, 1:] = False
    return result


def first_rgb_conv(policy):
    return next(module for module in policy.obs_encoder.modules()
                if isinstance(module, nn.Conv2d) and module.in_channels == 3)


@pytest.fixture(autouse=True)
def cpu_and_offline(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise AssertionError('Direct ResNet action flow must not construct DINO/OAT or download weights')
    monkeypatch.setattr(DINOv3PatchEncoder, '__init__', forbidden)
    from oat.tokenizer.oat.tokenizer import OATTok
    monkeypatch.setattr(OATTok, '__init__', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    import torchvision.models._api
    monkeypatch.setattr(torchvision.models._api, 'load_state_dict_from_url', forbidden)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('gate', [False, True])
def test_actual_resnet_full_gradients_optimizer_and_context(gate):
    torch.manual_seed(71)
    student = make_resnet_action_flow(gate).train()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    batch = make_resnet_action_batch(student)
    assert any(isinstance(m, RobomimicRgbEncoder) for m in student.obs_encoder.modules())
    assert not any(isinstance(m, (DINOv3PatchEncoder, nn.BatchNorm2d))
                   for m in student.obs_encoder.modules())
    assert any(isinstance(m, nn.GroupNorm) for m in student.obs_encoder.modules())
    camera_convs = [m for m in student.obs_encoder.modules()
                    if isinstance(m, nn.Conv2d) and m.in_channels == 3]
    assert len(camera_convs) == 2
    assert camera_convs[0].weight.data_ptr() != camera_convs[1].weight.data_ptr()
    conv = first_rgb_conv(student)
    before = conv.weight.detach().clone()
    optimizer = student.get_optimizer(policy_lr=.005, obs_enc_lr=.0005)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in student.parameters() if p.requires_grad}
    assert {id(p) for p in parameters}.isdisjoint({id(p) for p in teacher.parameters()})
    def learning_rate(parameter):
        return next(group['lr'] for group in optimizer.param_groups
                    if any(value is parameter for value in group['params']))
    assert learning_rate(conv.weight) == .0005
    assert learning_rate(student.obs_encoder.state_projection.weight) == .005
    if not gate:
        assert not any('history_encoder' in name or 'history_gate' in name or
                       'observation_pool' in name for name, _ in student.named_parameters())
    generator = torch.Generator().manual_seed(912)
    for update in range(4):
        prepared = student.prepare_training_batch(batch, teacher=teacher, generator=generator)
        assert prepared.prepared_visual.shape == (4, 2, 2, 3, 64, 64)
        assert prepared.prepared_visual.grad_fn is None and not prepared.prepared_visual.requires_grad
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = student(prepared)
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        missing = [name for name, p in student.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f'Unused trainable parameters: {missing}'
        assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.requires_grad)
        if update == 3:
            zero = [name for name, p in student.named_parameters()
                    if p.requires_grad and not torch.count_nonzero(p.grad)]
            assert not zero, f'Zero gradients after leaving zero initialization: {zero}'
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        student.on_optimizer_step()
    assert not torch.equal(before, conv.weight)
    assert all(p.grad is None for p in teacher.parameters())
    context = student.build_context(batch['obs'], batch['past_action'], batch['past_action_valid'])
    assert context.memory.shape == (4, 19 if gate else 15, 16)
    assert int(context.segment_mask(Segment.VISUAL).sum()) == 4
    assert int(context.segment_mask(Segment.PROPRIO).sum()) == 2
    context.validate_variant(student.variant)
    assert student.self_past_step == 4
    assert student._last_flow_losses.keys() == {'loss', 'fm_loss', 'ct_loss'}
    assert not any('tokenizer' in name or 'quantizer' in name or 'latent_adapter' in name
                   for name in student.state_dict())
    assert not hasattr(student, 'action_tokenizer')


def test_consistency_teacher_uses_own_backbone_and_identical_crops(monkeypatch):
    from oat.model.flow.consistency_flow import FlowSchedule
    import oat.policy.p2n_action_flow_resnet_common as common
    student = make_resnet_action_flow().train()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    student_conv, teacher_conv = first_rgb_conv(student), first_rgb_conv(teacher)
    with torch.no_grad():
        teacher_conv.weight.add_(.1)
    assert student_conv.weight.data_ptr() != teacher_conv.weight.data_ptr()
    batch = make_resnet_action_batch(student)
    fm, ct = torch.tensor([0, 1, 2]), torch.tensor([3])
    monkeypatch.setattr(common, 'sample_flow_schedule', lambda *args, **kwargs:
        FlowSchedule(torch.full((4,), .2), torch.tensor([0., 0., 0., .3]), fm, ct))
    seen = {'student': [], 'teacher': []}
    hooks = [student_conv.register_forward_pre_hook(
        lambda _, args: seen['student'].append(args[0].detach().clone())),
        teacher_conv.register_forward_pre_hook(
        lambda _, args: seen['teacher'].append(args[0].detach().clone()))]
    try:
        prepared = prepare_action_flow_resnet_training_batch(batch, student=student,
            teacher=teacher, generator=torch.Generator().manual_seed(83))
        assert not seen['student'] and len(seen['teacher']) == 1
        assert torch.equal(seen['teacher'][0], prepared.prepared_visual[ct, :, 0].flatten(0, 1))
        student(prepared).backward()
        assert seen['student']
        assert all(torch.equal(value, prepared.prepared_visual[:, :, 0].flatten(0, 1))
                   for value in seen['student'])
        assert all(p.grad is None for p in teacher.parameters())
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize('gate', [False, True])
def test_eval_center_crop_one_encoding_and_deterministic_validation(gate):
    policy = make_resnet_action_flow(gate).eval()
    batch = make_resnet_action_batch(policy, 1)
    seen = []
    hook = first_rgb_conv(policy).register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone()))
    try:
        def predict():
            return policy.predict_action(batch['obs'], past_actions=batch['past_action'],
                past_action_valid=batch['past_action_valid'], num_flow_steps=8,
                generator=torch.Generator().manual_seed(51))
        result = predict()
        assert len(seen) == 1
        expected = batch['obs']['camera_a'][:, :, 8:72, 8:72].float()
        expected = (expected / 127.5 - 1.).flatten(0, 1).permute(0, 3, 1, 2)
        assert torch.allclose(seen[0], expected, atol=1e-6)
        second = predict()
        assert len(seen) == 2 and torch.equal(seen[0], seen[1])
        assert torch.equal(result['action_pred'], second['action_pred'])
        assert result['action'].shape == (1, 8, 7) and result['action_pred'].shape == (1, 16, 7)
        assert policy._past_buffer is None and policy._pending_execution_steps is None
        a = policy.validation_metrics(batch, torch.Generator().manual_seed(11), 'generated')
        b = policy.validation_metrics(batch, torch.Generator().manual_seed(11), 'generated')
        assert all(torch.equal(a[key][field], b[key][field]) for key in a for field in ('sum', 'count'))
        assert a['action_mse']['count'] == a['normalized_action_mse']['count'] == 7
        assert policy._past_buffer is None and policy._pending_execution_steps is None
    finally:
        hook.remove()


@pytest.mark.parametrize('gate', [False, True])
def test_state_normalized_once_for_context(gate):
    policy = make_resnet_action_flow(gate).eval()
    batch = make_resnet_action_batch(policy, 1)
    for key in policy.obs_encoder.state_ports:
        with torch.no_grad():
            policy.action_normalizer[key].params_dict['scale'].fill_(2.)
            policy.action_normalizer[key].params_dict['offset'].fill_(3.)
    expected = torch.cat([batch['obs'][key].float() * 2. + 3.
                          for key in policy.obs_encoder.state_ports], dim=-1)
    seen = []
    hook = policy.obs_encoder.state_projection.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone()))
    try:
        policy.build_context(batch['obs'], batch['past_action'], batch['past_action_valid'])
        assert len(seen) == 1 and torch.equal(seen[0], expected)
    finally:
        hook.remove()


@pytest.mark.parametrize('gate', [False, True])
def test_raw_self_past_execution_acknowledgement_and_training_mode(gate, monkeypatch):
    policy = make_resnet_action_flow(gate, self_past_p=1.).train()
    batch = make_resnet_action_batch(policy, 2)
    batch['prev_window_valid'][1] = False
    generated = torch.full((1, 16, 7), 4.25)
    calls = []
    def generate(obs, past, valid, **kwargs):
        calls.append((len(past), policy.training, first_rgb_conv(policy).training))
        return generated.expand(len(past), -1, -1)
    monkeypatch.setattr(policy, '_generate_actions', generate)
    replaced = policy._maybe_self_past(batch, batch['past_action'], probability=1.)
    assert calls == [(1, False, False)]
    assert torch.equal(replaced[0], generated[0, :7])
    assert torch.equal(replaced[1], batch['past_action'][1])
    assert policy.training and first_rgb_conv(policy).training
    assert not replaced.requires_grad
    obs = policy.create_dummy_observation(1)
    result = policy.predict_action(obs)
    assert not policy._past_valid_buffer.any()
    policy.record_executed_actions(result['action'], executed_lengths=[3])
    assert int(policy._past_valid_buffer.sum()) == 3
    assert torch.equal(policy._past_buffer[0, -3:], generated[0, :3])
    policy.reset()
    assert policy._past_buffer is None


def test_missing_rgb_normalizer_rejected_and_actions_inverse_once(monkeypatch):
    policy = make_resnet_action_flow().eval()
    normalizer = LinearNormalizer()
    for key in ['action', *policy.obs_encoder.state_ports]:
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    with pytest.raises(KeyError, match='RGB|camera'):
        policy.set_normalizer(normalizer)
    for key in policy.obs_encoder.rgb_ports:
        normalizer[key] = SingleFieldLinearNormalizer.create_fit(
            torch.tensor([[0., 0., 0.], [255., 255., 255.]]), mode='limits')
    values = torch.randn(100, 7) * 3. + 8.
    values[:, 6] = .5
    normalizer['action'] = SingleFieldLinearNormalizer.create_fit(values)
    policy.set_normalizer(normalizer)
    raw = torch.randn(2, 16, 7) + 8.
    normalized = policy._encode_target(raw)
    assert normalized.dtype == torch.float32
    monkeypatch.setattr(policy, '_generate_normalized_actions', lambda *args, **kwargs: normalized)
    assert torch.allclose(policy._generate_actions({}, None, None), raw, atol=2e-6)


@pytest.mark.parametrize('gate', [False, True])
def test_offline_strict_standalone_artifact_and_rejections(gate, tmp_path):
    policy = make_resnet_action_flow(gate).eval()
    metadata, config = policy.artifact_metadata(), policy.export_config()
    assert metadata['policy_family'] == 'continuous_action_flow'
    assert metadata['obs_encoder_type'] == config['obs_encoder_type'] == 'resnet18'
    assert 'resnet18' in policy.get_policy_name()
    assert 'dino' not in config['obs_encoder_config']
    payload = dict(cfg=OmegaConf.create({'training': {'use_ema': True}, 'policy': config}),
        policy_config=config, metadata=metadata,
        state_dicts={'model': policy.state_dict(), 'ema_model': policy.state_dict()})
    path = tmp_path / 'resnet_action_flow.ckpt'
    torch.save(payload, path, pickle_module=dill)
    restored = type(policy).from_checkpoint(path)
    assert restored.obs_encoder_type == 'resnet18' and not restored.training
    assert restored._past_buffer is None
    for key, value in policy.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    with pytest.raises(ValueError, match='strict'):
        restored.load_state_dict(policy.state_dict(), strict=False)
    other = P2NActionFlowResNet18Policy if gate else P2NStateGateActionFlowResNet18Policy
    with pytest.raises(ValueError, match='family/variant|variant'):
        other.from_checkpoint(path)
    for field, invalid in (('obs_encoder_type', 'dinov3'), ('policy_family', 'continuous_latent_flow')):
        original = payload['metadata'][field]
        payload['metadata'][field] = invalid
        torch.save(payload, path, pickle_module=dill)
        with pytest.raises(ValueError, match='encoder|family/variant'):
            type(policy).from_checkpoint(path)
        payload['metadata'][field] = original
    path.unlink()


@pytest.mark.parametrize('teacher_problem', ['same', 'train', 'grad'])
def test_teacher_must_be_independent_frozen_eval(teacher_problem):
    student = make_resnet_action_flow(tiny_backbones=True).train()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    if teacher_problem == 'same':
        teacher = student
    elif teacher_problem == 'train':
        teacher.train()
    else:
        teacher.requires_grad_(True)
    with pytest.raises(ValueError, match='EMA|eval|independent'):
        student.prepare_training_batch(make_resnet_action_batch(student), teacher=teacher,
                                       generator=torch.Generator().manual_seed(8))


def test_closed_gate_has_no_history_bypass():
    policy = make_resnet_action_flow(True, history_gate_mode='closed').eval()
    for name, value in policy.model.named_parameters():
        if 'modulation' in name or 'output' in name:
            nn.init.normal_(value, std=.05)
    batch = make_resnet_action_batch(policy, 1)
    changed = copy.deepcopy(batch['obs'])
    changed['state_history__robot0_eef_pos'] += 50
    def predict(obs):
        return policy.predict_action(obs, past_actions=batch['past_action'],
            past_action_valid=batch['past_action_valid'],
            generator=torch.Generator().manual_seed(31))['action_pred']
    assert torch.equal(predict(batch['obs']), predict(changed))
