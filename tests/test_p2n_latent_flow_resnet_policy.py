"""CPU integration contracts with the actual trainable two-camera ResNet18.

The flow transformer and OAT are deliberately small; the vision backbones are
the original full ResNet18/GroupNorm/SpatialSoftmax implementations. No network,
GPU allocation, pretrained DINO, or production training is needed.
"""
import copy
from dataclasses import replace

import dill
import hydra
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from test_p2n_new_policy import META, observation, tokenizer_config
from oat.model.common.context_batch import Segment
from oat.model.common.normalizer import SingleFieldLinearNormalizer
from oat.perception.dinov3_patch_encoder import DINOv3PatchEncoder
from oat.perception.robomimic_vision_encoder import RobomimicRgbEncoder
from oat.policy.p2n_latent_flow import P2NLatentFlowPolicy
from oat.policy.p2n_state_gate_latent_flow import P2NStateGateLatentFlowPolicy
from oat.policy.p2n_latent_flow_common import prepare_flow_training_batch


def make_resnet_flow(gate=False, *, history_gate_mode='learned', self_past_p=0.):
    shape_meta = copy.deepcopy(META)
    for info in shape_meta['obs'].values():
        if info['type'] == 'rgb':
            # The 2x2 final spatial map keeps SpatialSoftmax sensitive to images.
            info['shape'] = [80, 80, 3]
    tc = tokenizer_config()
    for key in ('encoder', 'decoder'):
        tc[key].update(sample_horizon=16, latent_dim=5)
    tc['encoder']['num_registers'] = 8
    tc['decoder']['latent_horizon'] = 8
    tc['quantizer']['levels'] = [8, 5, 5, 5, 5]
    tokenizer = hydra.utils.instantiate(tc)
    tokenizer.normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
    kwargs = dict(shape_meta=shape_meta, obs_encoder_type='resnet18',
        resnet_config=dict(crop_shape=[64, 64], use_group_norm=True,
                           share_rgb_model=False, eval_fixed_crop=True),
        action_tokenizer=tokenizer, tokenizer_config=tc, embed_dim=16,
        n_layers=2, n_heads=2, ffn_dim=32, dropout=0.,
        activation_checkpointing=True, self_past_p=self_past_p,
        self_past_warmup_steps=0, self_past_ramp_steps=0, self_past_chunk_size=2)
    if gate:
        kwargs.update(history_embed_dim=16, history_n_heads=2, history_n_layers=1,
            history_dropout=0., history_gate_hidden_dim=16,
            history_gate_mode=history_gate_mode)
    return (P2NStateGateLatentFlowPolicy if gate else P2NLatentFlowPolicy)(**kwargs)


def make_resnet_batch(policy, size=4):
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
    dino_init = DINOv3PatchEncoder.__init__
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise AssertionError('ResNet latent flow must not load DINO or download weights')
    monkeypatch.setattr(DINOv3PatchEncoder, '__init__', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    import torchvision.models._api
    monkeypatch.setattr(torchvision.models._api, 'load_state_dict_from_url', forbidden)
    yield dino_init
    torch.set_num_threads(previous)


@pytest.mark.parametrize('gate', [False, True])
def test_actual_resnet_receives_gradients_and_oat_stays_frozen(gate):
    torch.manual_seed(71)
    student = make_resnet_flow(gate).train()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    batch = make_resnet_batch(student)
    assert any(isinstance(module, RobomimicRgbEncoder) for module in student.obs_encoder.modules())
    assert not any(isinstance(module, (DINOv3PatchEncoder, nn.BatchNorm2d))
                   for module in student.obs_encoder.modules())
    assert any(isinstance(module, nn.GroupNorm) for module in student.obs_encoder.modules())
    camera_convs = [module for module in student.obs_encoder.modules()
                    if isinstance(module, nn.Conv2d) and module.in_channels == 3]
    assert len(camera_convs) == 2
    assert camera_convs[0].weight.data_ptr() != camera_convs[1].weight.data_ptr()
    conv = first_rgb_conv(student)
    initial_conv = conv.weight.detach().clone()
    frozen = {name: value.detach().clone() for name, value in student.action_tokenizer.state_dict().items()}
    optimizer = student.get_optimizer(policy_lr=.005, obs_enc_lr=.0005)
    assert next(group['lr'] for group in optimizer.param_groups
                if any(value is conv.weight for value in group['params'])) == .0005
    state_weight = student.obs_encoder.state_projection.weight
    assert next(group['lr'] for group in optimizer.param_groups
                if any(value is state_weight for value in group['params'])) == .005
    generator = torch.Generator().manual_seed(912)
    had_backbone_gradient = False
    for _ in range(3):
        prepared = prepare_flow_training_batch(batch, student=student, teacher=teacher,
                                              generator=generator)
        assert prepared.frozen_patches is None
        assert prepared.obs_encoder_type == 'resnet18'
        assert prepared.prepared_visual.shape == (4, 2, 2, 3, 64, 64)
        assert not prepared.prepared_visual.requires_grad
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = student(prepared)
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        missing = [name for name, value in student.named_parameters()
                   if value.requires_grad and value.grad is None]
        assert not missing, f'Unused trainable parameters: {missing}'
        assert all(torch.isfinite(value.grad).all() for value in student.parameters()
                   if value.requires_grad)
        had_backbone_gradient |= bool(conv.weight.grad.abs().sum() > 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        student.on_optimizer_step()
    assert had_backbone_gradient and not torch.equal(initial_conv, conv.weight)
    assert all(torch.equal(value, frozen[name]) for name, value in student.action_tokenizer.state_dict().items())
    assert all(value.grad is None for value in teacher.parameters())
    assert all(value.grad is None for value in student.action_tokenizer.parameters())
    context = student.build_context(batch['obs'], batch['past_action'], batch['past_action_valid'])
    assert context.memory.shape == (4, 19 if gate else 15, 16)
    assert int(context.segment_mask(Segment.VISUAL).sum()) == 4
    assert int(context.segment_mask(Segment.PROPRIO).sum()) == 2
    context.validate_variant(student.variant)
    assert student.self_past_step == 3


def test_consistency_teacher_uses_own_backbone_on_identical_student_crops(monkeypatch):
    from oat.model.flow.consistency_flow import FlowSchedule
    import oat.policy.p2n_latent_flow_common as common
    student = make_resnet_flow().train()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    student_conv, teacher_conv = first_rgb_conv(student), first_rgb_conv(teacher)
    with torch.no_grad():
        teacher_conv.weight.add_(.1)
    assert student_conv.weight.data_ptr() != teacher_conv.weight.data_ptr()
    batch = make_resnet_batch(student)
    fm, ct = torch.tensor([0, 1, 2]), torch.tensor([3])
    monkeypatch.setattr(common, 'sample_flow_schedule', lambda *args, **kwargs:
        FlowSchedule(torch.full((4,), .2), torch.tensor([0., 0., 0., .3]), fm, ct))
    seen = {'student': [], 'teacher': []}
    hooks = [student_conv.register_forward_pre_hook(
        lambda _, args: seen['student'].append(args[0].detach().clone())),
        teacher_conv.register_forward_pre_hook(
        lambda _, args: seen['teacher'].append(args[0].detach().clone()))]
    try:
        prepared = prepare_flow_training_batch(batch, student=student, teacher=teacher,
                                              generator=torch.Generator().manual_seed(83))
        with pytest.raises(ValueError):
            replace(prepared, frozen_patches=torch.zeros(1)).validate()
        assert not seen['student'] and len(seen['teacher']) == 1
        assert torch.equal(seen['teacher'][0], prepared.prepared_visual[ct, :, 0].flatten(0, 1))
        student(prepared).backward()
        assert seen['student']
        assert all(torch.equal(value, prepared.prepared_visual[:, :, 0].flatten(0, 1))
                   for value in seen['student'])
        assert all(value.grad is None for value in teacher.parameters())
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize('gate', [False, True])
def test_state_enters_projection_and_global_condition_once_normalized(gate):
    policy = make_resnet_flow(gate).eval()
    batch = make_resnet_batch(policy, 1)
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
        assert torch.equal(policy.current_state_features(batch['obs']), expected.flatten(1))
    finally:
        hook.remove()


def test_euler_reuses_one_encoding_and_eval_uses_original_normalized_center_crop():
    policy = make_resnet_flow().eval()
    batch = make_resnet_batch(policy, 1)
    seen = []
    hook = first_rgb_conv(policy).register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone()))
    try:
        first = policy.predict_action(batch['obs'], past_actions=batch['past_action'],
            past_action_valid=batch['past_action_valid'], num_flow_steps=8,
            generator=torch.Generator().manual_seed(51))
        assert len(seen) == 1
        expected = batch['obs']['camera_a'][:, :, 8:72, 8:72].float()
        expected = (expected / 127.5 - 1.).flatten(0, 1).permute(0, 3, 1, 2)
        assert torch.allclose(seen[0], expected, atol=1e-6)
        second = policy.predict_action(batch['obs'], past_actions=batch['past_action'],
            past_action_valid=batch['past_action_valid'], num_flow_steps=8,
            generator=torch.Generator().manual_seed(51))
        assert len(seen) == 2 and torch.equal(seen[0], seen[1])
        assert first['action'].shape == (1, 8, 7)
        assert first['action_pred'].shape == (1, 16, 7)
        assert torch.equal(first['action_pred'], second['action_pred'])
    finally:
        hook.remove()


@pytest.mark.parametrize('gate', [False, True])
def test_self_contained_resnet_checkpoint_and_encoder_mismatch_rejection(gate, tmp_path):
    policy = make_resnet_flow(gate).eval()
    metadata = policy.artifact_metadata()
    config = policy.export_config()
    assert metadata['obs_encoder_type'] == config['obs_encoder_type'] == 'resnet18'
    assert 'resnet18' in policy.get_policy_name()
    assert 'dino' not in config['obs_encoder_config']
    payload = dict(cfg=OmegaConf.create({'training': {'use_ema': True}, 'policy': config}),
        policy_config=config, metadata=metadata,
        state_dicts={'model': policy.state_dict(), 'ema_model': policy.state_dict()})
    path = tmp_path / 'resnet_flow.ckpt'
    torch.save(payload, path, pickle_module=dill)
    restored = type(policy).from_checkpoint(path)
    assert restored.obs_encoder_type == 'resnet18' and not restored.training
    assert restored._past_buffer is None
    for key, value in policy.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    payload['metadata']['obs_encoder_type'] = 'dinov3'
    torch.save(payload, path, pickle_module=dill)
    with pytest.raises(ValueError, match='encoder'):
        type(policy).from_checkpoint(path)
    path.unlink()


def test_resnet_closed_gate_has_no_history_bypass():
    policy = make_resnet_flow(True, history_gate_mode='closed').eval()
    for name, value in policy.model.named_parameters():
        if 'modulation' in name or 'output' in name:
            nn.init.normal_(value, std=.05)
    batch = make_resnet_batch(policy, 1)
    changed = copy.deepcopy(batch['obs'])
    changed['state_history__robot0_eef_pos'] += 50
    def predict(obs):
        return policy.predict_action(obs, past_actions=batch['past_action'],
            past_action_valid=batch['past_action_valid'],
            generator=torch.Generator().manual_seed(31))['action_pred']
    assert torch.equal(predict(batch['obs']), predict(changed))


@pytest.mark.parametrize('gate', [False, True])
def test_resnet_tail_validation_and_all_invalid_self_past(gate, monkeypatch):
    policy = make_resnet_flow(gate, self_past_p=1.).train()
    batch = make_resnet_batch(policy, 1)
    batch['prev_window_valid'].zero_()
    if gate:
        batch['prev_obs']['state_history_valid'].zero_()
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid previous windows must be excluded before generation')
    monkeypatch.setattr(policy, '_generate_actions', forbidden)
    history = policy._maybe_self_past(batch, batch['past_action'], probability=1.)
    assert torch.equal(history, batch['past_action']) and policy.training
    metrics = policy.validation_metrics(batch, torch.Generator().manual_seed(11), 'generated')
    assert metrics['decoded_action_mse']['count'] == 7
    assert metrics['translation_mse']['count'] == 3
    assert metrics['rotation_mse']['count'] == 3
    assert metrics['gripper_mse']['count'] == 1
    assert metrics['projection_legal']['sum'] == metrics['projection_legal']['count']
    assert policy.training


def test_direct_cross_encoder_state_loading_fails_before_any_tensor_changes(cpu_and_offline, monkeypatch):
    # Construct the existing offline mocked DINO fixture only to test that these
    # two otherwise identical flow families cannot load one another's weights.
    import transformers
    from test_p2n_new_policy import MockDINOBackbone
    from test_p2n_latent_flow_policy import make_flow
    resnet = make_resnet_flow().eval()
    with monkeypatch.context() as local:
        local.setattr(DINOv3PatchEncoder, '__init__', cpu_and_offline)
        local.setattr(transformers, 'DINOv3ViTModel', MockDINOBackbone)
        dino = make_flow().eval()
    before_resnet = first_rgb_conv(resnet).weight.detach().clone()
    before_dino = dino.obs_encoder.patch_projection.weight.detach().clone()
    with pytest.raises(ValueError, match='observation encoder'):
        resnet.load_state_dict(dino.state_dict(), strict=True)
    with pytest.raises(ValueError, match='observation encoder'):
        dino.load_state_dict(resnet.state_dict(), strict=True)
    assert torch.equal(first_rgb_conv(resnet).weight, before_resnet)
    assert torch.equal(dino.obs_encoder.patch_projection.weight, before_dino)
