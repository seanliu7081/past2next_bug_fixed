"""Actual CPU ResNet-18 checks against the repository's original RGB encoder."""
import copy

import pytest
import torch
from torch import nn

from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.perception.latent_flow_resnet_obs_encoder import FlowResNetObservationEncoder


META = {
    'obs': {
        'camera_a': {'shape': [80, 80, 3], 'type': 'rgb'},
        'camera_b': {'shape': [80, 80, 3], 'type': 'rgb'},
        'robot_state': {'shape': [3], 'type': 'state'},
        'task_uid': {'shape': [1], 'type': 'state'},
    },
    'action': {'shape': [7]},
}


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def encoder_and_observation(**kwargs):
    encoder = FlowResNetObservationEncoder(META, n_emb=16, crop_shape=(64, 64), **kwargs)
    obs = {
        'camera_a': torch.randint(0, 256, (1, 2, 80, 80, 3), dtype=torch.uint8),
        'camera_b': torch.randint(0, 256, (1, 2, 80, 80, 3), dtype=torch.uint8),
        'robot_state': torch.randn(1, 2, 3),
        'task_uid': torch.randn(1, 2, 1),
    }
    return encoder, obs


@pytest.mark.parametrize('training', [False, True])
def test_original_rgb_feature_equivalence_after_separating_crops(training, monkeypatch):
    encoder, obs = encoder_and_observation()
    encoder.train(training)
    torch.manual_seed(173)
    prepared = encoder.prepare_conditioning(obs)
    assert prepared.shape == (1, 2, 2, 3, 64, 64)
    assert not prepared.requires_grad and prepared.grad_fn is None
    assert prepared.min() >= -1 and prepared.max() <= 1
    # Compare both feature paths on the exact same crop realization. The
    # original SpatialSoftmax consumes RNG even when its noise_std is zero,
    # so moving all crops before the networks changes global RNG interleaving.
    for camera, port in enumerate(encoder.rgb_ports):
        crop = prepared[:, :, camera].reshape(-1, 3, 64, 64)
        monkeypatch.setattr(encoder.vision_encoder.encoder.obs_randomizers[port],
                            'forward_in', lambda inputs, crop=crop: crop)
    with torch.no_grad():
        separate = encoder._encode_prepared_visual(prepared).flatten(2)
        original = encoder.vision_encoder(obs)
    torch.testing.assert_close(separate, original, rtol=0, atol=0)
    if not training:
        expected = (obs['camera_a'][:, :, 8:72, 8:72].float() * (2 / 255) - 1).permute(0, 1, 4, 2, 3)
        torch.testing.assert_close(prepared[:, :, 0], expected, rtol=0, atol=0)


def test_original_backbones_train_without_downloads_and_cameras_are_independent(monkeypatch):
    import torchvision.models
    original = torchvision.models.resnet18
    initialized = []

    def local_resnet(*args, **kwargs):
        assert kwargs.get('pretrained') is False
        initialized.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(torchvision.models, 'resnet18', local_resnet)
    encoder, obs = encoder_and_observation()
    assert len(initialized) == 2
    nets = encoder.vision_encoder.encoder.obs_nets
    assert nets['camera_a'] is not nets['camera_b']
    assert any(isinstance(module, nn.GroupNorm) for module in nets.modules())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in nets.modules())
    assert not hasattr(encoder, 'dino_encoder') and not hasattr(encoder, 'resampler')
    visual, proprio = encoder(obs)
    assert visual.shape == (1, 4, 16) and proprio.shape == (1, 2, 16)
    (visual.square().mean() + proprio.square().mean()).backward()
    for port in encoder.rgb_ports:
        first_conv = next(module for module in nets[port].modules() if isinstance(module, nn.Conv2d))
        assert first_conv.weight.requires_grad
        assert first_conv.weight.grad is not None
        assert torch.isfinite(first_conv.weight.grad).all()
        assert first_conv.weight.grad.abs().sum() > 0
    assert encoder.visual_projection.weight.grad.abs().sum() > 0
    assert encoder.state_projection.weight.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in encoder.vision_encoder.normalizer.parameters())


def test_normalizer_restoration_preserves_original_pixels_and_does_not_renormalize_states():
    encoder, obs = encoder_and_observation()
    encoder.eval()
    normalizer = LinearNormalizer()
    for port in encoder.rgb_ports:
        normalizer[port] = SingleFieldLinearNormalizer.create_fit(
            torch.tensor([[0., 0., 0.], [255., 255., 255.]]), output_min=0., output_max=1.)
    normalizer['robot_state'] = SingleFieldLinearNormalizer.create_fit(
        torch.tensor([[-20., -20., -20.], [20., 20., 20.]]))
    encoder.set_normalizer(normalizer)
    expected_states = torch.cat((obs['robot_state'], obs['task_uid']), dim=-1)
    torch.testing.assert_close(encoder.state_features(obs), expected_states)
    prepared = encoder.prepare_conditioning(obs)
    torch.testing.assert_close(prepared[:, :, 0],
        (obs['camera_a'][:, :, 8:72, 8:72].float() / 255).permute(0, 1, 4, 2, 3))
    with torch.no_grad():
        visual, proprio = encoder(obs, prepared)
        expected_proprio = encoder.state_projection(expected_states) + encoder.frame_embedding[None]
        torch.testing.assert_close(proprio, expected_proprio, rtol=0, atol=0)
    restored = FlowResNetObservationEncoder(**encoder.export_config()).eval()
    restored.load_state_dict(encoder.state_dict(), strict=True)
    with torch.no_grad():
        actual_visual, actual_proprio = restored(obs)
    torch.testing.assert_close(actual_visual, visual, rtol=0, atol=0)
    torch.testing.assert_close(actual_proprio, proprio, rtol=0, atol=0)


def test_invalid_pixels_states_or_prepared_crops_are_rejected():
    encoder, obs = encoder_and_observation()
    bad = copy.copy(obs)
    bad['camera_a'] = bad['camera_a'].float()
    bad['camera_a'][0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite RGB'):
        encoder.prepare_conditioning(bad)
    bad = copy.copy(obs)
    bad['robot_state'] = torch.full_like(bad['robot_state'], float('nan'))
    with pytest.raises(ValueError, match='Nonfinite current state'):
        encoder(bad)
    prepared = encoder.prepare_conditioning(obs)
    with pytest.raises(ValueError, match='no gradient'):
        encoder(obs, prepared.requires_grad_(True))
    with pytest.raises(ValueError, match='must have shape'):
        encoder(obs, prepared.detach()[:, :, :1])
