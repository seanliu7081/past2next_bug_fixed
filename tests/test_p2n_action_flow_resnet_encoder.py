"""Actual ResNet-18 and detached crop checks for the new direct-action adapter."""
import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from oat.common.action_flow_resnet_batch import PreparedActionFlowResNetBatch
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.perception.action_flow_resnet_obs_encoder import ActionFlowResNetObservationEncoder


META = {
    "obs": {
        "camera_a": {"shape": [80, 80, 3], "type": "rgb"},
        "camera_b": {"shape": [80, 80, 3], "type": "rgb"},
        "robot_state": {"shape": [3], "type": "state"},
        "task_uid": {"shape": [1], "type": "state"},
    },
    "action": {"shape": [7]},
}


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def encoder_and_observation():
    encoder = ActionFlowResNetObservationEncoder(META, n_emb=16, crop_shape=(64, 64))
    obs = {
        "camera_a": torch.randint(0, 256, (1, 2, 80, 80, 3), dtype=torch.uint8),
        "camera_b": torch.randint(0, 256, (1, 2, 80, 80, 3), dtype=torch.uint8),
        "robot_state": torch.randn(1, 2, 3),
        "task_uid": torch.randn(1, 2, 1),
    }
    return encoder, obs


def test_backbones_train_independently_with_groupnorm_and_no_downloads(monkeypatch):
    import torchvision.models
    original = torchvision.models.resnet18
    initialized = []

    def local_resnet(*args, **kwargs):
        assert kwargs.get("pretrained") is False
        initialized.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(torchvision.models, "resnet18", local_resnet)
    encoder, obs = encoder_and_observation()
    assert len(initialized) == 2
    nets = encoder.vision_encoder.encoder.obs_nets
    assert nets["camera_a"] is not nets["camera_b"]
    camera_parameters = [{id(p) for p in nets[port].parameters()} for port in encoder.rgb_ports]
    assert not camera_parameters[0].intersection(camera_parameters[1])
    assert any(isinstance(module, nn.GroupNorm) for module in nets.modules())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in nets.modules())
    assert not hasattr(encoder, "dino_encoder") and not hasattr(encoder, "resampler")
    visual, proprio = encoder(obs)
    assert visual.shape == (1, 4, 16)
    assert proprio.shape == (1, 2, 16)
    assert encoder.num_queries == 1 and encoder.num_visual_tokens == 4
    (visual.square().mean() + proprio.square().mean()).backward()
    for port in encoder.rgb_ports:
        conv = next(module for module in nets[port].modules() if isinstance(module, nn.Conv2d))
        assert conv.weight.requires_grad
        assert conv.weight.grad is not None
        assert torch.isfinite(conv.weight.grad).all()
        assert conv.weight.grad.abs().sum() > 0
    assert encoder.visual_projection.weight.grad.abs().sum() > 0
    assert encoder.state_projection.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad for p in encoder.vision_encoder.normalizer.parameters())


def test_student_and_teacher_reuse_crops_and_encode_with_their_own_backbones(monkeypatch):
    student, obs = encoder_and_observation()
    teacher = copy.deepcopy(student).requires_grad_(False)
    crops = student.prepare_conditioning(obs)
    assert crops.shape == (1, 2, 2, 3, 64, 64)
    assert crops.grad_fn is None and not crops.requires_grad
    assert crops.min() >= -1 and crops.max() <= 1
    seen = {"student": [], "teacher": []}
    hooks = []
    for name, encoder in (("student", student), ("teacher", teacher)):
        monkeypatch.setattr(encoder, "prepare_conditioning",
            lambda obs: pytest.fail("Prepared image crops must not be sampled again."))
        for port in encoder.rgb_ports:
            hooks.append(encoder.vision_encoder.encoder.obs_nets[port].register_forward_pre_hook(
                lambda module, inputs, name=name: seen[name].append(inputs[0].detach().clone())))
    try:
        with torch.no_grad():
            teacher_visual, _ = teacher(obs, prepared_visual=crops)
        student_visual, _ = student(obs, prepared_visual=crops)
        torch.testing.assert_close(student_visual, teacher_visual, rtol=0, atol=0)
        student_visual.square().mean().backward()
    finally:
        for hook in hooks:
            hook.remove()
    assert len(seen["student"]) == len(seen["teacher"]) == 2
    for student_images, teacher_images in zip(seen["student"], seen["teacher"]):
        torch.testing.assert_close(student_images, teacher_images, rtol=0, atol=0)
    for port in student.rgb_ports:
        source = next(student.vision_encoder.encoder.obs_nets[port].parameters())
        target = next(teacher.vision_encoder.encoder.obs_nets[port].parameters())
        assert source.data_ptr() != target.data_ptr()
        assert source.grad is not None and source.grad.abs().sum() > 0
        assert target.grad is None


def test_random_training_crops_and_deterministic_eval_center_crop():
    encoder, obs = encoder_and_observation()
    encoder.train()
    torch.manual_seed(173)
    first = encoder.prepare_conditioning(obs)
    torch.manual_seed(174)
    second = encoder.prepare_conditioning(obs)
    assert not torch.equal(first, second)
    encoder.eval()
    first = encoder.prepare_conditioning(obs)
    second = encoder.prepare_conditioning(obs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    expected = (obs["camera_a"][:, :, 8:72, 8:72].float() * (2 / 255) - 1).permute(0, 1, 4, 2, 3)
    torch.testing.assert_close(first[:, :, 0], expected, rtol=0, atol=0)
    with torch.no_grad():
        first_visual, first_state = encoder(obs)
        second_visual, second_state = encoder(obs)
    torch.testing.assert_close(first_visual, second_visual, rtol=0, atol=0)
    torch.testing.assert_close(first_state, second_state, rtol=0, atol=0)


def test_normalizer_restoration_matches_encoder_device_dtype_and_stays_frozen():
    encoder, obs = encoder_and_observation()
    encoder.double().eval()
    normalizer = LinearNormalizer()
    for port in encoder.rgb_ports:
        normalizer[port] = SingleFieldLinearNormalizer.create_fit(
            torch.tensor([[0., 0., 0.], [255., 255., 255.]]), output_min=0., output_max=1.)
    normalizer["robot_state"] = SingleFieldLinearNormalizer.create_fit(
        torch.tensor([[-20., -20., -20.], [20., 20., 20.]]))
    encoder.set_normalizer(normalizer)
    encoder.requires_grad_(True)
    pixel_params = list(encoder.vision_encoder.normalizer.parameters())
    assert pixel_params and all(not p.requires_grad for p in pixel_params)
    assert all(p.dtype == torch.float64 and p.device == encoder.visual_projection.weight.device for p in pixel_params)
    expected_states = torch.cat((obs["robot_state"], obs["task_uid"]), dim=-1)
    torch.testing.assert_close(encoder.state_features(obs), expected_states)
    crops = encoder.prepare_conditioning(obs)
    assert crops.dtype == torch.float64
    expected = (obs["camera_a"][:, :, 8:72, 8:72].double() / 255).permute(0, 1, 4, 2, 3)
    torch.testing.assert_close(crops[:, :, 0], expected, rtol=1e-6, atol=1e-8)
    # Exercise loading through a containing policy, where overriding just the
    # encoder's load_state_dict would not repair the normalizer dictionary.
    source = nn.ModuleDict({"obs_encoder": encoder.float()})
    target_encoder = ActionFlowResNetObservationEncoder(**encoder.export_config()).double().eval()
    target = nn.ModuleDict({"obs_encoder": target_encoder})
    target.load_state_dict(source.state_dict(), strict=True)
    pixels = target_encoder.vision_encoder.normalizer
    assert all(p.dtype == torch.float64 and p.device == target_encoder.visual_projection.weight.device for p in pixels.parameters())
    assert all(not p.requires_grad for p in pixels.parameters())
    with torch.no_grad():
        source_visual, source_state = source["obs_encoder"](obs)
        restored_visual, restored_state = target_encoder(obs)
    torch.testing.assert_close(restored_visual.float(), source_visual, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(restored_state.float(), source_state, rtol=1e-6, atol=1e-6)


def test_invalid_crops_or_nonfinite_observations_are_rejected():
    encoder, obs = encoder_and_observation()
    bad = dict(obs)
    bad["camera_a"] = bad["camera_a"].float()
    bad["camera_a"][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite RGB"):
        encoder.prepare_conditioning(bad)
    bad = dict(obs)
    bad["robot_state"] = torch.full_like(bad["robot_state"], float("nan"))
    with pytest.raises(ValueError, match="Nonfinite current state"):
        encoder(bad)
    crops = encoder.prepare_conditioning(obs)
    with pytest.raises(ValueError, match="no gradient"):
        encoder(obs, crops.requires_grad_(True))
    with pytest.raises(ValueError, match="must have shape"):
        encoder(obs, crops.detach()[:, :, :1])


def prepared_batch():
    return PreparedActionFlowResNetBatch(
        noisy_actions=torch.zeros(4, 8, 7), time=torch.tensor([0., .1, .2, .3]),
        step_size=torch.tensor([0., 0., 0., .2]), velocity_targets=torch.ones(4, 8, 7),
        fm_indices=torch.tensor([0, 1, 2]), ct_indices=torch.tensor([3]),
        obs={"robot_state": torch.zeros(4, 2, 3)},
        past_actions=torch.zeros(4, 6, 7), past_action_valid=torch.ones(4, 6, dtype=torch.bool),
        prepared_visual=torch.zeros(4, 2, 2, 3, 8, 8))


def test_prepared_batch_requires_detached_crops_not_cached_backbone_features():
    batch = prepared_batch()
    assert batch.validate() is batch
    with pytest.raises(ValueError, match="ResNet crops"):
        replace(batch, prepared_visual=torch.zeros(4, 2, 2, 64)).validate()
    with pytest.raises(ValueError, match="detached"):
        replace(batch, prepared_visual=batch.prepared_visual.requires_grad_(True)).validate()


@pytest.mark.parametrize("field,value,match", [
    ("fm_indices", torch.tensor([0, 1, 3]), "partition"),
    ("step_size", torch.tensor([.1, 0., 0., .2]), "FM rows"),
    ("past_actions", torch.full((4, 6, 7), float("nan")), "finite FP32"),
    ("prepared_visual", torch.zeros(4, 2, 2, 3, 8, 8, dtype=torch.uint8), "floating-point"),
])
def test_prepared_batch_rejects_invalid_training_inputs(field, value, match):
    with pytest.raises(ValueError, match=match):
        replace(prepared_batch(), **{field: value}).validate()


def test_prepared_batch_accepts_masked_nan_history():
    batch = prepared_batch()
    batch.past_actions[:, 0] = float("nan")
    batch.past_action_valid[:, 0] = False
    batch.obs["past_robot_state"] = torch.full((4, 6, 3), float("nan"))
    assert batch.validate() is batch


def test_checkpoint_load_restores_pixel_normalizer_on_destination_device_without_gpu():
    encoder, _ = encoder_and_observation()
    checkpoint = encoder.state_dict()
    encoder.to("meta")
    # Meta exercises a real source/destination device mismatch without using a
    # GPU. Ordinary weights stay on meta; the normalizer rebuilds CPU parameters
    # while loading, which the adapter's post-hook must move back to meta.
    with pytest.warns(UserWarning, match="meta parameter"):
        encoder.load_state_dict(checkpoint, strict=True)
    pixels = list(encoder.vision_encoder.normalizer.parameters())
    assert pixels and all(p.device.type == "meta" for p in pixels)
    assert all(not p.requires_grad for p in pixels)
