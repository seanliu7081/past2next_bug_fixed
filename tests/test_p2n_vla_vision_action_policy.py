"""Vision/action training gradients, ownership, modes and artifact compatibility."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from oat.model.vla.siglip import SiglipEncoder
from oat.model.vla.specs import MODEL_SIZES
from oat.model.vla.trainable_siglip import TrainableSiglipEncoder
from oat.policy.p2n_vla import P2NVLAPolicy
from oat.policy.p2n_vla_vision_action import (
    P2NVLAVisionActionPolicy, P2NVLAStateGateVisionActionPolicy,
)
from oat.workspace.train_p2n_vla import validate_parameter_partition
from p2n_vla_vision_action_fixtures import make_test_tokenizer_checkpoint
import test_p2n_vla_policy as policy_fixtures
from test_p2n_vla_policy import SHAPE_META, build, make_batch


POLICIES = (P2NVLAVisionActionPolicy, P2NVLAStateGateVisionActionPolicy)


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    return make_test_tokenizer_checkpoint(tmp_path_factory.mktemp("vision_action_oat") / "tokenizer.ckpt")


@pytest.fixture(autouse=True)
def local_policy_tokenizer(monkeypatch, tiny_tokenizer):
    monkeypatch.setattr(policy_fixtures, "TOKENIZER", tiny_tokenizer)


def test_encoder_reuses_modules_and_keys():
    spec, vlm_spec, _ = MODEL_SIZES["tiny"]
    frozen = SiglipEncoder(spec, vlm_spec.width, frozen_dtype=torch.bfloat16)
    expected_keys = set(frozen.state_dict())
    parameters = list(frozen.parameters())
    encoder = TrainableSiglipEncoder(frozen)
    assert encoder.vision is frozen.vision and encoder.projector is frozen.projector
    assert set(encoder.state_dict()) == expected_keys
    assert all(a is b for a, b in zip(parameters, encoder.parameters()))
    assert all(p.requires_grad and p.dtype == torch.float32 for p in encoder.parameters())
    assert encoder.vision.is_gradient_checkpointing
    with pytest.raises(ValueError, match="pixels must be"):
        encoder(torch.zeros(1, 3, 1, 1))


@pytest.mark.parametrize("cls", POLICIES)
@pytest.mark.parametrize("checkpointing", (False, True))
def test_vision_and_expert_receive_gradients_language_stays_frozen(cls, checkpointing):
    policy = build(cls, vision_activation_checkpointing=checkpointing).train()
    batch = make_batch(batch=1, gate=cls is P2NVLAStateGateVisionActionPolicy)
    loss = policy(batch, history_mode="expert")
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    for module in (policy.siglip.vision, policy.siglip.projector, policy.joint.expert):
        assert any(p.grad is not None and p.grad.count_nonzero() for p in module.parameters())
    assert all(not p.requires_grad and p.grad is None for p in policy.joint.vlm.parameters())
    assert all(not p.requires_grad and p.grad is None for p in policy.action_tokenizer.parameters())
    optimizer = policy.get_optimizer()
    named, clip = validate_parameter_partition(policy, optimizer)
    assert {group["name"] for group in optimizer.param_groups} == {"pretrained", "new"}
    pretrained = next(group for group in optimizer.param_groups if group["name"] == "pretrained")
    vision_ids = {id(p) for p in policy.siglip.parameters()}
    assert vision_ids <= {id(p) for p in pretrained["params"]}
    assert vision_ids <= {id(p) for p in clip["ki"]}
    assert not any(name.startswith("joint.vlm.") for name, _ in named)
    before = policy.siglip.projector.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(policy.siglip.projector.weight, before)


def test_vision_gradient_comes_from_ki_not_the_detached_expert():
    policy = build(P2NVLAVisionActionPolicy).train()
    batch = make_batch(batch=1)
    targets = policy.encode_targets(batch["action"])
    prefix = policy.build_prefix(batch["obs"], train_aug=False)
    prefix_out, layout, ki_logits = policy.vlm_pass(prefix, targets)
    cond = policy.build_conditions(batch["obs"], batch["past_action"], batch["past_action_valid"])
    logits, _ = policy.teacher_forcing_logits(prefix_out, layout, cond, targets, None, False)
    loss_ar = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss_ki = F.cross_entropy(ki_logits.reshape(-1, policy.N_CODES), targets.reshape(-1))
    vision = list(policy.siglip.parameters())
    grads_ar = torch.autograd.grad(loss_ar, vision, allow_unused=True, retain_graph=True)
    assert all(grad is None or not grad.count_nonzero() for grad in grads_ar)
    grads_ki = torch.autograd.grad(loss_ki, vision)
    assert any(grad.count_nonzero() for grad in grads_ki)


@pytest.mark.parametrize("cls", POLICIES)
def test_rollout_restores_vision_training_and_keeps_language_and_oat_eval(cls):
    policy = build(cls).train()
    assert policy.siglip.training and policy.siglip.vision.training
    assert not policy.joint.vlm.training and not policy.latent_adapter.training
    with pytest.raises(RuntimeError, match="intentional"):
        with policy._rollout_mode():
            assert not any(module.training for module in policy.modules())
            raise RuntimeError("intentional")
    assert policy.siglip.training and policy.siglip.vision.training
    assert not policy.joint.vlm.training and not policy.latent_adapter.training
    policy.eval()
    with policy._rollout_mode():
        assert not any(module.training for module in policy.modules())
    assert not any(module.training for module in policy.modules())


@pytest.mark.parametrize("cls", POLICIES)
def test_artifact_round_trip_includes_trained_vision_and_rejects_old_scope(cls, tmp_path):
    policy = build(cls, vision_activation_checkpointing=False).eval()
    with torch.no_grad():
        policy.siglip.projector.weight.add_(0.07)
    batch = make_batch(batch=1, gate=cls is P2NVLAStateGateVisionActionPolicy)
    state = policy.artifact_state_dict()
    assert {"siglip." + name for name, _ in policy.siglip.named_parameters()} <= set(state)
    assert not any(name.startswith("siglip.") for name in policy.frozen_base_keys())
    assert not (set(state) & set(policy.frozen_base_keys()))
    assert int(state["_artifact_schema"]) == 2
    config, metadata = policy.export_config(), policy.artifact_metadata()
    assert config["training_scope"] == "vision_action"
    assert config["vision_activation_checkpointing"] is False
    assert metadata["training_scope"] == "vision_action" and not metadata["language_trainable"]
    assert metadata["vision_gradient_source"] == "ki_loss"
    assert str(Path("oat/policy/p2n_vla_vision_action.py")) in metadata["source_sha256"]
    with torch.no_grad():
        expected = policy(batch, history_mode="expert")
    path = tmp_path / "vision_action.ckpt"
    torch.save({"policy_config": config, "metadata": metadata, "state_dicts": {"model": state}}, path)
    restored = cls.from_checkpoint(path, weights="model")
    assert torch.equal(restored.siglip.projector.weight, policy.siglip.projector.weight)
    assert all(not p.requires_grad for p in restored.joint.vlm.parameters())
    assert all(p.requires_grad for p in restored.siglip.parameters())
    with torch.no_grad():
        assert torch.allclose(restored(batch, history_mode="expert"), expected, atol=1e-6)
    without_vision = {key: value for key, value in state.items() if not key.startswith("siglip.")}
    with pytest.raises(KeyError, match="Artifact key mismatch"):
        restored.load_artifact_state(without_vision)
    wrong_schema = dict(state, _artifact_schema=torch.tensor(1))
    with pytest.raises(ValueError, match="_artifact_schema"):
        restored.load_artifact_state(wrong_schema)
    with pytest.raises(ValueError, match="artifact schema"):
        restored.load_state_dict(dict(policy.state_dict(), _artifact_schema=torch.tensor(1)))


@pytest.mark.parametrize("lambda_ki", (0.0, -1.0, float("nan")))
def test_rejects_disabled_vision_learning_signal(lambda_ki):
    with pytest.raises(ValueError, match="lambda_ki > 0"):
        P2NVLAVisionActionPolicy(shape_meta=SHAPE_META, lambda_ki=lambda_ki)


def test_original_policy_vision_and_lora_behavior_unchanged():
    original = build(P2NVLAPolicy).train()
    assert not original.siglip.training
    assert all(not p.requires_grad for p in original.siglip.parameters())
    assert any(p.requires_grad for p in original.joint.vlm.parameters())
    assert not any(key.startswith("siglip.") for key in original.artifact_state_dict())
