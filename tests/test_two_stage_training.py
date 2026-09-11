"""Exercise the tokenizer-to-policy checkpoint handoff with real CPU models."""
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
import pytest
import torch
from torch import nn

from oat.model.common.normalizer import LinearNormalizer
from oat.model.diffusion.ema_model import EMAModel
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy
from oat.tokenizer.oat.tokenizer_so3_aug import OATTokSO3Aug
from oat.workspace.train_oattok import TrainOATTokWorkspace


class TinyObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(7, 8)

    def modalities(self):
        return ["state"]

    def output_feature_dim(self):
        return 8

    def set_normalizer(self, normalizer):
        pass

    def forward(self, obs):
        return self.project(obs["state"])


def clone_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_state_equal(module, expected):
    actual = module.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("use_ema", [False, True])
def test_trained_tokenizer_checkpoint_stays_frozen_during_policy_training(tmp_path, use_ema):
    torch.manual_seed(42)
    config_dir = str(Path(__file__).resolve().parents[1] / "oat/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        tokenizer_cfg = compose(config_name="train_oattok_so3aug", overrides=[
            f"training.use_ema={str(use_ema).lower()}",
            "tokenizer.encoder.emb_dim=16", "tokenizer.encoder.head_dim=8",
            "tokenizer.encoder.depth=1", "tokenizer.encoder.num_registers=2",
            "tokenizer.encoder.pdropout=0.2",
            "tokenizer.decoder.emb_dim=16", "tokenizer.decoder.head_dim=8",
            "tokenizer.decoder.depth=1", "tokenizer.decoder.pdropout=0.2",
            "tokenizer.quantizer.levels=[3,3]", "tokenizer.action_aug.p=1.0",
        ])

    # Stage 1 trains the real tokenizer and writes the same payload as a full run.
    stage1 = TrainOATTokWorkspace(
        tokenizer_cfg, output_dir=str(tmp_path), lazy_instantiation=False)
    assert isinstance(stage1.model, OATTokSO3Aug)
    actions = torch.linspace(-0.8, 0.8, 2 * 16 * 7).reshape(2, 16, 7)
    tokenizer_normalizer = LinearNormalizer()
    tokenizer_normalizer.fit({"action": actions})
    stage1.model.set_normalizer(tokenizer_normalizer)
    initial_tokenizer = clone_state(stage1.model)
    ema = None
    if use_ema:
        stage1.ema_model.set_normalizer(tokenizer_normalizer)
        ema = EMAModel(stage1.ema_model)
    stage1.model.train()
    for _ in range(3):
        stage1.optimizer.zero_grad(set_to_none=True)
        loss = stage1.model({"action": actions})
        assert torch.isfinite(loss)
        loss.backward()
        stage1.optimizer.step()
        if ema is not None:
            ema.step(stage1.model)
        stage1.global_step += 1
    stage1.epoch = 1
    assert any(not torch.equal(value, initial_tokenizer[name])
               for name, value in stage1.model.named_parameters() if value.requires_grad)
    trained_tokenizer = stage1.ema_model if use_ema else stage1.model
    saved_tokenizer = clone_state(trained_tokenizer)
    saved_normalizer = clone_state(trained_tokenizer.normalizer)
    checkpoint = stage1.save_checkpoint(tag="tokenizer", use_thread=False)

    # Stage 2 instantiates a fresh policy, loading only its action tokenizer.
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        policy_cfg = compose(config_name="train_past2next_scratch")
    assert policy_cfg.training.init_checkpoint is None
    assert not policy_cfg.training.resume
    policy_cfg.shape_meta = {
        "action": {"shape": [7]},
        "obs": {"state": {"shape": [7], "type": "state"}},
    }
    policy_cfg.policy.obs_encoder = {"_target_": f"{__name__}.TinyObservationEncoder"}
    policy_cfg.policy.action_tokenizer.checkpoint = checkpoint
    policy_cfg.policy.embed_dim = 16
    policy_cfg.policy.n_layers = 1
    policy_cfg.policy.n_heads = 2
    policy_cfg.policy.self_past_p = 1.0
    policy_cfg.policy.self_past_warmup_steps = 0
    policy_cfg.policy.self_past_ramp_steps = 0
    policy_cfg.policy.self_past_temperature = 0.0
    policy_cfg.policy.self_past_topk = 4
    policy = hydra.utils.instantiate(policy_cfg.policy)
    assert isinstance(policy, Past2NextSelfPastPolicy)
    assert isinstance(policy.action_tokenizer, OATTokSO3Aug)
    assert_state_equal(policy.action_tokenizer, saved_tokenizer)
    assert all(not value.requires_grad for value in policy.action_tokenizer.parameters())
    assert policy.self_past_step == 0

    # Fitting the policy's dataset normalizer must not replace the tokenizer's.
    policy_normalizer = LinearNormalizer()
    policy_normalizer.fit({"action": actions * 3 + 2})
    policy.set_normalizer(policy_normalizer)
    assert_state_equal(policy.action_tokenizer.normalizer, saved_normalizer)
    assert_state_equal(policy.action_normalizer, clone_state(policy_normalizer))
    assert any(not torch.equal(value, policy.action_normalizer.state_dict()[name])
               for name, value in saved_normalizer.items())

    optimizer = policy.get_optimizer(**policy_cfg.optimizer)
    tokenizer_ids = {id(value) for value in policy.action_tokenizer.parameters()}
    optimized_ids = {id(value) for group in optimizer.param_groups for value in group["params"]}
    assert tokenizer_ids.isdisjoint(optimized_ids)
    assert not optimizer.state
    initial_policy = {name: value.detach().clone()
                      for name, value in policy.named_parameters() if value.requires_grad}

    policy.train()
    assert policy.training and policy.model.training and policy.obs_encoder.training
    assert all(not module.training for module in policy.action_tokenizer.modules())
    with policy._rollout_mode():
        assert not policy.model.training and not policy.obs_encoder.training
        assert all(not module.training for module in policy.action_tokenizer.modules())
        with torch.no_grad():
            tokens = policy.action_tokenizer.tokenize(actions)
            decoded = policy.action_tokenizer.detokenize(tokens)
            assert decoded.shape == actions.shape and torch.isfinite(decoded).all()
    assert policy.model.training and policy.obs_encoder.training
    assert all(not module.training for module in policy.action_tokenizer.modules())

    batch = {
        "action": actions,
        "obs": {"state": torch.randn(2, 2, 7)},
        "past_action": torch.randn(2, 7, 7),
        "prev_obs": {"state": torch.randn(2, 2, 7)},
        "prev_past_action": torch.randn(2, 7, 7),
    }
    loss = policy(batch)  # Includes real tokenization and generated-history decoding.
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is None for value in policy.action_tokenizer.parameters())
    optimizer.step()
    policy.on_optimizer_step()
    assert policy.self_past_step == 1
    assert any(not torch.equal(value, initial_policy[name])
               for name, value in policy.named_parameters() if value.requires_grad)
    assert_state_equal(policy.action_tokenizer, saved_tokenizer)
    assert all(not module.training for module in policy.action_tokenizer.modules())
