"""The observation-only OAT baseline must compose, train, and generate actions."""

from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
import pytest
import torch

from oat.common.hydra_util import register_new_resolvers
from oat.common.replay_buffer import ReplayBuffer
from oat.model.common.normalizer import LinearNormalizer
from oat.policy.oatpolicy import OATPolicy
from oat.workspace.train_policy import TrainPolicyWorkspace
from test_action_token_generation import prefer_bos
from test_history_training import TinyObservationEncoder, TinyTokenizer


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "oat/config")
SHAPE_META = {
    "action": {"shape": [7]},
    "obs": {"state": {"shape": [7], "type": "state"}},
}


def make_policy(tokenizer=None):
    return OATPolicy(
        shape_meta=SHAPE_META,
        obs_encoder=TinyObservationEncoder(),
        action_tokenizer=TinyTokenizer() if tokenizer is None else tokenizer,
        n_obs_steps=2, n_action_steps=8,
        embed_dim=16, n_layers=1, n_heads=2, dropout=0,
    )


def test_oatpolicy_launch_config_resolves_all_targets():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train_oatpolicy", overrides=[
            "policy.action_tokenizer.checkpoint=/tmp/tokenizer.ckpt",
            "task.policy.lazy_eval=false", "logging.mode=online",
            "training.resume=false", "logging.resume=false",
        ])
    assert hydra.utils.get_class(cfg.policy._target_) is OATPolicy
    assert hydra.utils.get_class(cfg._target_) is TrainPolicyWorkspace
    hydra.utils.get_class(cfg.task.policy.dataset._target_)
    hydra.utils.get_class(cfg.task.policy.env_runner._target_)
    assert cfg.policy.shape_meta.action.shape == [7]
    assert cfg.task.policy.dataset.zarr_path == "data/libero/libero10_N500.zarr"
    assert cfg.task.policy.env_runner.protocol == "corrected"
    assert not cfg.task.policy.lazy_eval
    assert cfg.logging.mode == "online"
    assert not cfg.training.resume and not cfg.logging.resume


def test_baseline_dataset_needs_only_observations_and_actions(tmp_path):
    replay = ReplayBuffer.create_empty_numpy()
    for episode in range(10):
        actions = np.full((20, 7), episode, dtype=np.float32)
        replay.add_episode({"state": actions.copy(), "action": actions})
    path = tmp_path / "baseline.zarr"
    replay.save_to_path(str(path))
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train_oatpolicy")
    dataset = hydra.utils.instantiate(
        cfg.task.policy.dataset, zarr_path=str(path), obs_keys=["state"],
    )
    assert dataset.train_mask.sum() == 9
    assert dataset.get_validation_dataset().train_mask.sum() == 1
    sample = dataset[0]
    assert set(sample) == {"obs", "action"}
    assert sample["obs"]["state"].shape == (2, 7)
    assert sample["action"].shape == (16, 7)


def test_training_with_real_so3_tokenizer_preserves_frozen_weights_and_normalizer():
    register_new_resolvers()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train_oattok_so3aug", overrides=[
            "tokenizer.encoder.emb_dim=16", "tokenizer.encoder.head_dim=8",
            "tokenizer.encoder.depth=1", "tokenizer.encoder.num_registers=2",
            "tokenizer.encoder.pdropout=0.2",
            "tokenizer.decoder.emb_dim=16", "tokenizer.decoder.head_dim=8",
            "tokenizer.decoder.depth=1", "tokenizer.decoder.pdropout=0.2",
            "tokenizer.quantizer.levels=[3,3]", "tokenizer.action_aug.p=1.0",
            "tokenizer.action_aug.mode=left_noise",
        ])
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    actions = torch.linspace(-0.8, 0.8, 2 * 16 * 7).reshape(2, 16, 7)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": actions})
    tokenizer.set_normalizer(normalizer)
    policy = make_policy(tokenizer)
    before = {key: value.detach().clone() for key, value in tokenizer.state_dict().items()}

    different_normalizer = LinearNormalizer()
    different_normalizer.fit({"action": actions * 3 + 2})
    policy.set_normalizer(different_normalizer)
    optimizer = policy.get_optimizer(
        policy_lr=5e-5, obs_enc_lr=1e-5, weight_decay=0, betas=(0.9, 0.95),
    )
    optimized_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized_ids.isdisjoint(id(p) for p in tokenizer.parameters())
    trainable_before = {key: value.detach().clone()
                        for key, value in policy.named_parameters() if value.requires_grad}

    policy.train()
    assert policy.model.training and policy.obs_encoder.training
    assert all(not module.training for module in tokenizer.modules())
    batch = {"obs": {"state": torch.randn(2, 2, 7)}, "action": actions}
    loss = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in tokenizer.parameters())
    optimizer.step()
    for key, value in tokenizer.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert any(not torch.equal(value, trainable_before[key])
               for key, value in policy.named_parameters() if value.requires_grad)

    policy.eval()
    with torch.inference_mode():
        result = TrainPolicyWorkspace._predict_validation_action(policy, batch)
    assert result["action"].shape == (2, 8, 7)
    assert result["action_pred"].shape == (2, 16, 7)
    assert torch.isfinite(result["action_pred"]).all()


@pytest.mark.parametrize("temperature", [0, 1])
def test_baseline_masks_bos_before_action_sampling(monkeypatch, temperature):
    policy = make_policy().eval()
    prefer_bos(policy.model, policy.bos_id, action_id=1)
    decoded = []

    def detokenize(tokens):
        decoded.append(tokens.detach().clone())
        return torch.zeros(tokens.shape[0], 16, 7)

    monkeypatch.setattr(policy.action_tokenizer, "detokenize", detokenize)
    with torch.inference_mode():
        result = policy.predict_action(
            {"state": torch.zeros(2, 2, 7)}, temperature=temperature, topk=1,
        )
    assert result["action"].shape == (2, 8, 7)
    torch.testing.assert_close(decoded[0], torch.ones(2, 2, dtype=torch.long))
