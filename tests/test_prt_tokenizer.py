from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from oat.common.hydra_util import register_new_resolvers
from oat.dataset.prt_dataset import PRTDataset


@pytest.fixture
def config():
    register_new_resolvers()
    config_dir = Path(__file__).resolve().parents[1] / "oat/config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="PRT_Tok", overrides=[
            "past_n=2", "horizon=4", "tokenizer.execution_steps=2",
            "tokenizer.encoder.num_registers=2", "tokenizer.quantizer.levels=[3,3]",
            "tokenizer.encoder.emb_dim=16", "tokenizer.decoder.emb_dim=16",
            "tokenizer.encoder.head_dim=8", "tokenizer.decoder.head_dim=8",
            "tokenizer.encoder.depth=1", "tokenizer.decoder.depth=1",
        ])


@pytest.fixture
def codec(config):
    model = instantiate(config.tokenizer)
    weights = torch.zeros(2, 4, 2)
    weights[:, :, -1] = 1
    model.set_prior_statistics(weights, torch.ones(4, 2))
    return model


def test_tokens_decode_relative_to_history_and_allow_short_prefixes(codec):
    history = torch.zeros(2, 2, 7)
    target = torch.zeros(2, 4, 7)
    target[..., :6] = torch.arange(4).reshape(1, -1, 1) * 0.1
    target[..., 6] = -1
    # Give the zero-initialized output head one update so suffix masking is observable.
    optimizer = torch.optim.Adam(codec.parameters(), lr=1e-3)
    codec({"action": target, "past_action": history,
           "past_action_valid": torch.zeros(2, 2, dtype=torch.bool)}).backward()
    optimizer.step()
    codec.eval()
    with torch.no_grad():
        arm, grip = codec.tokenize(target, history)
        assert arm.shape == (2, 2) and grip.shape == (2,)
        decoded = codec.detokenize(arm[:, :1], grip, history)
        torch.testing.assert_close(decoded, codec.autoencode(target, history, eval_keep_k=[1, 1]))
        changed = arm.clone()
        changed[:, 1] = (changed[:, 1] + 1) % codec.quantizer.codebook_size
        torch.testing.assert_close(decoded, codec.detokenize(changed, grip, history, [1, 1]))
        assert not torch.equal(codec.detokenize(arm, grip, history),
                               codec.detokenize(changed, grip, history))
        moved = history.clone()
        moved[:, -1, :6] = torch.arange(6)
        delta = codec.detokenize(arm[:, :1], grip, moved) - decoded
    torch.testing.assert_close(delta[..., :6], moved[:, -1:, :6].expand(-1, 4, -1))
    assert torch.all(delta[..., 6] == 0)
    with pytest.raises(ValueError):
        codec.detokenize(arm[:, :1], grip, history, eval_keep_k=[2, 1])
    with pytest.raises(ValueError):
        codec.detokenize(arm + codec.quantizer.codebook_size, grip, history)
    with pytest.raises(ValueError):
        codec.detokenize(arm, torch.full_like(grip, 9), history)


def test_gripper_events_cover_reset_switch_hold_and_multiple_switches(codec):
    codec.eval()
    history = torch.zeros(5, 2, 7)
    history[2, -1, 6] = 1
    target = torch.tensor([
        [-1, -1, -1, -1], [-1, -1, -1, 1], [1, -1, -1, -1],
        [-1, 1, -1, -1], [1, -1, 1, 1],
    ])
    grip = codec.encode_gripper(target, history)
    assert grip.tolist() == [0, 4, 6, 0, 1]
    with torch.no_grad():
        reconstructed = codec.detokenize(torch.zeros(5, 2, dtype=torch.long), grip, history)
    assert (target != reconstructed[..., 6]).sum(1).tolist() == [0, 0, 0, 1, 1]
    with pytest.raises(ValueError):
        codec.encode_gripper(torch.zeros_like(target), history)


def test_training_updates_arm_codec_and_evaluation_uses_clean_history(codec):
    past = torch.zeros(3, 2, 7)
    past[:, 1, 6] = -1
    valid = torch.tensor([[False, True]]).expand(3, -1)
    target = torch.linspace(-0.4, 0.4, 84).reshape(3, 4, 7)
    target[..., 6] = 1
    batch = {"action": target, "past_action": past, "past_action_valid": valid}
    perturbed = codec.perturb_history(past, valid)
    assert torch.equal(perturbed[~valid], past[~valid])
    assert torch.equal(perturbed[..., 6], past[..., 6])
    assert torch.any(perturbed[..., :6][valid] != past[..., :6][valid])
    optimizer = torch.optim.Adam(codec.parameters(), lr=1e-3)
    # The OAT decoder starts at zero. Its first update opens the path to the encoder.
    for _ in range(2):
        optimizer.zero_grad()
        loss = codec(batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    for module in (codec.encoder, codec.decoder):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert all(g.isfinite().all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    codec.eval()
    with torch.no_grad():
        assert torch.equal(codec.perturb_history(past, valid), past)
        torch.testing.assert_close(codec(batch), codec(batch), atol=0, rtol=0)
        error = (codec.autoencode(target, past) - target)[..., :6]
        per_step = error.square().mean(-1) + 0.1 * error.cumsum(1).square().mean(-1)
        expected = (per_step * torch.tensor([2, 2, 1, 1])).sum(-1) / 6
        torch.testing.assert_close(codec(batch, reduction="none"), expected)


def test_checkpoint_keeps_residual_scales_and_prior(tmp_path, config, codec):
    path = tmp_path / "prt.ckpt"
    codec.set_prior_statistics(torch.arange(16).reshape(2, 4, 2) * 0.1,
                               torch.arange(8).reshape(4, 2) + 1.5)
    torch.save({"cfg": config, "state_dicts": {"ema_model": codec.state_dict()}}, path)
    loaded = type(codec).from_checkpoint(path)
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    loaded.train()
    assert not any(module.training for module in loaded.modules())
    past = torch.ones(1, 2, 7)
    arm = torch.ones(1, 4, 6)
    torch.testing.assert_close(loaded.prior(past), codec.prior(past))
    torch.testing.assert_close(loaded.compand(arm), codec.compand(arm))
    torch.testing.assert_close(loaded.expand(loaded.compand(arm)), arm)

    # A common change of frame must commute with both triads of the fitted prior.
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    history = torch.arange(14, dtype=torch.float32).reshape(1, 2, 7)
    rotated = history.clone()
    rotated[..., :6] = (history[..., :6].reshape(1, 2, 2, 3) @ rotation.T).flatten(2)
    expected = (loaded.prior(history).reshape(1, 4, 2, 3) @ rotation.T).flatten(2)
    torch.testing.assert_close(loaded.prior(rotated), expected)


def test_ridge_uses_only_training_episodes_and_resets_history(tmp_path):
    path = tmp_path / "commands.zarr"
    store = zarr.open_group(str(path), mode="w")
    lengths = [6, 7, 8, 9]
    ends = np.cumsum(lengths)
    store.create_group("meta").array("episode_ends", ends)
    rates = np.array([0.8] * 3 + [0.5] * 3)
    episodes = []
    for i, length in enumerate(lengths):
        arm = (i + 1) * rates ** np.arange(length)[:, None]
        episodes.append(np.column_stack((arm, -np.ones(length))).astype(np.float32))
    action = store.create_group("data").array("action", np.concatenate(episodes))
    original = PRTDataset(path, past_n=1, horizon=3, seed=42, val_ratio=0.25,
                          max_train_episodes=2)
    offset = 0
    for i in np.flatnonzero(original.train_mask):
        assert not original[offset]["past_action_valid"].any()
        assert torch.all(original[offset]["past_action"] == 0)
        torch.testing.assert_close(original[offset + 1]["past_action"][0],
                                   torch.from_numpy(episodes[i][0]))
        offset += lengths[i] - 2
    assert len(original) == offset

    weights, scale = original.fit_prior(ridge=0)
    expected = torch.tensor([[0.8, 0.8**2, 0.8**3], [0.5, 0.5**2, 0.5**3]])
    torch.testing.assert_close(weights[:, :, 0], expected)
    assert scale.shape == (3, 2) and torch.all(scale.isfinite() & (scale > 0))
    starts = np.r_[0, ends[:-1]]
    for i in np.flatnonzero(~original.train_mask):
        action[starts[i]:ends[i], :6] = 100
    modified = PRTDataset(path, past_n=1, horizon=3, seed=42, val_ratio=0.25,
                          max_train_episodes=2)
    validation = modified.get_validation_dataset()
    assert torch.all(validation[0]["action"][:, :6] == 100)
    for view in (modified, validation):
        clean_weights, clean_scale = view.fit_prior(ridge=0, batch_size=2)
        torch.testing.assert_close(clean_weights, weights)
        torch.testing.assert_close(clean_scale, scale)
