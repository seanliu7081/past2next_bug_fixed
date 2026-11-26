"""Exercise PRT behavior cloning with a fitted Stage 1 checkpoint on CPU."""
import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from oat.common.hydra_util import register_new_resolvers
from oat.model.common.normalizer import LinearNormalizer
from oat.model.diffusion.ema_model import EMAModel
from oat.tokenizer.prt import PredictiveResidualTokenizer
from oat.workspace.train_policy import TrainPolicyWorkspace
from test_history_training import TinyObservationEncoder


def recipe(name, overrides=()):
    register_new_resolvers()
    with initialize_config_dir(version_base=None,
                               config_dir=str(Path(__file__).resolve().parents[1] / "oat/config")):
        return compose(config_name=name, overrides=list(overrides))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    cfg = recipe("PRT_Tok", [
        "past_n=3", "horizon=4", "tokenizer.execution_steps=2",
        "tokenizer.encoder.num_registers=2", "tokenizer.quantizer.levels=[3,3]",
        "tokenizer.encoder.emb_dim=16", "tokenizer.decoder.emb_dim=16",
        "tokenizer.encoder.head_dim=8", "tokenizer.decoder.head_dim=8",
        "tokenizer.encoder.depth=1", "tokenizer.decoder.depth=1",
    ])
    codec = instantiate(cfg.tokenizer)
    prior = torch.zeros(2, 4, 3)
    prior[:, :, -1] = 1
    codec.set_prior_statistics(prior, torch.ones(4, 2))
    path = tmp_path_factory.mktemp("prt-stage1") / "fitted.ckpt"
    torch.save({"cfg": cfg, "state_dicts": {"ema_model": codec.state_dict()}}, path)
    return path


@pytest.fixture
def policy(checkpoint):
    cfg = recipe("PRT_Policy", [
        f"policy.action_tokenizer.checkpoint={checkpoint}", "past_n=3", "horizon=4",
        "n_action_steps=2", "policy.embed_dim=16", "policy.n_layers=1", "policy.n_heads=2",
        "policy.dropout=0", "policy.self_past_p=1", "policy.self_past_warmup_steps=0",
        "policy.self_past_ramp_steps=0", "policy.topk=2", "policy.self_past_topk=2",
    ])
    shape = {"obs": {"state": {"shape": [7], "type": "state"}}, "action": {"shape": [7]}}
    p = instantiate(cfg.policy, obs_encoder=TinyObservationEncoder(), shape_meta=shape)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.tensor([-1., 1.])[:, None].expand(2, 7)})
    p.set_normalizer(normalizer)
    return p


@pytest.fixture
def batch():
    target = torch.linspace(-0.2, 0.5, 56).reshape(2, 4, 7)
    target[..., 6] = torch.tensor([[-1, -1, 1, 1], [1, 1, -1, -1]])
    past = torch.full((2, 3, 7), 0.2)
    past[..., 6] = -1
    past[0, :2] = 0
    previous = torch.full_like(past, -0.1)
    previous[..., 6] = 1
    return {
        "obs": {"state": torch.randn(2, 2, 7)},
        "prev_obs": {"state": torch.randn(2, 2, 7)},
        "action": target, "past_action": past, "prev_past_action": previous,
        "past_action_valid": torch.tensor([[False, False, True], [True, True, True]]),
        "prev_window_valid": torch.tensor([False, True]),
    }


def test_bc_step_preserves_codec_and_updates_both_policy_heads(policy, batch):
    policy.train()
    codec = policy.action_tokenizer
    assert isinstance(codec, PredictiveResidualTokenizer)
    before = copy.deepcopy(codec.state_dict())
    ema_policy = copy.deepcopy(policy)
    ema = EMAModel(ema_policy)
    optimizer = policy.get_optimizer(1e-3, 1e-3, 1e-4, (0.9, 0.95))
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len({id(p) for p in optimized})
    assert {id(p) for p in optimized} == {id(p) for p in policy.parameters() if p.requires_grad}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in optimized)
    grip_weights = policy.model.grip_head.weight.detach().clone()
    assert policy.model.grip_head.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(policy.model.grip_head.weight, grip_weights)
    policy.on_optimizer_step()
    ema.step(policy)
    for tested in (policy, ema_policy):
        assert not any(m.training for m in tested.action_tokenizer.modules())
        assert not any(p.requires_grad or p.grad is not None for p in tested.action_tokenizer.parameters())
        for key, tensor in tested.action_tokenizer.state_dict().items():
            torch.testing.assert_close(tensor, before[key], atol=0, rtol=0)
    assert policy.self_past_step == 1


def test_explicit_validation_and_acknowledged_execution(policy, batch):
    policy.eval()
    obs, past = batch["obs"], batch["past_action"]
    policy.reset()
    with torch.inference_mode():
        validation = TrainPolicyWorkspace._predict_validation_action(policy, batch)
        assert validation["action"].shape == (2, 2, 7)
        assert validation["action_pred"].shape == (2, 4, 7)
        decoded = policy.action_tokenizer.detokenize(
            validation["arm_tokens"], validation["grip_tokens"], past)
        torch.testing.assert_close(validation["action_pred"], decoded)
        torch.testing.assert_close(validation["action"], decoded[:, :2])
        short = policy.predict_action(obs, past_actions=past, use_k_tokens=1)
        torch.testing.assert_close(short["arm_tokens"], validation["arm_tokens"][:, :1])
        torch.testing.assert_close(short["grip_tokens"], validation["grip_tokens"])
        first = policy.predict_action(obs)
        with pytest.raises(RuntimeError):
            policy.predict_action(obs)
        commands = first["action"].clone()
        commands[:, :, :6] = 0.7
        commands[:, :, 6] = 1
        policy.record_executed_actions(commands, executed_lengths=[0, 1])
        expected = torch.zeros_like(past)
        expected[1, -1] = commands[1, 0]
        explicit = policy.predict_action(obs, past_actions=expected)
        next_action = policy.predict_action(obs)
        torch.testing.assert_close(next_action["action_pred"], explicit["action_pred"])
        policy.reset()
        torch.testing.assert_close(policy.predict_action(obs)["action_pred"], first["action_pred"])


def test_history_dropout_hides_reference_only_from_policy(policy, batch, monkeypatch):
    policy.train()
    policy.history_dropout = 1
    observed = {}
    tokenize = policy.action_tokenizer.tokenize
    teacher_logits = policy.model.teacher_logits

    def labels(action, past):
        observed["past"] = past.clone()
        return tokenize(action, past)

    def predict(cond, arm, bos_id):
        observed["condition"] = cond.detach().clone()
        return teacher_logits(cond, arm, bos_id)

    monkeypatch.setattr(policy.action_tokenizer, "tokenize", labels)
    monkeypatch.setattr(policy.model, "teacher_logits", predict)
    before = batch["past_action"].clone()
    policy(batch, history_mode="expert")
    valid = batch["past_action_valid"]
    torch.testing.assert_close(observed["past"][~valid], before[~valid])
    torch.testing.assert_close(observed["past"][..., 6], before[..., 6])
    assert torch.any(observed["past"][valid][:, :6] != before[valid][:, :6])
    torch.testing.assert_close(batch["past_action"], before)
    features = policy.obs_encoder(batch["obs"])
    torch.testing.assert_close(observed["condition"][:, :2], features)
    assert torch.all(observed["condition"][:, 2:] == 0)
    policy.eval()
    policy(batch)
    torch.testing.assert_close(observed["past"], before)
    assert torch.any(observed["condition"][:, 2:] != 0)


def test_generated_history_relabels_the_current_expert_chunk(policy, batch, monkeypatch):
    policy.eval()
    with torch.inference_mode():
        previous = policy.predict_action(batch["prev_obs"], past_actions=batch["prev_past_action"])
    # The first batch row has no previous decision in this episode.
    expected = batch["past_action"].clone()
    expected[1] = torch.cat((batch["prev_past_action"][1], previous["action"][1]))[-3:]
    observed = []
    tokenize = policy.action_tokenizer.tokenize

    def labels(action, past):
        observed.append(past.clone())
        torch.testing.assert_close(action, batch["action"])
        return tokenize(action, past)

    monkeypatch.setattr(policy.action_tokenizer, "tokenize", labels)
    with torch.no_grad():
        loss = policy(batch, history_mode="generated")
    assert torch.isfinite(loss)
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], expected)


def test_bc_weights_arm_likelihood_and_scores_gripper_separately(policy, batch):
    policy.eval()
    policy.innovation_alpha = 2
    policy.innovation_cap = 0.35
    policy.grip_weight = 0.4
    batch["action"][1, :, :6] = 5
    with torch.no_grad():
        target, past = batch["action"], batch["past_action"]
        codec = policy.action_tokenizer
        arm, grip = codec.tokenize(target, past)
        cond = policy._build_condition(policy.obs_encoder(batch["obs"]), past)
        arm_logits, grip_logits = policy.model.teacher_logits(cond, arm, policy.bos_id)
        # Check the probability of the full expert arm sequence, with no BOS class.
        assert arm_logits.shape == (2, 2, 9) and grip_logits.shape == (2, 9)
        arm_nll = -arm_logits.log_softmax(-1).gather(-1, arm[..., None]).squeeze(-1).sum(1)
        grip_nll = F.cross_entropy(grip_logits, grip, reduction="none")
        # This checkpoint's prior holds the last arm command and its scales are one.
        energy = (target[..., :6] - past[:, -1:, :6]).square().mean((1, 2)).sqrt()
        assert energy[0] < 0.35 < energy[1]
        expected = ((1 + 2 * energy.clamp(max=0.35)) * arm_nll + 0.4 * grip_nll).mean()
        torch.testing.assert_close(policy(batch, history_mode="expert"), expected)


def test_cached_generation_matches_teacher_forced_logits(policy, batch, monkeypatch):
    policy.eval()
    sample = policy.model._sample
    cached_logits = []

    def capture(logits, temperature, top_k):
        cached_logits.append(logits.clone())
        return sample(logits, temperature, top_k)

    monkeypatch.setattr(policy.model, "_sample", capture)
    with torch.no_grad():
        condition = policy._build_condition(policy.obs_encoder(batch["obs"]), batch["past_action"])
        arm, grip = policy.model.sample_tokens(condition, policy.max_seq_len, policy.bos_id, 0, None)
        arm_logits, grip_logits = policy.model.teacher_logits(condition, arm, policy.bos_id)
    torch.testing.assert_close(torch.stack(cached_logits[:-1], dim=1), arm_logits)
    torch.testing.assert_close(cached_logits[-1], grip_logits)
    assert torch.equal(arm, arm_logits.argmax(-1)) and torch.equal(grip, grip_logits.argmax(-1))


def test_policy_rejects_incompatible_stage_one_horizon(policy):
    with pytest.raises(ValueError, match="horizon"):
        type(policy)(action_tokenizer=policy.action_tokenizer, obs_encoder=TinyObservationEncoder(),
                     shape_meta={"obs": {"state": {"shape": [7], "type": "state"}},
                                 "action": {"shape": [7]}},
                     past_n=3, n_obs_steps=2, n_action_steps=2, horizon=5,
                     embed_dim=16, n_layers=1, n_heads=2)
