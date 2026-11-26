"""Sparse PRT supervision on fitted, deterministic CPU codecs."""
import pytest
import torch
from hydra.utils import instantiate

from oat.policy.prt_sparse_policy import sparse_position_loss
from oat.tokenizer.prt import PredictiveResidualTokenizer
from oat.tokenizer.prt_sparse import SparsePRTTokenizer, SparseTokenTargets
from test_history_training import TinyObservationEncoder
from test_prt_policy import batch, checkpoint, policy as base_policy, recipe


@pytest.fixture
def sparse_codec(checkpoint):
    model = SparsePRTTokenizer.from_checkpoint(checkpoint)
    # The fixture's decoder head starts at zero. Give it a nonconstant
    # reconstruction for the physical-cost comparisons.
    with torch.no_grad():
        weights = model.decoder.head.proj.weight
        rng = torch.Generator().manual_seed(107)
        weights.copy_(torch.randn(weights.shape, generator=rng) * 0.08)
    return model


@pytest.fixture
def sparse_policy(checkpoint, base_policy):
    cfg = recipe("PRT_Sparse_Policy", [
        f"policy.action_tokenizer.checkpoint={checkpoint}", "past_n=3", "horizon=4",
        "n_action_steps=2", "policy.embed_dim=16", "policy.n_layers=1", "policy.n_heads=2",
        "policy.dropout=0", "policy.self_past_warmup_steps=0", "policy.self_past_ramp_steps=0",
        "policy.self_past_p=1", "policy.topk=2", "policy.self_past_topk=2",
    ])
    shape = {"obs": {"state": {"shape": [7], "type": "state"}}, "action": {"shape": [7]}}
    policy = instantiate(cfg.policy, obs_encoder=TinyObservationEncoder(), shape_meta=shape)
    policy.set_normalizer(base_policy.action_normalizer)
    policy.load_state_dict(base_policy.state_dict())
    return policy


def test_native_and_original_checkpoints_use_identical_frozen_codec(checkpoint, tmp_path, batch):
    cfg = recipe("PRT_Sparse_Tok", [
        "past_n=3", "horizon=4", "tokenizer.execution_steps=2",
        "tokenizer.encoder.num_registers=2", "tokenizer.quantizer.levels=[3,3]",
        "tokenizer.encoder.emb_dim=16", "tokenizer.decoder.emb_dim=16",
        "tokenizer.encoder.head_dim=8", "tokenizer.decoder.head_dim=8",
        "tokenizer.encoder.depth=1", "tokenizer.decoder.depth=1",
    ])
    assert cfg.task.tokenizer.lazy_eval is False and cfg.logging.mode == "online"
    original = PredictiveResidualTokenizer.from_checkpoint(checkpoint)
    native = instantiate(cfg.tokenizer)
    assert isinstance(native, SparsePRTTokenizer)
    native.load_state_dict(original.state_dict())
    path = tmp_path / "sparse.ckpt"
    torch.save({"cfg": cfg, "state_dicts": {"ema_model": native.state_dict()}}, path)
    past = batch["past_action"]
    arm, grip = original.tokenize(batch["action"], past)
    for source in (checkpoint, path):
        frozen, _ = SparsePRTTokenizer.from_checkpoint(source, return_configuration=True)
        assert isinstance(frozen, SparsePRTTokenizer)
        frozen.train()
        assert not any(m.training for m in frozen.modules())
        assert not any(p.requires_grad for p in frozen.parameters())
        torch.testing.assert_close(frozen.tokenize(batch["action"], past), (arm, grip))
        torch.testing.assert_close(frozen.detokenize(arm, grip, past), original.detokenize(arm, grip, past))


def test_sparse_targets_match_individual_full_decodes_and_deduplicate_boundaries(sparse_codec, batch):
    codec = sparse_codec
    past = batch["past_action"]
    arm = torch.tensor([[0, 5], [7, 8]])
    grip = torch.tensor([4, 8])
    pos = torch.tensor([0, 1])
    proposed = torch.tensor([[1, 8, 0, 1], [5, 0, 8, 5]])
    desired = arm.clone()
    desired[0, 0], desired[1, 1] = 1, 7
    end = codec.execution_steps
    action = codec.detokenize(desired, grip, past).clone()
    # Exact prefix matches remain acceptable even if the rest of the chunk and
    # the gripper disagree with the expert.
    action[:, end:, :6] += 50.
    action[:, :, 6] *= -1
    targets = codec.sparse_targets(action, past, arm, grip, pos, proposed,
                                   decode_batch_size=1, relative_tolerance=0., absolute_tolerance=1e-7)
    expected_sets = [{0, 1, 3, 8}, {0, 5, 7, 8}]
    assert not targets.costs.requires_grad
    assert targets.valid[:, 0].all()
    torch.testing.assert_close(targets.ids[:, 0], arm[torch.arange(2), pos])
    for b in range(2):
        valid = targets.valid[b]
        ids = targets.ids[b, valid]
        assert len(ids) == len(set(ids.tolist())) and set(ids.tolist()) == expected_sets[b]
        values = []
        for code in ids.tolist():
            trial = arm[b:b+1].clone()
            trial[0, pos[b]] = code
            decoded = codec.detokenize(trial, grip[b:b+1], past[b:b+1])
            values.append((decoded[0, :end, :6] - action[b, :end, :6]).square().mean())
        costs = torch.stack(values)
        torch.testing.assert_close(targets.costs[b, valid], costs, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(targets.reference_cost[b], costs[0], atol=1e-6, rtol=1e-4)
        acceptable = targets.acceptable[b, valid]
        assert desired[b, pos[b]] in ids[acceptable] and not acceptable.all()
        torch.testing.assert_close(acceptable, costs <= costs.min() + 1e-7)
        torch.testing.assert_close(targets.hard_negative[b, valid], torch.isin(ids, proposed[b]) & ~acceptable)
    assert not (targets.acceptable & ~targets.valid).any()
    assert not (targets.hard_negative & ~targets.valid).any()
    grouped = codec.sparse_targets(action, past, arm, grip, pos, proposed,
                                   decode_batch_size=32, relative_tolerance=0., absolute_tolerance=1e-7)
    torch.testing.assert_close(grouped.ids, targets.ids)
    torch.testing.assert_close(grouped.costs, targets.costs)


def test_action_only_candidates_and_execution_length(sparse_codec, batch):
    codec = sparse_codec
    past = batch["past_action"]
    arm = torch.tensor([[4, 2], [1, 4]])  # Center of the 3x3 FSQ grid at each chosen position.
    pos = torch.tensor([0, 1])
    grip = torch.tensor([0, 6])
    targets = codec.sparse_targets(batch["action"], past, arm, grip, pos, None, execution_steps=1)
    for b in range(2):
        assert set(targets.ids[b, targets.valid[b]].tolist()) == {1, 3, 4, 5, 7}
    assert not targets.hard_negative.any()
    decoded = codec.detokenize(arm, grip, past)
    expected = (decoded[:, 0, :6] - batch["action"][:, 0, :6]).square().mean(dim=1)
    torch.testing.assert_close(targets.reference_cost, expected)


def test_sparse_objective_uses_full_vocab_and_penalizes_expensive_confounder():
    logits = torch.zeros(1, 5, requires_grad=True)
    targets = SparseTokenTargets(
        ids=torch.tensor([[1, 2, 3, 3]]), costs=torch.tensor([[0., 0., .8, 0.]]),
        valid=torch.tensor([[True, True, True, False]]),
        acceptable=torch.tensor([[True, True, False, False]]),
        hard_negative=torch.tensor([[False, False, True, False]]),
        reference_cost=torch.zeros(1), threshold=torch.zeros(1),
    )
    base = sparse_position_loss(logits, targets, cost_scale=1., risk_weight=0.)
    torch.testing.assert_close(base, -torch.log(torch.tensor([2/5])))
    grad = torch.autograd.grad(base.sum(), logits)[0]
    assert grad[0, 1] < 0 and grad[0, 2] < 0
    assert grad[0, 0] > 0 and grad[0, 4] > 0
    penalized = sparse_position_loss(logits, targets, cost_scale=1., risk_weight=1.)
    additional = torch.autograd.grad(penalized.sum(), logits)[0] - grad
    assert penalized > base and additional[0, 3] > 0
    assert additional[0, 1] < 0 and additional[0, 2] < 0


def test_policy_uses_actual_retokenized_history_and_detached_topk(sparse_policy, batch, monkeypatch):
    policy = sparse_policy.train()
    codec = policy.action_tokenizer
    observed = {}
    original_tokenize, original_score = codec.tokenize, codec.sparse_targets
    teacher_logits = policy.model.teacher_logits

    def tokenize(action, past):
        observed["past"] = past.detach().clone()
        return original_tokenize(action, past)

    def logits(*args, **kwargs):
        result = teacher_logits(*args, **kwargs)
        observed["logits"] = result[0].detach()
        observed["grip_logits"] = result[1].detach()
        return result

    def score(actions, past, arm, grip, positions, policy_ids, **kwargs):
        torch.testing.assert_close(past, observed["past"])
        assert not policy_ids.requires_grad
        selected = observed["logits"][torch.arange(len(positions)), positions]
        torch.testing.assert_close(selected.topk(policy_ids.shape[1]).indices, policy_ids)
        output = original_score(actions, past, arm, grip, positions, policy_ids, **kwargs)
        observed.update(targets=output, arm=arm, grip=grip, positions=positions)
        return output

    monkeypatch.setattr(codec, "tokenize", tokenize)
    monkeypatch.setattr(policy.model, "teacher_logits", logits)
    monkeypatch.setattr(codec, "sparse_targets", score)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = policy(batch, history_mode="generated")
    # Only the sampled arm position receives a set target. All other arm
    # positions and the grip event retain their respective full-vocabulary CE.
    targets, positions = observed["targets"], observed["positions"]
    rows = torch.arange(len(positions))
    log_prob = observed["logits"].float().log_softmax(dim=-1)
    arm_nll = -log_prob.gather(-1, observed["arm"].unsqueeze(-1)).squeeze(-1)
    chosen = log_prob[rows, positions].gather(1, targets.ids)
    prob = chosen.masked_fill(~targets.valid, -torch.inf).softmax(dim=1)
    excess = (targets.costs - targets.threshold[:, None]).clamp_min(0)
    scale = codec.residual_scale[:policy.n_action_steps].square().mean()
    arm_nll[rows, positions] = (-chosen.masked_fill(~targets.acceptable, -torch.inf).logsumexp(dim=1)
                                + policy.sparse_risk_weight * (prob * excess).sum(dim=1) / scale)
    grip_nll = -observed["grip_logits"].float().log_softmax(dim=-1)[rows, observed["grip"]]
    weights = policy._innovation_weights(batch["action"], observed["past"])
    torch.testing.assert_close(loss, (weights * arm_nll.sum(dim=1) + policy.grip_weight * grip_nll).mean())
    loss.backward()
    assert targets.acceptable.any(dim=1).all() and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in policy.parameters() if p.requires_grad)
    assert all(p.grad is None and not p.requires_grad for p in codec.parameters())


def test_sparse_off_matches_parent_loss_and_inference(sparse_policy, base_policy, batch, monkeypatch):
    policy, baseline = sparse_policy, base_policy
    policy.sparse_loss_weight = 0.

    def fail(*args, **kwargs):
        pytest.fail("No candidate evaluation is needed for ordinary CE or deployment")

    monkeypatch.setattr(policy.action_tokenizer, "sparse_targets", fail)
    with torch.no_grad():
        for model in (policy, baseline):
            model.train()
            model.history_dropout = model.history_noise_std = 0.
        torch.testing.assert_close(policy(batch, history_mode="expert"), baseline(batch, history_mode="expert"))
        policy.sparse_loss_weight = 1.
        policy.eval()
        baseline.eval()
        for mode in ("expert", "generated"):
            torch.testing.assert_close(policy(batch, history_mode=mode), baseline(batch, history_mode=mode))
        for count in (1, 2):
            kwargs = {"past_actions": batch["past_action"], "use_k_tokens": count}
            current = policy.predict_action(batch["obs"], **kwargs)
            expected = baseline.predict_action(batch["obs"], **kwargs)
            for key in ("action", "action_pred", "arm_tokens", "grip_tokens"):
                torch.testing.assert_close(current[key], expected[key])
