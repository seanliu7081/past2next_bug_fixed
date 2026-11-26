import os

import numpy as np
import pytest
import torch

from oat.tokenizer.influence_quantization import (
    assign_nearby_codes, prt_last_token_costs, requantize_prt_last_token, shortlist_risk_gradient,
)


def test_label_dependent_term_matches_an_actual_sgd_step():
    gen = torch.Generator().manual_seed(621)
    weight = torch.randn(6, 3, dtype=torch.float64, generator=gen, requires_grad=True)
    design = torch.randn(5, 3, dtype=torch.float64, generator=gen)
    train_feature = torch.tensor([0.8, -0.4, 1.2], dtype=torch.float64)
    ids = (design @ weight.T).detach().topk(4, dim=1).indices
    costs = torch.rand(5, 4, dtype=torch.float64, generator=gen)

    def risk(w):
        logits = (design @ w.T).gather(1, ids)
        return (logits.softmax(dim=1) * costs).sum(dim=1).mean()

    gradient = shortlist_risk_gradient(design, ids, (design @ weight.T).gather(1, ids),
                                       costs, vocab_size=6)
    torch.testing.assert_close(gradient, torch.autograd.grad(risk(weight), weight)[0])
    step = 1e-5
    updated_risks = []
    for i in range(6):
        loss = -(weight @ train_feature).log_softmax(dim=0)[i]
        update = torch.autograd.grad(loss, weight)[0]
        updated_risks.append(risk(weight.detach() - step * update))
    observed = torch.stack(updated_risks) / step
    predicted = gradient @ train_feature
    torch.testing.assert_close(observed - observed[0], predicted - predicted[0],
                               atol=1e-6, rtol=1e-5)


def test_budget_excludes_far_code_even_if_teacher_favors_it():
    ids = torch.tensor([[0, 1, 2], [1, 2, 0]])
    costs = torch.tensor([[1., 1.1, 1.6], [0., 0.01, 0.2]])
    features = torch.ones(2, 1)
    gradient = torch.tensor([[1.], [-1.], [-100.]])
    torch.testing.assert_close(assign_nearby_codes(costs, ids, features, gradient, 0., 0.25),
                               torch.tensor([0, 1]))
    torch.testing.assert_close(assign_nearby_codes(costs, ids, features, gradient, 1., 0.25),
                               torch.tensor([1, 1]))


def test_real_prt_checkpoint_interface():
    """Opt-in CPU integration test on eight LIBERO episodes; no policy training."""
    path, data = os.getenv("PRT_INFLUENCE_POLICY"), os.getenv("PRT_INFLUENCE_DATA")
    if not path or not data:
        pytest.skip("Set PRT_INFLUENCE_POLICY and PRT_INFLUENCE_DATA to run on a real checkpoint")
    import zarr
    from oat.policy.base_policy import BasePolicy

    policy = BasePolicy.from_checkpoint(path)
    codec = policy.action_tokenizer
    root = zarr.open_group(data, mode="r")
    ends = root["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    episode_ids = np.linspace(0, len(starts) - 1, 8, dtype=int)
    times = [int((starts[e] + ends[e]) // 2) for e in episode_ids]

    def stack(key, left, right):
        array = root["data"][key]
        return torch.from_numpy(np.stack([array[t + left:t + right] for t in times]))

    actions = stack("action", 0, codec.horizon).float()
    past = stack("action", -codec.past_n, 0).float()
    obs = {key: stack(key, 1 - policy.n_obs_steps, 1) for key in policy.obs_ports}

    with torch.inference_mode():
        cond = policy._build_condition(policy.obs_encoder(obs), past)
        sampled_arm, sampled_grip = policy.model.sample_tokens(cond, codec.latent_horizon,
                                                               policy.bos_id, 0., None)
        ref_arm, ref_grip = codec.tokenize(actions, past)

        def feature_and_proposals(prefix, condition):
            bos = prefix.new_full((len(prefix), 1), policy.bos_id)
            hidden = policy.model.backbone(torch.cat((bos, prefix), dim=1), condition)[:, -1]
            values, ids = policy.model._arm_logits(hidden).topk(8, dim=1)
            return hidden, values, ids

        # First four episodes estimate the teacher's risk on generated prefixes.
        first = slice(0, 4)
        prefix = sampled_arm[first, :-1]
        f, logits, ids = feature_and_proposals(prefix, cond[first])
        d = prt_last_token_costs(codec, actions[first], past[first], prefix, sampled_grip[first], ids)
        gradient = shortlist_risk_gradient(f, ids, logits, d, policy.bos_id)

        # Quantize different episodes. Both baseline and candidate use the same history.
        second = slice(4, 8)
        arm, grip = ref_arm[second], ref_grip[second]
        f, _, alternatives = feature_and_proposals(arm[:, :-1], cond[second])
        updated, event = requantize_prt_last_token(
            codec, actions[second], past[second], arm, grip, alternatives, f, gradient,
            weight=1., relative_budget=0.25,
        )
        torch.testing.assert_close(updated[:, :-1], arm[:, :-1])
        torch.testing.assert_close(event, grip)
        pair = torch.stack((arm[:, -1], updated[:, -1]), dim=1)
        costs = prt_last_token_costs(codec, actions[second], past[second], arm[:, :-1], grip, pair)
        assert torch.isfinite(costs).all() and (costs[:, 1] <= 1.25 * costs[:, 0] + 1e-7).all()
