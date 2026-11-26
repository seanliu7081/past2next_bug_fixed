"""Experimental quantization with a bounded reconstruction cost and a BC teacher.

The teacher estimates how a hypothetical update to its *linear readout* would
change expected decoded action error on independent demonstrations. The actual
PRT decoder and its on-robot sampling path require no additional inputs.
"""

import math

import torch


@torch.no_grad()
def shortlist_risk_gradient(features, codes, logits, costs, vocab_size):
    """Readout gradient of mean expected cost under a fixed shortlist softmax.

    Supply [N,D] causal teacher features and [N,C] distinct candidate IDs,
    logits, and decoded costs from the design split. Candidates should come
    from the teacher without looking at the target action. Costs may use the
    target. Treat the support and the feature extractor as fixed.
    """
    if features.ndim != 2 or codes.ndim != 2 or codes.size(0) != features.size(0):
        raise ValueError("Expected one feature vector and shortlist per example")
    if costs.shape != codes.shape or logits.shape != codes.shape or not codes.numel():
        raise ValueError("Expected nonempty and equally shaped codes, logits, and costs")
    if codes.dtype not in (torch.int32, torch.int64):
        raise ValueError("Shortlists must contain integer code IDs")
    ids = codes.long()
    if (ids < 0).any() or (ids >= vocab_size).any():
        raise ValueError("Shortlist code is outside the readout vocabulary")
    sorted_ids = ids.sort(dim=1).values
    if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
        raise ValueError("Each design shortlist must have distinct codes")
    probability = logits.softmax(dim=-1)
    expected_cost = (probability * costs).sum(dim=-1, keepdim=True)
    logit_gradient = probability * (costs - expected_cost)
    rows = logit_gradient.unsqueeze(-1) * features.unsqueeze(1)
    gradient = rows.new_zeros(vocab_size, features.size(-1))
    gradient.index_add_(0, ids.flatten(), rows.flatten(0, 1))
    return gradient / features.size(0)


@torch.no_grad()
def assign_nearby_codes(costs, codes, features, risk_gradient, weight, relative_budget):
    """Choose one code per example. The nearest candidate is always feasible.

    For candidate i an infinitesimal SGD step on cross-entropy shifts expected
    design risk by a common term plus eta * (risk_gradient @ feature)[i].
    The nonnegative weight absorbs eta and the trade-off with reconstruction.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    if not math.isfinite(relative_budget) or relative_budget < 0:
        raise ValueError("relative_budget must be finite and nonnegative")
    if costs.ndim != 2 or codes.shape != costs.shape or not costs.numel():
        raise ValueError("Expected nonempty aligned [batch,candidates] costs and codes")
    if (features.ndim != 2 or risk_gradient.ndim != 2
            or features.shape != (codes.size(0), risk_gradient.size(1))):
        raise ValueError("Expected aligned [batch,features] and [vocabulary,features] matrices")
    if codes.dtype not in (torch.int32, torch.int64):
        raise ValueError("Candidates must contain integer code IDs")
    if not torch.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Reconstruction costs must be finite and nonnegative")
    ids = codes.long()
    # Zero gradient can occur for codes absent from all design shortlists.
    influence = (features @ risk_gradient.T).gather(1, ids)
    eligible = costs <= costs.amin(dim=1, keepdim=True) * (1 + relative_budget)
    score = (costs + weight * influence).masked_fill(~eligible, torch.inf)
    index = score.argmin(dim=1, keepdim=True)
    return ids.gather(1, index).squeeze(1)


@torch.no_grad()
def prt_last_token_costs(codec, actions, past, prefix, grip, last_ids):
    """Decode each completion of a common arm prefix using the frozen PRT.

    A row is one decision. Keep its command history, previous arm IDs and
    gripper event fixed. Cost is MSE in raw 6D arm command units across only
    the steps that would be sent to the controller.
    """
    if codec.training:
        raise ValueError("Use a frozen codec in eval mode to compare labels")
    if last_ids.ndim != 2 or not last_ids.numel():
        raise ValueError("Expected nonempty last_ids [batch,candidates]")
    batch_size, count = last_ids.shape
    if prefix.shape != (batch_size, codec.latent_horizon - 1):
        raise ValueError("Supply the preceding arm token IDs for each decision")
    if (actions.shape != (batch_size, codec.horizon, 7)
            or past.shape != (batch_size, codec.past_n, 7) or grip.shape != (batch_size,)):
        raise ValueError("Expected matching PRT actions, historical commands, and grip events")
    previous = prefix[:, None, :].expand(-1, count, -1)
    tokens = torch.cat((previous, last_ids.unsqueeze(-1)), dim=-1)
    decoded = codec.detokenize(tokens.flatten(0, 1), grip.repeat_interleave(count),
                               past.repeat_interleave(count, dim=0))
    end = codec.execution_steps
    arm = decoded[:, :end, :6].reshape(batch_size, count, end, 6)
    return (arm - actions[:, None, :end, :6]).square().mean(dim=(-1, -2))


@torch.no_grad()
def requantize_prt_last_token(codec, actions, past, arm, grip, alternatives, features,
                              risk_gradient, *, weight, relative_budget):
    """Return a PRT label pair after refitting the last arm-token decision.

    Start with arm, grip = codec.tokenize(actions, past). Extract the teacher
    feature for the decision following arm[:, :-1], and pass a shortlist of
    alternatives. Including the original ID ensures the budget is also a
    bound relative to the original reconstruction error.
    """
    if arm.ndim != 2 or arm.size(1) != codec.latent_horizon:
        raise ValueError("Expected the complete reference sequence of arm IDs")
    candidates = torch.cat((arm[:, -1:], alternatives), dim=1)
    costs = prt_last_token_costs(codec, actions, past, arm[:, :-1], grip, candidates)
    last = assign_nearby_codes(costs, candidates, features, risk_gradient, weight, relative_budget)
    return torch.cat((arm[:, :-1], last[:, None]), dim=1), grip
