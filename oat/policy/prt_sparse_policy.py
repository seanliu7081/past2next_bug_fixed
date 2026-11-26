"""Train a PRT action policy on local sets of executable arm codes."""

import math

import torch
import torch.nn.functional as F

from oat.policy.prt_policy import PRTPolicy
from oat.tokenizer.prt_sparse import SparsePRTTokenizer


def sparse_position_loss(logits, targets, cost_scale, risk_weight):
    """Set NLL over the whole vocabulary plus expected excess error within the shortlist."""
    # Score the rare good tokens in FP32 even when the vision/AR backbone uses BF16.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    candidates = logits.gather(1, targets.ids.long())
    good = candidates.masked_fill(~(targets.valid & targets.acceptable), -torch.inf)
    set_nll = logits.logsumexp(dim=1) - good.logsumexp(dim=1)
    if risk_weight == 0:
        return set_nll

    conditional = candidates.masked_fill(~targets.valid, -torch.inf).softmax(dim=1)
    excess = (targets.costs - targets.threshold[:, None]).clamp_min(0)
    excess = excess.masked_fill(~targets.valid, 0)
    risk = (conditional * excess).sum(dim=1) / cost_scale
    return set_nll + risk_weight * risk


class SparsePRTPolicy(PRTPolicy):
    """Replace one randomly selected arm target per sample with a sparse set loss.

    The ordinary PRT loss remains the validation metric, making runs comparable
    across candidate budgets. Rollout and execution feedback use the base policy.
    """

    def __init__(
        self, *, action_tokenizer, sparse_topk=4, sparse_relative_tolerance=0.25,
        sparse_absolute_tolerance=1e-7, sparse_decode_batch_size=128,
        sparse_loss_weight=1.0, sparse_risk_weight=1.0, **kwargs,
    ):
        if not isinstance(action_tokenizer, SparsePRTTokenizer):
            raise TypeError("SparsePRTPolicy requires a SparsePRTTokenizer")
        if not isinstance(sparse_topk, int) or sparse_topk < 0:
            raise ValueError("sparse_topk must be a nonnegative integer")
        if not isinstance(sparse_decode_batch_size, int) or sparse_decode_batch_size < 1:
            raise ValueError("sparse_decode_batch_size must be a positive integer")
        if not 0 <= sparse_loss_weight <= 1:
            raise ValueError("sparse_loss_weight must be in [0, 1]")
        for name, value in (("sparse_risk_weight", sparse_risk_weight),
                            ("sparse_relative_tolerance", sparse_relative_tolerance),
                            ("sparse_absolute_tolerance", sparse_absolute_tolerance)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        super().__init__(action_tokenizer=action_tokenizer, **kwargs)
        self.sparse_topk = min(sparse_topk, self.bos_id)
        self.sparse_relative_tolerance = sparse_relative_tolerance
        self.sparse_absolute_tolerance = sparse_absolute_tolerance
        self.sparse_decode_batch_size = sparse_decode_batch_size
        self.sparse_loss_weight = sparse_loss_weight
        self.sparse_risk_weight = sparse_risk_weight

    def get_policy_name(self):
        return "prt_sparse_policy"

    def forward(self, batch, history_mode=None):
        if not self.training or self.sparse_loss_weight == 0:
            return super().forward(batch, history_mode)

        action = batch["action"]
        codec = self.action_tokenizer
        with torch.no_grad():
            past = self._selected_history(batch, history_mode)
            past = self._perturb_history(past, batch.get("past_action_valid"))
            arm, grip = codec.tokenize(action, past)
            weights = self._innovation_weights(action, past)

        cond = self._policy_condition(self.obs_encoder(batch["obs"]), past, apply_dropout=True)
        arm_logits, grip_logits = self.model.teacher_logits(cond, arm, self.bos_id)
        arm_ce = F.cross_entropy(arm_logits.flatten(0, 1), arm.flatten(),
                                 reduction="none").reshape_as(arm)
        grip_ce = F.cross_entropy(grip_logits, grip, reduction="none")
        rows = torch.arange(len(arm), device=arm.device)
        positions = torch.randint(arm.size(1), (len(arm),), device=arm.device)
        # Teacher-forced logits at this position see only preceding arm IDs.
        selected_logits = arm_logits[rows, positions]
        with torch.no_grad():
            policy_ids = selected_logits.detach().topk(self.sparse_topk, dim=1).indices
            targets = codec.sparse_targets(
                action, past, arm, grip, positions, policy_ids,
                relative_tolerance=self.sparse_relative_tolerance,
                absolute_tolerance=self.sparse_absolute_tolerance,
                decode_batch_size=self.sparse_decode_batch_size,
                execution_steps=self.n_action_steps,
            )
            # Express the risk in units of the fitted Stage 1 arm residual variance.
            cost_scale = codec.residual_scale[:self.n_action_steps].float().square().mean().clamp_min(1e-12)
        selected_loss = sparse_position_loss(selected_logits, targets, cost_scale, self.sparse_risk_weight)
        arm_loss = arm_ce.sum(dim=1) + self.sparse_loss_weight * (selected_loss - arm_ce[rows, positions])
        if self.self_past_schedule == "legacy":
            self._train_step += 1
        return (weights * arm_loss + self.grip_weight * grip_ce).mean()
