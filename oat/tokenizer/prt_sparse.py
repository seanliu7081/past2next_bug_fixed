"""PRT action coding with sparse, execution-cost-aware supervision."""

import math
import operator
from dataclasses import dataclass

import dill
import hydra
import torch

from oat.common.hydra_util import register_new_resolvers
from oat.tokenizer.prt import PredictiveResidualTokenizer


@dataclass(frozen=True)
class SparseTokenTargets:
    """Per-example candidate IDs and raw arm command MSE over executed steps.

    Column zero is the original FSQ ID. Each distinct ID has one valid slot.
    Duplicate slots contain legal IDs and zero costs; use the valid mask when
    aggregating. The final two scalar fields have one value per example.
    """

    ids: torch.Tensor
    costs: torch.Tensor
    valid: torch.Tensor
    acceptable: torch.Tensor
    hard_negative: torch.Tensor
    reference_cost: torch.Tensor
    threshold: torch.Tensor


class SparsePRTTokenizer(PredictiveResidualTokenizer):
    """Keep the PRT wire format and label alternatives by their decoded effects."""

    @classmethod
    def from_checkpoint(cls, checkpoint, output_dir=None, return_configuration=False):
        """Restore fitted ordinary or sparse PRT Stage 1 weights, including EMA."""
        with open(checkpoint, "rb") as stream:
            payload = torch.load(stream, pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        register_new_resolvers()
        source_class = hydra.utils.get_class(cfg.tokenizer._target_)
        if source_class not in (PredictiveResidualTokenizer, cls):
            raise TypeError("Expected a Stage 1 PredictiveResidualTokenizer or SparsePRTTokenizer")
        # Candidate supervision introduces no new learned parameters or state.
        codec = hydra.utils.instantiate(cfg.tokenizer, _target_=f"{cls.__module__}.{cls.__name__}")
        state_key = "ema_model" if getattr(cfg.training, "use_ema", False) else "model"
        codec.load_state_dict(payload["state_dicts"][state_key], strict=True)
        if not codec.prior_fitted.item():
            raise ValueError("The checkpoint does not contain a fitted PRT prior")
        codec._frozen_for_policy = True
        codec.requires_grad_(False)
        codec.eval()
        return (codec, cfg) if return_configuration else codec

    @torch.no_grad()
    def sparse_targets(
        self, actions, past, arm, grip, positions, policy_ids=None, *,
        relative_tolerance=0.25, absolute_tolerance=1e-7, decode_batch_size=128,
        execution_steps=None,
    ):
        """Score alternative IDs at one chosen register per example.

        Obtain arm [B,L] and grip [B] from tokenize(actions, past). positions
        [B] can cover any arm register. Pass optional policy_ids [B,K] from
        logits conditioned on observation, history and arm IDs before that
        position. For each counterfactual, all other arm IDs, including the
        expert suffix, stay fixed. The scores measure this single substitution.
        """
        if self.training:
            raise ValueError("Sparse supervision requires the PRT decoder in eval mode")
        decode_batch_size = operator.index(decode_batch_size)
        steps = self.execution_steps if execution_steps is None else operator.index(execution_steps)
        if decode_batch_size <= 0 or not 1 <= steps <= self.horizon:
            raise ValueError("Require positive decode_batch_size and execution_steps within the horizon")
        if any(not math.isfinite(x) or x < 0 for x in (relative_tolerance, absolute_tolerance)):
            raise ValueError("Cost tolerances must be finite and nonnegative")
        batch = actions.size(0)
        if not batch or actions.shape != (batch, self.horizon, 7) or past.shape != (batch, self.past_n, 7):
            raise ValueError("Expected raw actions [B,H,7] and command history [B,K,7]")
        if arm.shape != (batch, self.latent_horizon) or grip.shape != (batch,) or positions.shape != (batch,):
            raise ValueError("Expected arm [B,L], grip [B] and positions [B]")
        if policy_ids is None:
            policy_ids = arm.new_empty((batch, 0))
        if policy_ids.ndim != 2 or policy_ids.size(0) != batch:
            raise ValueError("policy_ids must be an integer [B,K] shortlist")
        if any(x.dtype not in (torch.int32, torch.int64) for x in (arm, grip, positions, policy_ids)):
            raise ValueError("Token IDs and positions must be integers")
        if any(x.device != actions.device for x in (arm, grip, past, positions, policy_ids)):
            raise ValueError("All sparse supervision inputs must be on the same device")
        if ((positions < 0) | (positions >= self.latent_horizon)).any():
            raise ValueError("Arm register position outside the reference sequence")
        if ((policy_ids < 0) | (policy_ids >= self.quantizer.codebook_size)).any():
            raise ValueError("Policy proposed a code outside the arm vocabulary")

        positions = positions.long()
        reference = arm.gather(1, positions[:, None]).long()
        basis = self.quantizer._basis
        levels = self.quantizer._levels
        digits = (reference // basis) % levels
        # Mixed-radix FSQ: change one digit only. Clamp at lattice boundaries
        # by reusing the original ID; it will be deduplicated below.
        lower = torch.where(digits > 0, reference - basis, reference)
        upper = torch.where(digits + 1 < levels, reference + basis, reference)
        ids = torch.cat((reference, lower, upper, policy_ids.long()), dim=1)
        sorted_ids, order = ids.sort(dim=1, stable=True)
        starts = torch.ones_like(ids, dtype=torch.bool)
        starts[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
        valid = torch.zeros_like(starts).scatter_(1, order, starts)
        proposed = torch.zeros_like(valid)
        if policy_ids.size(1):
            picks = policy_ids.long().sort(dim=1).values
            offsets = torch.searchsorted(picks, ids).clamp_max(picks.size(1) - 1)
            proposed = picks.gather(1, offsets) == ids

        rows, columns = valid.nonzero(as_tuple=True)
        costs = actions.new_zeros(ids.shape, dtype=torch.float32)
        # The frozen PRT checkpoint runs at native FP32 in rollout. A surrounding
        # policy AMP context can otherwise move a close pair across the threshold.
        with torch.autocast(device_type=actions.device.type, enabled=False):
            for row, col in zip(rows.split(decode_batch_size), columns.split(decode_batch_size)):
                candidate_arm = arm.index_select(0, row)
                candidate_arm.scatter_(1, positions[row, None], ids[row, col, None].to(arm.dtype))
                decoded = self.detokenize(candidate_arm, grip[row], past[row])
                error = decoded[:, :steps, :6].float() - actions[row, :steps, :6].float()
                costs[row, col] = error.square().mean(dim=(1, 2))
        if not torch.isfinite(costs).all():
            raise FloatingPointError("Non-finite PRT executed-prefix reconstruction cost")

        best = costs.masked_fill(~valid, torch.inf).amin(dim=1)
        threshold = best * (1 + relative_tolerance) + absolute_tolerance
        acceptable = valid & (costs <= threshold[:, None])
        return SparseTokenTargets(
            ids=ids, costs=costs, valid=valid, acceptable=acceptable,
            hard_negative=valid & proposed & ~acceptable, reference_cost=costs[:, 0],
            threshold=threshold,
        )
