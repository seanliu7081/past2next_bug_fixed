"""Independent, observation-conditioned gate for state-history summary attention.

Only the appended summary tokens are gated. Original observation, raw action,
and command-difference conditions remain available. No existing policy or
attention implementation is modified by this module.
"""

import contextlib
import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from oat.model.autoregressive.transformer_cache_history_gate import (
    HistoryGatedAutoregressiveModel,
)
from oat.policy.past2next_state_history import Past2NextStateHistoryPolicy


class Past2NextStateHistoryGatePolicy(Past2NextStateHistoryPolicy):
    """Add a shared scalar attention gate to the existing history summaries.

The gate observes pooled current observation features, pooled state/action
memory, and the fraction of valid history states. ``log(sigmoid(logit))`` is
added to summary-key attention scores in every AR layer, including cached
generation and offline self-past generation. It is a relative attention prior,
not the probability that the history is correct or a fraction of the action.

Modes ``open`` and ``closed`` are fixed diagnostic controls. A fully open gate
preserves the original C model's computation; closed masks only its appended
history summaries. The learned gate starts at ``history_gate_init``.
"""

    def __init__(self, *args, history_gate_mode="learned",
                 history_gate_hidden_dim=128, history_gate_init=0.9, **kwargs):
        if history_gate_mode not in {"learned", "open", "closed"}:
            raise ValueError("history_gate_mode must be learned, open, or closed")
        if (isinstance(history_gate_hidden_dim, bool)
                or not isinstance(history_gate_hidden_dim, int)
                or history_gate_hidden_dim < 1):
            raise ValueError("history_gate_hidden_dim must be a positive integer")
        if (isinstance(history_gate_init, bool)
                or not isinstance(history_gate_init, (int, float))
                or not math.isfinite(history_gate_init)
                or not 0 < history_gate_init < 1):
            raise ValueError("history_gate_init must be finite and strictly between 0 and 1")
        super().__init__(*args, **kwargs)
        self.history_gate_mode = history_gate_mode
        self.history_gate_init = float(history_gate_init)
        self.history_summary_tokens = self.history_encoder.n_summary_tokens
        self.model = HistoryGatedAutoregressiveModel.from_base(
            self.model, history_token_count=self.history_summary_tokens,
        )
        gate_input_dim = 2 * self.obs_feature_dim + 1
        self.history_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, history_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(history_gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.history_gate[-1].weight)
        nn.init.constant_(self.history_gate[-1].bias,
                          math.log(self.history_gate_init / (1 - self.history_gate_init)))
        # Keep identical checkpoint keys for all modes, without unused trainable
        # gate parameters in the fixed controls under DDP.
        self.history_gate.requires_grad_(history_gate_mode == "learned")
        self.register_buffer("_history_gate_schema_version", torch.tensor(1, dtype=torch.long))
        self._last_history_gate = None

    def get_policy_name(self):
        return "past2next_state_history_gate_" + "|".join(
            modality for modality in self.modalities if modality != "state"
        )

    def get_optimizer(self, policy_lr, obs_enc_lr, weight_decay, betas):
        optimizer = super().get_optimizer(policy_lr, obs_enc_lr, weight_decay, betas)
        params = [p for p in self.history_gate.parameters() if p.requires_grad]
        for decay, group in ((weight_decay, [p for p in params if p.ndim >= 2]),
                             (0.0, [p for p in params if p.ndim < 2])):
            if group:
                optimizer.add_param_group({"params": group, "lr": policy_lr,
                                           "weight_decay": decay})
        return optimizer

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        schema_key = prefix + "_history_gate_schema_version"
        gate_prefix = prefix + "history_gate."
        is_legacy_c = (
            schema_key not in state_dict
            and not any(key.startswith(gate_prefix) for key in state_dict)
            and any(key.startswith(prefix + "history_encoder.") for key in state_dict)
        )
        if is_legacy_c:
            # Only a complete old C -> gate migration receives fresh gate
            # weights. Partially missing gate checkpoints still fail strictly.
            state_dict[schema_key] = self._history_gate_schema_version.detach().clone()
            for name, value in self.history_gate.state_dict().items():
                state_dict[gate_prefix + name] = value.detach().clone()
            warnings.warn(
                "Loading a state-history C checkpoint: initializing new history gate "
                f"at {self.history_gate_init:g}; all existing weights load strictly. "
                "Use training.init_checkpoint for a new run, not optimizer resume.",
                UserWarning,
                stacklevel=2,
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _condition_and_gate(self, obs, past_actions):
        # The parent validates history, masks padding, and computes observation
        # and history features exactly once. Original observations occupy the
        # first n_obs_steps positions; appended summaries occupy the last ones.
        cond = super()._condition(obs, past_actions)
        batch_size = cond.shape[0]
        if self.history_gate_mode == "open":
            log_gate = None  # Also retains the original SDPA path when open.
            gate_value = cond.new_ones(batch_size, 1)
        elif self.history_gate_mode == "closed":
            log_gate = cond.new_full((batch_size, 1), -float("inf"))
            gate_value = cond.new_zeros(batch_size, 1)
        else:
            observed = cond[:, :self.n_obs_steps].mean(dim=1)
            history = cond[:, -self.history_summary_tokens:].mean(dim=1)
            valid_fraction = obs["state_history_valid"].to(
                device=cond.device, dtype=cond.dtype,
            ).mean(dim=1, keepdim=True)
            inputs = torch.cat((observed, history, valid_fraction), dim=-1)
            logits = self.history_gate(inputs)
            # Stable for large negative logits, including mixed precision;
            # unlike log(sigmoid(logits)), this does not underflow to -inf.
            log_gate = F.logsigmoid(logits.float())
            gate_value = log_gate.exp()
        self._last_history_gate = gate_value.detach()
        return cond, log_gate

    def get_history_gate_metrics(self):
        """Inspect the last call; synchronization occurs only on explicit use.

        The unchanged training workspace does not automatically log these
        diagnostics. No graph or persistent checkpoint statistics are retained.
        """
        if self._last_history_gate is None:
            return {}
        gate = self._last_history_gate.float()
        return {"history_gate/mean": gate.mean().item(),
                "history_gate/min": gate.min().item(),
                "history_gate/max": gate.max().item()}

    @contextlib.contextmanager
    def _rollout_mode(self):
        was_training = self.history_gate.training
        try:
            self.history_gate.eval()
            with super()._rollout_mode():
                yield
        finally:
            self.history_gate.train(was_training)

    def _generate_actions(self, obs, past_actions, n_tokens, temperature, topk):
        cond, log_gate = self._condition_and_gate(obs, past_actions)
        tokens = torch.full((cond.shape[0], 1), self.bos_id, dtype=torch.long,
                            device=cond.device)
        tokens = self.model.generate(
            tokens, cond=cond, max_new_tokens=n_tokens,
            temperature=temperature, top_k=topk, bos_id=self.bos_id,
            history_log_gate=log_gate,
        )[:, 1:]
        with torch.inference_mode():
            return self.action_tokenizer.detokenize(tokens=tokens)

    def forward(self, batch, history_mode=None):
        with torch.no_grad():
            targets = self.action_tokenizer.tokenize(batch["action"])
        past = batch["past_action"]
        if history_mode is None:
            history_mode = ("expert" if not self.training and
                            self.self_past_schedule == "optimizer_step" else "configured")
        if history_mode == "generated":
            past = self._maybe_self_past(batch, past, probability=1.0)
        elif history_mode == "configured":
            past = self._maybe_self_past(batch, past)
        elif history_mode != "expert":
            raise ValueError("history_mode must be 'expert', 'generated', or 'configured'")
        cond, log_gate = self._condition_and_gate(batch["obs"], past)
        bos = torch.full((targets.shape[0], 1), self.bos_id, dtype=torch.long,
                         device=targets.device)
        logits = self.model(torch.cat((bos, targets[:, :-1]), dim=1),
                            cond=cond, history_log_gate=log_gate)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if self.training and self.self_past_schedule == "legacy":
            self._train_step += 1
        return loss
