"""Stage 2 behavior cloning with a frozen predictive residual action tokenizer."""

import math

import torch
import torch.nn.functional as F

from oat.model.autoregressive.prt_transformer import PRTActionModel
from oat.policy.past2next_self_past_executed import Past2NextSelfPastExecutedPolicy
from oat.tokenizer.prt import PredictiveResidualTokenizer


class PRTPolicy(Past2NextSelfPastExecutedPolicy):
    """Predict an arm FSQ sequence and a gripper event relative to the same past commands."""

    def __init__(
        self, *, action_tokenizer, horizon=None, history_dropout=0.25, history_noise_std=0.5,
        history_noise_rho=0.8, innovation_alpha=1.0, innovation_cap=3.0,
        grip_weight=1.0, **kwargs,
    ):
        if not isinstance(action_tokenizer, PredictiveResidualTokenizer):
            raise TypeError("PRTPolicy requires a fitted PredictiveResidualTokenizer")
        if not action_tokenizer.prior_fitted.item():
            raise ValueError("The PRT checkpoint must include fitted prior weights and scales")
        if horizon is not None and horizon != action_tokenizer.horizon:
            raise ValueError("The Stage 2 action horizon must match the fitted PRT checkpoint horizon")
        super().__init__(action_tokenizer=action_tokenizer, **kwargs)
        if self.action_dim != 7 or self.past_n != action_tokenizer.past_n:
            raise ValueError("PRT requires seven-dimensional commands and matching policy/codec past_n")
        if not 1 <= self.n_action_steps <= action_tokenizer.horizon:
            raise ValueError("n_action_steps must lie within the PRT prediction horizon")
        if not 0 <= history_dropout <= 1 or not -1 < history_noise_rho < 1:
            raise ValueError("Require history_dropout in [0,1] and history_noise_rho in (-1,1)")
        for name, value in (("history_noise_std", history_noise_std),
                            ("innovation_alpha", innovation_alpha),
                            ("innovation_cap", innovation_cap), ("grip_weight", grip_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.model = PRTActionModel(self.model, action_tokenizer.grip_vocab_size)
        self.history_dropout = history_dropout
        self.history_noise_std = history_noise_std
        self.history_noise_rho = history_noise_rho
        self.innovation_alpha = innovation_alpha
        self.innovation_cap = innovation_cap
        self.grip_weight = grip_weight

    def get_policy_name(self):
        return "prt_policy"

    def _selected_history(self, batch, history_mode):
        past = batch["past_action"]
        if history_mode is None:
            history_mode = ("expert" if not self.training and self.self_past_schedule == "optimizer_step"
                            else "configured")
        if history_mode == "expert":
            return past
        if history_mode == "generated":
            return self._maybe_self_past(batch, past, probability=1.0)
        if history_mode == "configured":
            return self._maybe_self_past(batch, past)
        raise ValueError("history_mode must be 'expert', 'generated', or 'configured'")

    def _perturb_history(self, past, valid):
        """Sample stationary AR(1) arm noise in units of the final horizon's fitted scales."""
        if not self.training or self.history_noise_std == 0:
            return past
        if valid is None or valid.dtype != torch.bool or valid.shape != past.shape[:2]:
            raise ValueError("PRT history perturbation requires boolean past_action_valid [B,K]")
        innovations = torch.randn_like(past[..., :6], dtype=torch.float32)
        rho = self.history_noise_rho
        innovation_scale = math.sqrt(1 - rho * rho)
        noise = [innovations[:, 0]]
        for step in range(1, self.past_n):
            noise.append(rho * noise[-1] + innovation_scale * innovations[:, step])
        scale = self.history_noise_std * self.action_tokenizer.residual_scale[-1]
        delta = torch.stack(noise, dim=1) * scale
        result = past.clone()
        result[..., :6] += delta.masked_fill(~valid[..., None], 0)
        return result

    def _policy_condition(self, features, past, apply_dropout=False):
        if features.ndim != 3 or features.shape[1:] != (self.n_obs_steps, self.obs_feature_dim):
            raise ValueError("Expected encoded observations shaped [B,n_obs_steps,obs_feature_dim]")
        if past.shape != (features.shape[0], self.past_n, self.action_dim):
            raise ValueError("Past commands must match the observation batch and PRT history dimensions")
        cond = self._build_condition(features, past)
        if apply_dropout and self.training and self.history_dropout:
            # Mask after projection so raw commands, differences and their biases all disappear.
            history_tokens = cond[:, features.shape[1]:]
            drop = torch.rand((features.shape[0], 1, 1), device=features.device) < self.history_dropout
            cond = torch.cat((features, history_tokens.masked_fill(drop, 0)), dim=1)
        return cond

    def _innovation_weights(self, action, past):
        residual = action[..., :6].float() - self.action_tokenizer.prior(past)
        standardized = residual / self.action_tokenizer.residual_scale
        energy = standardized.square().mean(dim=(1, 2)).sqrt()
        return 1 + self.innovation_alpha * energy.clamp(max=self.innovation_cap)

    def forward(self, batch, history_mode=None):
        action = batch["action"]
        with torch.no_grad():
            past = self._selected_history(batch, history_mode)
            past = self._perturb_history(past, batch.get("past_action_valid"))
            arm, grip = self.action_tokenizer.tokenize(action, past)
            weights = self._innovation_weights(action, past)
        cond = self._policy_condition(self.obs_encoder(batch["obs"]), past, apply_dropout=True)
        arm_logits, grip_logits = self.model.teacher_logits(cond, arm, self.bos_id)
        arm_ce = F.cross_entropy(
            arm_logits.flatten(0, 1), arm.flatten(), reduction="none",
        ).reshape_as(arm)
        grip_ce = F.cross_entropy(grip_logits, grip, reduction="none")
        loss = (weights * arm_ce.sum(dim=1) + self.grip_weight * grip_ce).mean()
        if self.training and self.self_past_schedule == "legacy":
            self._train_step += 1
        return loss

    def _generate_commands(self, features, past, count, temperature, topk):
        cond = self._policy_condition(features, past)
        # The gripper is conditioned on the full sequence, as in the teacher-forced loss.
        arm, grip = self.model.sample_tokens(cond, self.max_seq_len, self.bos_id, temperature, topk)
        arm = arm[:, :count]
        action = self.action_tokenizer.detokenize(arm, grip, past)
        return {"arm_tokens": arm, "grip_tokens": grip, "action_pred": action,
                "action": action[:, :self.n_action_steps]}

    def _generate_prev_past(self, batch):
        """Unroll one previous decision on its recorded observations and its own prior."""
        previous = batch["prev_past_action"]
        with self._rollout_mode(), self._clean_autocast_cache(), torch.inference_mode():
            output = self._generate_commands(
                self.obs_encoder(batch["prev_obs"]), previous, self.max_seq_len,
                self.self_past_temperature, self.self_past_topk,
            )
            history = torch.cat((previous, output["action"]), dim=1)[:, -self.past_n:]
        with torch.inference_mode(False):
            return history.detach().clone().to(dtype=previous.dtype)

    @torch.no_grad()
    def predict_action(self, obs_dict, use_k_tokens=None, temperature=None, topk=None,
                       past_actions=None):
        """Use explicit offline history or require acknowledgement of each stateful prediction."""
        stateful = past_actions is None
        if stateful and self._pending_execution_steps is not None:
            raise RuntimeError("Execution feedback is pending. Call record_executed_actions or reset.")
        count = self.max_seq_len if use_k_tokens is None else use_k_tokens
        if not isinstance(count, int) or not 1 <= count <= self.max_seq_len:
            raise ValueError("use_k_tokens must be a valid number of arm tokens")
        features = self.obs_encoder(obs_dict)
        expected = (features.shape[0], self.past_n, self.action_dim)
        if stateful:
            if self._past_buffer is None:
                self._past_buffer = torch.zeros(expected, device=self.device, dtype=torch.float32)
            elif self._past_buffer.shape != expected:
                raise RuntimeError("Execution-history batch size changed; call reset first")
            past_actions = self._past_buffer.to(device=self.device)
            self._past_buffer = past_actions
        elif past_actions.shape != expected:
            raise ValueError(f"past_actions must have shape {expected}")
        output = self._generate_commands(
            features, past_actions, count, self.temperature if temperature is None else temperature,
            self.topk if topk is None else topk,
        )
        if stateful:
            self._pending_execution_steps = output["action"].shape[1]
        return output
