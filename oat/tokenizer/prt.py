"""Predictive residual coding for LIBERO-style six-axis arm and binary gripper commands."""

import math

import torch
import torch.nn.functional as F

from oat.tokenizer.oat.tokenizer import OATTok


class PredictiveResidualTokenizer(OATTok):
    """Condition the physical action representation on raw, zero-padded command history.

    Arm FSQ IDs and the independent gripper event ID form separate vocabularies.
    The gripper event is 0 for hold, 1..H for switching to +1, and H+1..2H for
    switching to -1. In LIBERO the reset/open command is -1. History uses zero
    padding, which resolves to the configured initial command at episode start.
    """

    def __init__(
        self, encoder, decoder, quantizer, past_n=7, gamma=0.5, execution_steps=8,
        prefix_weight=2.0, integration_weight=0.1, history_noise_std=0.0,
        history_noise_rho=0.8, initial_gripper=-1.0,
    ):
        super().__init__(encoder, decoder, quantizer)
        self.horizon = decoder.sample_horizon
        if decoder.sample_dim != 6:
            raise ValueError("PRT's decoder must return six arm command channels")
        if self.latent_horizon != encoder.num_registers:
            raise ValueError("Encoder registers must match decoder latent_horizon")
        if past_n < 1 or not 1 <= execution_steps <= self.horizon:
            raise ValueError("Require past_n >= 1 and 1 <= execution_steps <= horizon")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        if not math.isfinite(prefix_weight) or prefix_weight <= 0:
            raise ValueError("prefix_weight must be finite and positive")
        if not math.isfinite(integration_weight) or integration_weight < 0:
            raise ValueError("integration_weight must be finite and nonnegative")
        if not math.isfinite(history_noise_std) or history_noise_std < 0:
            raise ValueError("history_noise_std must be finite and nonnegative")
        if not -1 < history_noise_rho < 1:
            raise ValueError("history_noise_rho must be between -1 and 1")
        if initial_gripper not in (-1.0, 1.0):
            raise ValueError("initial_gripper must be a binary command (-1 or +1)")
        self.past_n = past_n
        self.gamma = gamma
        self.execution_steps = execution_steps
        self.prefix_weight = prefix_weight
        self.integration_weight = integration_weight
        self.history_noise_std = history_noise_std
        self.history_noise_rho = history_noise_rho
        self.initial_gripper = float(initial_gripper)
        self.grip_vocab_size = 2 * self.horizon + 1
        self.register_buffer("prior_weights", torch.zeros(2, self.horizon, past_n))
        self.register_buffer("residual_scale", torch.ones(self.horizon, 6))
        self.register_buffer("prior_fitted", torch.tensor(False))
        step_weights = torch.ones(self.horizon)
        step_weights[:execution_steps] = prefix_weight
        self.register_buffer("step_weights", step_weights, persistent=False)

    @classmethod
    def from_checkpoint(cls, checkpoint, output_dir=None, return_configuration=False):
        """Export a fitted codec for policy training or inference."""
        codec, cfg = super().from_checkpoint(
            checkpoint, output_dir=output_dir, return_configuration=True,
        )
        if not isinstance(codec, cls):
            raise TypeError("The checkpoint configuration does not specify a PRT tokenizer")
        if not codec.prior_fitted.item():
            raise ValueError("The checkpoint does not contain a fitted PRT prior")
        codec._frozen_for_policy = True
        codec.requires_grad_(False)
        codec.eval()
        return (codec, cfg) if return_configuration else codec

    def train(self, mode=True):
        # Parent policies call train() recursively. Exported codec labels must stay deterministic.
        return super().train(False if getattr(self, "_frozen_for_policy", False) else mode)

    @torch.no_grad()
    def set_prior_statistics(self, weights, scales):
        """Install training-split ridge weights [2,H,K] and positive scales [H,2] or [H,6]."""
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        scales = torch.as_tensor(scales, device=self.device, dtype=torch.float32)
        if scales.shape == (self.horizon, 2):
            scales = scales.repeat_interleave(3, dim=1)
        if weights.shape != self.prior_weights.shape or scales.shape != self.residual_scale.shape:
            raise ValueError("Expected prior weights [2,H,K] and residual scales [H,2] or [H,6]")
        if not torch.isfinite(weights).all() or not (torch.isfinite(scales) & (scales > 0)).all():
            raise ValueError("Prior weights must be finite and residual scales finite and positive")
        self.prior_weights.copy_(weights)
        self.residual_scale.copy_(scales)
        self.prior_fitted.fill_(True)

    def prior(self, past_actions):
        """Predict raw arm commands, applying one temporal filter per 3-axis group."""
        if past_actions.ndim != 3 or past_actions.shape[1:] != (self.past_n, 7):
            raise ValueError("past_actions must contain raw commands shaped [B,K,7]")
        if not self.prior_fitted.item():
            raise RuntimeError("Fit and install the PRT prior statistics before using the tokenizer")
        with torch.autocast(device_type=past_actions.device.type, enabled=False):
            arm = past_actions[..., :6].float().reshape(-1, self.past_n, 2, 3)
            predicted = torch.einsum("ghk,bkgc->bhgc", self.prior_weights.float(), arm)
            return predicted.flatten(start_dim=2)

    def compand(self, residuals):
        return torch.asinh(residuals.float() / (self.gamma * self.residual_scale))

    def expand(self, compact):
        return compact.float().sinh() * (self.gamma * self.residual_scale)

    def initial_command(self, past_actions):
        if past_actions.ndim != 3 or past_actions.shape[1:] != (self.past_n, 7):
            raise ValueError("past_actions must contain raw commands shaped [B,K,7]")
        last = past_actions[:, -1, 6]
        if not ((last == -1) | (last == 0) | (last == 1)).all():
            raise ValueError("Past gripper commands must be -1/+1, with zero allowed at reset")
        return torch.where(last == 0, self.initial_gripper, last)

    def _gripper_trajectory(self, events, initial):
        # The trailing axis is time; leading axes can also enumerate candidate events.
        steps = torch.arange(self.horizon, device=events.device)
        switch_step = (events - 1).remainder(self.horizon)
        after = (events[..., None] != 0) & (steps >= switch_step[..., None])
        command = torch.where(events <= self.horizon, 1.0, -1.0)
        return torch.where(after, command[..., None], initial[..., None])

    def encode_gripper(self, target, past_actions):
        """Choose the event with the fewest wrong command steps. Ties prefer hold."""
        if target.shape != (past_actions.shape[0], self.horizon):
            raise ValueError("Gripper targets must have shape [B,H]")
        if not ((target == -1) | (target == 1)).all():
            raise ValueError("PRT requires binary -1/+1 gripper targets")
        initial = self.initial_command(past_actions)
        events = torch.arange(self.grip_vocab_size, device=target.device)
        candidates = self._gripper_trajectory(events[None, :], initial[:, None])
        return (candidates != target[:, None, :]).sum(dim=-1).argmin(dim=1)

    def _encoded_arm(self, samples, past_actions):
        if samples.shape != (past_actions.shape[0], self.horizon, 7):
            raise ValueError("samples must contain raw commands shaped [B,H,7]")
        predicted = self.prior(past_actions)
        latents = self.encoder(self.compand(samples[..., :6] - predicted))
        quantized, arm_tokens = self.quantizer(latents)
        return predicted, quantized, arm_tokens

    def encode(self, samples, past_actions):
        """Return quantized arm latents and the pair (arm IDs [B,L], gripper IDs [B])."""
        _, quantized, arm_tokens = self._encoded_arm(samples, past_actions)
        grip_tokens = self.encode_gripper(samples[..., 6], past_actions)
        return quantized, (arm_tokens, grip_tokens)

    def tokenize(self, samples, past_actions):
        return self.encode(samples, past_actions)[1]

    def decode(self, latents, grip_tokens, past_actions, eval_keep_k=None):
        """Decode arm latents and an event relative to the history used for encoding."""
        if latents.shape != (past_actions.shape[0], self.latent_horizon, self.quantizer.dim):
            raise ValueError("Quantized arm latents must have shape [B,L,FSQ dimensions]")
        if grip_tokens.shape != (past_actions.shape[0],):
            raise ValueError("grip_tokens must have shape [B]")
        if grip_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("grip_tokens must contain integer event IDs")
        if not ((grip_tokens >= 0) & (grip_tokens < self.grip_vocab_size)).all():
            raise ValueError("Gripper event ID outside the PRT gripper vocabulary")
        predicted = self.prior(past_actions)
        arm = predicted + self.expand(self.decoder(latents, eval_keep_k=eval_keep_k))
        grip = self._gripper_trajectory(grip_tokens, self.initial_command(past_actions))
        return torch.cat((arm, grip.unsqueeze(-1)), dim=-1)

    def detokenize(self, arm_tokens, grip_tokens, past_actions, eval_keep_k=None):
        """Decode a common prefix [B,k], optionally retaining a shorter prefix per row."""
        if arm_tokens.ndim != 2 or arm_tokens.shape[0] != past_actions.shape[0]:
            raise ValueError("arm_tokens must have shape [B,k]")
        if arm_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("arm_tokens must contain integer FSQ IDs")
        if not ((arm_tokens >= 0) & (arm_tokens < self.quantizer.codebook_size)).all():
            raise ValueError("Arm token ID outside the FSQ vocabulary")
        batch_size, count = arm_tokens.shape
        if not 1 <= count <= self.latent_horizon:
            raise ValueError("Require 1 <= arm prefix length <= latent_horizon")
        if eval_keep_k is None:
            lengths = [count] * batch_size
        else:
            lengths = torch.as_tensor(eval_keep_k)
            if lengths.shape != (batch_size,) or lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("eval_keep_k must be an integer length per batch row")
            lengths = lengths.tolist()
            if any(k < 1 or k > count for k in lengths):
                raise ValueError("Requested arm prefix exceeds the supplied arm tokens")
        if any(k < self.latent_horizon for k in lengths):
            if self.training:
                raise ValueError("Explicit arm prefix decoding requires tokenizer.eval()")
            if self.decoder.nested_dropout.size_sampling_mode == "disable":
                raise ValueError("Arm prefix decoding requires a decoder trained with nested dropout")
        padded = F.pad(arm_tokens, (0, self.latent_horizon - count))
        latents = self.quantizer.indices_to_embedding(padded)
        return self.decode(latents, grip_tokens, past_actions, eval_keep_k=lengths)

    def autoencode(self, samples, past_actions, eval_keep_k=None):
        arm_tokens, grip_tokens = self.tokenize(samples, past_actions)
        return self.detokenize(arm_tokens, grip_tokens, past_actions, eval_keep_k)

    reconstruct = autoencode

    def perturb_history(self, past_actions, valid):
        """Add stationary AR(1) arm noise in units of the last horizon's fitted residual SD."""
        if not self.training or self.history_noise_std == 0:
            return past_actions
        if valid is None or valid.dtype != torch.bool or valid.shape != past_actions.shape[:2]:
            raise ValueError("Training history perturbation requires boolean past_action_valid [B,K]")
        innovations = torch.randn_like(past_actions[..., :6], dtype=torch.float32)
        noise = [innovations[:, 0]]
        rho = self.history_noise_rho
        innovation_scale = math.sqrt(1 - rho * rho)
        for step in range(1, self.past_n):
            noise.append(rho * noise[-1] + innovation_scale * innovations[:, step])
        noise = torch.stack(noise, dim=1) * (self.history_noise_std * self.residual_scale[-1])
        result = past_actions.clone()
        result[..., :6] += noise.masked_fill(~valid.unsqueeze(-1), 0)
        return result

    def loss_per_sample(self, batch):
        action = batch["action"]
        past = self.perturb_history(batch["past_action"], batch.get("past_action_valid"))
        predicted, latents, _ = self._encoded_arm(action, past)
        # The binary gripper uses a deterministic codec, without learned parameters.
        if not ((action[..., 6] == -1) | (action[..., 6] == 1)).all():
            raise ValueError("PRT requires binary -1/+1 gripper targets")
        recovered = predicted + self.expand(self.decoder(latents))
        error = recovered - action[..., :6].float()
        instantaneous = error.square().mean(dim=-1)
        integrated = error.cumsum(dim=1).square().mean(dim=-1)
        distortion = instantaneous + self.integration_weight * integrated
        return (distortion * self.step_weights).sum(dim=1) / self.step_weights.sum()

    def forward(self, batch, reduction="mean"):
        distortion = self.loss_per_sample(batch)
        if reduction == "none":
            return distortion
        if reduction == "mean":
            return distortion.mean()
        raise ValueError("reduction must be 'mean' or 'none'")
