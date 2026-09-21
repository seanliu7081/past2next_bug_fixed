"""Offline self-past with measured state history and execution-aware rollout."""

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from oat.model.state_action_history import StateActionHistoryEncoder
from oat.policy.past2next_self_past_executed import Past2NextSelfPastExecutedPolicy


DEFAULT_STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


class Past2NextStateHistoryPolicy(Past2NextSelfPastExecutedPolicy):
    """Append temporal state/action memory to the existing AR conditions.

    H measured states align with H-1 past commands. Offline self-past can replace
    those commands with predictions, while observations remain demonstrations.
    At inference both states and acknowledged commands come from execution.
    The frozen tokenizer and discrete token autoregressive objective are reused.
    """

    def __init__(self, *args, state_history_steps=8,
                 state_history_keys=DEFAULT_STATE_KEYS, history_embed_dim=128,
                 history_n_heads=4, history_n_layers=2, history_summary_tokens=4,
                 history_dropout=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if state_history_steps != self.past_n + 1:
            raise ValueError("state_history_steps must equal past_n + 1")
        keys = tuple(state_history_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("state_history_keys must be nonempty and unique")
        for key in keys:
            if key not in self.obs_key_shapes or len(self.obs_key_shapes[key]) != 1:
                raise ValueError(f"History key {key!r} must be a vector observation")
        self.state_history_steps = state_history_steps
        self.state_history_keys = keys
        self.history_encoder = StateActionHistoryEncoder(
            state_shapes={key: tuple(self.obs_key_shapes[key]) for key in keys},
            action_dim=self.action_dim, history_steps=state_history_steps,
            output_dim=self.obs_feature_dim, embed_dim=history_embed_dim,
            n_heads=history_n_heads, n_layers=history_n_layers,
            n_summary_tokens=history_summary_tokens, dropout=history_dropout,
        )
        # Fresh-policy architecture: retain all original conditions and append
        # fixed-size history memory. The tokenizer's architecture is unchanged.
        positions = self.model.cond_pos_emb
        self.model.cond_pos_emb = nn.Parameter(torch.cat((
            positions.detach(), positions.new_zeros(1, history_summary_tokens,
                                                   positions.shape[-1]),
        ), dim=1))
        history_params = sum(p.numel() for p in self.history_encoder.parameters())
        print(f"  state history: {state_history_steps} states, {self.past_n} commands, "
              f"{history_summary_tokens} summary tokens; {history_params / 1e6:.2f}M params; "
              f"total cond_len={self.model.cond_pos_emb.shape[1]}\n")

    def get_policy_name(self):
        return "past2next_state_history_" + "|".join(
            modality for modality in self.modalities if modality != "state"
        )

    def get_observation_ports(self):
        return list(super().get_observation_ports()) + [
            "state_history__" + key for key in self.state_history_keys
        ] + ["state_history_valid"]

    def create_dummy_observation(self, batch_size=1, device=None):
        device = self.device if device is None else device
        obs = super().create_dummy_observation(batch_size=batch_size, device=device)
        obs = {key: value.to(dtype=self.dtype) if isinstance(value, torch.Tensor)
               and value.is_floating_point() else value for key, value in obs.items()}
        for key in self.state_history_keys:
            value = torch.zeros(batch_size, self.state_history_steps,
                                *self.obs_key_shapes[key], device=device, dtype=self.dtype)
            if key.endswith("_quat"):
                value[:, -1, 3] = 1
            obs["state_history__" + key] = value
        valid = torch.zeros(batch_size, self.state_history_steps, device=device,
                            dtype=torch.bool)
        valid[:, -1] = True
        obs["state_history_valid"] = valid
        return obs

    def get_optimizer(self, policy_lr, obs_enc_lr, weight_decay, betas):
        optimizer = super().get_optimizer(policy_lr, obs_enc_lr, weight_decay, betas)
        params = [p for p in self.history_encoder.parameters() if p.requires_grad]
        for decay, group in ((weight_decay, [p for p in params if p.ndim >= 2]),
                             (0.0, [p for p in params if p.ndim < 2])):
            optimizer.add_param_group({"params": group, "lr": policy_lr,
                                       "weight_decay": decay})
        return optimizer

    def _condition(self, obs, past_actions):
        required = ["state_history__" + key for key in self.state_history_keys]
        required.append("state_history_valid")
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"Missing state history ports {missing}; use "
                           "ZarrDatasetWithStateHistory / LiberoStateHistoryRunner")
        features = self.obs_encoder(obs)
        expected = (features.shape[0], self.past_n, self.action_dim)
        if tuple(past_actions.shape) != expected:
            raise ValueError(f"past_actions must have shape {expected}")
        validity = obs["state_history_valid"]
        # LiberoRunner casts all ndarray ports (including boolean metadata) to
        # the policy dtype. Accept its exact 0/1 representation, then restore bool.
        if validity.dtype != torch.bool:
            if not ((validity == 0) | (validity == 1)).all():
                raise ValueError("state_history_valid must contain only zero/one flags")
            validity = validity.to(dtype=torch.bool)
        original = self._build_condition(features, past_actions)
        memory = self.history_encoder(
            {key: obs["state_history__" + key] for key in self.state_history_keys},
            validity,
            self.action_normalizer["action"].normalize(past_actions),
            self.action_normalizer,
        )
        return torch.cat((original, memory), dim=1)

    @contextlib.contextmanager
    def _rollout_mode(self):
        was_training = self.history_encoder.training
        try:
            self.history_encoder.eval()
            with super()._rollout_mode():
                yield
        finally:
            self.history_encoder.train(was_training)

    def _generate_actions(self, obs, past_actions, n_tokens, temperature, topk):
        cond = self._condition(obs, past_actions)
        tokens = torch.full((cond.shape[0], 1), self.bos_id, dtype=torch.long,
                            device=cond.device)
        tokens = self.model.generate(
            tokens, cond=cond, max_new_tokens=n_tokens,
            temperature=temperature, top_k=topk, bos_id=self.bos_id,
        )[:, 1:]
        with torch.inference_mode():
            return self.action_tokenizer.detokenize(tokens=tokens)

    def _generate_prev_past(self, batch):
        previous = batch["prev_past_action"]
        with self._rollout_mode(), self._clean_autocast_cache():
            with torch.inference_mode():
                prediction = self._generate_actions(
                    batch["prev_obs"], previous, self.max_seq_len,
                    self.self_past_temperature, self.self_past_topk,
                )
                generated = torch.cat((previous, prediction[:, :self.n_action_steps]),
                                      dim=1)[:, -self.past_n:]
        with torch.inference_mode(False):
            return generated.detach().clone().to(dtype=previous.dtype)

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
        cond = self._condition(batch["obs"], past)
        bos = torch.full((targets.shape[0], 1), self.bos_id, dtype=torch.long,
                         device=targets.device)
        logits = self.model(torch.cat((bos, targets[:, :-1]), dim=1), cond=cond)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if self.training and self.self_past_schedule == "legacy":
            self._train_step += 1
        return loss

    def predict_action(self, obs_dict, use_k_tokens=None, temperature=None,
                       topk=None, past_actions=None):
        stateful = past_actions is None
        if stateful and self._pending_execution_steps is not None:
            raise RuntimeError("Execution feedback is pending. Call record_executed_actions "
                               "or reset before another stateful prediction.")
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        if (isinstance(use_k_tokens, bool) or not isinstance(use_k_tokens, int)
                or use_k_tokens < 1):
            raise ValueError("use_k_tokens must be a positive integer")
        use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if stateful:
            observation = next((value for value in obs_dict.values()
                                if isinstance(value, torch.Tensor)), None)
            if observation is None or observation.ndim < 1 or observation.shape[0] < 1:
                raise ValueError("A nonempty batched tensor observation is required")
            expected = (observation.shape[0], self.past_n, self.action_dim)
            if self._past_buffer is None:
                self._past_buffer = torch.zeros(expected, device=self.device, dtype=self.dtype)
            elif tuple(self._past_buffer.shape) != expected:
                raise RuntimeError("Execution-history batch size changed; call reset first")
            else:
                self._past_buffer = self._past_buffer.to(device=self.device, dtype=self.dtype)
            past_actions = self._past_buffer
        prediction = self._generate_actions(
            obs_dict, past_actions, use_k_tokens,
            self.temperature if temperature is None else temperature,
            self.topk if topk is None else topk,
        )
        action = prediction[:, :self.n_action_steps]
        if stateful:
            self._pending_execution_steps = action.shape[1]
        return {"action": action, "action_pred": prediction}
