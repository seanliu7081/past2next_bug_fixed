"""Past2Next conditioned on recorded or acknowledged executed commands.

Offline training inherits the base policy's demonstrated ``past_action`` path.
During rollout, prediction never assumes its proposed chunk has been executed:
the caller must acknowledge the actual executed prefix before predicting again.
Both raw history and command-difference features use this same history.
"""

from typing import Dict, Optional

import torch

from oat.policy.past2next import Past2NextPolicy


class Past2NextExecutedPastPolicy(Past2NextPolicy):
    """Use executed commands, without synthetic self-past substitutions.

    ``record_executed_actions`` records commands acknowledged by the execution
    layer, not measured displacements or actuator torques. Execution history is
    transient episode state and is intentionally absent from ``state_dict``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.past_n < 3:
            raise ValueError("past_n must be at least 3 for command-difference features")
        if self.n_action_steps < 1:
            raise ValueError("n_action_steps must be positive")
        self._pending_execution_steps = None

    def get_policy_name(self):
        return "past2next_executedpast_" + "|".join(
            modality for modality in self.modalities if modality != "state"
        )

    def reset(self):
        super().reset()
        self._pending_execution_steps = None

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        past_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict from explicit offline history or the acknowledged buffer.

        Explicit ``past_actions`` is a stateless interface, as used by offline
        validation. Stateful calls require reset at episode/batch boundaries and
        one execution acknowledgement between successive predictions.
        """
        if past_actions is not None:
            return super().predict_action(
                obs_dict, use_k_tokens=use_k_tokens, temperature=temperature,
                topk=topk, past_actions=past_actions,
            )

        if self._pending_execution_steps is not None:
            raise RuntimeError(
                "Execution feedback is pending. Call record_executed_actions "
                "after execution, or reset at an episode boundary, before "
                "another stateful prediction."
            )
        observation = next(
            (value for value in obs_dict.values() if isinstance(value, torch.Tensor)),
            None,
        )
        if observation is None or observation.ndim < 1 or observation.shape[0] < 1:
            raise ValueError("A nonempty batched tensor observation is required")
        batch_size = observation.shape[0]
        expected = (batch_size, self.past_n, self.action_dim)
        if self._past_buffer is None:
            self._past_buffer = torch.zeros(expected, device=self.device, dtype=self.dtype)
        elif tuple(self._past_buffer.shape) != expected:
            raise RuntimeError("Execution-history batch size changed; call reset first")
        elif self._past_buffer.device != self.device or self._past_buffer.dtype != self.dtype:
            self._past_buffer = self._past_buffer.to(device=self.device, dtype=self.dtype)

        # Passing history explicitly suppresses the parent's predicted-chunk update.
        result = super().predict_action(
            obs_dict, use_k_tokens=use_k_tokens, temperature=temperature,
            topk=topk, past_actions=self._past_buffer,
        )
        self._pending_execution_steps = result["action"].shape[1]
        return result

    def record_executed_actions(self, actions, executed_lengths=None):
        """Append only commands confirmed as executed, then permit prediction.

        Args:
            actions: Tensor/array of shape (B, T, action_dim), in the same raw
                command coordinates as the training dataset. Values may differ
                from predictions when an execution layer modifies commands.
            executed_lengths: Integer prefix length per batch row, shape (B,).
                Omit only when every supplied command was actually executed.
                Zero leaves that row's history unchanged; unexecuted suffixes
                never contribute to history or its dynamic features.

        All inputs are validated before any history is changed. Acknowledgements
        describe a single proposed chunk, not arbitrary replay or future actions.
        """
        if self._pending_execution_steps is None or self._past_buffer is None:
            raise RuntimeError("No stateful prediction is awaiting execution feedback")

        commands = torch.as_tensor(actions, device=self.device)
        batch_size = self._past_buffer.shape[0]
        if (commands.ndim != 3 or commands.shape[0] != batch_size
                or commands.shape[2] != self.action_dim):
            raise ValueError("actions must have shape (batch_size, steps, action_dim)")
        steps = commands.shape[1]
        if steps > self._pending_execution_steps:
            raise ValueError("Acknowledged chunk exceeds the predicted execution horizon")
        if commands.is_complex() or commands.dtype == torch.bool:
            raise ValueError("Executed commands must contain real numeric values")
        commands = commands.detach().to(dtype=self._past_buffer.dtype)

        if executed_lengths is None:
            lengths = torch.full((batch_size,), steps, device=self.device, dtype=torch.long)
        else:
            lengths = torch.as_tensor(executed_lengths, device=self.device)
            if lengths.shape != (batch_size,) or lengths.dtype == torch.bool or lengths.is_complex():
                raise ValueError("executed_lengths must be an integer vector of shape (batch_size,)")
            if lengths.is_floating_point():
                if not torch.isfinite(lengths).all() or not torch.equal(lengths, lengths.floor()):
                    raise ValueError("executed_lengths must contain finite integer counts")
            if (lengths < 0).any() or (lengths > steps).any():
                raise ValueError("executed_lengths must lie between zero and the supplied chunk length")
            lengths = lengths.to(dtype=torch.long)

        executed = torch.arange(steps, device=self.device)[None, :] < lengths[:, None]
        if not torch.isfinite(commands).all(dim=-1)[executed].all():
            raise ValueError("Executed commands must be finite")

        # Clone outside inference mode so cached commands are ordinary detached
        # tensors even when acknowledgement is called inside inference_mode().
        with torch.inference_mode(False), torch.no_grad():
            history = self._past_buffer.detach().clone()
            for index, count in enumerate(lengths.cpu().tolist()):
                if count:
                    history[index] = torch.cat(
                        (history[index], commands[index, :count]), dim=0,
                    )[-self.past_n:]
        self._past_buffer = history
        self._pending_execution_steps = None
