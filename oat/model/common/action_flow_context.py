"""Explicit direct-action context layouts without changing existing families."""

from oat.model.common.context_batch import ContextBatch, Segment


ACTION_FLOW_CONTEXT_LAYOUTS = {
    "p2n_action_flow": "plain",
    "p2n_state_gate_action_flow": "state_gate",
}


class ActionFlowContextBatch(ContextBatch):
    """The existing mask contract, with independently named action variants."""

    def validate_variant(self, variant, num_summary_tokens=None):
        self.validate()
        if variant not in ACTION_FLOW_CONTEXT_LAYOUTS:
            raise ValueError(f"Unknown continuous_action_flow variant: {variant!r}.")
        count = int(self.segment_mask(Segment.HISTORY_SUMMARY).sum())
        metadata = (
            self.observation_summary, self.history_summary_pool,
            self.history_valid_fraction, self.history_log_gate,
        )
        if ACTION_FLOW_CONTEXT_LAYOUTS[variant] == "plain":
            if count or any(value is not None for value in metadata):
                raise ValueError("Plain action flow must not contain summaries or gate metadata.")
        elif count == 0 or any(value is None for value in metadata):
            raise ValueError("State-gate action flow requires summaries and all gate metadata.")
        if num_summary_tokens is not None and count != num_summary_tokens:
            raise ValueError(f"Expected {num_summary_tokens} history summaries, received {count}.")
        return self
