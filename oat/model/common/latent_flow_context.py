"""Flow layouts layered over the existing ContextBatch without changing AR."""
from oat.model.common.context_batch import ContextBatch, Segment

class FlowContextBatch(ContextBatch):
    def validate_variant(self, variant, num_summary_tokens=None):
        self.validate()
        layouts = {'p2n_latent_flow': 'plain', 'p2n_state_gate_latent_flow': 'state_gate'}
        if variant not in layouts:
            raise ValueError(f'Unknown flow variant: {variant}')
        count = int(self.segment_mask(Segment.HISTORY_SUMMARY).sum())
        metadata = (self.observation_summary, self.history_summary_pool,
                    self.history_valid_fraction, self.history_log_gate)
        if layouts[variant] == 'plain' and (count or any(x is not None for x in metadata)):
            raise ValueError('Plain flow context cannot contain history summaries or gate metadata')
        if layouts[variant] == 'state_gate' and (not count or any(x is None for x in metadata)):
            raise ValueError('Gate flow context requires summaries and gate metadata')
        if num_summary_tokens is not None and count != num_summary_tokens:
            raise ValueError('Incorrect number of history summaries')
        return self
