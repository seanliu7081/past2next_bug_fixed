"""Additive schema for original fused observations; legacy layouts stay intact.

The original ContextBatch remains the attention/cache implementation. Validation
maps the new observation segment onto its existing observation contract in a
separate view, without changing the actual segment IDs or any legacy module.
"""
from enum import IntEnum

import torch

from oat.model.common.context_batch import (
    ContextBatch as _LegacyContextBatch,
    Segment as _LegacySegment,
    build_attention_bias,
)


class Segment(IntEnum):
    VISUAL = int(_LegacySegment.VISUAL)
    PROPRIO = int(_LegacySegment.PROPRIO)
    RAW_ACTION = int(_LegacySegment.RAW_ACTION)
    ACTION_DIFF = int(_LegacySegment.ACTION_DIFF)
    HISTORY_SUMMARY = int(_LegacySegment.HISTORY_SUMMARY)
    FUSED_OBSERVATION = 5


ContextSegment = Segment
CONTEXT_SCHEMA_VERSION = 2
CONTEXT_LAYOUT = "original_fused_v1"


class ContextBatch(_LegacyContextBatch):
    def validate(self) -> "ContextBatch":
        # Keep the authoritative checks in the unchanged shared class. A view
        # lets its observation test recognize fused tokens while its shape,
        # metadata, device, range and validity checks still run verbatim.
        mapped_segments = torch.where(
            self.segment_ids == int(Segment.FUSED_OBSERVATION),
            int(_LegacySegment.VISUAL), self.segment_ids,
        )
        _LegacyContextBatch(
            memory=self.memory,
            valid_mask=self.valid_mask,
            segment_ids=mapped_segments,
            observation_summary=self.observation_summary,
            history_summary_pool=self.history_summary_pool,
            history_valid_fraction=self.history_valid_fraction,
            history_log_gate=self.history_log_gate,
        ).validate()
        return self
