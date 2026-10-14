"""Fully train vision and the P2N action expert while keeping Gemma frozen.

The existing knowledge-insulation boundary is retained: KI token loss trains
SigLIP and its projector through frozen Gemma, while AR loss trains the action
expert. The auxiliary KI token table also remains trainable. No language LoRA
parameters are updated in this mode.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

from torch import nn

from oat.model.vla.pi05_checkpoint import file_sha256
from oat.model.vla.trainable_siglip import TrainableSiglipEncoder
from oat.policy.p2n_vla import P2NVLAPolicy
from oat.policy.p2n_vla_common import P2NVLACommonPolicy, REPO_ROOT
from oat.policy.p2n_vla_state_gate import P2NVLAStateGatePolicy


class _VisionActionTraining:
    """Add the training scope without altering the original policy contracts."""

    ARTIFACT_SCHEMA_VERSION = 2

    def __init__(self, *args, training_scope="vision_action", vision_activation_checkpointing=True, **kwargs):
        if training_scope != "vision_action":
            raise ValueError("This policy requires training_scope='vision_action'")
        lambda_ki = float(kwargs.get("lambda_ki", 1.0))
        if not math.isfinite(lambda_ki) or lambda_ki <= 0:
            raise ValueError("vision_action requires lambda_ki > 0: KI loss supplies the vision gradient")
        if not isinstance(vision_activation_checkpointing, bool):
            raise TypeError("vision_activation_checkpointing must be a bool")
        super().__init__(*args, **kwargs)
        self.training_scope = training_scope
        self.vision_activation_checkpointing = vision_activation_checkpointing
        self.siglip = TrainableSiglipEncoder(
            self.siglip, activation_checkpointing=vision_activation_checkpointing)
        # This includes the adapters inserted by the original policy. Their zero
        # initialized B matrices remain frozen, so fresh runs use the base Gemma.
        self.joint.vlm.requires_grad_(False)
        self._construction.update(training_scope=training_scope,
                                  vision_activation_checkpointing=vision_activation_checkpointing)
        self.train(self.training)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.joint.vlm.eval()
        self.latent_adapter.eval()
        return self

    @contextlib.contextmanager
    def _rollout_mode(self):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            yield
        finally:
            for module, mode in modes:
                module.training = mode
            self.joint.vlm.eval()
            self.latent_adapter.eval()

    @staticmethod
    def _is_pretrained_group(name: str) -> bool:
        # Preserve the workspace's two optimizer groups and learning-rate API.
        return name.startswith("siglip.") or P2NVLACommonPolicy._is_pretrained_group(name)

    @staticmethod
    def _is_ki_clip(name: str) -> bool:
        return name.startswith("siglip.") or P2NVLACommonPolicy._is_ki_clip(name)

    def frozen_base_keys(self):
        # Trainable vision weights must be serialized, including their EMA.
        # Frozen LoRA remains embedded as in the original artifact contract.
        return [name for name in super().frozen_base_keys() if not name.startswith("siglip.")]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if int(state_dict.get("_artifact_schema", -1)) != self.ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Checkpoint artifact schema does not match the vision_action policy")
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def artifact_metadata(self):
        metadata = super().artifact_metadata()
        metadata.update(format="p2n_vla_artifact_v2", training_scope=self.training_scope,
                        language_trainable=False, vision_gradient_source="ki_loss",
                        vision_activation_checkpointing=self.vision_activation_checkpointing)
        source = Path(__file__).resolve()
        metadata["source_sha256"][str(source.relative_to(REPO_ROOT))] = file_sha256(source)
        return metadata


class P2NVLAVisionActionPolicy(_VisionActionTraining, P2NVLAPolicy):
    """P2N-VLA with full vision/action training and a frozen language model."""


class P2NVLAStateGateVisionActionPolicy(_VisionActionTraining, P2NVLAStateGatePolicy):
    """Vision/action training with the original measured-state history gate."""
