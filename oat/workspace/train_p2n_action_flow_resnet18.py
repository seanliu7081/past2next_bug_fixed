"""ResNet-18 artifact guard over the shared direct-action-flow training loop."""
from __future__ import annotations

from oat.workspace.train_p2n_action_flow import TrainP2NActionFlowWorkspace


def _validate_resnet_policy(config):
    variant = config.get("variant")
    targets = {
        "p2n_action_flow": "oat.policy.p2n_action_flow_resnet18.P2NActionFlowResNet18Policy",
        "p2n_state_gate_action_flow": "oat.policy.p2n_state_gate_action_flow_resnet18.P2NStateGateActionFlowResNet18Policy",
    }
    if config.get("obs_encoder_type") != "resnet18" or config.get("_target_") != targets.get(variant):
        raise ValueError("ResNet-18 workspace requires a matching resnet18 observation encoder and policy target")


class TrainP2NActionFlowResNet18Workspace(TrainP2NActionFlowWorkspace):
    """Retain FM/CT, normalization, EMA/DDP and continuation behavior unchanged."""

    def __init__(self, cfg, output_dir=None, lazy_instantiation=True):
        _validate_resnet_policy(cfg.policy)
        super().__init__(cfg, output_dir=output_dir, lazy_instantiation=lazy_instantiation)

    @staticmethod
    def validate_resume_payload(payload, cfg):
        _validate_resnet_policy(cfg.policy)
        saved = payload.get("cfg", {})
        _validate_resnet_policy(saved.get("policy", {}))
        _validate_resnet_policy(payload.get("policy_config", {}))
        if payload.get("metadata", {}).get("obs_encoder_type") != "resnet18":
            raise ValueError("Resume observation encoder must be resnet18; DINO artifacts are incompatible")
        return TrainP2NActionFlowWorkspace.validate_resume_payload(payload, cfg)
