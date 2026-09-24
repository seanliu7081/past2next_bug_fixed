"""New policy boundary adapters; existing simulator runners remain unchanged."""
import torch

from oat.common.p2n_new_capabilities import policy_capability
from oat.env_runner.executed_action_runner import LiberoExecutedPastRunner
from oat.env_runner.state_history_runner import LiberoStateHistoryRunner


class ByteObservationPolicy:
    """Undo the old runner's blanket floating cast at its documented byte boundary."""
    def __init__(self, policy):
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def predict_action(self, obs_dict, **kwargs):
        obs = dict(obs_dict)
        shape_meta = self.policy.shape_meta
        for key, spec in shape_meta["obs"].items():
            if spec.get("type") == "rgb":
                value = obs[key]
                if value.dtype != torch.uint8:
                    if not torch.isfinite(value).all() or (value < 0).any() or (value > 255).any() or not torch.equal(value, value.round()):
                        raise ValueError(f"Runner RGB {key} must contain byte-range integer pixels")
                    obs[key] = value.to(torch.uint8)
        if "state_history_valid" in obs:
            valid = obs["state_history_valid"]
            if not ((valid == 0) | (valid == 1)).all():
                raise ValueError("Runner state validity must contain only 0/1 values")
            obs["state_history_valid"] = valid.bool()
        return self.policy.predict_action(obs, **kwargs)


class P2NNewLiberoRunner(LiberoExecutedPastRunner):
    def run(self, policy, **kwargs):
        if not policy_capability(policy, "requires_execution_acknowledgement"):
            raise ValueError("This runner requires execution acknowledgement")
        if policy_capability(policy, "requires_state_history"):
            raise ValueError("State-history policy requires P2NStateGateNewLiberoRunner")
        return super().run(ByteObservationPolicy(policy), **kwargs)


class P2NStateGateNewLiberoRunner(LiberoStateHistoryRunner):
    def run(self, policy, **kwargs):
        if not policy_capability(policy, "requires_execution_acknowledgement") or not policy_capability(policy, "requires_state_history"):
            raise ValueError("Gate runner requires measured state history and execution acknowledgement")
        return super().run(ByteObservationPolicy(policy), **kwargs)
