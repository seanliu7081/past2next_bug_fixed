"""Policy adapter shared by the P2N runners; importable without any simulator."""
import torch


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
