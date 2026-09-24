"""Explicit capabilities for new policies, with conservative legacy fallbacks."""
import inspect
import math

import torch


def policy_capability(policy, name):
    value = getattr(policy, name, None)
    if value is not None:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool or None")
        return value
    lineage = {cls.__name__ for cls in type(policy).__mro__}
    past = "Past2NextPolicy" in lineage
    if name == "supports_explicit_past_actions":
        return past or _accepts(policy.predict_action, "past_actions")
    if name == "supports_explicit_past_action_valid":
        return _accepts(policy.predict_action, "past_action_valid")
    if name == "supports_generated_history_validation":
        return "Past2NextSelfPastPolicy" in lineage or _accepts(policy.forward, "history_mode")
    if name == "requires_execution_acknowledgement":
        return callable(getattr(policy, "record_executed_actions", None))
    if name == "requires_state_history":
        return "Past2NextStateHistoryPolicy" in lineage
    if name == "supports_history_summary_gate":
        return "Past2NextStateHistoryGatePolicy" in lineage
    raise KeyError(name)


def _accepts(method, name):
    # **kwargs alone does not confirm that a legacy method supports a new field.
    try:
        return name in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def predict_validation_action(policy, batch):
    kwargs = {}
    if policy_capability(policy, "supports_explicit_past_actions"):
        kwargs["past_actions"] = batch["past_action"]
        if policy_capability(policy, "supports_explicit_past_action_valid"):
            kwargs["past_action_valid"] = batch["past_action_valid"]
    return policy.predict_action(batch["obs"], **kwargs)


def validate_history_batch(policy, batch):
    """Validate metadata before training or generating any previous window."""
    if not policy_capability(policy, "supports_explicit_past_action_valid"):
        return
    for prefix, obs_key in (("", "obs"), ("prev_", "prev_obs")):
        action = batch[prefix + "past_action"]
        valid = batch[prefix + "past_action_valid"]
        if valid.dtype != torch.bool or valid.shape != action.shape[:2]:
            raise ValueError(f"{prefix}past_action_valid must be bool [B, past_n]")
        if policy_capability(policy, "requires_state_history"):
            state_valid = batch[obs_key]["state_history_valid"]
            if state_valid.dtype != torch.bool or state_valid.shape != (action.shape[0], action.shape[1] + 1):
                raise ValueError("state_history_valid must be bool [B, past_n + 1]")
            transition = state_valid[:, :-1] & state_valid[:, 1:]
            if not torch.equal(valid, transition):
                raise ValueError(f"{prefix}action validity disagrees with aligned measured state transitions")


def resolve_update_schedule(batches_per_rank, epochs, accumulation, *, max_train_steps=None,
                            warmup_steps=None, warmup_ratio=0.05):
    """Count tail accumulation groups once per epoch, after distributed sharding."""
    if min(int(batches_per_rank), int(epochs), int(accumulation)) < 1:
        raise ValueError("Training batches, epochs and accumulation must be positive")
    batches = int(batches_per_rank)
    if max_train_steps is not None:
        if int(max_train_steps) < 1:
            raise ValueError("max_train_steps must be positive")
        batches = min(batches, int(max_train_steps))
    updates_per_epoch = math.ceil(batches / int(accumulation))
    total = updates_per_epoch * int(epochs)
    if warmup_steps is None:
        if not 0 <= float(warmup_ratio) <= 1:
            raise ValueError("lr_warmup_ratio must be in [0, 1]")
        warmup_steps = math.ceil(total * float(warmup_ratio))
    if not 0 <= int(warmup_steps) <= total:
        raise ValueError("lr_warmup_steps must be between zero and planned updates")
    return {"batches_per_rank": batches, "updates_per_epoch": updates_per_epoch,
            "planned_optimizer_updates": total, "lr_warmup_steps": int(warmup_steps)}
