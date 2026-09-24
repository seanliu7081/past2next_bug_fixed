"""FP32 Euler integration over continuous actions, without output transforms."""

from typing import Optional

import torch
from torch import Tensor


@torch.no_grad()
def euler_sample(model, *, context, num_steps: int = 8,
                 generator: Optional[torch.Generator] = None, initial_noise: Optional[Tensor] = None,
                 use_kv_cache: bool = True) -> Tensor:
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    if model.training:
        raise RuntimeError("Euler generation requires an eval-mode velocity model.")
    batch = context.memory.shape[0]
    shape = (batch, model.horizon, model.action_dim)
    device = context.memory.device
    if initial_noise is None:
        actions = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
    else:
        if initial_noise.shape != shape or initial_noise.device != device or not initial_noise.is_floating_point():
            raise ValueError(f"initial_noise must be floating point {shape} on {device}.")
        actions = initial_noise.float().clone()
    if not torch.isfinite(actions).all():
        raise ValueError("Initial noise contains NaN or Inf.")
    cache = model.build_kv_cache(context) if use_kv_cache else None
    dt = torch.full((batch,), 1.0 / num_steps, dtype=torch.float32, device=device)
    for index in range(num_steps):
        time = torch.full((batch,), index / num_steps, dtype=torch.float32, device=device)
        kwargs = {"time": time, "step_size": dt, "context": context}
        if use_kv_cache:
            kwargs["kv_cache"] = cache
        velocity = model(actions, **kwargs)
        if velocity.shape != shape or not velocity.is_floating_point() or not torch.isfinite(velocity).all():
            raise ValueError("Euler velocity must be finite and match the continuous-action shape.")
        with torch.autocast(device_type=device.type, enabled=False):
            actions = actions + dt[:, None, None] * velocity.float()
        if not torch.isfinite(actions).all():
            raise ValueError("Euler integration produced NaN or Inf.")
    return actions


sample_actions_euler = euler_sample
