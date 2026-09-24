"""Uniform-grid, FP32 Euler integration. Projection belongs only at the endpoint."""
from typing import Optional

import torch
from torch import Tensor


@torch.no_grad()
def euler_sample(model, *, context, current_state: Tensor, num_steps: int = 8,
                 generator: Optional[torch.Generator] = None, initial_noise: Optional[Tensor] = None,
                 use_kv_cache: bool = True) -> Tensor:
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    if model.training:
        raise RuntimeError("Euler generation requires an eval-mode velocity model.")
    batch = context.memory.shape[0]
    shape = (batch, model.num_slots, model.code_dim)
    device = context.memory.device
    if initial_noise is None:
        z = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
    else:
        if initial_noise.shape != shape or initial_noise.device != device:
            raise ValueError(f"initial_noise must have shape {shape} on {device}.")
        if not initial_noise.is_floating_point():
            raise ValueError("initial_noise must be floating point.")
        z = initial_noise.float().clone()
    if not torch.isfinite(z).all():
        raise ValueError("Initial noise contains NaN or Inf.")
    cache = model.build_kv_cache(context) if use_kv_cache else None
    dt = torch.full((batch,), 1.0 / num_steps, dtype=torch.float32, device=device)
    for index in range(num_steps):
        time = torch.full((batch,), index / num_steps, dtype=torch.float32, device=device)
        kwargs = {"time": time, "step_size": dt, "context": context, "current_state": current_state}
        if use_kv_cache:
            kwargs["kv_cache"] = cache
        velocity = model(z, **kwargs)
        if velocity.shape != shape or not torch.isfinite(velocity).all():
            raise ValueError("Euler velocity must be finite and match the latent shape.")
        with torch.autocast(device_type=device.type, enabled=False):
            z = z + dt[:, None, None] * velocity.float()
        if not torch.isfinite(z).all():
            raise ValueError("Euler integration produced NaN or Inf.")
    return z
