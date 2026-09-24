"""FP32 flow matching and EMA consistency arithmetic, independent of policy code."""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class FlowSchedule:
    time: Tensor
    step_size: Tensor
    fm_indices: Tensor
    ct_indices: Tensor


def _finite(name: str, value: Tensor) -> None:
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating point values.")


def _time(name: str, value: Tensor, batch: int, *, allow_one: bool = True) -> Tensor:
    if value.shape != (batch,):
        raise ValueError(f"{name} must have shape [{batch}], including a singleton batch axis.")
    _finite(name, value)
    value = value.float()
    if (value < 0).any() or ((value > 1) if allow_one else (value >= 1)).any():
        raise ValueError(f"{name} must be in [0, {'1' if allow_one else '1)'}].")
    return value


def validate_training_microbatch(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 4 or batch_size % 4:
        raise ValueError("The 3:1 FM/CT recipe requires each training microbatch to be a multiple of four, at least four.")


def sample_fm_times(batch_size: int, *, device, generator: Optional[torch.Generator] = None) -> Tensor:
    """Exactly 0.999 * Beta(1, 1.5), via its inverse CDF and an explicit RNG."""
    if batch_size < 0:
        raise ValueError("batch_size cannot be negative.")
    uniform = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
    return 0.999 * (1.0 - (1.0 - uniform).pow(1.0 / 1.5))


def sample_flow_schedule(batch_size: int, *, device, generator: Optional[torch.Generator] = None) -> FlowSchedule:
    validate_training_microbatch(batch_size)
    indices = torch.randperm(batch_size, device=device, generator=generator)
    fm_indices, ct_indices = indices[:3 * batch_size // 4], indices[3 * batch_size // 4:]
    time = torch.empty(batch_size, device=device, dtype=torch.float32)
    dt = torch.zeros_like(time)
    time[fm_indices] = sample_fm_times(fm_indices.numel(), device=device, generator=generator)
    time[ct_indices] = torch.randint(0, 10, (ct_indices.numel(),), device=device, generator=generator).float() / 10.0
    dt[ct_indices] = torch.rand(ct_indices.numel(), device=device, dtype=torch.float32, generator=generator)
    return FlowSchedule(time, dt, fm_indices, ct_indices)


def _latents(z1: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
    if z1.ndim != 3 or noise.shape != z1.shape or z1.device != noise.device:
        raise ValueError("Target and noise must have identical [B, slots, code_dim] shapes and devices.")
    _finite("Target latents", z1)
    _finite("Noise", noise)
    return z1.float(), noise.float()


def interpolate_latents(z1: Tensor, noise: Tensor, time: Tensor) -> Tensor:
    z1, noise = _latents(z1, noise)
    t = _time("time", time, z1.shape[0])[:, None, None]
    with torch.autocast(device_type=z1.device.type, enabled=False):
        return (1.0 - t) * noise + t * z1


def fm_velocity_target(z1: Tensor, noise: Tensor) -> Tensor:
    z1, noise = _latents(z1, noise)
    with torch.autocast(device_type=z1.device.type, enabled=False):
        return (z1 - noise).detach()


@torch.no_grad()
def consistency_velocity_target(z1: Tensor, noise: Tensor, time: Tensor, step_size: Tensor,
                                teacher_velocity: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]) -> Tensor:
    """Compute CT targets, calling teacher only for rows with ``t_next < 1``.

    The callback takes ``(z_next, t_next, original_dt, row_indices)``. Its row
    indices select from this function's input batch, so the caller can select
    the matching EMA context without sharing the student's trainable features.
    """
    z1, noise = _latents(z1, noise)
    t = _time("CT time", time, z1.shape[0], allow_one=False)
    dt = _time("CT step_size", step_size, z1.shape[0], allow_one=False)
    with torch.autocast(device_type=z1.device.type, enabled=False):
        t_next = torch.minimum(t + dt, torch.ones_like(t))
        z_t = interpolate_latents(z1, noise, t)
        z_next = interpolate_latents(z1, noise, t_next)
        endpoint = z1.clone()
        active = torch.nonzero(t_next < 1, as_tuple=False).flatten()
    if active.numel():
        # Preserve surrounding network autocast; only the arithmetic is FP32.
        v_next = teacher_velocity(z_next[active], t_next[active], dt[active], active)
        if v_next.shape != z_next[active].shape:
            raise ValueError("Teacher velocity has the wrong latent shape.")
        _finite("Teacher velocity", v_next)
        with torch.autocast(device_type=z1.device.type, enabled=False):
            endpoint[active] = z_next[active] + (1.0 - t_next[active, None, None]) * v_next.float()
    with torch.autocast(device_type=z1.device.type, enabled=False):
        target = (endpoint - z_t) / (1.0 - t[:, None, None])
    _finite("CT target", target)
    return target.detach()


def packed_flow_loss(prediction: Tensor, velocity_target: Tensor, fm_indices: Tensor, ct_indices: Tensor) -> dict:
    if prediction.ndim != 3 or prediction.shape != velocity_target.shape:
        raise ValueError("Prediction and velocity_target must have matching [B, slots, code_dim] shapes.")
    _finite("Predicted velocity", prediction)
    _finite("Velocity target", velocity_target)
    batch = prediction.shape[0]
    for name, rows in (("FM", fm_indices), ("CT", ct_indices)):
        if rows.ndim != 1 or rows.dtype != torch.long or rows.numel() == 0:
            raise ValueError(f"{name} indices must be a nonempty int64 vector.")
        if rows.device != prediction.device:
            raise ValueError("Loss row indices must be on the prediction device.")
    rows = torch.cat((fm_indices, ct_indices)).sort().values
    if not torch.equal(rows, torch.arange(batch, device=prediction.device)):
        raise ValueError("FM and CT rows must partition the entire batch exactly once.")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        error = (prediction.float() - velocity_target.detach().float()).square().mean(dim=(1, 2))
        fm_loss, ct_loss = error[fm_indices].mean(), error[ct_indices].mean()
        loss = fm_loss + ct_loss
    return {"loss": loss, "fm_loss": fm_loss, "ct_loss": ct_loss}
