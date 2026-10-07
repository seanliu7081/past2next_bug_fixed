"""Exponential moving average of the trainable parameters only (P2N-VLA, M4).

The deep-copy ``EMAModel`` used by older workspaces duplicates the whole policy;
at PI0.5 scale that is a second 3B-parameter model. :class:`TrainableEMA` keeps a
float32 shadow of exactly the parameters the optimizer trains, de-duplicated by
identity, on the parameters' own device. Frozen backbone weights, normalizers,
buffers and counters are state, never averages, so they are not tracked.

Usage contract (see ``docs/P2N_VLA_IMPLEMENTATION.md``):

- ``step`` is called once per *successful optimizer step*, never per micro-batch.
- ``swap_in`` exposes the averaged values through the live parameters (for
  validation or export) and restores the live values and every module's
  ``training`` flag on exit, even if the body raises. It must run on every
  rank at optimizer-step boundaries and never inside ``torch.inference_mode``.
- ``trainable_override`` maps parameter names to shadows for
  ``policy.artifact_state_dict(trainable_override=...)``.
"""
from __future__ import annotations

import contextlib
import math
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch
from torch import nn

NamedParameters = Iterable[Tuple[str, torch.Tensor]]


class TrainableEMA:
    """``shadow = d * shadow + (1 - d) * param`` over trainable parameters, in fp32.

    ``decay`` is the constant EMA decay ``d``. With ``warmup_power=p`` the decay
    used for update ``n`` (0-based) is ``min(decay, 1 - (1 + n) ** -p)``, so the
    first update copies the parameters; ``None`` keeps ``d`` constant from the
    first update (the shadow starts at the parameters' initial values).
    """

    def __init__(self, named_parameters: NamedParameters, decay: float = 0.999,
                 warmup_power: Optional[float] = None):
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise TypeError("EMA decay must be a real number")
        decay = float(decay)
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        if warmup_power is not None:
            if isinstance(warmup_power, bool) or not isinstance(warmup_power, (int, float)):
                raise TypeError("EMA warmup_power must be a real number or None")
            warmup_power = float(warmup_power)
            if not math.isfinite(warmup_power) or warmup_power <= 0:
                raise ValueError("EMA warmup_power must be positive")
        names, parameters = self._collect(named_parameters)
        if not names:
            raise ValueError("TrainableEMA needs at least one trainable parameter")
        self.decay = decay
        self.warmup_power = warmup_power
        self.updates = 0
        self._names: Tuple[str, ...] = tuple(names)
        with torch.no_grad():
            self.shadow: Dict[str, torch.Tensor] = {
                name: parameter.detach().to(dtype=torch.float32, copy=True).contiguous()
                for name, parameter in zip(names, parameters)
            }
        self._swapped = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _collect(named_parameters: NamedParameters) -> Tuple[List[str], List[torch.Tensor]]:
        """Validate (name, parameter) pairs and de-duplicate tied parameters by identity."""
        if isinstance(named_parameters, Mapping):
            named_parameters = named_parameters.items()
        names: List[str] = []
        parameters: List[torch.Tensor] = []
        by_name: Dict[str, int] = {}
        seen = set()
        for item in named_parameters:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("TrainableEMA expects (name, parameter) pairs")
            name, parameter = item
            if not isinstance(name, str) or not name:
                raise TypeError("Parameter names must be nonempty strings")
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(f"{name} is not a tensor")
            if not parameter.requires_grad:
                raise ValueError(f"TrainableEMA tracks trainable parameters only; {name} is frozen")
            if not parameter.is_floating_point():
                raise ValueError(f"{name} must be a floating-point parameter")
            if name in by_name:
                if by_name[name] != id(parameter):
                    raise ValueError(f"Parameter name {name} refers to two different tensors")
                continue
            by_name[name] = id(parameter)
            if id(parameter) in seen:
                continue  # tied weights: the first name owns the shadow
            seen.add(id(parameter))
            names.append(name)
            parameters.append(parameter)
        return names, parameters

    def _ordered(self, named_parameters: NamedParameters) -> List[torch.Tensor]:
        names, parameters = self._collect(named_parameters)
        if tuple(names) != self._names:
            missing = sorted(set(self._names) - set(names))
            unexpected = sorted(set(names) - set(self._names))
            if missing or unexpected:
                raise ValueError(f"EMA parameter set changed: missing={missing[:5]} unexpected={unexpected[:5]}")
            lookup = dict(zip(names, parameters))
            parameters = [lookup[name] for name in self._names]
        for name, parameter in zip(self._names, parameters):
            shadow = self.shadow[name]
            if parameter.shape != shadow.shape:
                raise ValueError(f"EMA shape mismatch for {name}: {tuple(parameter.shape)} != {tuple(shadow.shape)}")
            if parameter.device != shadow.device:
                raise ValueError(f"EMA device mismatch for {name}: {parameter.device} != {shadow.device}")
        return parameters

    @property
    def names(self) -> Tuple[str, ...]:
        return self._names

    def num_elements(self) -> int:
        return sum(int(tensor.numel()) for tensor in self.shadow.values())

    def current_decay(self) -> float:
        """Decay that the next ``step`` applies."""
        if self.warmup_power is None:
            return self.decay
        return min(self.decay, 1.0 - (1.0 + self.updates) ** (-self.warmup_power))

    # ------------------------------------------------------------------- update
    @torch.no_grad()
    def step(self, named_parameters: NamedParameters) -> float:
        """Fold the current parameter values into the shadow; returns the decay used."""
        if self._swapped:
            raise RuntimeError("The EMA cannot be updated while its weights are swapped in")
        parameters = self._ordered(named_parameters)
        decay = self.current_decay()
        shadows = [self.shadow[name] for name in self._names]
        sources = [parameter.detach() if parameter.dtype == torch.float32 else parameter.detach().float()
                   for parameter in parameters]
        # Literal form of the contract: shadow = d * shadow + (1 - d) * param.
        torch._foreach_mul_(shadows, decay)
        torch._foreach_add_(shadows, sources, alpha=1.0 - decay)
        self.updates += 1
        return decay

    # ---------------------------------------------------------------- swapping
    @contextlib.contextmanager
    def swap_in(self, policy: nn.Module, zero_copy: bool = True) -> Iterator[nn.Module]:
        """Temporarily load the shadow into ``policy``'s parameters.

        On exit the live parameter values and the ``training`` flag of every
        module are restored exactly. With ``zero_copy`` (default) a parameter
        whose dtype, device and shape match its shadow is re-pointed at the
        shadow storage instead of copied, so no second copy of the trainable
        weights is allocated; the body must therefore not modify parameters in
        place. Other parameters are copied and restored from a stash.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError("TrainableEMA.swap_in must not run inside torch.inference_mode")
        if self._swapped:
            raise RuntimeError("EMA weights are already swapped in")
        if not isinstance(policy, nn.Module):
            raise TypeError("swap_in expects the (unwrapped) policy module")
        lookup = dict(policy.named_parameters())
        missing = [name for name in self._names if name not in lookup]
        if missing:
            hint = (" (pass the unwrapped policy, not the DDP wrapper)"
                    if any(("module." + name) in lookup for name in missing) else "")
            raise KeyError(f"Policy lacks EMA parameters {missing[:5]}{hint}")
        modes = [(module, module.training) for module in policy.modules()]
        stash = []
        try:
            with torch.no_grad():
                for name in self._names:
                    parameter, shadow = lookup[name], self.shadow[name]
                    if parameter.shape != shadow.shape:
                        raise ValueError(f"EMA shape mismatch for {name}")
                    if (zero_copy and parameter.dtype == shadow.dtype
                            and parameter.device == shadow.device):
                        stash.append((parameter, parameter.data, True))
                        parameter.data = shadow
                    else:
                        stash.append((parameter, parameter.detach().clone(), False))
                        parameter.copy_(shadow)
            self._swapped = True
            yield policy
        finally:
            with torch.no_grad():
                for parameter, original, pointer in reversed(stash):
                    if pointer:
                        parameter.data = original
                    else:
                        parameter.copy_(original)
            self._swapped = False
            for module, mode in modes:
                module.training = mode

    def trainable_override(self) -> Dict[str, torch.Tensor]:
        """Name -> fp32 shadow for ``artifact_state_dict(trainable_override=...)`` (do not mutate)."""
        return {name: self.shadow[name] for name in self._names}

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {"decay": self.decay, "updates": int(self.updates), "warmup_power": self.warmup_power,
                "shadow": {name: self.shadow[name] for name in self._names}}

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping) -> None:
        for key in ("decay", "updates", "shadow"):
            if key not in state_dict:
                raise KeyError(f"EMA state is missing {key!r}")
        if float(state_dict["decay"]) != self.decay:
            raise ValueError(f"EMA decay differs: saved {state_dict['decay']} != configured {self.decay}")
        saved_power = state_dict.get("warmup_power")
        if (None if saved_power is None else float(saved_power)) != self.warmup_power:
            raise ValueError("EMA warmup_power differs from the saved state")
        updates = state_dict["updates"]
        if isinstance(updates, bool) or int(updates) != updates or int(updates) < 0:
            raise ValueError("EMA update count must be a nonnegative integer")
        shadow = state_dict["shadow"]
        if set(shadow) != set(self._names):
            missing = sorted(set(self._names) - set(shadow))
            unexpected = sorted(set(shadow) - set(self._names))
            raise ValueError(f"EMA shadow names differ: missing={missing[:5]} unexpected={unexpected[:5]}")
        for name in self._names:
            value = shadow[name]
            if not isinstance(value, torch.Tensor) or value.shape != self.shadow[name].shape:
                raise ValueError(f"EMA shadow {name} has the wrong shape")
            if not torch.isfinite(value).all():
                raise ValueError(f"EMA shadow {name} is not finite")
        for name in self._names:
            self.shadow[name].copy_(shadow[name].to(device=self.shadow[name].device, dtype=torch.float32))
        self.updates = int(updates)
