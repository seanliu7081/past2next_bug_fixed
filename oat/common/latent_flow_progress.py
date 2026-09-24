"""Throttled progress reporting without reading tensors on every batch."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from typing import Any


class FlowProgress:
    """Emit phase progress and evaluate optional metrics only when due.

    Timing uses a monotonic clock and includes work since construction. The
    average throughput of the whole phase determines the ETA. An ETA of None
    means no positive throughput is measurable yet; an empty or finished phase
    has ETA zero. Disabled instances do not call the clock or either callback.
    """

    def __init__(
        self,
        phase: str,
        total: int,
        epoch: int,
        interval: float,
        emit: Callable[[dict[str, Any]], None],
        *,
        unit: str = "batches",
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(total, Integral) or total < 0:
            raise ValueError("total must be a nonnegative integer")
        if not isinstance(interval, Real) or not math.isfinite(interval) or interval < 0:
            raise ValueError("interval must be a finite nonnegative number")
        self.phase = phase
        self.total = int(total)
        self.epoch = epoch
        self.interval = float(interval)
        self.unit = unit
        self.enabled = enabled
        self._emit = emit
        self._clock = clock
        self._started_at = clock() if enabled else None
        self._last_emitted_at = None

    def update(
        self,
        completed: int,
        metrics: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
        force: bool = False,
    ) -> bool:
        if not isinstance(completed, Integral) or completed < 0:
            raise ValueError("completed must be a nonnegative integer")
        if not self.enabled:
            return False
        now = self._clock()
        finished = completed >= self.total
        if not (self._last_emitted_at is None or finished or force
                or now - self._last_emitted_at >= self.interval):
            return False

        elapsed = max(0.0, now - self._started_at)
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = 0.0 if finished else ((self.total - completed) / rate if rate > 0 else None)
        values = metrics() if callable(metrics) else metrics
        payload = dict(values) if values is not None else {}
        # Progress fields remain authoritative if metric names overlap them.
        payload.update(
            phase=self.phase,
            epoch=self.epoch,
            completed=int(completed),
            total=self.total,
            unit=self.unit,
            elapsed_seconds=elapsed,
            rate_per_second=rate,
            eta_seconds=eta,
        )
        self._emit(payload)
        self._last_emitted_at = now
        return True
