"""Deterministic CPU-only checks for throttled latent-flow progress."""
import pytest

from oat.common.latent_flow_progress import FlowProgress


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.now


def test_first_update_interval_force_and_completion_have_lazy_metrics():
    clock = FakeClock()
    reports = []
    metric_reads = []

    def metrics():
        metric_reads.append(True)
        return {"loss": 1.25}

    progress = FlowProgress("train", 20, 3, 10, reports.append, clock=clock)
    assert reports == []
    clock.now += 2
    assert progress.update(1, metrics)
    first = reports[-1]
    assert first == dict(phase="train", total=20, completed=1, epoch=3,
                         unit="batches", elapsed_seconds=2., rate_per_second=.5,
                         eta_seconds=38., loss=1.25)
    clock.now += 9
    assert not progress.update(4, metrics)
    assert len(metric_reads) == 1
    clock.now += 1
    assert progress.update(6, metrics)
    assert reports[-1]["elapsed_seconds"] == 12
    assert reports[-1]["rate_per_second"] == .5
    assert reports[-1]["eta_seconds"] == 28
    clock.now += 1
    assert progress.update(8, metrics, force=True)
    assert progress.update(20, metrics)
    assert reports[-1]["eta_seconds"] == 0
    assert len(metric_reads) == len(reports) == 4


def test_empty_phase_and_initial_zero_throughput_are_well_defined():
    clock = FakeClock()
    reports = []
    progress = FlowProgress("val", 0, 0, 60, reports.append, unit="samples", clock=clock)
    assert progress.update(0)
    assert reports[-1]["unit"] == "samples"
    assert reports[-1]["elapsed_seconds"] == 0
    assert reports[-1]["rate_per_second"] == 0
    assert reports[-1]["eta_seconds"] == 0

    progress = FlowProgress("train", 10, 0, 60, reports.append, clock=clock)
    assert progress.update(0)
    assert reports[-1]["eta_seconds"] is None
    clock.now += 5
    assert progress.update(0, force=True)
    assert reports[-1]["rate_per_second"] == 0
    assert reports[-1]["eta_seconds"] is None


def test_disabled_rank_never_touches_clock_or_callbacks():
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled progress must not call clock or callbacks")

    progress = FlowProgress("val", 0, 4, 10, forbidden, enabled=False, clock=forbidden)
    assert not progress.update(0, forbidden, force=True)
    progress = FlowProgress("train", 100, 4, 10, forbidden, enabled=False, clock=forbidden)
    assert not progress.update(50, forbidden)
    assert not progress.update(100, forbidden)


@pytest.mark.parametrize("total", [-1, 1.5])
def test_invalid_total_is_rejected(total):
    with pytest.raises(ValueError, match="total"):
        FlowProgress("train", total, 0, 1, lambda _: None)


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf")])
def test_invalid_interval_is_rejected(interval):
    with pytest.raises(ValueError, match="interval"):
        FlowProgress("train", 1, 0, interval, lambda _: None)


@pytest.mark.parametrize("completed", [-1, 1.5])
def test_invalid_completed_is_rejected(completed):
    progress = FlowProgress("train", 10, 0, 1, lambda _: None)
    with pytest.raises(ValueError, match="completed"):
        progress.update(completed)


def test_mapping_metrics_zero_interval_and_completion_beyond_total():
    clock = FakeClock()
    reports = []
    progress = FlowProgress("val", 10, 7, 0, reports.append, clock=clock)
    assert progress.update(0, {"loss": .5, "completed": 999})
    assert reports[-1]["completed"] == 0
    clock.now += 5
    assert progress.update(5)
    assert reports[-1]["rate_per_second"] == 1
    assert reports[-1]["eta_seconds"] == 5
    clock.now += 6
    assert progress.update(11)
    assert reports[-1]["completed"] == 11
    assert reports[-1]["eta_seconds"] == 0
