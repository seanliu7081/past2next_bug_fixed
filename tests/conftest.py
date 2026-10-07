"""Pytest markers for the P2N-VLA suites. Unmarked existing tests are unaffected."""


def pytest_configure(config):
    for marker, description in (
        ("requires_pi05", "needs the 14.5 GB lerobot/pi05_base weights"),
        ("requires_data", "needs the LIBERO-10 zarr dataset"),
        ("gpu", "needs a CUDA device"),
        ("slow", "slow test (large model load or dataset load)"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")
