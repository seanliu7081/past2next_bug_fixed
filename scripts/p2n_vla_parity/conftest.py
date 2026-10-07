"""Pytest setup for the PI0.5 parity-reference checks in scripts/p2n_vla_parity.

Registers the P2N-VLA markers, which tests/conftest.py does not cover here, and makes the shared
spec module importable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    for marker, description in (
        ("requires_pi05", "needs the 14.5 GB lerobot/pi05_base weights"),
        ("requires_data", "needs the LIBERO-10 zarr dataset"),
        ("gpu", "needs a CUDA device"),
        ("slow", "slow test (large model load or dataset load)"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")
