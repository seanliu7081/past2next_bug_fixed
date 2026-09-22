"""Real-robot state-history windows with training-only normalization.

The sampler retains raw rotation-6D observations and physical units. Pass the
real-robot state_history_keys explicitly; rotation geometry belongs to the policy.
"""

from oat.dataset.real_robot_dataset import _RealRobotNormalization
from oat.dataset.zarr_dataset_with_state_history import ZarrDatasetWithStateHistory


class RealRobotZarrDatasetWithStateHistory(
    _RealRobotNormalization, ZarrDatasetWithStateHistory
):
    """Causal current/previous histories, fixed RGB range, and an isolated holdout."""
