"""LIBERO runner with measured-state history.

The simulator-independent collector, vector-env enrichment and runner mixin live in
``oat.env_runner.state_history_common`` (importable without LIBERO); they are re-exported here.
"""

from oat.env_runner.executed_action_runner import LiberoExecutedPastRunner
from oat.env_runner.state_history_common import (  # noqa: F401
    DEFAULT_STATE_HISTORY_KEYS,
    STATE_HISTORY_PREFIX,
    STATE_HISTORY_VALID,
    LowDimStateHistoryWrapper,
    StateHistoryInitializer,
    StateHistoryVectorEnv,
    _snapshot_state_history,
    _StateHistoryRunnerMixin,
    _validate_history_settings,
)


class LiberoStateHistoryRunner(_StateHistoryRunnerMixin, LiberoExecutedPastRunner):
    """Use continuous measured-state history and confirmed action acknowledgments."""

    def __init__(self, *args, state_history_steps=8,
                 state_history_keys=DEFAULT_STATE_HISTORY_KEYS,
                 protocol="corrected", **kwargs):
        super().__init__(*args, state_history_steps=state_history_steps,
                         state_history_keys=state_history_keys, protocol=protocol, **kwargs)
