"""LIBERO and RoboCasa executed-action runners.

The simulator-independent adapter and mixin live in
``oat.env_runner.executed_action_common`` (importable without LIBERO, e.g. with robosuite 1.5);
they are re-exported here for existing imports.
"""

from oat.env_runner.executed_action_common import ExecutedActionVectorEnv, _ExecutedPastRunnerMixin  # noqa: F401
from oat.env_runner.libero_runner import LiberoRunner
from oat.env_runner.robocasa_multitask_runner import RoboCasaMultiTaskRunner


class LiberoExecutedPastRunner(_ExecutedPastRunnerMixin, LiberoRunner):
    """LIBERO runner with confirmed action history and corrected/official resets."""

    def __init__(self, *args, protocol="corrected", **kwargs):
        if protocol not in ("corrected", "official"):
            raise ValueError("Executed-action LIBERO requires corrected or official protocol")
        super().__init__(*args, protocol=protocol, **kwargs)


class RoboCasaExecutedPastRunner(_ExecutedPastRunnerMixin, RoboCasaMultiTaskRunner):
    """Reuse the corrected RoboCasa loop with executed-action acknowledgement."""
