"""New policy boundary adapters; existing simulator runners remain unchanged."""
from oat.common.p2n_new_capabilities import policy_capability
from oat.env_runner.byte_observation_policy import ByteObservationPolicy  # noqa: F401
from oat.env_runner.executed_action_runner import LiberoExecutedPastRunner
from oat.env_runner.state_history_runner import LiberoStateHistoryRunner


class P2NNewLiberoRunner(LiberoExecutedPastRunner):
    def run(self, policy, **kwargs):
        if not policy_capability(policy, "requires_execution_acknowledgement"):
            raise ValueError("This runner requires execution acknowledgement")
        if policy_capability(policy, "requires_state_history"):
            raise ValueError("State-history policy requires P2NStateGateNewLiberoRunner")
        return super().run(ByteObservationPolicy(policy), **kwargs)


class P2NStateGateNewLiberoRunner(LiberoStateHistoryRunner):
    def run(self, policy, **kwargs):
        if not policy_capability(policy, "requires_execution_acknowledgement") or not policy_capability(policy, "requires_state_history"):
            raise ValueError("Gate runner requires measured state history and execution acknowledgement")
        return super().run(ByteObservationPolicy(policy), **kwargs)
