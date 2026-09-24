"""Capability-based runner adapters for flow; execution feedback is unchanged."""
from oat.env_runner.p2n_new_runner import P2NNewLiberoRunner, P2NStateGateNewLiberoRunner

class P2NActionFlowLiberoRunner(P2NNewLiberoRunner):
    pass

class P2NStateGateActionFlowLiberoRunner(P2NStateGateNewLiberoRunner):
    pass
