"""Past2Next action-history continuous flow, without state-summary modules."""
from oat.policy.p2n_action_flow_common import P2NActionFlowCommonPolicy


class P2NActionFlowPolicy(P2NActionFlowCommonPolicy):
    VARIANT = 'p2n_action_flow'
