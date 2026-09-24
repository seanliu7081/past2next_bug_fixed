"""Direct action flow with trainable ResNet18 and action history."""
from oat.policy.p2n_action_flow_resnet_common import P2NActionFlowResNetCommonPolicy


class P2NActionFlowResNet18Policy(P2NActionFlowResNetCommonPolicy):
    VARIANT = 'p2n_action_flow'
