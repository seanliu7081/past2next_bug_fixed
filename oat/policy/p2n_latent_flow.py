"""Plain Past2Next latent flow, without history-summary modules."""
from oat.policy.p2n_latent_flow_common import P2NLatentFlowCommonPolicy

class P2NLatentFlowPolicy(P2NLatentFlowCommonPolicy):
    VARIANT = 'p2n_latent_flow'
