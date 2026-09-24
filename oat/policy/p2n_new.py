"""Base DINOv3 policy: spatial observations and acknowledged command history."""
from oat.policy.p2n_new_common import P2NNewCommonPolicy


class P2NNewPolicy(P2NNewCommonPolicy):
    VARIANT = 'p2n_new'
