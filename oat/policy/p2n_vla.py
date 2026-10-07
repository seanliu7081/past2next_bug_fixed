"""P2N-VLA with past-command conditioning and self-past (no measured state history)."""
from oat.policy.p2n_vla_common import P2NVLACommonPolicy


class P2NVLAPolicy(P2NVLACommonPolicy):
    VARIANT = "p2n_vla"
    VARIANT_CODE = 10
