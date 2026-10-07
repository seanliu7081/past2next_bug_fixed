"""Architecture specifications for the P2N-VLA backbone (PaliGemma + Gemma action expert)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GemmaSpec:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: Optional[int] = None
    eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self):
        for name in ("width", "depth", "mlp_dim", "num_heads", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"GemmaSpec.{name} must be a positive integer")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.vocab_size is not None and self.vocab_size < 2:
            raise ValueError("vocab_size must be at least two")


@dataclass(frozen=True)
class SiglipSpec:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    patch: int
    image_size: int
    eps: float = 1e-6

    def __post_init__(self):
        if self.image_size % self.patch:
            raise ValueError("image_size must be a multiple of the patch size")
        if self.hidden % self.heads:
            raise ValueError("SigLIP hidden size must be divisible by the head count")

    @property
    def num_tokens(self) -> int:
        return (self.image_size // self.patch) ** 2


GEMMA_2B = GemmaSpec(2048, 18, 16384, 8, 1, 256, vocab_size=257152)
GEMMA_300M = GemmaSpec(1024, 18, 4096, 8, 1, 256)
# Tiny shapes keep the full vocabulary so real PaliGemma token ids and the KI
# row mapping (vocab - 1 - 1152 - code) work unchanged in CPU tests.
TINY_VLM = GemmaSpec(64, 2, 128, 8, 1, 16, vocab_size=257152)
TINY_EXPERT = GemmaSpec(32, 2, 64, 8, 1, 16)

SIGLIP_SO400M = SiglipSpec(1152, 4304, 27, 16, 14, 224)
TINY_SIGLIP = SiglipSpec(32, 64, 2, 4, 14, 28)

MODEL_SIZES = {
    "full": (SIGLIP_SO400M, GEMMA_2B, GEMMA_300M),
    "tiny": (TINY_SIGLIP, TINY_VLM, TINY_EXPERT),
}


def check_joint_compatible(vlm: GemmaSpec, expert: GemmaSpec) -> None:
    """Both streams share one attention per layer, so their geometry must match."""
    for field in ("depth", "num_heads", "num_kv_heads", "head_dim"):
        if getattr(vlm, field) != getattr(expert, field):
            raise ValueError(f"VLM and expert must share {field}: "
                             f"{getattr(vlm, field)} != {getattr(expert, field)}")
    if vlm.vocab_size is None:
        raise ValueError("The VLM spec needs a vocabulary size")
