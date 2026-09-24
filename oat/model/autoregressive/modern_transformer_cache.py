"""Modern action decoder for p2n_new and p2n_state_gate_new.

Both cached and full execution share all projections, QK normalization and
attention-mask handling. Caches belong to a single caller-owned generation
session, never to the module or its state dict.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from oat.model.common.context_batch import ContextBatch

KeyValue = Tuple[Tensor, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, n_emb: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(n_emb, ffn_dim, bias=False)
        self.up = nn.Linear(n_emb, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, n_emb, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Attention(nn.Module):
    """Bias-free multi-head attention with per-head RMS QK normalization."""
    def __init__(self, n_emb: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if n_emb <= 0 or n_head <= 0 or n_emb % n_head:
            raise ValueError("n_emb must be positive and divisible by positive n_head.")
        self.n_emb, self.n_head, self.head_dim = n_emb, n_head, n_emb // n_head
        self.q_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.kv_proj = nn.Linear(n_emb, 2 * n_emb, bias=False)
        self.out_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = dropout

    def _heads(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.n_head, self.head_dim).transpose(1, 2)

    def project_kv(self, memory: Tensor) -> KeyValue:
        key, value = self.kv_proj(memory).chunk(2, dim=-1)
        return self.k_norm(self._heads(key)), self._heads(value)

    def forward(
        self, x: Tensor, memory: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None, causal: bool = False,
        past_key_value: Optional[KeyValue] = None,
        memory_key_value: Optional[KeyValue] = None, use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KeyValue]]:
        if past_key_value is not None and memory_key_value is not None:
            raise ValueError("An attention operation cannot use both self and cross caches.")
        query = self.q_norm(self._heads(self.q_proj(x)))
        key, value = memory_key_value if memory_key_value is not None else self.project_kv(x if memory is None else memory)
        previous_length = 0
        if past_key_value is not None:
            previous_length = past_key_value[0].shape[-2]
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
        bias = attention_bias.to(dtype=query.dtype) if attention_bias is not None else None
        # SDPA's is_causal aligns to the upper-left corner. Cached chunks need
        # the absolute query offset, including chunks longer than one token.
        use_causal_kernel = causal and previous_length == 0 and bias is None
        if causal and not use_causal_kernel:
            query_positions = previous_length + torch.arange(x.shape[1], device=x.device)
            key_positions = torch.arange(key.shape[-2], device=x.device)
            future = key_positions[None, :] > query_positions[:, None]
            causal_bias = torch.zeros(future.shape, device=x.device, dtype=query.dtype).masked_fill(future, float("-inf"))
            bias = causal_bias if bias is None else bias + causal_bias
        result = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, is_causal=use_causal_kernel,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        result = result.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], self.n_emb)
        return self.dropout(self.out_proj(result)), (key, value) if use_cache else None


class ModernTransformerBlock(nn.Module):
    def __init__(self, n_emb: int, n_head: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_norm, self.cross_norm, self.ffn_norm = RMSNorm(n_emb), RMSNorm(n_emb), RMSNorm(n_emb)
        self.self_attn = Attention(n_emb, n_head, dropout)
        self.cross_attn = Attention(n_emb, n_head, dropout)
        self.ffn = SwiGLU(n_emb, ffn_dim, dropout)

    def forward(
        self, x: Tensor, memory: Optional[Tensor], attention_bias: Optional[Tensor] = None,
        causal: bool = True, past_key_value: Optional[KeyValue] = None,
        memory_key_value: Optional[KeyValue] = None, use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KeyValue]]:
        residual, present = self.self_attn(self.self_norm(x), causal=causal,
                                         past_key_value=past_key_value, use_cache=use_cache)
        x = x + residual
        residual, _ = self.cross_attn(self.cross_norm(x), memory, attention_bias=attention_bias,
                                     memory_key_value=memory_key_value)
        x = x + residual
        return x + self.ffn(self.ffn_norm(x)), present


@dataclass(frozen=True)
class DecoderCache:
    self_key_values: Tuple[Optional[KeyValue], ...]
    cross_key_values: Tuple[KeyValue, ...]
    attention_bias: Tensor
    position: int


class ModernAutoregressiveModel(nn.Module):
    """16 x 768 action transformer, configurable to small shapes for tests.

    Initialization: Gaussian std=0.02 for embeddings/input projections; all
    three residual outputs per block use std=0.02/sqrt(3 * n_layer).
    ``max_seq_len`` counts action targets: position zero holds the BOS input.
    """
    def __init__(
        self, vocab_size: int, max_seq_len: int = 8, n_layer: int = 16,
        n_emb: int = 768, n_head: int = 12, ffn_dim: int = 2048,
        dropout: float = 0.0, activation_checkpointing: bool = False,
    ):
        super().__init__()
        if vocab_size < 2 or max_seq_len < 1 or n_layer < 1 or ffn_dim < 1:
            raise ValueError("Decoder requires an action vocabulary and positive sequence/layer/FFN sizes.")
        self.vocab_size, self.bos_id = vocab_size, vocab_size - 1
        self.max_seq_len, self.n_layer = max_seq_len, n_layer
        self.n_emb, self.n_head, self.ffn_dim = n_emb, n_head, ffn_dim
        self.activation_checkpointing = activation_checkpointing
        self.tok_emb = nn.Embedding(vocab_size, n_emb)
        self.slot_emb = nn.Parameter(torch.empty(1, max_seq_len, n_emb))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ModernTransformerBlock(n_emb, n_head, ffn_dim, dropout) for _ in range(n_layer)])
        self.final_norm = RMSNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._initialize()

    def _initialize(self) -> None:
        seen = set()
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and id(module.weight) not in seen:
                nn.init.normal_(module.weight, std=0.02)
                seen.add(id(module.weight))
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        nn.init.normal_(self.slot_emb, std=0.02)
        residual_std = 0.02 / math.sqrt(3 * self.n_layer)
        for block in self.blocks:
            for projection in (block.self_attn.out_proj, block.cross_attn.out_proj, block.ffn.down):
                nn.init.normal_(projection.weight, std=residual_std)

    def _embed(self, tokens: Tensor, position: int = 0) -> Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("Action tokens must be int64 [B, T].")
        if tokens.shape[1] < 1 or position < 0 or position + tokens.shape[1] > self.max_seq_len:
            raise ValueError("Action token positions exceed the configured OAT latent length.")
        return self.dropout(self.tok_emb(tokens) + self.slot_emb[:, position:position + tokens.shape[1]])

    def _validate_context(self, context: ContextBatch, batch_size: Optional[int] = None) -> None:
        context.validate()
        if context.memory.shape[-1] != self.n_emb:
            raise ValueError(f"Decoder expects context width {self.n_emb}.")
        if batch_size is not None and context.memory.shape[0] != batch_size:
            raise ValueError("Tokens and context must have the same batch size.")
        if context.memory.device != self.tok_emb.weight.device:
            raise ValueError("Context and decoder must be on the same device.")

    def forward(self, tokens: Tensor, context: ContextBatch) -> Tensor:
        self._validate_context(context, tokens.shape[0])
        x = self._embed(tokens)
        memory, bias = context.sanitized_memory(), context.attention_bias()
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                # Bind each block so backward recomputation cannot use the last
                # loop iteration's module. Preserve dropout RNG (default).
                def run(value, condition, mask, current_block=block):
                    return current_block(value, condition, attention_bias=mask)[0]
                x = checkpoint(run, x, memory, bias, use_reentrant=False)
            else:
                x, _ = block(x, memory, attention_bias=bias)
        return self.head(self.final_norm(x))

    @torch.no_grad()
    def prefill(self, tokens: Tensor, context: ContextBatch) -> Tuple[Tensor, DecoderCache]:
        """Start an independent cache; logits include every prefill position."""
        self._validate_context(context, tokens.shape[0])
        memory = context.sanitized_memory()
        cache = DecoderCache(
            self_key_values=(None,) * self.n_layer,
            cross_key_values=tuple(block.cross_attn.project_kv(memory) for block in self.blocks),
            attention_bias=context.attention_bias(), position=0,
        )
        return self.decode(tokens, cache)

    @torch.no_grad()
    def decode(self, tokens: Tensor, cache: DecoderCache) -> Tuple[Tensor, DecoderCache]:
        """Append one or more teacher-forcing tokens using absolute positions."""
        if len(cache.self_key_values) != self.n_layer or len(cache.cross_key_values) != self.n_layer:
            raise ValueError("Decoder cache layer count does not match this model.")
        if tokens.shape[0] != cache.attention_bias.shape[0]:
            raise ValueError("Token batch size does not match the generation cache.")
        x = self._embed(tokens, cache.position)
        presents = []
        for block, past, cross in zip(self.blocks, cache.self_key_values, cache.cross_key_values):
            x, present = block(x, None, attention_bias=cache.attention_bias,
                               past_key_value=past, memory_key_value=cross, use_cache=True)
            presents.append(present)
        return self.head(self.final_norm(x)), DecoderCache(
            tuple(presents), cache.cross_key_values, cache.attention_bias,
            cache.position + tokens.shape[1],
        )

    @torch.no_grad()
    def generate(
        self, context: ContextBatch, max_new_tokens: Optional[int] = None,
        temperature: float = 0.0, top_k: Optional[int] = None,
    ) -> Tensor:
        """Return action IDs only; BOS is supplied internally and never sampled.

        Disable autocast's cast cache inside this no-grad operation. Otherwise
        generation inside an outer training autocast scope can retain detached
        weight casts and silently suppress the subsequent training gradients.
        """
        device_type = context.memory.device.type
        with torch.autocast(
            device_type=device_type,
            enabled=torch.is_autocast_enabled(device_type),
            dtype=torch.get_autocast_dtype(device_type),
            cache_enabled=False,
        ):
            return self._generate(context, max_new_tokens, temperature, top_k)

    def _generate(
        self, context: ContextBatch, max_new_tokens: Optional[int],
        temperature: float, top_k: Optional[int],
    ) -> Tensor:
        self._validate_context(context)
        count = self.max_seq_len if max_new_tokens is None else max_new_tokens
        if not isinstance(count, int) or not 0 <= count <= self.max_seq_len:
            raise ValueError("max_new_tokens must be between zero and the OAT latent length.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative.")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer.")
        batch = context.memory.shape[0]
        if count == 0:
            return torch.empty(batch, 0, dtype=torch.long, device=context.memory.device)
        prefix = torch.full((batch, 1), self.bos_id, dtype=torch.long, device=context.memory.device)
        logits, cache = self.prefill(prefix, context)
        generated = []
        for index in range(count):
            scores = logits[:, -1].float().clone()
            scores[:, self.bos_id] = float("-inf")
            if temperature == 0:
                next_token = scores.argmax(-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k is not None:
                    cutoff = scores.topk(min(top_k, self.vocab_size - 1), dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < cutoff, float("-inf"))
                next_token = torch.multinomial(scores.softmax(-1), num_samples=1)
            generated.append(next_token)
            if index + 1 < count:
                logits, cache = self.decode(next_token, cache)
        return torch.cat(generated, dim=1)
