"""Gemma VLM with complete native autoregressive K/V caching and no expert.

The legacy GemmaJoint prefix cache intentionally excludes its KI suffix. This
module retains every physical row, including masked history/padding, and is a
separate path so existing flow/expert checkpoints and attention stay unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from oat.model.vla.gemma_joint import GemmaLayer, GemmaStack, KeyValue, apply_rope, rope_cos_sin
from oat.model.vla.native_ar_layout import NativeARLayout
from oat.model.vla.specs import GemmaSpec


@dataclass(frozen=True)
class VLMCache:
    kv: Tuple[KeyValue, ...]  # per layer, post-RoPE [B, kv_heads, physical_length, head_dim]
    length: int
    valid: Tensor            # [B, physical_length]


@dataclass(frozen=True)
class VLMOutput:
    hidden: Tensor
    cache: Optional[VLMCache]


class GemmaVLM(nn.Module):
    def __init__(self, vlm: GemmaSpec, frozen_dtype: torch.dtype = torch.bfloat16,
                 activation_checkpointing: bool = True):
        super().__init__()
        if vlm.vocab_size is None:
            raise ValueError("The VLM spec needs a vocabulary size")
        self.vlm_spec = vlm
        self.activation_checkpointing = bool(activation_checkpointing)
        self.vlm = GemmaStack(vlm)
        for name, parameter in self.vlm.named_parameters():
            parameter.requires_grad_(False)
            if not name.endswith("layernorm.weight") and name != "norm.weight":
                parameter.data = parameter.data.to(frozen_dtype)

    def embed_text(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ids must be a long tensor [B, L]")
        embeds = self.vlm.embed_tokens(ids)
        return embeds * torch.tensor(math.sqrt(self.vlm_spec.width), dtype=embeds.dtype, device=embeds.device)

    def _layer(self, layer: GemmaLayer, x: Tensor, cos: Tensor, sin: Tensor,
               bias: Tensor, cached: Optional[KeyValue]):
        spec = self.vlm_spec
        batch, length, _ = x.shape
        h = layer.input_layernorm(x)
        q = layer.self_attn.q_proj(h).view(batch, length, spec.num_heads, spec.head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(h).view(batch, length, spec.num_kv_heads, spec.head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(h).view(batch, length, spec.num_kv_heads, spec.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cached is not None:
            k = torch.cat((cached[0], k), dim=2)
            v = torch.cat((cached[1], v), dim=2)
        key_length = k.shape[2]
        if spec.num_kv_heads == 1:
            keys = k.expand(batch, spec.num_heads, key_length, spec.head_dim)
            values = v.expand(batch, spec.num_heads, key_length, spec.head_dim)
        else:
            repeat = spec.num_heads // spec.num_kv_heads
            keys, values = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=bias.to(q.dtype))
        out = out.transpose(1, 2).reshape(batch, length, spec.num_heads * spec.head_dim)
        x = x + layer.self_attn.o_proj(out)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x, k, v

    def native_forward(self, embeds: Tensor, layout: NativeARLayout, *,
                       cache: Optional[VLMCache] = None, use_cache: bool = False) -> VLMOutput:
        if embeds.ndim != 3 or embeds.shape[-1] != self.vlm_spec.width:
            raise ValueError(f"VLM embeddings must be [B, Q, {self.vlm_spec.width}]")
        batch, length, _ = embeds.shape
        old_length = cache.length if cache is not None else 0
        key_length = old_length + length
        if (layout.positions.shape != (batch, length)
                or layout.bias.shape != (batch, 1, length, key_length)
                or layout.valid.shape != (batch, key_length)
                or layout.query_start != old_length):
            raise ValueError("Native AR layout does not match the query and cached key lengths")
        if cache is not None:
            if len(cache.kv) != self.vlm_spec.depth or cache.valid.shape != (batch, old_length):
                raise ValueError("Cache layer count or validity shape does not match the VLM")
            if not torch.equal(cache.valid, layout.valid[:, :old_length]):
                raise ValueError("Cached key validity cannot change during decoding")
            expected = (batch, self.vlm_spec.num_kv_heads, old_length, self.vlm_spec.head_dim)
            if any(k.shape != expected or v.shape != expected for k, v in cache.kv):
                raise ValueError("Cached VLM key/value shapes do not match the model")
        cos, sin = rope_cos_sin(layout.positions, self.vlm_spec.head_dim, self.vlm_spec.rope_theta)
        use_checkpoint = self.activation_checkpointing and self.training and torch.is_grad_enabled()
        x, kv = embeds, []
        for index, layer in enumerate(self.vlm.layers):
            cached = None if cache is None else cache.kv[index]
            if use_checkpoint:
                def run(value, current=layer, previous=cached):
                    return self._layer(current, value, cos, sin, layout.bias, previous)
                x, k, v = checkpoint(run, x, use_reentrant=False)
            else:
                x, k, v = self._layer(layer, x, cos, sin, layout.bias, cached)
            if use_cache:
                kv.append((k, v))
        new_cache = VLMCache(tuple(kv), key_length, layout.valid.clone()) if use_cache else None
        return VLMOutput(self.vlm.norm(x), new_cache)

    def forward(self, embeds: Tensor, layout: NativeARLayout, *,
                cache: Optional[VLMCache] = None, use_cache: bool = False) -> VLMOutput:
        return self.native_forward(embeds, layout, cache=cache, use_cache=use_cache)
