"""Shared spatial resampling for every image, preserving learned query slots."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from oat.model.autoregressive.modern_transformer_cache import ModernTransformerBlock, RMSNorm


class VisualResampler(nn.Module):
    def __init__(self, n_emb: int = 768, n_head: int = 12, ffn_dim: int = 2048,
                 num_queries: int = 64, depth: int = 2, grid_size: int = 14,
                 dropout: float = 0.0, activation_checkpointing: bool = False):
        super().__init__()
        if min(n_emb, num_queries, depth, grid_size) <= 0:
            raise ValueError("Resampler dimensions, query count and depth must be positive.")
        self.n_emb = int(n_emb)
        self.num_queries = int(num_queries)
        self.grid_size = int(grid_size)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.queries = nn.Parameter(torch.empty(num_queries, n_emb))
        self.row_position = nn.Parameter(torch.empty(grid_size, n_emb))
        self.column_position = nn.Parameter(torch.empty(grid_size, n_emb))
        self.blocks = nn.ModuleList([ModernTransformerBlock(n_emb, n_head, ffn_dim, dropout)
                                    for _ in range(depth)])
        self.output_norm = RMSNorm(n_emb)
        for embedding in (self.queries, self.row_position, self.column_position):
            nn.init.normal_(embedding, mean=0.0, std=0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or patches.shape[1:] != (self.grid_size ** 2, self.n_emb):
            raise ValueError(f"Projected patches must be [N,{self.grid_size ** 2},{self.n_emb}], got {tuple(patches.shape)}.")
        positions = (self.row_position[:, None, :] + self.column_position[None, :, :]).reshape(-1, self.n_emb)
        memory = patches + positions.to(dtype=patches.dtype)
        query = self.queries.to(dtype=patches.dtype)[None].expand(patches.shape[0], -1, -1)
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                def run(value, source, layer=block):
                    return layer(value, source, causal=False)[0]
                query = checkpoint(run, query, memory, use_reentrant=False, preserve_rng_state=True)
            else:
                query = block(query, memory, causal=False)[0]
        return self.output_norm(query)
