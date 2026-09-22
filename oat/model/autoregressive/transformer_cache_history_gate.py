"""Cross-attention history gates for the existing cached autoregressive decoder.

This module deliberately keeps the base model's parameter names and tied token
embedding/head.  The gate network belongs to the policy; this decoder only adds
its log weight to the trailing history-summary positions in cross-attention.
"""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from oat.model.common.module_attr_mixin import ModuleAttrMixin


class HistoryGatedAutoregressiveModel(AutoregressiveModel):
    """An independent copy of a decoder with per-example history attention bias.

    ``history_log_gate`` is a floating tensor of shape ``[B]`` or ``[B, 1]``.
    Its value is added to every attention score for the last
    ``history_token_count`` condition positions, across all heads, layers, and
    action queries.  A value of zero leaves attention unchanged; negative
    infinity completely masks those positions.  The remaining condition tokens
    always retain their original scores.

    A log weight is used rather than multiplying summary vectors: position
    embeddings and attention normalization cannot bypass a hard-closed gate.
    This does not mask any original past-action or command-difference tokens.
    """

    def __init__(self, base_model: AutoregressiveModel, history_token_count: int):
        # Copy the entire module graph together so tied Parameters remain tied.
        # Avoid constructing/reinitializing another randomly initialized model:
        # conversion should neither change the weights nor advance the RNG.
        ModuleAttrMixin.__init__(self)
        if not isinstance(base_model, AutoregressiveModel):
            raise TypeError("base_model must be an AutoregressiveModel")
        if isinstance(history_token_count, bool) or not isinstance(history_token_count, int):
            raise TypeError("history_token_count must be an integer")
        if history_token_count < 1:
            raise ValueError("history_token_count must be positive")
        copied = deepcopy(base_model)
        self.history_token_count = history_token_count
        self.n_layer = copied.n_layer
        self.n_head = copied.n_head
        self.n_emb = copied.n_emb
        self._dummy_variable = copied._dummy_variable
        self.tok_pos_emb = copied.tok_pos_emb
        self.cond_pos_emb = copied.cond_pos_emb
        for name in ("tok_emb", "cond_emb", "drop", "encoder", "blocks", "ln_f", "head"):
            self.add_module(name, getattr(copied, name))
        self.training = copied.training

    @classmethod
    def from_base(cls, base_model: AutoregressiveModel, history_token_count: int):
        """Copy a base decoder without modifying or sharing its parameters."""
        return cls(base_model=base_model, history_token_count=history_token_count)

    def _attention_bias(self, cond, history_log_gate):
        """Build a broadcastable bias without an ``-inf * 0`` operation."""
        if history_log_gate is None:
            return None
        if not isinstance(history_log_gate, torch.Tensor) or not torch.is_floating_point(history_log_gate):
            raise TypeError("history_log_gate must be a floating point Tensor")
        batch_size, memory_length = cond.shape[:2]
        if history_log_gate.ndim == 1:
            history_log_gate = history_log_gate.unsqueeze(-1)
        if history_log_gate.shape != (batch_size, 1):
            raise ValueError("history_log_gate must have shape [B] or [B, 1]")
        if memory_length <= self.history_token_count:
            raise ValueError("conditions must include at least one non-history token")
        history_log_gate = history_log_gate.to(device=cond.device)
        base_bias = history_log_gate.new_zeros(
            (batch_size, memory_length - self.history_token_count)
        )
        history_bias = history_log_gate.expand(batch_size, self.history_token_count)
        return torch.cat((base_bias, history_bias), dim=-1)[:, None, None, :]

    @staticmethod
    def _block_forward(block, x, memory, attention_bias, layer_past=None, memory_kv_cache=None):
        if attention_bias is None:
            return block(x, memory, layer_past=layer_past, memory_kv_cache=memory_kv_cache)

        attn_output, present = block.attn(block.ln_1(x), layer_past=layer_past)
        x = x + attn_output
        cross = block.cross_attn
        query = block.ln_2(x)
        batch_size, query_length, embedding_width = query.shape
        memory_batch, memory_length = memory.shape[:2]
        q = cross.q_proj(query).view(
            batch_size, query_length, cross.n_head, cross.head_dim
        ).transpose(1, 2)
        if memory_kv_cache is None:
            k, v = cross.kv_proj(memory).split(cross.n_emb, dim=-1)
            k, v = (
                tensor.view(memory_batch, memory_length, cross.n_head, cross.head_dim).transpose(1, 2)
                for tensor in (k, v)
            )
        else:
            k, v = memory_kv_cache
        # q may have been autocast independently of the policy's gate network.
        # Casting keeps SDPA dtype requirements while preserving gate gradients.
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias.to(dtype=q.dtype, device=q.device),
            dropout_p=cross.p_attn_dropout if cross.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, query_length, embedding_width)
        x = x + cross.resid_dropout(cross.c_proj(output))
        x = x + block.mlp(block.ln_3(x))
        return x, present

    def forward(self, tokens: torch.LongTensor, cond: torch.Tensor, history_log_gate=None):
        """Predict next-token logits, with differentiable history gate weights."""
        attention_bias = self._attention_bias(cond, history_log_gate)
        token_length, condition_length = tokens.shape[1], cond.shape[1]
        x = self.drop(self.tok_emb(tokens) + self.tok_pos_emb[:, :token_length, :])
        memory = self.drop(self.cond_emb(cond) + self.cond_pos_emb[:, :condition_length, :])
        memory = self.encoder(memory)
        for block in self.blocks:
            x, _ = self._block_forward(block, x, memory, attention_bias)
        return self.head(self.ln_f(x))

    @torch.inference_mode()
    def generate(
        self,
        prefix: torch.LongTensor,
        cond: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        bos_id: Optional[int] = None,
        history_log_gate=None,
    ) -> torch.LongTensor:
        """Generate with the same fixed gate in prefix and incremental KV paths.

        BOS is allowed in the prefix but suppressed from generated tokens, and
        completed rows are filled with EOS exactly as in the original decoder.
        """
        if prefix.ndim != 2 or prefix.shape[1] == 0:
            raise ValueError("prefix must have shape [B, T] with at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        attention_bias = self._attention_bias(cond, history_log_gate)
        if max_new_tokens == 0:
            return prefix

        condition_length = cond.shape[1]
        memory = self.drop(self.cond_emb(cond) + self.cond_pos_emb[:, :condition_length, :])
        memory = self.encoder(memory)
        memory_batch, memory_length = memory.shape[:2]
        memory_kv_cache = []
        for block in self.blocks:
            cross = block.cross_attn
            k, v = cross.kv_proj(memory).split(cross.n_emb, dim=-1)
            k, v = (
                tensor.view(memory_batch, memory_length, cross.n_head, cross.head_dim).transpose(1, 2)
                for tensor in (k, v)
            )
            memory_kv_cache.append((k, v))

        prefix_length = prefix.shape[1]
        x = self.drop(self.tok_emb(prefix) + self.tok_pos_emb[:, :prefix_length, :])
        past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.n_layer
        for index, block in enumerate(self.blocks):
            x, present = self._block_forward(
                block, x, memory, attention_bias, memory_kv_cache=memory_kv_cache[index]
            )
            past_key_values[index] = present
        logits = self.head(self.ln_f(x[:, -1:, :]))

        out_tokens = prefix
        finished = (
            torch.zeros(prefix.shape[0], dtype=torch.bool, device=prefix.device)
            if eos_id is not None else None
        )
        for index in range(max_new_tokens):
            if bos_id is not None:
                logits[..., bos_id] = -float("inf")
            if temperature > 0:
                step_logits = logits.squeeze(1) / temperature
                if top_k is not None:
                    values, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
                    step_logits[step_logits < values[:, [-1]]] = -float("inf")
                next_token = torch.multinomial(F.softmax(step_logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)

            if eos_id is not None:
                next_token = torch.where(
                    finished.view(-1, 1), torch.full_like(next_token, eos_id), next_token
                )
                finished = finished | (next_token.squeeze(-1) == eos_id)
            out_tokens = torch.cat((out_tokens, next_token), dim=1)
            if (eos_id is not None and finished.all()) or index == max_new_tokens - 1:
                break

            current_position = prefix_length + index
            x = self.drop(
                self.tok_emb(next_token) + self.tok_pos_emb[:, current_position:current_position + 1, :]
            )
            for layer_index, block in enumerate(self.blocks):
                x, present = self._block_forward(
                    block,
                    x,
                    memory,
                    attention_bias,
                    layer_past=past_key_values[layer_index],
                    memory_kv_cache=memory_kv_cache[layer_index],
                )
                past_key_values[layer_index] = present
            logits = self.head(self.ln_f(x))
        return out_tokens
