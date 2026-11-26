"""A Past2Next decoder for an arm-token sequence followed by a gripper event."""

import torch
from torch import nn
from torch.nn import functional as F

from oat.model.autoregressive.transformer_cache import AutoregressiveModel


class PRTActionModel(nn.Module):
    def __init__(self, base_model: AutoregressiveModel, grip_vocab_size: int):
        super().__init__()
        # The original readout is tied to tok_emb. Keep that tie for arm logits
        # while exposing the final hidden state to both output heads.
        base_model.head = nn.Identity()
        self.backbone = base_model
        self.grip_head = nn.Linear(base_model.n_emb, grip_vocab_size)
        nn.init.xavier_uniform_(self.grip_head.weight)
        nn.init.zeros_(self.grip_head.bias)

    def _arm_logits(self, hidden):
        # BOS is the final embedding row and is never a valid action label.
        return F.linear(hidden, self.backbone.tok_emb.weight[:-1])

    def _bos(self, cond, bos_id):
        if bos_id != self.backbone.tok_emb.num_embeddings - 1:
            raise ValueError("Expected BOS immediately after the arm vocabulary")
        return torch.full((cond.shape[0], 1), bos_id, device=cond.device, dtype=torch.long)

    def teacher_logits(self, cond, arm_tokens, bos_id):
        tokens = torch.cat((self._bos(cond, bos_id), arm_tokens), dim=1)
        hidden = self.backbone(tokens, cond)
        return self._arm_logits(hidden[:, :-1]), self.grip_head(hidden[:, -1])

    @staticmethod
    def _sample(logits, temperature, top_k):
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits.float() / temperature
        if top_k is not None:
            values, indices = logits.topk(min(top_k, logits.size(-1)), dim=-1)
            rank = torch.multinomial(values.softmax(dim=-1), 1)
            return indices.gather(-1, rank)
        return torch.multinomial(logits.softmax(dim=-1), 1)

    @torch.no_grad()
    def sample_tokens(self, cond, n_arm, bos_id, temperature=0.0, top_k=None):
        """Decode once per position, reusing self- and cross-attention keys/values."""
        base = self.backbone
        if not 1 <= n_arm < base.tok_pos_emb.size(1):
            raise ValueError("Arm token count exceeds the decoder's positional capacity")
        if temperature < 0 or (top_k is not None and top_k < 1):
            raise ValueError("Expected nonnegative temperature and positive top_k")
        token = self._bos(cond, bos_id)
        memory = base.encoder(base.drop(base.cond_emb(cond) + base.cond_pos_emb[:, :cond.size(1)]))
        memory_cache = []
        for block in base.blocks:
            key, value = block.cross_attn.kv_proj(memory).chunk(2, dim=-1)
            heads = (base.n_head, base.n_emb // base.n_head)
            memory_cache.append((key.unflatten(-1, heads).transpose(1, 2),
                                 value.unflatten(-1, heads).transpose(1, 2)))
        past = [None] * len(base.blocks)
        arm_tokens = []
        for position in range(n_arm + 1):
            hidden = base.drop(base.tok_emb(token) + base.tok_pos_emb[:, position:position + 1])
            for i, block in enumerate(base.blocks):
                hidden, past[i] = block(hidden, memory, layer_past=past[i],
                                        memory_kv_cache=memory_cache[i])
            hidden = base.ln_f(hidden[:, 0])
            if position < n_arm:
                token = self._sample(self._arm_logits(hidden), temperature, top_k)
                arm_tokens.append(token)
        grip = self._sample(self.grip_head(hidden), temperature, top_k)
        return torch.cat(arm_tokens, dim=1), grip.squeeze(-1)
