"""pi05_p2n: Pi05_past with its flow-matching action head replaced by the Past2Next self-past head.

Everything except the action head is ``Pi05PastPolicy``, inherited unchanged and run with the same config values:
- the PaliGemma prefix and its VLM regime: trainable SigLIP, Gemma LoRA, un-detached K/V scaled by
  ``kv_grad_scale``, no KI loss;
- the Past2Next condition features (``history_tokens``) and their static-memory pass (``history_kv``): self-only
  attention, one shared RoPE position after the chunk, fp32, K/V appended to the VLM K/V;
- self-past (``P2NVLACommonPolicy._maybe_self_past``) and the eval-only history interventions.

The head is P2N-VLA's autoregressive OAT-token head on the same pi05_base Gemma-300M expert:

* Tokens. The frozen OAT maps a 16 x 7 chunk to 8 FSQ indices (5000 codes). Teacher forcing feeds
  ``[BOS, z0..z6]`` through a new tied ``tok_emb`` (5001 x 1024) and trains cross-entropy on ``z0..z7``.
  Deployment and self-past decode the tokens with a KV cache (``P2NVLACommonPolicy._sample``), and the OAT decoder
  turns them back into actions.
* Rows. The 8 token rows are causal and sit at RoPE positions ``pos0 + k``, where Pi05_past's first 8 flow rows
  sat. They read every valid VLM key plus the 9 condition keys, which are at ``pos0 + horizon``.
* Norms. Without a flow time the adaRMS conditioning is a constant, so the pi05 loader folds it at
  ``adarms_t0 = cond_t0`` into per-segment modulations (``LOADER_MODE='p2n'``, as P2N-VLA). The condition pass
  uses ``SEG_RAWDIFF``, the token rows ``SEG_AR``. At initialisation the condition K/V equal Pi05_past's exactly.
* Samplers. ``temperature`` / ``topk`` are the deployment sampler (greedy at 0); ``self_past_temperature`` /
  ``self_past_topk`` are the self-past sampler (null = the deployment sampler, as in the original yaml).

See docs/PI05_P2N.md.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from oat.model.vla.layout import (NEG, SEG_AR, SEG_RAWDIFF, SuffixLayout, _bool_matrix, _validated_pos0,
                                  decode_view, prefill_view)
from oat.model.vla.pi05_checkpoint import file_sha256
from oat.policy.p2n_vla_common import REPO_ROOT
from oat.policy.pi05_past import Pi05PastPolicy


def build_ar_suffix_layout(pos0: Tensor, prefix_valid: Tensor, n_ar: int) -> SuffixLayout:
    """Causal token rows at ``pos0 + k``. Row i sees the valid ``prefix_valid`` keys (VLM, then the condition)
    and token keys j <= i. It is a ``SuffixLayout`` without RAWDIFF/HIST rows, so ``prefill_view`` and
    ``decode_view`` apply."""
    prefix_valid = _bool_matrix(prefix_valid, "prefix_valid")
    batch, nonki_len = prefix_valid.shape
    pos0 = _validated_pos0(pos0, batch)
    if isinstance(n_ar, bool) or not isinstance(n_ar, int) or n_ar < 1:
        raise ValueError("n_ar must be a positive integer")
    device = prefix_valid.device
    positions = pos0[:, None] + torch.arange(n_ar, device=device)[None]
    bias = torch.full((batch, n_ar, nonki_len + n_ar), NEG, dtype=torch.float32, device=device)
    bias[:, :, :nonki_len].masked_fill_(prefix_valid[:, None, :], 0.0)
    bias[:, :, nonki_len:].masked_fill_(torch.ones(n_ar, n_ar, dtype=torch.bool, device=device).tril(), 0.0)
    seg_ids = torch.full((n_ar,), SEG_AR, dtype=torch.long, device=device)
    return SuffixLayout(positions, bias[:, None], seg_ids, 0, 0, n_ar, nonki_len)


class Pi05P2NPolicy(Pi05PastPolicy):
    VARIANT = "pi05_p2n"
    VARIANT_CODE = 14
    NORM_MODE = "segment"
    LOADER_MODE = "p2n"
    HAS_AR_HEAD = True

    def __init__(self, *args, flow_num_steps=None, flow_time_beta=None, **kwargs):
        # Validate the head's knobs before the base class reads 14.5 GB of pi05 weights.
        if flow_num_steps is not None or flow_time_beta is not None:
            raise ValueError("pi05_p2n has no flow head: flow_num_steps and flow_time_beta must be null")
        cond_t0 = float(kwargs.get("cond_t0", 0.6))
        if float(kwargs.setdefault("adarms_t0", cond_t0)) != cond_t0:
            raise ValueError("adarms_t0 (the adaRMS fold time) must equal cond_t0: the condition pass keeps "
                             "Pi05_past's constant time")
        # null = the deployment sampler (the original yaml's meaning). Resolve it here, before Pi05PastPolicy
        # replaces null with the flow head's inert placeholders (T=1, top-k 10), which this head would use.
        for key, source, default in (("self_past_temperature", "temperature", 0.0), ("self_past_topk", "topk", 10)):
            if key in kwargs and kwargs[key] is None:
                kwargs[key] = kwargs.get(source, default)
        for key, default in (("temperature", 0.0), ("self_past_temperature", 1.0)):
            value = kwargs.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative (0 = greedy)")
        for key in ("topk", "self_past_topk"):
            value = kwargs.get(key, 10)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        super().__init__(*args, **kwargs)
        # P2NVLACommonPolicy builds P2N-VLA's RAWDIFF slot embeddings for every AR head with past input. The
        # Pi05_past condition (history_tokens) never reads them, and an unused parameter fails the first-step
        # gradient-coverage check and DDP (find_unused_parameters=False).
        for name in ("type_embedding", "action_time_embedding"):
            if hasattr(self, name):
                delattr(self, name)
        self.flow_num_steps = self.flow_time_beta = None
        for key in ("flow_num_steps", "flow_time_beta"):
            self._construction.pop(key)
        self._last_self_past_mse = None

    # ------------------------------------------------------------------ construction helpers
    def _make_flow_heads(self, width):
        return None  # the time conditioning is folded (load_pi05 mode 'p2n' needs flow_heads=None)

    # ------------------------------------------------------------------ condition (Past2Next)
    def history_kv(self, tokens: Tensor, pos0: Tensor):
        """``Pi05PastPolicy.history_kv`` with the folded condition segment in place of ``flow_condition(cond_t0)``:
        self-only attention, one shared position after the chunk, fp32."""
        batch, length, width = tokens.shape
        spec = self.joint.expert_spec
        if width != spec.width:
            raise ValueError(f"Condition tokens must have width {spec.width}")
        empty = tokens.new_zeros(batch, spec.num_kv_heads, 0, spec.head_dim)
        bias = torch.full((batch, 1, length, length), NEG, dtype=torch.float32, device=tokens.device)
        index = torch.arange(length, device=tokens.device)
        bias[:, :, index, index] = 0.0
        positions = (pos0.long() + self.horizon)[:, None].expand(batch, length)
        seg_ids = torch.full((length,), SEG_RAWDIFF, dtype=torch.long, device=tokens.device)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            out = self.joint.expert_forward(tokens.float(), tuple((empty, empty) for _ in range(spec.depth)), bias,
                                            positions, seg_ids=seg_ids, use_cache=True)
        return out.cache.kv

    # ------------------------------------------------------------------ token head
    def _token_inputs(self, ids: Tensor) -> Tensor:
        return (self.tok_emb(ids) * self.tok_scale).float()

    def ar_logits(self, kv, pos0: Tensor, prefix_valid: Tensor, targets: Tensor,
                  probe_cols: Optional[Tuple[int, int]] = None):
        """Teacher forcing: inputs ``[BOS, z0..z6]``, fp32 logits for ``z0..z7``, and the attention mass on
        ``probe_cols`` averaged over every token row."""
        ids = torch.cat((targets.new_full((targets.shape[0], 1), self.BOS_ID), targets[:, :-1]), dim=1)
        layout = build_ar_suffix_layout(pos0, prefix_valid, ids.shape[1])
        out = self.joint.expert_forward(self._token_inputs(ids), kv, layout.bias, layout.positions,
                                        seg_ids=layout.seg_ids, probe_cols=probe_cols, probe_rows="all")
        return self._head(out.hidden), out.probe_mass

    @torch.no_grad()
    def ar_generate(self, kv, pos0: Tensor, prefix_valid: Tensor, n_tokens: int, temperature: float, topk):
        """KV-cached decoding of the first ``n_tokens`` OAT tokens: ``P2NVLACommonPolicy.generate_tokens`` over the
        joined [VLM | condition] keys."""
        if not 1 <= n_tokens <= self.max_seq_len:
            raise ValueError(f"n_tokens must be in [1, {self.max_seq_len}]")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        layout = build_ar_suffix_layout(pos0, prefix_valid, n_tokens)
        bos = torch.full((pos0.shape[0], 1), self.BOS_ID, dtype=torch.long, device=pos0.device)
        bias, positions, seg_ids = prefill_view(layout, 1)
        out = self.joint.expert_forward(self._token_inputs(bos), kv, bias, positions, seg_ids=seg_ids, use_cache=True)
        segment = torch.tensor([SEG_AR], dtype=torch.long, device=pos0.device)
        generated = []
        for index in range(n_tokens):
            token = self._sample(self._head(out.hidden[:, -1]), temperature, topk, self.BOS_ID)
            generated.append(token)
            if index + 1 < n_tokens:
                bias, position = decode_view(layout, index + 1)
                out = self.joint.expert_forward(self._token_inputs(token[:, None]), kv, bias, position,
                                                seg_ids=segment, cache=out.cache, use_cache=True)
        return torch.stack(generated, dim=1)

    # ------------------------------------------------------------------ self-past
    @torch.no_grad()
    def _maybe_self_past(self, batch, past, probability=None):
        """``P2NVLACommonPolicy._maybe_self_past``, plus the normalized-action MSE between the substituted history
        and the ground truth it replaced (``self_past_hist_mse``: how far the head's own past is from the demos)."""
        history = super()._maybe_self_past(batch, past, probability)
        substituted = (history != past).any(dim=-1)  # [B, past_n]: only valid slots of selected rows change
        if substituted.any():
            normalize = self.action_normalizer["action"].normalize
            with torch.autocast(device_type=past.device.type, enabled=False):
                error = (normalize(history.float()).float() - normalize(past.float()).float()).square().mean(-1)
            self._last_self_past_mse = float(error[substituted].mean())
        return history

    # ------------------------------------------------------------------ training
    def forward(self, batch, history_mode=None):
        self._check_ready()
        self._last_self_past_rows, self._last_self_past_mse = 0, None  # expert passes generate nothing
        past, probability = self._resolve_history(batch, history_mode)
        targets = self.encode_targets(batch["action"])
        if targets.shape != (past.shape[0], self.max_seq_len):
            raise ValueError("OAT returned an unexpected token shape")
        use_ki = self.lambda_ki > 0
        coupled = self.kv_grad_scale > 0
        with self._autocast():
            prefix = self.build_prefix(batch["obs"], train_aug=self.training and self.train_image_aug)
            with contextlib.nullcontext() if (use_ki or coupled) else torch.no_grad():
                prefix_out, layout, ki_logits = self.vlm_pass(prefix, targets if use_ki else None)
            kv, valid = self._expert_kv(prefix_out, layout, past, batch["past_action_valid"])
            probe = (layout.nonki_len, layout.nonki_len + self.n_cond_tokens) if self.use_past else None
            logits, probe_mass = self.ar_logits(kv, prefix_out.pos0, valid, targets, probe_cols=probe)
        loss_ar = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss, loss_ki = loss_ar, None
        if use_ki:
            loss_ki = F.cross_entropy(ki_logits.reshape(-1, ki_logits.shape[-1]).float(), targets.reshape(-1))
            loss = loss_ar + self.lambda_ki * loss_ki
        with torch.no_grad():
            self.last_loss_components = dict(
                loss=loss.detach().float().item(), loss_ar=loss_ar.detach().float().item(), loss_flow=None,
                loss_ki=None if loss_ki is None else loss_ki.detach().float().item(),
                ar_token_acc=(logits.argmax(-1) == targets).float().mean().item(),
                ki_token_acc=None if ki_logits is None else (ki_logits.argmax(-1) == targets).float().mean().item(),
                self_past_p=float(probability), self_past_rows=float(self._last_self_past_rows),
                self_past_hist_mse=self._last_self_past_mse,
                hist_attention_mass=None if probe_mass is None else probe_mass.float().mean().item(),
                gate_mean=None, gate_min=None, gate_max=None)
        return loss

    # ------------------------------------------------------------------ inference (rollout and self-past)
    def _generate_actions(self, obs, past, valid, n_tokens, temperature, topk):
        """The first ``n_tokens`` OAT tokens decoded at ``temperature`` / ``topk`` (greedy at 0), then the frozen
        OAT decoder. Deployment passes the deployment sampler, self-past the self-past sampler."""
        self._check_ready()
        if self._acting and self.history_intervention is not None:
            past, valid = self._intervened_history(past, valid)
        with torch.no_grad(), self._autocast():
            prefix = self.build_prefix(obs, train_aug=False)
            prefix_out, layout, _ = self.vlm_pass(prefix, None)
            kv, prefix_valid = self._expert_kv(prefix_out, layout, past, valid)
            tokens = self.ar_generate(kv, prefix_out.pos0, prefix_valid, n_tokens, temperature, topk)
        return self._detokenize(tokens)

    # ------------------------------------------------------------------ artifacts
    def artifact_metadata(self) -> dict:
        metadata = super().artifact_metadata()
        block = metadata.pop("pi05_past")
        block.pop("flow_num_steps")
        block.update(head="past2next_ar_oat8", tokens=self.max_seq_len, vocab=self.N_CODES + 1,
                     norm_fold_t0=self.cond_t0, temperature=self.temperature, topk=self.topk,
                     self_past_temperature=self.self_past_temperature, self_past_topk=self.self_past_topk)
        metadata["pi05_p2n"] = block
        source = Path(__file__).resolve()
        metadata["source_sha256"][str(source.relative_to(REPO_ROOT))] = file_sha256(source)
        return metadata
