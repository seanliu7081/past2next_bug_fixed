"""Pi05_past: pi05_base's original flow-matching action head plus the Past2Next condition.

The stack is the same-stack pi05 baseline (``PI05KIFlowPolicy``): PaliGemma prefix, the pi05_base Gemma-300M
expert with its unfolded adaRMS time conditioning, ``FlowHeads``, Beta(1.5, 1) flow time and Euler sampling.
Added is the condition of ``oat/config/train_past2next_self_past.yaml`` (``Past2NextPolicy._build_condition``
and ``Past2NextSelfPastPolicy``):

* Condition features. The 7 past commands are normalized with the dataset action normalizer. Zero-padded
  history is normalized as is and never masked, exactly like the original. Then
  ``acc = a[t-1] - a[t-2]`` and ``jerk = a[t-1] - 2 a[t-2] + a[t-3]``. The tokens ``[acc, jerk, raw_0..raw_6]``
  go through Linear-GELU-Linear MLPs (``raw_proj`` shared over the lags), plus a per-slot embedding
  (the original ``cond_pos_emb``, zero-initialised) and dropout in training.
* Static memory. Once per sample, the 9 tokens run through the expert with self-only attention, at a
  constant time condition ``cond_t0`` and one shared RoPE position just after the action chunk. Their
  per-layer keys and values are appended to the VLM prefix keys and values. The unchanged
  ``flow_velocity`` / ``sample_actions`` math then reads ``[prefix | condition]`` keys at every Euler step,
  like the original cross-attention memory that is computed once.
* Self-past. ``P2NVLACommonPolicy._maybe_self_past`` regenerates the history from the previous window with
  the deployment sampler (fresh noise and ``flow_num_steps`` Euler steps).

VLM regime (defaults = pi05 fine-tune style): the flow loss reaches SigLIP and Gemma LoRA through
un-detached per-layer keys and values (``kv_grad_scale=1``), and there is no knowledge-insulation loss
(``lambda_ki=0``). ``kv_grad_scale=0`` restores the insulated (detached) coupling. A fractional scale is applied
in the K/V dtype (bf16 under autocast), so it is exact only for powers of two. See docs/PI05_PAST.md.

Eval-only history diagnostics (``set_history_intervention``) change what the head reads in ``predict_action``:
no history, another environment's history, or noisy history. Training, self-past and validation never see them.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from oat.model.vla.flow_head import FLOW_ACTION_DIM, FlowHeads, flow_condition, sample_actions
from oat.model.vla.layout import NEG, build_flow_suffix_layout
from oat.model.vla.lora import lora_parameters
from oat.model.vla.pi05_checkpoint import PROJECTOR_PREFIX, SIGLIP_PREFIX, default_pi05_path, file_sha256
from oat.model.vla.trainable_siglip import TrainableSiglipEncoder
from oat.policy.p2n_new_common import bool_mask
from oat.policy.p2n_vla_common import P2NVLACommonPolicy, REPO_ROOT

KI_INITS = ("embedding", "random")
HISTORY_INTERVENTIONS = ("zero", "swap", "noise")
# Without the KI block nothing reads the last VLM layer's output, so these LoRA adapters get no gradient.
UNREACHABLE_LAST_LAYER = ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _is_bool(value) -> bool:
    return isinstance(value, bool)


class Pi05PastPolicy(P2NVLACommonPolicy):
    VARIANT = "pi05_past"
    VARIANT_CODE = 13
    NORM_MODE = "adarms"
    LOADER_MODE = "flow"
    HAS_AR_HEAD = False
    supports_generated_history_validation = True
    supports_history_intervention = True
    N_EXPLICIT_FEATURES = 2  # acc, jerk (Past2NextPolicy.N_EXPLICIT_FEATURES)

    def __init__(self, *args, flow_num_steps=10, flow_time_beta=(1.5, 1.0), kv_grad_scale=1.0,
                 train_vision=True, train_lora=True, vision_activation_checkpointing=True,
                 cond_t0=0.6, cond_dropout=0.1, ki_init="embedding", **kwargs):
        # Validate the new knobs before the base class reads 14.5 GB of pi05 weights.
        if isinstance(flow_num_steps, bool) or not isinstance(flow_num_steps, int) or flow_num_steps < 1:
            raise ValueError("flow_num_steps must be a positive integer")
        beta = tuple(float(x) for x in flow_time_beta)
        if len(beta) != 2 or not all(math.isfinite(x) and x > 0 for x in beta):
            raise ValueError("flow_time_beta must be two positive numbers (Beta concentration parameters)")
        kv_grad_scale = float(kv_grad_scale)
        if not math.isfinite(kv_grad_scale) or not 0.0 <= kv_grad_scale <= 1.0:
            raise ValueError("kv_grad_scale must lie in [0, 1] (1 = end to end, 0 = insulated)")
        for name, value in (("train_vision", train_vision), ("train_lora", train_lora),
                            ("vision_activation_checkpointing", vision_activation_checkpointing)):
            if not _is_bool(value):
                raise TypeError(f"{name} must be a bool")
        cond_t0, cond_dropout = float(cond_t0), float(cond_dropout)
        if not 0.0 <= cond_t0 <= 1.0:
            raise ValueError("cond_t0 must lie in [0, 1]")
        if not 0.0 <= cond_dropout < 1.0:
            raise ValueError("cond_dropout must lie in [0, 1)")
        if ki_init not in KI_INITS:
            raise ValueError(f"ki_init must be one of {KI_INITS}")
        lambda_ki = float(kwargs.get("lambda_ki", 1.0))
        if lambda_ki == 0 and kv_grad_scale == 0 and (train_vision or train_lora):
            raise ValueError("kv_grad_scale=0 (insulated) with lambda_ki=0 leaves the trainable SigLIP/LoRA without "
                             "any loss: set kv_grad_scale>0, lambda_ki>0, or train_vision=train_lora=false")
        # The original yaml writes null for these, meaning "the deployment sampler". The flow sampler ignores
        # temperature and top-k, but the VLA base casts them with float()/int(), so give it placeholders.
        for key, placeholder in (("self_past_temperature", 1.0), ("self_past_topk", 10)):
            if key in kwargs and kwargs[key] is None:
                kwargs[key] = placeholder
        self.flow_num_steps = int(flow_num_steps)
        self.flow_time_beta = beta
        self.kv_grad_scale = kv_grad_scale
        self.train_vision = bool(train_vision)
        self.train_lora = bool(train_lora)
        self.history_intervention, self.history_noise_std, self._acting = None, 0.0, False
        construction_mode = kwargs.get("construction_mode", "fresh")
        super().__init__(*args, **kwargs)
        if self.action_dim > FLOW_ACTION_DIM:
            raise ValueError("Action dimension exceeds the PI0.5 padded action width")
        width = self.expert_width
        self.cond_t0 = cond_t0
        self.n_cond_tokens = self.N_EXPLICIT_FEATURES + self.past_n if self.use_past else 0

        # ------------------------------------------------------------ condition modules (Past2NextPolicy)
        if self.use_past:
            def mlp():
                return nn.Sequential(nn.Linear(self.action_dim, width), nn.GELU(), nn.Linear(width, width))
            self.acc_proj, self.jerk_proj, self.raw_proj = mlp(), mlp(), mlp()
            self.cond_pos_emb = nn.Parameter(torch.zeros(self.n_cond_tokens, width))
            self.cond_dropout = nn.Dropout(cond_dropout)

        # ------------------------------------------------------------ trainable scope
        if self.train_vision:
            self.siglip = TrainableSiglipEncoder(self.siglip, activation_checkpointing=vision_activation_checkpointing)
            if self.model_size == "full" and construction_mode == "fresh":
                # The frozen encoder was loaded in bf16; start the fp32 masters from the exact fp32 weights.
                self._reload_vision_fp32(kwargs.get("pi05_weights") or default_pi05_path())
        for parameter in lora_parameters(self.joint):
            parameter.requires_grad_(self.train_lora)
        self.dead_lora = []
        if self.train_lora and self.lambda_ki == 0:
            self.dead_lora = self._freeze_unreachable_lora()
        if self.lambda_ki > 0 and ki_init == "random":
            with torch.no_grad():
                std = float(self.joint.vlm.embed_tokens.weight.float().std())
                nn.init.normal_(self.ki_table.rows, std=std)
        self._construction.update(
            flow_num_steps=self.flow_num_steps, flow_time_beta=list(self.flow_time_beta),
            kv_grad_scale=self.kv_grad_scale, train_vision=self.train_vision, train_lora=self.train_lora,
            vision_activation_checkpointing=bool(vision_activation_checkpointing), cond_t0=self.cond_t0,
            cond_dropout=cond_dropout, ki_init=ki_init)
        self.train(self.training)

    # ------------------------------------------------------------------ construction helpers
    def _make_flow_heads(self, width) -> Optional[FlowHeads]:
        return FlowHeads(width)

    @torch.no_grad()
    def _reload_vision_fp32(self, path) -> None:
        from safetensors import safe_open
        targets = {SIGLIP_PREFIX + name: tensor
                   for name, tensor in self.siglip.vision.state_dict(keep_vars=True).items()}
        for name in ("weight", "bias"):
            targets[PROJECTOR_PREFIX + name] = getattr(self.siglip.projector, name)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            missing = sorted(set(targets) - set(handle.keys()))
            if missing:
                raise KeyError(f"pi05 checkpoint lacks {len(missing)} SigLIP tensors, e.g. {missing[:3]}")
            for key, destination in targets.items():
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != tuple(destination.shape):
                    raise ValueError(f"{key}: checkpoint shape {tuple(tensor.shape)} != {tuple(destination.shape)}")
                destination.data.copy_(tensor.to(device=destination.device, dtype=destination.dtype))

    def _freeze_unreachable_lora(self) -> list:
        """Freeze the last-layer LoRA that no loss reaches without the KI block (only K/V of each layer are read)."""
        layer = self.joint.vlm.layers[-1]
        index = len(self.joint.vlm.layers) - 1
        frozen = []
        for path in UNREACHABLE_LAST_LAYER:
            module = layer.get_submodule(path)
            for name in ("lora_A", "lora_B"):
                getattr(module, name).requires_grad_(False)
                frozen.append(f"joint.vlm.layers.{index}.{path}.{name}")
        return frozen

    def _pad(self, actions: Tensor) -> Tensor:
        return F.pad(actions, (0, FLOW_ACTION_DIM - actions.shape[-1]))

    # ------------------------------------------------------------------ modes
    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if not getattr(self, "train_vision", False):
            self.siglip.eval()  # frozen SigLIP is deterministic in every mode
        self.latent_adapter.eval()
        return self

    @contextlib.contextmanager
    def _rollout_mode(self):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            yield
        finally:
            for module, mode in modes:
                module.training = mode
            if not self.train_vision:
                self.siglip.eval()
            self.latent_adapter.eval()

    # ------------------------------------------------------------------ coupling
    def _detached_kv(self, prefix_out):
        """VLM keys/values for the expert, with the action gradient into the VLM scaled by ``kv_grad_scale``."""
        scale = self.kv_grad_scale
        if scale <= 0.0 or not torch.is_grad_enabled():
            return tuple((key.detach(), value.detach()) for key, value in prefix_out.kv)
        if scale >= 1.0:
            return tuple(prefix_out.kv)
        return tuple((key.detach() + scale * (key - key.detach()), value.detach() + scale * (value - value.detach()))
                     for key, value in prefix_out.kv)

    # ------------------------------------------------------------------ condition (Past2Next)
    def history_tokens(self, past: Tensor, past_valid: Tensor) -> Tensor:
        """``[B, 2 + past_n, W]`` condition tokens: Past2NextPolicy._build_condition's features and order."""
        if (not isinstance(past, Tensor) or past.ndim != 3 or tuple(past.shape[1:]) != (self.past_n, self.action_dim)
                or not past.is_floating_point()):
            raise ValueError(f"past_actions must be floating [B,{self.past_n},{self.action_dim}]")
        valid = bool_mask(past_valid, past.shape[:2], "past_action_valid", past.device)
        if not torch.isfinite(past[valid]).all():
            raise ValueError("Valid historical commands must be finite")
        # Raw zeros at unavailable slots, normalized as is (no masking): identical to the original on the
        # zero-padded dataset, rollout-buffer and self-past paths.
        raw = torch.where(valid[..., None], past, torch.zeros_like(past))
        # The whole condition path runs in fp32 (9 tokens, negligible cost): under autocast every call would
        # save another bf16 copy of the weights it uses for backward.
        with torch.autocast(device_type=past.device.type, enabled=False):
            normalized = self.action_normalizer["action"].normalize(raw.float()).float()
            acc = normalized[:, -1] - normalized[:, -2]
            jerk = normalized[:, -1] - 2.0 * normalized[:, -2] + normalized[:, -3]
            tokens = torch.cat((self.acc_proj(acc)[:, None], self.jerk_proj(jerk)[:, None],
                                self.raw_proj(normalized)), dim=1)
            return self.cond_dropout(tokens.float() + self.cond_pos_emb)

    def history_kv(self, tokens: Tensor, pos0: Tensor):
        """Per-layer expert K/V of the condition tokens: self-only attention, constant time condition."""
        batch, length, width = tokens.shape
        spec = self.joint.expert_spec
        if width != spec.width:
            raise ValueError(f"Condition tokens must have width {spec.width}")
        empty = tokens.new_zeros(batch, spec.num_kv_heads, 0, spec.head_dim)
        bias = torch.full((batch, 1, length, length), NEG, dtype=torch.float32, device=tokens.device)
        index = torch.arange(length, device=tokens.device)
        bias[:, :, index, index] = 0.0
        # One shared position just after the action chunk: slot identity comes from cond_pos_emb only.
        positions = (pos0.long() + self.horizon)[:, None].expand(batch, length)
        t0 = torch.full((batch,), self.cond_t0, dtype=torch.float32, device=tokens.device)
        # fp32 (autocast off): the action pass already saves one bf16 copy of the expert weights for backward;
        # a second autocast pass here would save another ~0.6 GiB set. The K/V are cast to the prefix dtype later.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            out = self.joint.expert_forward(tokens.float(), tuple((empty, empty) for _ in range(spec.depth)), bias,
                                            positions, cond=flow_condition(self.flow_heads, t0), use_cache=True)
        return out.cache.kv

    @staticmethod
    def _join_history(prefix_kv, cond_kv, prefix_valid: Tensor):
        if cond_kv is None:
            return prefix_kv, prefix_valid
        joined = tuple((torch.cat((pk, ck.to(pk.dtype)), dim=2), torch.cat((pv, cv.to(pv.dtype)), dim=2))
                       for (pk, pv), (ck, cv) in zip(prefix_kv, cond_kv))
        extra = prefix_valid.new_ones(prefix_valid.shape[0], cond_kv[0][0].shape[2])
        return joined, torch.cat((prefix_valid, extra), dim=1)

    def _expert_kv(self, prefix_out, layout, past: Optional[Tensor], past_valid: Optional[Tensor]):
        kv, valid = self._detached_kv(prefix_out), layout.nonki_valid
        if self.use_past:
            cond_kv = self.history_kv(self.history_tokens(past, past_valid), prefix_out.pos0)
            kv, valid = self._join_history(kv, cond_kv, valid)
        return kv, valid

    def _velocity(self, prefix_kv, pos0, prefix_valid, x_t, t, probe_cols=None):
        """``flow_velocity`` with an optional attention-mass probe (same computation)."""
        heads = self.flow_heads
        layout = build_flow_suffix_layout(pos0, prefix_valid, x_t.shape[1])
        tokens = heads.action_in_proj(x_t.to(heads.action_in_proj.weight.dtype))
        out = self.joint.expert_forward(tokens, prefix_kv, layout.bias, layout.positions,
                                        cond=flow_condition(heads, t), probe_cols=probe_cols, probe_rows="all")
        return heads.action_out_proj(out.hidden), out.probe_mass

    # ------------------------------------------------------------------ training
    def forward(self, batch, history_mode=None):
        self._check_ready()
        self._last_self_past_rows = 0  # expert-history passes generate nothing (no stale count)
        past, probability = self._resolve_history(batch, history_mode)
        use_ki = self.lambda_ki > 0
        targets = self.encode_targets(batch["action"]) if use_ki else None
        with torch.autocast(device_type=batch["action"].device.type, enabled=False):
            actions = self._pad(self.action_normalizer["action"].normalize(batch["action"].float()).float())
        batch_size = actions.shape[0]
        beta = torch.distributions.Beta(*[torch.tensor(x, device=actions.device) for x in self.flow_time_beta])
        t = beta.sample((batch_size,)) * 0.999 + 0.001
        noise = torch.randn_like(actions)
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        velocity_target = noise - actions
        coupled = self.kv_grad_scale > 0
        with self._autocast():
            prefix = self.build_prefix(batch["obs"], train_aug=self.training and self.train_image_aug)
            with contextlib.nullcontext() if (use_ki or coupled) else torch.no_grad():
                prefix_out, layout, ki_logits = self.vlm_pass(prefix, targets if use_ki else None)
            kv, valid = self._expert_kv(prefix_out, layout, past, batch["past_action_valid"])
            probe = (layout.nonki_len, layout.nonki_len + self.n_cond_tokens) if self.use_past else None
            velocity, probe_mass = self._velocity(kv, prefix_out.pos0, valid, x_t, t, probe_cols=probe)
        loss_flow = F.mse_loss(velocity.float()[..., :self.action_dim], velocity_target[..., :self.action_dim])
        loss, loss_ki = loss_flow, None
        if use_ki:
            loss_ki = F.cross_entropy(ki_logits.reshape(-1, ki_logits.shape[-1]).float(), targets.reshape(-1))
            loss = loss_flow + self.lambda_ki * loss_ki
        with torch.no_grad():
            self.last_loss_components = dict(
                loss=loss.detach().float().item(), loss_flow=loss_flow.detach().float().item(),
                loss_ki=None if loss_ki is None else loss_ki.detach().float().item(),
                ki_token_acc=None if ki_logits is None else (ki_logits.argmax(-1) == targets).float().mean().item(),
                self_past_p=float(probability), self_past_rows=float(self._last_self_past_rows),
                hist_attention_mass=None if probe_mass is None else probe_mass.float().mean().item(),
                gate_mean=None, gate_min=None, gate_max=None, ar_token_acc=None, loss_ar=None)
        return loss

    # ------------------------------------------------------------------ inference (rollout and self-past)
    def set_history_intervention(self, mode=None, noise_std=0.0):
        """Eval-only: change the history the head reads in ``predict_action``; the execution buffer keeps the
        true commands. ``zero`` = no history (zeros, invalid), ``swap`` = the previous batch row's history
        (needs >= 2 parallel environments), ``noise`` = Gaussian noise of ``noise_std`` (normalized action
        units) on the valid commands. ``None`` restores normal inference."""
        if mode is not None and mode not in HISTORY_INTERVENTIONS:
            raise ValueError(f"history intervention must be one of {HISTORY_INTERVENTIONS} or None")
        noise_std = float(noise_std)
        if not math.isfinite(noise_std) or (mode == "noise") != (noise_std > 0) or noise_std < 0:
            raise ValueError("noise_std must be positive for the noise intervention and 0 otherwise")
        if mode is not None and not self.use_past:
            raise ValueError("The no-past control reads no history to intervene on")
        self.history_intervention, self.history_noise_std = mode, noise_std

    def predict_action(self, obs_dict, use_k_tokens=None, temperature=None, topk=None,
                       past_actions=None, past_action_valid=None):
        self._acting = True  # interventions apply to rollout inference only, never to self-past
        try:
            return super().predict_action(obs_dict, use_k_tokens=use_k_tokens, temperature=temperature, topk=topk,
                                          past_actions=past_actions, past_action_valid=past_action_valid)
        finally:
            self._acting = False

    def _intervened_history(self, past: Tensor, valid: Tensor):
        mode = self.history_intervention
        if mode == "zero":
            return torch.zeros_like(past), torch.zeros_like(valid)
        if mode == "swap":
            if past.shape[0] < 2:
                raise ValueError("The history swap needs at least 2 parallel environments")
            return past.roll(1, dims=0), valid.roll(1, dims=0)
        normalizer = self.action_normalizer["action"]
        with torch.autocast(device_type=past.device.type, enabled=False):
            normalized = normalizer.normalize(past.float()).float()
            noisy = normalizer.unnormalize(normalized + self.history_noise_std * torch.randn_like(normalized))
        mask = bool_mask(valid, past.shape[:2], "past_action_valid", past.device)
        return torch.where(mask[..., None], noisy.to(past.dtype), past), valid

    def _generate_actions(self, obs, past, valid, n_tokens, temperature, topk):
        """The deployment sampler (fresh noise + Euler steps); temperature, top-k and n_tokens do not apply."""
        self._check_ready()
        if self._acting and self.history_intervention is not None:
            past, valid = self._intervened_history(past, valid)
        with torch.no_grad(), self._autocast():
            prefix = self.build_prefix(obs, train_aug=False)
            prefix_out, layout, _ = self.vlm_pass(prefix, None)
            kv, prefix_valid = self._expert_kv(prefix_out, layout, past, valid)
            noise = torch.randn(prefix.embeds.shape[0], self.horizon, FLOW_ACTION_DIM, device=prefix.embeds.device)
            sampled = sample_actions(self.joint, self.flow_heads, kv, prefix_out.pos0, prefix_valid, noise,
                                     self.flow_num_steps)
        with torch.autocast(device_type=sampled.device.type, enabled=False):
            actions = self.action_normalizer["action"].unnormalize(sampled[..., :self.action_dim].float())
        with torch.inference_mode(False):
            return actions.float().detach().clone()

    # ------------------------------------------------------------------ optimizer and artifacts
    @staticmethod
    def _is_pretrained_group(name: str) -> bool:
        # 'pretrained' (policy_lr): expert, flow heads, SigLIP. 'new' (new_module_lr): condition modules, LoRA, KI.
        return name.startswith(("joint.expert.", "flow_heads.", "siglip."))

    @staticmethod
    def _is_ki_clip(name: str) -> bool:
        # 'ki' clip group = the VLM side (SigLIP, LoRA, KI table); 'ar' = expert, flow heads, condition modules.
        return name.startswith(("siglip.", "ki_table.")) or ".lora_" in name

    def frozen_base_keys(self):
        keys = super().frozen_base_keys()
        if self.train_vision:  # a trained SigLIP belongs in every artifact
            keys = [key for key in keys if not key.startswith("siglip.")]
        return keys

    def artifact_metadata(self) -> dict:
        metadata = super().artifact_metadata()
        metadata["pi05_past"] = {
            "condition": "past2next_self_past" if self.use_past else None,
            "condition_tokens": self.n_cond_tokens, "cond_t0": self.cond_t0,
            "kv_grad_scale": self.kv_grad_scale, "lambda_ki": self.lambda_ki,
            "train_vision": self.train_vision, "train_lora": self.train_lora,
            "frozen_unreachable_lora": list(self.dead_lora), "flow_num_steps": self.flow_num_steps}
        source = Path(__file__).resolve()
        metadata["source_sha256"][str(source.relative_to(REPO_ROOT))] = file_sha256(source)
        return metadata
