"""P2N-VLA: a PI0.5-style VLA whose Gemma action expert runs Past2Next over frozen OAT tokens.

The VLM (SigLIP + Gemma-2B with LoRA) encodes [images | "Task: ..., State: ...;
Action: "] and, during training, predicts the OAT tokens of the target chunk in
a self-contained causal KI block (knowledge insulation). The action expert
(PI0.5's Gemma-300M, adaRMS folded at a constant condition) reads the VLM's
per-layer keys/values *detached*, plus static past-command tokens, and decodes
8 OAT tokens autoregressively. Rollout, acknowledgement and self-past contracts
are inherited from ``P2NNewCommonPolicy``; see docs/P2N_VLA_IMPLEMENTATION.md.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
import importlib.metadata
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dill
import hydra
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from oat.common.hydra_util import register_new_resolvers
from oat.model.common.dict_of_tensor_mixin import DictOfTensorMixin
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.vla.flow_head import FlowHeads
from oat.model.vla.gemma_joint import GemmaJoint, PrefixOutput
from oat.model.vla.image_preprocess import ImagePreprocessor
from oat.model.vla.layout import (SEG_AR, PrefixLayout, build_prefix_layout, build_suffix_layout,
                                  decode_view, prefill_view)
from oat.model.vla.lora import inject_lora, lora_parameters
from oat.model.vla.oat_ki import OATKITable
from oat.model.vla.paligemma_prompt import PaliGemmaTokenizer, PromptBuilder, discretize_state, uid_from_obs
from oat.model.vla.pi05_checkpoint import default_pi05_path, file_sha256, load_pi05
from oat.model.vla.siglip import SiglipEncoder
from oat.model.vla.specs import MODEL_SIZES
from oat.model.vla.state_transforms import LIBERO_PROMPT_STATE, PromptStateSpec
from oat.policy.base_policy import BasePolicy
from oat.policy.p2n_new_common import P2NNewCommonPolicy, _plain, bool_mask
from oat.tokenizer.oat.latent_adapter import FrozenOATLatentAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPM = REPO_ROOT / "data/pretrained/p2n_vla/paligemma_tokenizer.model"
TINY_BASE_SEED = 1234


@dataclass
class PrefixBatch:
    embeds: Tensor            # [B, P0, Wv]
    img_valid: Tensor         # [B, Ni] bool
    prompt_valid: Tensor      # [B, Lp] bool
    prompt_state: Tensor      # [B, D] fp32, normalized to [-1, 1]


@dataclass
class ConditionBatch:
    tokens: Tensor                      # [B, n_rawdiff, We] (n_rawdiff is 0 or 9)
    rawdiff_valid: Optional[Tensor]     # [B, 9] bool, or None without past conditioning
    hist: Optional[Tensor] = None       # [B, n_hist, We]
    state_valid: Optional[Tensor] = None


class P2NVLACommonPolicy(P2NNewCommonPolicy):
    VARIANT = "p2n_vla"
    VARIANT_CODE = 10
    NORM_MODE = "segment"
    LOADER_MODE = "p2n"
    HAS_AR_HEAD = True
    CONTEXT_SCHEMA_VERSION = 101
    ARTIFACT_SCHEMA_VERSION = 1
    policy_family = "p2n_vla"
    requires_state_history = False
    supports_history_summary_gate = False
    supports_explicit_past_actions = True
    supports_explicit_past_action_valid = True
    supports_generated_history_validation = True
    requires_execution_acknowledgement = True
    N_CODES = 5000
    BOS_ID = 5000
    AR_INIT_RMS = 0.76

    def __init__(
        self, shape_meta, n_action_steps=8, n_obs_steps=1, past_n=7, horizon=16, variant=None, task="libero",
        construction_mode="fresh", model_size="full", pi05_weights=None, pi05_sha256=None,
        spm_path=None, spm_sha256=None, tokenizer_checkpoint=None, tokenizer_config=None, tokenizer_metadata=None,
        rgb_ports=("agentview_rgb", "robot0_eye_in_hand_rgb"), wrist_ports=("robot0_eye_in_hand_rgb",),
        train_image_aug=True, prompt=None, lora_rank=16, lora_alpha=16.0, lambda_ki=1.0, ki_skip=1152,
        use_past=True, adarms_t0=0.6, temperature=0.0, topk=10, activation_checkpointing=True,
        frozen_dtype=None, self_past_p=0.5, self_past_warmup_steps=1000, self_past_ramp_steps=4000,
        self_past_chunk_size=4, self_past_temperature=1.0, self_past_topk=10,
        self_past_schedule="optimizer_step",
    ):
        BasePolicy.__init__(self)
        if variant is not None and variant != self.VARIANT:
            raise ValueError(f"{type(self).__name__} requires variant={self.VARIANT!r}")
        if construction_mode not in ("fresh", "restore"):
            raise ValueError("construction_mode must be fresh or restore")
        if model_size not in MODEL_SIZES:
            raise ValueError(f"model_size must be one of {sorted(MODEL_SIZES)}")
        if past_n < 3 or n_obs_steps < 1 or not 1 <= n_action_steps <= horizon:
            raise ValueError("Invalid observation, action, or history horizon")
        if self_past_schedule != "optimizer_step":
            raise ValueError("P2N-VLA advances self-past only on successful optimizer updates")
        if not 0 <= self_past_p <= 1 or min(self_past_warmup_steps, self_past_ramp_steps) < 0:
            raise ValueError("Invalid self-past probability or schedule")
        if self_past_chunk_size < 1:
            raise ValueError("self_past_chunk_size must be positive")
        if lambda_ki < 0 or not math.isfinite(lambda_ki):
            raise ValueError("lambda_ki must be finite and nonnegative")
        if not 0.0 <= adarms_t0 <= 1.0:
            raise ValueError("adarms_t0 must lie in [0, 1]")
        rgb_ports, wrist_ports = list(rgb_ports), list(wrist_ports)
        if not rgb_ports or not set(wrist_ports) <= set(rgb_ports):
            raise ValueError("wrist_ports must be a subset of the nonempty rgb_ports")

        self.variant, self.task, self.model_size = self.VARIANT, task, model_size
        self.shape_meta = _plain(shape_meta)
        self.obs_key_shapes = {k: tuple(v["shape"]) for k, v in self.shape_meta["obs"].items()}
        for port in rgb_ports:
            if port not in self.obs_key_shapes:
                raise ValueError(f"RGB port {port!r} missing from shape_meta")
        action_shape = self.shape_meta["action"]["shape"]
        if len(action_shape) != 1:
            raise ValueError("Action schema must be a vector")
        self.action_dim = int(action_shape[0])
        self.horizon, self.past_n = int(horizon), int(past_n)
        self.n_obs_steps, self.n_action_steps = int(n_obs_steps), int(n_action_steps)
        self.use_past = bool(use_past)
        self.lambda_ki = float(lambda_ki)
        self.train_image_aug = bool(train_image_aug)
        self.rgb_ports, self.wrist_ports = rgb_ports, wrist_ports
        register_new_resolvers()

        # ---------------------------------------------------------------- frozen OAT
        self._tokenizer_config = _plain(tokenizer_config)
        self._tokenizer_metadata = _plain(tokenizer_metadata) or {}
        action_tokenizer = self._load_action_tokenizer(construction_mode, tokenizer_checkpoint)
        self.latent_adapter = FrozenOATLatentAdapter(
            action_tokenizer, levels=(8, 5, 5, 5, 5), num_slots=8,
            action_horizon=self.horizon, action_dim=self.action_dim)
        self.max_seq_len = int(action_tokenizer.latent_horizon)
        if int(action_tokenizer.quantizer.codebook_size) != self.N_CODES:
            raise ValueError("P2N-VLA expects the 5000-code [8,5,5,5,5] OAT tokenizer")
        self.bos_id = self.BOS_ID

        # ---------------------------------------------------------------- prompt pipeline
        prompt = dict(_plain(prompt) or {})
        prompt.setdefault("instruction_source", "libero10")
        prompt.setdefault("state_keys", list(LIBERO_PROMPT_STATE["keys"]))
        prompt.setdefault("state_transforms", dict(LIBERO_PROMPT_STATE["transforms"]))
        prompt.setdefault("max_len", 96)
        prompt.setdefault("dummy_uid", 30)
        self.prompt_config = prompt
        self.spm_path = str(spm_path or DEFAULT_SPM)
        if not Path(self.spm_path).is_file():
            raise FileNotFoundError(f"PaliGemma SentencePiece model not found: {self.spm_path}")
        self.spm_sha256 = file_sha256(self.spm_path)
        if spm_sha256 is not None and spm_sha256 != self.spm_sha256:
            raise ValueError("SentencePiece model sha256 does not match the recorded artifact")
        self.prompt_state_spec = PromptStateSpec(list(prompt["state_keys"]), dict(prompt["state_transforms"]))
        for key in self.prompt_state_spec.keys:
            if key not in self.obs_key_shapes:
                raise ValueError(f"Prompt-state key {key!r} missing from shape_meta")
        self.prompt_builder = PromptBuilder(PaliGemmaTokenizer(self.spm_path), prompt["instruction_source"],
                                            int(prompt["max_len"]))
        self.uses_task_uid = bool(self.prompt_builder.requires_uids)
        self.prompt_state_dim = int(self.prompt_state_spec.output_dim(self.obs_key_shapes))

        # ---------------------------------------------------------------- backbone
        siglip_spec, vlm_spec, expert_spec = MODEL_SIZES[model_size]
        if frozen_dtype is None:
            frozen_dtype = torch.bfloat16 if model_size == "full" else torch.float32
        elif isinstance(frozen_dtype, str):
            frozen_dtype = getattr(torch, frozen_dtype)
        self.frozen_dtype = frozen_dtype
        self.vlm_width, self.expert_width = vlm_spec.width, expert_spec.width
        rng = torch.random.fork_rng(devices=[]) if model_size == "tiny" else contextlib.nullcontext()
        with rng:
            if model_size == "tiny":
                torch.manual_seed(TINY_BASE_SEED)  # deterministic frozen base for artifact round trips
            self.siglip = SiglipEncoder(siglip_spec, vlm_spec.width, frozen_dtype=frozen_dtype)
            self.joint = GemmaJoint(vlm_spec, expert_spec, self.NORM_MODE, frozen_dtype=frozen_dtype,
                                    activation_checkpointing=activation_checkpointing)
            self.flow_heads = self._make_flow_heads(expert_spec.width)
        self._pi05_report = None
        if model_size == "full":
            if not pi05_weights:
                pi05_weights = default_pi05_path()
            self._pi05_report = load_pi05(pi05_weights, siglip=self.siglip, joint=self.joint,
                                          mode=self.LOADER_MODE, t0=adarms_t0, flow_heads=self.flow_heads,
                                          expected_sha256=pi05_sha256)
        elif pi05_weights:
            raise ValueError("Tiny models are randomly initialized; pi05_weights must be unset")
        self.pi05_sha256 = self._pi05_report.sha256 if self._pi05_report is not None else None
        self.ki_table = OATKITable(vlm_spec.width, self.N_CODES, ki_skip)
        self.ki_table.init_from_embedding(self.joint.vlm.embed_tokens.weight)
        inject_lora(self.joint.vlm.layers, int(lora_rank), float(lora_alpha))
        if self.lambda_ki == 0:
            # Without the KI loss nothing reaches the VLM side (stop-grad), so freeze it outright.
            for parameter in lora_parameters(self.joint) + list(self.ki_table.parameters()):
                parameter.requires_grad_(False)
        self.image_preprocessor = ImagePreprocessor(rgb_ports, wrist_ports, siglip_spec.image_size)

        # ---------------------------------------------------------------- expert inputs
        width = expert_spec.width
        if self.use_past and self.HAS_AR_HEAD:
            self.raw_proj = nn.Linear(self.action_dim, width)
            self.acc_proj = nn.Linear(self.action_dim, width)
            self.jerk_proj = nn.Linear(self.action_dim, width)
            for projection in (self.raw_proj, self.acc_proj, self.jerk_proj):
                nn.init.xavier_uniform_(projection.weight)
                nn.init.zeros_(projection.bias)
            self.type_embedding = nn.Parameter(torch.empty(3, width))
            self.action_time_embedding = nn.Parameter(torch.empty(self.past_n, width))
            nn.init.normal_(self.type_embedding, std=0.02)
            nn.init.normal_(self.action_time_embedding, std=0.02)
        if self.HAS_AR_HEAD:
            self.tok_emb = nn.Embedding(self.N_CODES + 1, width)
            nn.init.normal_(self.tok_emb.weight, std=self.AR_INIT_RMS / math.sqrt(width))
        self.tok_scale = math.sqrt(width)

        # ---------------------------------------------------------------- normalizer + buffers
        self.action_normalizer = LinearNormalizer()
        for key in self._normalizer_fields():
            self.action_normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        self.action_normalizer.requires_grad_(False)
        self.register_buffer("_normalizer_fitted", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_self_past_optimizer_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_context_schema", torch.tensor(self.CONTEXT_SCHEMA_VERSION))
        self.register_buffer("_variant_code", torch.tensor(self.VARIANT_CODE))
        self.register_buffer("_artifact_schema", torch.tensor(self.ARTIFACT_SCHEMA_VERSION))

        self.temperature, self.topk = float(temperature), int(topk)
        self.self_past_p = float(self_past_p) if self.use_past else 0.0
        self.self_past_warmup_steps = int(self_past_warmup_steps)
        self.self_past_ramp_steps = int(self_past_ramp_steps)
        self.self_past_chunk_size = int(self_past_chunk_size)
        self.self_past_temperature = float(self_past_temperature)
        self.self_past_topk = int(self_past_topk)
        self.self_past_schedule = self_past_schedule
        self.obs_ports = list(rgb_ports) + list(self.prompt_state_spec.keys) + (
            ["task_uid"] if self.uses_task_uid else [])
        self.last_loss_components: Dict[str, Optional[float]] = {}
        self._last_self_past_rows = 0
        self._construction = dict(
            shape_meta=self.shape_meta, n_action_steps=n_action_steps, n_obs_steps=n_obs_steps, past_n=past_n,
            horizon=horizon, variant=self.VARIANT, task=task, model_size=model_size, pi05_sha256=self.pi05_sha256,
            spm_path=self.spm_path, spm_sha256=self.spm_sha256, rgb_ports=list(rgb_ports),
            wrist_ports=list(wrist_ports), train_image_aug=train_image_aug, prompt=copy.deepcopy(prompt),
            lora_rank=lora_rank, lora_alpha=lora_alpha, lambda_ki=lambda_ki, ki_skip=ki_skip, use_past=use_past,
            adarms_t0=adarms_t0, temperature=temperature, topk=topk,
            activation_checkpointing=activation_checkpointing,
            frozen_dtype=str(frozen_dtype).replace("torch.", ""), self_past_p=self_past_p,
            self_past_warmup_steps=self_past_warmup_steps, self_past_ramp_steps=self_past_ramp_steps,
            self_past_chunk_size=self_past_chunk_size, self_past_temperature=self_past_temperature,
            self_past_topk=self_past_topk, self_past_schedule=self_past_schedule)
        self.reset()

    # ------------------------------------------------------------------ construction helpers
    def _load_action_tokenizer(self, construction_mode, tokenizer_checkpoint):
        if construction_mode == "restore":
            if not self._tokenizer_config:
                raise ValueError("Offline restore requires an embedded tokenizer_config")
            return hydra.utils.instantiate(self._tokenizer_config)
        if not tokenizer_checkpoint or not Path(tokenizer_checkpoint).is_file():
            raise FileNotFoundError(f"Frozen OAT checkpoint not found: {tokenizer_checkpoint}")
        with open(tokenizer_checkpoint, "rb") as stream:
            payload = torch.load(stream, map_location="cpu", pickle_module=dill)
        self._tokenizer_config = _plain(payload["cfg"].tokenizer)
        tokenizer = hydra.utils.instantiate(self._tokenizer_config)
        if "ema_model" not in payload["state_dicts"]:
            raise ValueError("The selected frozen OAT must contain EMA weights")
        tokenizer.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
        self._tokenizer_metadata = {
            "source": str(tokenizer_checkpoint), "sha256": file_sha256(tokenizer_checkpoint), "weights": "ema",
            "training_task": _plain(payload["cfg"].get("task", {})),
            "normalizer_source": "frozen_tokenizer_checkpoint",
        }
        return tokenizer

    def _make_flow_heads(self, width) -> Optional[FlowHeads]:
        return None

    def _normalizer_fields(self) -> List[str]:
        return ["action", "prompt_state"]

    def _zero_state_obs(self, batch_size, device=None):
        obs = {}
        for key in self.prompt_state_spec.keys:
            value = torch.zeros(batch_size, self.n_obs_steps, *self.obs_key_shapes[key], device=device)
            if key.endswith("_quat"):
                value[..., 3] = 1
            elif key.endswith("_rot6d"):
                value[...] = value.new_tensor([1, 0, 0, 0, 1, 0])
            obs[key] = value
        return obs

    @property
    def action_tokenizer(self):
        return self.latent_adapter.tokenizer

    # ------------------------------------------------------------------ modes and readiness
    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.siglip.eval()
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
            self.siglip.eval()
            self.latent_adapter.eval()

    def _autocast(self):
        device_type = self.device.type
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda",
                              cache_enabled=False)

    def _check_ready(self):
        if not bool(self._normalizer_fitted):
            raise RuntimeError("Call set_normalizer(dataset.get_normalizer()) before forward/predict_action")

    def set_normalizer(self, normalizer):
        if isinstance(normalizer, (list, tuple)):
            if len(normalizer) != 1:
                raise ValueError("Expected one task normalizer")
            normalizer = normalizer[0]
        missing = [key for key in self._normalizer_fields() if key not in normalizer.params_dict]
        if missing:
            raise KeyError(f"Missing training-set normalizer fields: {missing}")
        # LinearNormalizer reshapes inputs to (-1, width): a width mismatch would silently mis-normalize.
        for key, width in (("action", self.action_dim), ("prompt_state", self.prompt_state_dim)):
            shape = tuple(normalizer.params_dict[key]["scale"].shape)
            if shape != (width,):
                raise ValueError(f"Normalizer field {key!r} has scale shape {shape}, expected ({width},)")
        self.action_normalizer.load_state_dict(normalizer.state_dict())
        self.action_normalizer.requires_grad_(False)
        self._normalizer_fitted.fill_(True)

    def get_observation_ports(self):
        return list(self.obs_ports)

    def get_policy_name(self):
        return f"{self.variant}_paligemma_{self.model_size}_{self.task}"

    def create_dummy_observation(self, batch_size=1, device=None):
        device = self.device if device is None else device
        obs = self._zero_state_obs(batch_size, device)
        for port in self.rgb_ports:
            obs[port] = torch.zeros(batch_size, self.n_obs_steps, *self.obs_key_shapes[port],
                                    dtype=torch.uint8, device=device)
        if self.uses_task_uid:
            obs["task_uid"] = torch.full((batch_size, self.n_obs_steps, 1), int(self.prompt_config["dummy_uid"]),
                                         dtype=torch.long, device=device)
        return obs

    # ------------------------------------------------------------------ prefix (VLM)
    def encode_targets(self, actions: Tensor) -> Tensor:
        return self.latent_adapter.encode_actions(actions).indices

    def build_prefix(self, obs, *, train_aug: bool) -> PrefixBatch:
        pixels = self.image_preprocessor(obs, train_aug=train_aug)
        batch, n_cams = pixels.shape[:2]
        tokens = self.siglip(pixels.flatten(0, 1))
        tokens = tokens.reshape(batch, n_cams * tokens.shape[1], tokens.shape[2])
        state = self.prompt_state_spec.extract(obs).float()
        state = self.action_normalizer["prompt_state"].normalize(state).float()
        bins = discretize_state(state)
        uids = uid_from_obs(obs) if self.uses_task_uid else None
        ids, prompt_valid = self.prompt_builder.build(uids, bins)
        ids, prompt_valid = ids.to(tokens.device), prompt_valid.to(tokens.device)
        text = self.joint.embed_text(ids)
        embeds = torch.cat((tokens.to(text.dtype), text), dim=1)
        img_valid = torch.ones(batch, tokens.shape[1], dtype=torch.bool, device=tokens.device)
        return PrefixBatch(embeds, img_valid, prompt_valid, state)

    def vlm_pass(self, prefix: PrefixBatch, ki_targets: Optional[Tensor] = None
                 ) -> Tuple[PrefixOutput, PrefixLayout, Optional[Tensor]]:
        if ki_targets is None:
            layout = build_prefix_layout(prefix.img_valid, prefix.prompt_valid, 0)
            return self.joint.prefix_forward(prefix.embeds, layout), layout, None
        ki = self.ki_table.embed_block(ki_targets).to(prefix.embeds.dtype)
        layout = build_prefix_layout(prefix.img_valid, prefix.prompt_valid, ki.shape[1])
        out = self.joint.prefix_forward(torch.cat((prefix.embeds, ki), dim=1), layout)
        return out, layout, self.ki_table.logits(out.hidden[:, -ki.shape[1]:])

    # ------------------------------------------------------------------ suffix (expert)
    def build_conditions(self, obs, past: Tensor, past_valid: Tensor) -> ConditionBatch:
        if not self.use_past:
            batch = past.shape[0]
            return ConditionBatch(past.new_zeros(batch, 0, self.expert_width), None)
        normalized, valid = self._safe_past(past, past_valid)
        normalized = normalized.float()
        raw = self.raw_proj(normalized) + self.action_time_embedding + self.type_embedding[0]
        diff_valid = torch.stack((valid[:, -2:].all(1), valid[:, -3:].all(1)), dim=1)
        first = normalized[:, -1] - normalized[:, -2]
        second = normalized[:, -1] - 2 * normalized[:, -2] + normalized[:, -3]
        first = torch.where(diff_valid[:, :1], first, torch.zeros_like(first))
        second = torch.where(diff_valid[:, 1:], second, torch.zeros_like(second))
        diff = torch.stack((self.acc_proj(first) + self.type_embedding[1],
                            self.jerk_proj(second) + self.type_embedding[2]), dim=1)
        tokens = torch.cat((raw, diff), dim=1)
        rawdiff_valid = torch.cat((valid, diff_valid), dim=1)
        tokens = torch.where(rawdiff_valid[..., None], tokens, torch.zeros_like(tokens))
        return ConditionBatch(tokens, rawdiff_valid)

    def compute_log_gate(self, obs, prefix: PrefixBatch, prefix_out: PrefixOutput, layout: PrefixLayout,
                         cond: ConditionBatch) -> Tuple[Optional[Tensor], bool]:
        return None, False

    def _suffix_inputs(self, cond: ConditionBatch, ar_ids: Tensor) -> Tensor:
        ar = self.tok_emb(ar_ids) * self.tok_scale
        parts = [cond.tokens] + ([cond.hist] if cond.hist is not None else []) + [ar]
        return torch.cat([part.float() for part in parts], dim=1)

    def _head(self, hidden: Tensor) -> Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return hidden.float() @ self.tok_emb.weight.float().T

    @staticmethod
    def _detached_kv(prefix_out: PrefixOutput):
        return tuple((key.detach(), value.detach()) for key, value in prefix_out.kv)

    def teacher_forcing_logits(self, prefix_out, layout, cond, targets, log_gate, hist_closed, probe=False):
        batch = targets.shape[0]
        ar_ids = torch.cat((targets.new_full((batch, 1), self.BOS_ID), targets[:, :-1]), dim=1)
        n_hist = 0 if cond.hist is None else cond.hist.shape[1]
        suffix = build_suffix_layout(prefix_out.pos0, layout.nonki_valid, cond.rawdiff_valid, n_hist,
                                     ar_ids.shape[1], log_gate=log_gate, hist_closed=hist_closed)
        out = self.joint.expert_forward(
            self._suffix_inputs(cond, ar_ids), self._detached_kv(prefix_out), suffix.bias, suffix.positions,
            seg_ids=suffix.seg_ids, probe_cols=suffix.hist_columns if (probe and n_hist) else None)
        return self._head(out.hidden[:, -ar_ids.shape[1]:]), out.probe_mass

    @staticmethod
    def _sample(logits: Tensor, temperature: float, topk: Optional[int], bos_id: int) -> Tensor:
        scores = logits.float().clone()
        scores[:, bos_id] = float("-inf")
        if temperature == 0:
            return scores.argmax(dim=-1)
        scores = scores / temperature
        if topk is not None and topk > 0:
            cutoff = scores.topk(min(int(topk), scores.shape[-1] - 1), dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < cutoff, float("-inf"))
        return torch.multinomial(scores.softmax(dim=-1), num_samples=1)[:, 0]

    @torch.no_grad()
    def generate_tokens(self, prefix_out, layout, cond, log_gate, hist_closed, n_tokens, temperature, topk):
        if not 1 <= n_tokens <= self.max_seq_len:
            raise ValueError(f"n_tokens must be in [1, {self.max_seq_len}]")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        batch = cond.tokens.shape[0]
        n_hist = 0 if cond.hist is None else cond.hist.shape[1]
        suffix = build_suffix_layout(prefix_out.pos0, layout.nonki_valid, cond.rawdiff_valid, n_hist, n_tokens,
                                     log_gate=log_gate, hist_closed=hist_closed)
        kv = self._detached_kv(prefix_out)
        bos = torch.full((batch, 1), self.BOS_ID, dtype=torch.long, device=cond.tokens.device)
        bias, positions, seg_ids = prefill_view(suffix, suffix.n_cond + 1)
        out = self.joint.expert_forward(self._suffix_inputs(cond, bos), kv, bias, positions, seg_ids=seg_ids,
                                        use_cache=True)
        ar_segment = torch.tensor([SEG_AR], dtype=torch.long, device=bos.device)
        generated = []
        for index in range(n_tokens):
            token = self._sample(self._head(out.hidden[:, -1]), temperature, topk, self.BOS_ID)
            generated.append(token)
            if index + 1 < n_tokens:
                bias, position = decode_view(suffix, index + 1)
                embedded = (self.tok_emb(token[:, None]) * self.tok_scale).float()
                out = self.joint.expert_forward(embedded, kv, bias, position, seg_ids=ar_segment,
                                                cache=out.cache, use_cache=True)
        return torch.stack(generated, dim=1)

    def _detokenize(self, tokens: Tensor) -> Tensor:
        with torch.autocast(device_type=tokens.device.type, enabled=False), torch.no_grad():
            actions = self.action_tokenizer.detokenize(tokens=tokens.long())
        with torch.inference_mode(False):
            return actions.float().detach().clone()

    def _generate_actions(self, obs, past, valid, n_tokens, temperature, topk):
        self._check_ready()
        with torch.no_grad(), self._autocast():
            prefix = self.build_prefix(obs, train_aug=False)
            prefix_out, layout, _ = self.vlm_pass(prefix, None)
            cond = self.build_conditions(obs, past, valid)
            log_gate, hist_closed = self.compute_log_gate(obs, prefix, prefix_out, layout, cond)
            tokens = self.generate_tokens(prefix_out, layout, cond, log_gate, hist_closed, n_tokens,
                                          temperature, topk)
        return self._detokenize(tokens)

    # ------------------------------------------------------------------ self-past
    @torch.no_grad()
    def _maybe_self_past(self, batch, past, probability=None):
        probability = self.self_past_probability() if probability is None else probability
        self._last_self_past_rows = 0
        if not self.use_past or probability <= 0:
            return past
        if not 0 <= probability <= 1:
            raise ValueError("Self-past probability must be in [0, 1]")
        required = ("prev_obs", "prev_past_action", "prev_past_action_valid", "prev_window_valid")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Self-past needs previous-window metadata: {missing}")
        windows = bool_mask(batch["prev_window_valid"].reshape(-1), (past.shape[0],), "prev_window_valid",
                            past.device)
        selected = windows.clone()
        if probability < 1:
            selected &= torch.rand(past.shape[0], device=past.device) < probability
        indices = selected.nonzero(as_tuple=True)[0]
        self._last_self_past_rows = int(indices.numel())
        if indices.numel() == 0:
            return past
        generated = past.detach().clone()
        with self._rollout_mode():
            for chunk in indices.split(self.self_past_chunk_size):
                previous = batch["prev_past_action"][chunk]
                prediction = self._generate_actions(
                    {key: value[chunk] for key, value in batch["prev_obs"].items()}, previous,
                    batch["prev_past_action_valid"][chunk], self.max_seq_len,
                    self.self_past_temperature, self.self_past_topk)
                generated[chunk] = torch.cat(
                    (previous, prediction[:, :self.n_action_steps].to(previous.dtype)), dim=1)[:, -self.past_n:]
        mask = bool_mask(batch["past_action_valid"], past.shape[:2], "past_action_valid", past.device)
        use_self = selected[:, None] & mask
        return torch.where(use_self[..., None], generated, past)

    # ------------------------------------------------------------------ training forward
    def _resolve_history(self, batch, history_mode):
        history_mode = ("configured" if self.training else "expert") if history_mode is None else history_mode
        if history_mode not in ("expert", "generated", "configured"):
            raise ValueError("history_mode must be expert, generated, or configured")
        if "past_action_valid" not in batch:
            raise KeyError("P2N-VLA requires top-level past_action_valid")
        past, probability = batch["past_action"], 0.0
        if history_mode != "expert" and self.use_past:
            probability = 1.0 if history_mode == "generated" else self.self_past_probability()
            past = self._maybe_self_past(batch, past, probability)
        return past, probability

    def forward(self, batch, history_mode=None):
        self._check_ready()
        past, probability = self._resolve_history(batch, history_mode)
        targets = self.encode_targets(batch["action"])
        if targets.shape != (past.shape[0], self.max_seq_len):
            raise ValueError("OAT returned an unexpected token shape")
        use_ki = self.lambda_ki > 0
        with self._autocast():
            prefix = self.build_prefix(batch["obs"], train_aug=self.training and self.train_image_aug)
            with contextlib.nullcontext() if use_ki else torch.no_grad():
                prefix_out, layout, ki_logits = self.vlm_pass(prefix, targets if use_ki else None)
            cond = self.build_conditions(batch["obs"], past, batch["past_action_valid"])
            log_gate, hist_closed = self.compute_log_gate(batch["obs"], prefix, prefix_out, layout, cond)
            logits, probe = self.teacher_forcing_logits(prefix_out, layout, cond, targets, log_gate, hist_closed,
                                                        probe=self.supports_history_summary_gate)
        loss_ar = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss, loss_ki = loss_ar, None
        if use_ki:
            loss_ki = F.cross_entropy(ki_logits.reshape(-1, ki_logits.shape[-1]).float(), targets.reshape(-1))
            loss = loss_ar + self.lambda_ki * loss_ki
        with torch.no_grad():
            components = dict(
                loss=loss.detach().float().item(), loss_ar=loss_ar.detach().float().item(),
                loss_ki=None if loss_ki is None else loss_ki.detach().float().item(),
                ar_token_acc=(logits.argmax(-1) == targets).float().mean().item(),
                ki_token_acc=None if ki_logits is None else (ki_logits.argmax(-1) == targets).float().mean().item(),
                self_past_p=float(probability), self_past_rows=float(self._last_self_past_rows),
                hist_attention_mass=None if probe is None else probe.float().mean().item())
            components.update(self._gate_components(log_gate, hist_closed))
        self.last_loss_components = components
        return loss

    def _gate_components(self, log_gate, hist_closed):
        return dict(gate_mean=None, gate_min=None, gate_max=None)

    # ------------------------------------------------------------------ optimizer
    def trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        seen, result = set(), []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append((name, parameter))
        return result

    @staticmethod
    def _is_pretrained_group(name: str) -> bool:
        return name.startswith(("joint.expert.", "flow_heads.")) or ".lora_" in name

    @staticmethod
    def _is_ki_clip(name: str) -> bool:
        return ".lora_" in name or name.startswith("ki_table.")

    def get_optimizer(self, policy_lr=5e-5, new_module_lr=1e-4, weight_decay=1e-10, betas=(0.9, 0.95),
                      eps=1e-8, fused=None):
        pretrained, new = [], []
        for name, parameter in self.trainable_named_parameters():
            (pretrained if self._is_pretrained_group(name) else new).append(parameter)
        params = pretrained + new
        if fused is None:
            fused = bool(params) and all(p.is_cuda for p in params)
        groups = [group for group in (
            {"params": pretrained, "lr": float(policy_lr), "weight_decay": float(weight_decay), "name": "pretrained"},
            {"params": new, "lr": float(new_module_lr), "weight_decay": float(weight_decay), "name": "new"},
        ) if group["params"]]
        return torch.optim.AdamW(groups, betas=tuple(betas), eps=float(eps), fused=bool(fused))

    def clip_groups(self) -> Dict[str, List[nn.Parameter]]:
        groups = {"ar": [], "ki": []}
        for name, parameter in self.trainable_named_parameters():
            groups["ki" if self._is_ki_clip(name) else "ar"].append(parameter)
        return groups

    # ------------------------------------------------------------------ artifacts
    def frozen_base_keys(self) -> List[str]:
        keys = []
        for name, parameter in self.named_parameters(remove_duplicate=False):
            frozen_vlm = name.startswith("joint.vlm.") and (
                name == "joint.vlm.embed_tokens.weight" or name.endswith(".base.weight")
                or name.endswith("layernorm.weight") or name == "joint.vlm.norm.weight")
            if name.startswith("siglip.") or frozen_vlm:
                keys.append(name)
        return sorted(set(keys))

    def artifact_state_dict(self, trainable_override: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        frozen = set(self.frozen_base_keys())
        parameters = dict(self.named_parameters())
        bad = [key for key in frozen if parameters[key].requires_grad]
        if bad:
            raise RuntimeError(f"Frozen base keys unexpectedly trainable: {bad[:5]}")
        state = {key: value for key, value in self.state_dict().items() if key not in frozen}
        if trainable_override:
            trainable = {name for name, _ in self.trainable_named_parameters()}
            for name, tensor in trainable_override.items():
                if name not in trainable:
                    raise KeyError(f"EMA override for a non-trainable key: {name}")
                if tuple(tensor.shape) != tuple(state[name].shape):
                    raise ValueError(f"EMA override shape mismatch for {name}")
                state[name] = tensor
        return state

    def _normalizer_prefixes(self) -> Tuple[str, ...]:
        return tuple(f"{name}.params_dict." for name, module in self.named_modules()
                     if isinstance(module, DictOfTensorMixin))

    def load_artifact_state(self, state: Dict[str, Tensor]):
        # Normalizers rebuild their tensors from the payload (DictOfTensorMixin), so their
        # sub-keys are dynamic: check every static key exactly and every field explicitly.
        dynamic = self._normalizer_prefixes()
        static = lambda keys: {k for k in keys if not k.startswith(dynamic)}  # noqa: E731
        expected = static(set(self.state_dict()) - set(self.frozen_base_keys()))
        provided = static(set(state))
        if provided != expected:
            raise KeyError(f"Artifact key mismatch: missing {sorted(expected - provided)[:5]}, "
                           f"unexpected {sorted(provided - expected)[:5]}")
        required_fields = [f"action_normalizer.params_dict.{field}.scale" for field in self._normalizer_fields()]
        required_fields.append("latent_adapter.tokenizer.normalizer.params_dict.action.scale")
        missing_fields = [key for key in required_fields if key not in state]
        if missing_fields:
            raise KeyError(f"Artifact lacks normalizer fields: {missing_fields}")
        for key, wanted in (("_variant_code", self.VARIANT_CODE), ("_context_schema", self.CONTEXT_SCHEMA_VERSION),
                            ("_artifact_schema", self.ARTIFACT_SCHEMA_VERSION)):
            if int(state[key]) != wanted:
                raise ValueError(f"Artifact {key}={int(state[key])} does not match this policy ({wanted})")
        result = nn.Module.load_state_dict(self, state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != set(self.frozen_base_keys()):
            raise RuntimeError(f"Artifact load mismatch: {result}")
        self.action_normalizer.requires_grad_(False)
        self.reset()
        return result

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if int(state_dict.get("_variant_code", -1)) != self.VARIANT_CODE:
            raise ValueError("Checkpoint variant does not match this policy")
        if int(state_dict.get("_context_schema", -1)) != self.CONTEXT_SCHEMA_VERSION:
            raise ValueError("Checkpoint context schema does not match")
        result = nn.Module.load_state_dict(self, state_dict, strict=strict, **kwargs)
        self.reset()
        return result

    def export_config(self) -> dict:
        if not self._tokenizer_config:
            raise ValueError("Export requires the embedded OAT tokenizer_config")
        return dict(_target_=f"{type(self).__module__}.{type(self).__name__}", _recursive_=False,
                    **copy.deepcopy(self._construction), construction_mode="restore",
                    tokenizer_config=copy.deepcopy(self._tokenizer_config),
                    tokenizer_metadata=copy.deepcopy(self._tokenizer_metadata))

    def artifact_metadata(self) -> dict:
        versions = {}
        for package in ("torch", "transformers", "accelerate", "sentencepiece"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        sources = [Path(__file__).resolve()] + sorted((REPO_ROOT / "oat/model/vla").glob("*.py"))
        return dict(
            format="p2n_vla_artifact_v1", variant=self.variant, variant_code=self.VARIANT_CODE,
            policy_target=f"{type(self).__module__}.{type(self).__name__}", task=self.task,
            model_size=self.model_size, context_schema=self.CONTEXT_SCHEMA_VERSION,
            artifact_schema=self.ARTIFACT_SCHEMA_VERSION, execution_protocol="acknowledged_commands_v1",
            shape_meta=copy.deepcopy(self.shape_meta), horizon=self.horizon, n_action_steps=self.n_action_steps,
            vocab_size=self.N_CODES + 1, latent_horizon=self.max_seq_len,
            pi05={"repo": "lerobot/pi05_base", "sha256": self.pi05_sha256,
                  "loader_mode": self.LOADER_MODE,
                  "t0": None if self._pi05_report is None else self._pi05_report.t0},
            tokenizer=copy.deepcopy(self._tokenizer_metadata),
            sentencepiece={"path": self.spm_path, "sha256": self.spm_sha256},
            prompt=copy.deepcopy(self.prompt_config), frozen_base_keys=len(self.frozen_base_keys()),
            parameters=self.parameter_counts(), software=versions,
            self_past_optimizer_updates=self.self_past_step, architecture=self.export_config(),
            source_sha256={str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in sources})

    @classmethod
    def from_checkpoint(cls, path, *, base_weights=None, weights="ema", device="cpu", spm_path=None):
        with open(path, "rb") as stream:
            payload = torch.load(stream, map_location="cpu", pickle_module=dill, weights_only=False)
        config = dict(_plain(payload["policy_config"]))
        target = config.pop("_target_")
        config.pop("_recursive_", None)
        policy_cls = hydra.utils.get_class(target)
        if not issubclass(policy_cls, cls):
            raise ValueError(f"Checkpoint target {target} is not a {cls.__name__}")
        key = {"ema": "ema_model", "model": "model"}.get(weights)
        if key is None:
            raise ValueError("weights must be 'ema' or 'model'")
        if key not in payload["state_dicts"]:
            raise ValueError(f"Checkpoint does not contain {key}")
        config["construction_mode"] = "restore"
        if config.get("model_size", "full") == "full":
            config["pi05_weights"] = str(base_weights or default_pi05_path())
        if spm_path is not None:
            config["spm_path"] = str(spm_path)
        policy = policy_cls(**config)
        policy.load_artifact_state(payload["state_dicts"][key])
        return policy.to(device).eval()
