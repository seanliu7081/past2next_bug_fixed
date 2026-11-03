"""Native VLM autoregression over frozen OAT codes with continuous P2N history.

The Gemma VLM reads observation, nine continuous history slots, and a shifted
action-token suffix. There is no separate action expert or auxiliary KI branch.
Execution acknowledgements, self-past, normalizers and compact artifacts retain
the P2N-VLA contracts. See docs/PI05_P2N_VLM_V1.md.
"""
from __future__ import annotations

import contextlib
import copy
import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from oat.common.hydra_util import register_new_resolvers
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.vla.gemma_vlm import GemmaVLM
from oat.model.vla.image_preprocess import ImagePreprocessor
from oat.model.vla.lora import inject_lora, lora_parameters
from oat.model.vla.native_ar_layout import (
    build_native_ar_layout, native_decode_view, native_prefill_view,
)
from oat.model.vla.paligemma_prompt import PaliGemmaTokenizer, PromptBuilder
from oat.model.vla.pi05_checkpoint import default_pi05_path, file_sha256, load_pi05_vlm
from oat.model.vla.siglip import SiglipEncoder
from oat.model.vla.specs import MODEL_SIZES
from oat.model.vla.state_transforms import LIBERO_PROMPT_STATE, PromptStateSpec
from oat.model.vla.trainable_siglip import TrainableSiglipEncoder
from oat.policy.base_policy import BasePolicy
from oat.policy.p2n_new_common import _plain, bool_mask
from oat.policy.p2n_vla_common import DEFAULT_SPM, REPO_ROOT, TINY_BASE_SEED, P2NVLACommonPolicy
from oat.policy.pi05_past import Pi05PastPolicy
from oat.tokenizer.oat.latent_adapter import FrozenOATLatentAdapter


class Pi05P2NVLMV1Policy(P2NVLACommonPolicy):
    VARIANT = "pi05_p2n_vlm_v1"
    VARIANT_CODE = 15
    CONTEXT_SCHEMA_VERSION = 102
    LOADER_MODE = "vlm"
    HAS_AR_HEAD = True
    supports_history_intervention = True
    N_HISTORY_SLOTS = 9

    def __init__(
        self, shape_meta, n_action_steps=8, n_obs_steps=1, past_n=7, horizon=16,
        variant=None, task="libero", construction_mode="fresh", model_size="full",
        pi05_weights=None, pi05_sha256=None, spm_path=None, spm_sha256=None,
        tokenizer_checkpoint=None, tokenizer_config=None, tokenizer_metadata=None,
        rgb_ports=("agentview_rgb", "robot0_eye_in_hand_rgb"),
        wrist_ports=("robot0_eye_in_hand_rgb",), train_image_aug=True, prompt=None,
        lora_rank=32, lora_alpha=32.0, use_past=True, temperature=0.0, topk=10,
        activation_checkpointing=True, frozen_dtype=None, train_vision=True,
        train_lora=True, vision_activation_checkpointing=True, cond_dropout=0.1,
        self_past_p=0.5, self_past_warmup_steps=1000, self_past_ramp_steps=0,
        self_past_chunk_size=4, self_past_temperature=1.0, self_past_topk=10,
        self_past_schedule="optimizer_step", lambda_ki=None, ki_skip=None, adarms_t0=None,
    ):
        # Deliberately do not run the expert-based parent's constructor.
        BasePolicy.__init__(self)
        if variant is not None and variant != self.VARIANT:
            raise ValueError(f"{type(self).__name__} requires variant={self.VARIANT!r}")
        if construction_mode not in ("fresh", "restore"):
            raise ValueError("construction_mode must be fresh or restore")
        if model_size not in MODEL_SIZES:
            raise ValueError(f"model_size must be one of {sorted(MODEL_SIZES)}")
        for name, value in (("lambda_ki", lambda_ki), ("ki_skip", ki_skip), ("adarms_t0", adarms_t0)):
            if value is not None:
                raise ValueError(f"{name} must be null: native VLM AR has no KI or expert")
        for name, value in (("past_n", past_n), ("horizon", horizon), ("n_obs_steps", n_obs_steps),
                            ("n_action_steps", n_action_steps), ("lora_rank", lora_rank),
                            ("self_past_chunk_size", self_past_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if past_n != 7 or horizon != 16 or n_obs_steps != 1 or n_action_steps > horizon:
            raise ValueError("v1 requires past_n=7, horizon=16, n_obs_steps=1 and n_action_steps<=16")
        for name, value in (("use_past", use_past), ("train_vision", train_vision),
                            ("train_lora", train_lora), ("train_image_aug", train_image_aug),
                            ("activation_checkpointing", activation_checkpointing),
                            ("vision_activation_checkpointing", vision_activation_checkpointing)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        for name, value in (("self_past_warmup_steps", self_past_warmup_steps),
                            ("self_past_ramp_steps", self_past_ramp_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self_past_schedule != "optimizer_step":
            raise ValueError("Self-past advances only on successful optimizer updates")
        if not math.isfinite(self_past_p) or not 0 <= self_past_p <= 1:
            raise ValueError("self_past_p must lie in [0,1]")
        if not math.isfinite(cond_dropout) or not 0 <= cond_dropout < 1:
            raise ValueError("cond_dropout must lie in [0,1)")
        if not math.isfinite(lora_alpha) or lora_alpha <= 0:
            raise ValueError("lora_alpha must be finite and positive")
        self_past_temperature = temperature if self_past_temperature is None else self_past_temperature
        self_past_topk = topk if self_past_topk is None else self_past_topk
        for name, value in (("temperature", temperature), ("self_past_temperature", self_past_temperature)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("topk", topk), ("self_past_topk", self_past_topk)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        rgb_ports, wrist_ports = list(rgb_ports), list(wrist_ports)
        if not rgb_ports or not set(wrist_ports) <= set(rgb_ports):
            raise ValueError("wrist_ports must be a subset of nonempty rgb_ports")

        self.variant, self.task, self.model_size = self.VARIANT, task, model_size
        self.shape_meta = _plain(shape_meta)
        self.obs_key_shapes = {key: tuple(value["shape"]) for key, value in self.shape_meta["obs"].items()}
        if tuple(self.shape_meta["action"]["shape"]) != (7,):
            raise ValueError("v1 requires seven-dimensional LIBERO actions")
        for port in rgb_ports:
            if port not in self.obs_key_shapes:
                raise ValueError(f"RGB port {port!r} missing from shape_meta")
        self.action_dim, self.horizon, self.past_n = 7, horizon, past_n
        self.n_obs_steps, self.n_action_steps = n_obs_steps, n_action_steps
        self.use_past, self.train_image_aug = use_past, train_image_aug
        self.train_vision, self.train_lora = train_vision, train_lora
        self.rgb_ports, self.wrist_ports = rgb_ports, wrist_ports
        self.n_cond_tokens = self.N_HISTORY_SLOTS  # reserve positions even in the no-history control
        self.history_intervention, self.history_noise_std, self._acting = None, 0.0, False
        register_new_resolvers()

        self._tokenizer_config = _plain(tokenizer_config)
        self._tokenizer_metadata = _plain(tokenizer_metadata) or {}
        tokenizer = self._load_action_tokenizer(construction_mode, tokenizer_checkpoint)
        self.latent_adapter = FrozenOATLatentAdapter(
            tokenizer, levels=(8, 5, 5, 5, 5), num_slots=8,
            action_horizon=horizon, action_dim=self.action_dim)
        self.max_seq_len, self.bos_id = int(tokenizer.latent_horizon), self.BOS_ID
        if self.max_seq_len != 8 or int(tokenizer.quantizer.codebook_size) != self.N_CODES:
            raise ValueError("v1 requires eight tokens from the 5000-code OAT tokenizer")

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

        siglip_spec, vlm_spec, _ = MODEL_SIZES[model_size]
        if frozen_dtype is None:
            frozen_dtype = torch.bfloat16 if model_size == "full" else torch.float32
        elif isinstance(frozen_dtype, str):
            frozen_dtype = getattr(torch, frozen_dtype)
        self.frozen_dtype, self.vlm_width = frozen_dtype, vlm_spec.width
        with torch.random.fork_rng(devices=[]) if model_size == "tiny" else contextlib.nullcontext():
            if model_size == "tiny":
                torch.manual_seed(TINY_BASE_SEED)
            self.siglip = SiglipEncoder(siglip_spec, vlm_spec.width, frozen_dtype=frozen_dtype)
            self.joint = GemmaVLM(vlm_spec, frozen_dtype=frozen_dtype,
                                  activation_checkpointing=activation_checkpointing)
        # Wrap before loading so trainable vision masters retain the checkpoint's fp32 values.
        if train_vision:
            self.siglip = TrainableSiglipEncoder(self.siglip,
                                                 activation_checkpointing=vision_activation_checkpointing)
        self._pi05_report = None
        if model_size == "full":
            self._pi05_report = load_pi05_vlm(pi05_weights or default_pi05_path(), siglip=self.siglip,
                                               joint=self.joint, expected_sha256=pi05_sha256)
        elif pi05_weights:
            raise ValueError("Tiny models are randomly initialized; pi05_weights must be unset")
        self.pi05_sha256 = None if self._pi05_report is None else self._pi05_report.sha256
        inject_lora(self.joint.vlm.layers, lora_rank, float(lora_alpha))
        for parameter in lora_parameters(self.joint):
            parameter.requires_grad_(train_lora)
        self.image_preprocessor = ImagePreprocessor(rgb_ports, wrist_ports, siglip_spec.image_size)

        if self.use_past:
            def mlp():
                return nn.Sequential(nn.Linear(self.action_dim, self.vlm_width), nn.GELU(),
                                     nn.Linear(self.vlm_width, self.vlm_width))
            self.raw_proj, self.acc_proj, self.jerk_proj = mlp(), mlp(), mlp()
            self.cond_pos_emb = nn.Parameter(torch.zeros(self.N_HISTORY_SLOTS, self.vlm_width))
            self.cond_dropout = nn.Dropout(cond_dropout)
        self.tok_emb = nn.Embedding(self.N_CODES + 1, self.vlm_width)
        nn.init.normal_(self.tok_emb.weight, std=self.AR_INIT_RMS / math.sqrt(self.vlm_width))
        self.tok_scale = math.sqrt(self.vlm_width)

        self.action_normalizer = LinearNormalizer()
        for key in self._normalizer_fields():
            self.action_normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        self.action_normalizer.requires_grad_(False)
        self.register_buffer("_normalizer_fitted", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_self_past_optimizer_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_context_schema", torch.tensor(self.CONTEXT_SCHEMA_VERSION))
        self.register_buffer("_variant_code", torch.tensor(self.VARIANT_CODE))
        self.register_buffer("_artifact_schema", torch.tensor(self.ARTIFACT_SCHEMA_VERSION))
        self.temperature, self.topk = float(temperature), topk
        self.self_past_p = float(self_past_p) if use_past else 0.0
        self.self_past_warmup_steps, self.self_past_ramp_steps = self_past_warmup_steps, self_past_ramp_steps
        self.self_past_chunk_size, self.self_past_schedule = self_past_chunk_size, self_past_schedule
        self.self_past_temperature, self.self_past_topk = float(self_past_temperature), self_past_topk
        self.obs_ports = rgb_ports + list(self.prompt_state_spec.keys) + (["task_uid"] if self.uses_task_uid else [])
        self.last_loss_components = {}
        self._last_self_past_rows, self._last_self_past_mse = 0, None
        self._construction = dict(
            shape_meta=self.shape_meta, n_action_steps=n_action_steps, n_obs_steps=n_obs_steps,
            past_n=past_n, horizon=horizon, variant=self.VARIANT, task=task, model_size=model_size,
            pi05_sha256=self.pi05_sha256, spm_path=self.spm_path, spm_sha256=self.spm_sha256,
            rgb_ports=rgb_ports, wrist_ports=wrist_ports, train_image_aug=train_image_aug,
            prompt=copy.deepcopy(prompt), lora_rank=lora_rank, lora_alpha=lora_alpha, use_past=use_past,
            temperature=temperature, topk=topk, activation_checkpointing=activation_checkpointing,
            frozen_dtype=str(frozen_dtype).replace("torch.", ""), train_vision=train_vision,
            train_lora=train_lora, vision_activation_checkpointing=vision_activation_checkpointing,
            cond_dropout=cond_dropout, self_past_p=self_past_p, self_past_warmup_steps=self_past_warmup_steps,
            self_past_ramp_steps=self_past_ramp_steps, self_past_chunk_size=self_past_chunk_size,
            self_past_temperature=self.self_past_temperature, self_past_topk=self_past_topk,
            self_past_schedule=self_past_schedule)
        self.reset()
        self.train(self.training)

    # The deterministic tokenizer/vision modes and optional rollout interventions are shared,
    # but none of Pi05Past's expert construction, modulation or LoRA freezing is invoked.
    train = Pi05PastPolicy.train
    _rollout_mode = Pi05PastPolicy._rollout_mode
    set_history_intervention = Pi05PastPolicy.set_history_intervention
    _intervened_history = Pi05PastPolicy._intervened_history

    def history_tokens(self, past: Tensor, past_valid: Tensor):
        if (not isinstance(past, Tensor) or past.ndim != 3 or
                tuple(past.shape[1:]) != (self.past_n, self.action_dim) or not past.is_floating_point()):
            raise ValueError(f"past_actions must be floating [B,{self.past_n},{self.action_dim}]")
        valid = bool_mask(past_valid, past.shape[:2], "past_action_valid", past.device)
        if not self.use_past:
            return (past.new_zeros(past.shape[0], self.N_HISTORY_SLOTS, self.vlm_width),
                    valid.new_zeros(past.shape[0], self.N_HISTORY_SLOTS))
        if not torch.isfinite(past[valid]).all():
            raise ValueError("Valid historical commands must be finite")
        safe = torch.where(valid[..., None], past, torch.zeros_like(past))
        diff_valid = torch.stack((valid[:, -2:].all(1), valid[:, -3:].all(1)), dim=1)
        history_valid = torch.cat((diff_valid, valid), dim=1)
        with torch.autocast(device_type=past.device.type, enabled=False):
            normalized = self.action_normalizer["action"].normalize(safe.float()).float()
            normalized = torch.where(valid[..., None], normalized, torch.zeros_like(normalized))
            acc = normalized[:, -1] - normalized[:, -2]
            jerk = normalized[:, -1] - 2 * normalized[:, -2] + normalized[:, -3]
            acc = torch.where(diff_valid[:, :1], acc, torch.zeros_like(acc))
            jerk = torch.where(diff_valid[:, 1:], jerk, torch.zeros_like(jerk))
            tokens = torch.cat((self.acc_proj(acc)[:, None], self.jerk_proj(jerk)[:, None],
                                self.raw_proj(normalized)), dim=1)
            tokens = self.cond_dropout(tokens + self.cond_pos_emb)
            tokens = torch.where(history_valid[..., None], tokens, torch.zeros_like(tokens))
        return tokens, history_valid

    def build_native_context(self, obs, past, valid, *, train_aug=False):
        prefix = self.build_prefix(obs, train_aug=train_aug)
        history, history_valid = self.history_tokens(past, valid)
        if history.shape[0] != prefix.embeds.shape[0]:
            raise ValueError("Observation and history batch sizes differ")
        context = torch.cat((prefix.embeds, history.to(prefix.embeds.dtype)), dim=1)
        return context, torch.cat((prefix.img_valid, prefix.prompt_valid), dim=1), history_valid

    def _token_inputs(self, ids):
        return (self.tok_emb(ids) * self.tok_scale).float()

    def ar_logits(self, context, observation_valid, history_valid, targets):
        if (targets.dtype != torch.long or targets.ndim != 2 or
                tuple(targets.shape) != (context.shape[0], self.max_seq_len) or
                ((targets < 0) | (targets >= self.N_CODES)).any()):
            raise ValueError("targets must be long [B,8] OAT indices in [0,4999]")
        ids = torch.cat((targets.new_full((targets.shape[0], 1), self.BOS_ID), targets[:, :-1]), dim=1)
        layout = build_native_ar_layout(observation_valid, history_valid, ids.shape[1])
        embeds = torch.cat((context, self._token_inputs(ids).to(context.dtype)), dim=1)
        out = self.joint.native_forward(embeds, layout)
        return self._head(out.hidden[:, -ids.shape[1]:])

    @torch.no_grad()
    def generate_tokens(self, context, observation_valid, history_valid, n_tokens, temperature, topk):
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or not 1 <= n_tokens <= self.max_seq_len:
            raise ValueError(f"n_tokens must be in [1,{self.max_seq_len}]")
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if topk is not None and (isinstance(topk, bool) or not isinstance(topk, int) or topk < 1):
            raise ValueError("topk must be null or a positive integer")
        layout = build_native_ar_layout(observation_valid, history_valid, n_tokens)
        bos = torch.full((context.shape[0], 1), self.BOS_ID, dtype=torch.long, device=context.device)
        inputs = torch.cat((context, self._token_inputs(bos).to(context.dtype)), dim=1)
        out = self.joint.native_forward(inputs, native_prefill_view(layout, 1), use_cache=True)
        generated = []
        for index in range(n_tokens):
            token = self._sample(self._head(out.hidden[:, -1]), temperature, topk, self.BOS_ID)
            generated.append(token)
            if index + 1 < n_tokens:
                out = self.joint.native_forward(self._token_inputs(token[:, None]).to(context.dtype),
                                                native_decode_view(layout, index + 1),
                                                cache=out.cache, use_cache=True)
        return torch.stack(generated, dim=1)

    @torch.no_grad()
    def _maybe_self_past(self, batch, past, probability=None):
        history = super()._maybe_self_past(batch, past, probability)
        valid = bool_mask(batch["past_action_valid"], past.shape[:2], "past_action_valid", past.device)
        substituted = valid & (history != past).any(-1)
        if substituted.any():
            normalize = self.action_normalizer["action"].normalize
            with torch.autocast(device_type=past.device.type, enabled=False):
                error = (normalize(history.float()) - normalize(past.float())).square().mean(-1)
            self._last_self_past_mse = float(error[substituted].mean())
        return history

    def forward(self, batch, history_mode=None):
        self._check_ready()
        self._last_self_past_rows, self._last_self_past_mse = 0, None
        past, probability = self._resolve_history(batch, history_mode)
        targets = self.encode_targets(batch["action"])
        with self._autocast():
            context = self.build_native_context(batch["obs"], past, batch["past_action_valid"],
                                                 train_aug=self.training and self.train_image_aug)
            logits = self.ar_logits(*context, targets)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        self.last_loss_components = dict(
            loss=loss.detach().item(), loss_ar=loss.detach().item(), loss_flow=None, loss_ki=None,
            ar_token_acc=(logits.detach().argmax(-1) == targets).float().mean().item(), ki_token_acc=None,
            self_past_p=float(probability), self_past_rows=float(self._last_self_past_rows),
            self_past_hist_mse=self._last_self_past_mse, hist_attention_mass=None,
            gate_mean=None, gate_min=None, gate_max=None)
        return loss

    def predict_action(self, obs_dict, use_k_tokens=None, temperature=None, topk=None,
                       past_actions=None, past_action_valid=None):
        self._acting = True
        self._prediction_tokens = None
        try:
            result = super().predict_action(obs_dict, use_k_tokens=use_k_tokens, temperature=temperature,
                                            topk=topk, past_actions=past_actions,
                                            past_action_valid=past_action_valid)
            if self._prediction_tokens is not None:
                result["action_tokens"] = self._prediction_tokens
            return result
        finally:
            self._acting = False
            self._prediction_tokens = None

    def _generate_actions(self, obs, past, valid, n_tokens, temperature, topk):
        self._check_ready()
        if self._acting and self.history_intervention is not None:
            past, valid = self._intervened_history(past, valid)
        with torch.no_grad(), self._autocast():
            context = self.build_native_context(obs, past, valid, train_aug=False)
            tokens = self.generate_tokens(*context, n_tokens, temperature, topk)
        if self._acting:
            self._prediction_tokens = tokens.detach()
        return self._detokenize(tokens)

    @staticmethod
    def _is_pretrained_group(name):
        return name.startswith("siglip.")

    @staticmethod
    def _is_ki_clip(name):
        # The workspace keeps its historical clip-group names; there is no KI loss.
        return name.startswith("siglip.") or ".lora_" in name

    def frozen_base_keys(self):
        keys = super().frozen_base_keys()
        return [key for key in keys if not (self.train_vision and key.startswith("siglip."))]

    def artifact_metadata(self):
        metadata = super().artifact_metadata()
        metadata[self.VARIANT] = dict(
            head="native_vlm_ar_oat8", history_order=["acc", "jerk", *[f"a-{i}" for i in range(7, 0, -1)]],
            condition_tokens=self.N_HISTORY_SLOTS, use_past=self.use_past, history_width=self.vlm_width,
            attention="observation_only__history_reads_context__action_causal",
            positions="valid_observations_then_9_reserved_history_slots_then_actions",
            action_table_init=f"normal(0,{self.AR_INIT_RMS}/sqrt(width)); tied; 5001-way CE",
            train_vision=self.train_vision, train_lora=self.train_lora,
            temperature=self.temperature, topk=self.topk,
            self_past_temperature=self.self_past_temperature, self_past_topk=self.self_past_topk)
        for source in (Path(__file__).resolve(), Path(__file__).with_name("pi05_past.py"),
                       Path(__file__).with_name("p2n_new_common.py")):
            metadata["source_sha256"][str(source.relative_to(REPO_ROOT))] = file_sha256(source)
        return metadata
