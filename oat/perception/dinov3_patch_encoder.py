"""Frozen, explicitly sourced DINOv3-S/16 patch features for the new policies.

Fresh construction is strictly local. ``restore`` only allocates the architecture
saved in a complete policy artifact; its first forward requires loaded weights.
Neither path can silently replace a missing pretrained backbone with random weights.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


DINO_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _weight_digest(directory: Path) -> str:
    files = sorted(directory.glob("*.safetensors")) or sorted(directory.glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"No DINO weights in {directory}; prepare the approved DINOv3-S/16 snapshot first.")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class DINOv3PatchEncoder(nn.Module):
    """Return final normalized patches, excluding CLS and configured registers.

    ``backbone`` is an explicit dependency-injection hook for small contract tests.
    Production uses the fixed DINOv3-S/16 configuration and approved local weights.
    Float RGB input requires ``rgb_range='0_1'`` or ``'0_255'``; values are never
    classified by their observed maximum. Channel order is always RGB.
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        *,
        load_mode: str = "fresh",
        config: Optional[Mapping] = None,
        processor_config: Optional[Mapping] = None,
        revision: Optional[str] = None,
        weight_sha256: Optional[str] = None,
        model_id: str = DINO_MODEL_ID,
        rgb_range: str = "uint8",
        image_size: int = 224,
        geometry: str = "square_resize",
        interpolation: str = "bicubic",
        antialias: bool = True,
        brightness: float = 0.0,
        contrast: float = 0.0,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        if load_mode not in ("fresh", "restore"):
            raise ValueError("DINO load_mode must be 'fresh' or 'restore'.")
        if model_id != DINO_MODEL_ID:
            raise ValueError(f"Expected the fixed backbone {DINO_MODEL_ID}, got {model_id}.")
        if rgb_range not in ("uint8", "0_1", "0_255"):
            raise ValueError("rgb_range must explicitly be 'uint8', '0_1', or '0_255'.")
        if geometry not in ("square_resize", "stretch"):
            raise ValueError("Use square_resize, or explicitly opt into stretch for nonsquare inputs.")
        if interpolation not in ("bilinear", "bicubic"):
            raise ValueError("interpolation must be bilinear or bicubic.")
        if not 0 <= brightness <= 1 or not 0 <= contrast <= 1:
            raise ValueError("brightness and contrast must lie in [0, 1].")
        self.model_id = model_id
        self.rgb_range = rgb_range
        self.image_size = int(image_size)
        self.geometry = geometry
        self.interpolation = interpolation
        self.antialias = bool(antialias)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self._injected_backbone = backbone is not None
        self._restore_ready = load_mode == "fresh" or self._injected_backbone
        self._load_prefix = ""

        if backbone is not None:
            if config is None:
                raw_config = getattr(backbone, "config", None)
                if raw_config is None:
                    raise ValueError("An injected backbone must supply its explicit config.")
                config = raw_config.to_dict() if hasattr(raw_config, "to_dict") else vars(raw_config)
            if processor_config is None:
                raise ValueError("An injected test backbone requires explicit processor_config.")
            self.backbone = backbone
        else:
            try:
                from transformers import DINOv3ViTConfig, DINOv3ViTModel
            except ImportError as exc:
                raise ImportError("DINOv3 requires the installed Transformers DINOv3ViT implementation (4.57.6 supported).") from exc
            if load_mode == "fresh":
                if pretrained_path is None:
                    raise ValueError("Fresh training requires a local pretrained_path for approved DINOv3-S/16 weights.")
                directory = Path(pretrained_path).expanduser()
                if not directory.is_dir():
                    raise FileNotFoundError(f"DINO snapshot does not exist: {directory}")
                for filename in ("config.json", "preprocessor_config.json"):
                    if not (directory / filename).is_file():
                        raise FileNotFoundError(f"DINO snapshot lacks {filename}: {directory}")
                with (directory / "config.json").open() as stream:
                    source_config = json.load(stream)
                with (directory / "preprocessor_config.json").open() as stream:
                    source_processor = json.load(stream)
                if config is not None and dict(config) != source_config:
                    raise ValueError("Fresh DINO config must come from its local pretrained snapshot.")
                if processor_config is not None and dict(processor_config) != source_processor:
                    raise ValueError("Fresh DINO processor config must come from its local pretrained snapshot.")
                config, processor_config = source_config, source_processor
                revision = revision or source_config.get("_commit_hash")
                if revision is None and re.fullmatch(r"[0-9a-fA-F]{40}", directory.name):
                    revision = directory.name
                if revision is None or not re.fullmatch(r"[0-9a-fA-F]{40}", str(revision)):
                    raise ValueError("Supply the exact 40-character DINO commit revision, or use a local Hugging Face snapshots/<commit> directory.")
                actual_digest = _weight_digest(directory)
                if weight_sha256 is not None and weight_sha256 != actual_digest:
                    raise ValueError("DINO weight SHA256 does not match the approved snapshot.")
                weight_sha256 = actual_digest
                self._validate_production_config(config)
                self.backbone = DINOv3ViTModel.from_pretrained(str(directory), local_files_only=True)
            else:
                if config is None or processor_config is None:
                    raise ValueError("Offline DINO restore requires saved architecture and processor_config.")
                self._validate_production_config(config)
                self.backbone = DINOv3ViTModel(DINOv3ViTConfig.from_dict(dict(config)))

        self.config = copy.deepcopy(dict(config))
        self.processor_config = copy.deepcopy(dict(processor_config))
        self.revision = revision
        self.weight_sha256 = weight_sha256
        self.patch_size = int(self.config["patch_size"])
        self.hidden_size = int(self.config["hidden_size"])
        self.num_register_tokens = int(self.config["num_register_tokens"])
        if self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("image_size must be a positive multiple of the DINO patch size.")
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size ** 2
        if not self._injected_backbone and self.image_size != 224:
            raise ValueError("The production p2n_new DINO input is fixed at 224 x 224.")
        if not self.processor_config.get("do_normalize", True):
            raise ValueError("The DINO processor must provide its pretrained mean/std normalization.")
        if not self.processor_config.get("do_rescale", True):
            raise ValueError("The DINO processor must declare its byte-range rescale factor.")
        if "rescale_factor" not in self.processor_config:
            raise ValueError("processor_config must explicitly include rescale_factor.")
        self.rescale_factor = float(self.processor_config["rescale_factor"])
        if self.rescale_factor <= 0:
            raise ValueError("DINO processor rescale_factor must be positive.")
        mean = torch.tensor(self.processor_config["image_mean"], dtype=torch.float32)
        std = torch.tensor(self.processor_config["image_std"], dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("DINO processor image_mean/std must be three finite RGB values, with positive std.")
        self.register_buffer("image_mean", mean.reshape(1, 3, 1, 1))
        self.register_buffer("image_std", std.reshape(1, 3, 1, 1))
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.register_load_state_dict_pre_hook(self._before_load)
        self.register_load_state_dict_post_hook(self._after_load)

    @staticmethod
    def _validate_production_config(config: Mapping) -> None:
        expected = {"model_type": "dinov3_vit", "patch_size": 16, "hidden_size": 384,
                    "num_hidden_layers": 12, "num_attention_heads": 6, "num_register_tokens": 4}
        mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
        if mismatch:
            raise ValueError(f"Expected DINOv3-S/16 configuration; mismatched (actual, expected): {mismatch}")

    def _before_load(self, module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self._load_prefix = prefix

    def _after_load(self, module, incompatible_keys):
        self._restore_ready = not any(key.startswith(self._load_prefix) for key in incompatible_keys.missing_keys)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def preprocess(self, rgb: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        """Convert NHWC raw RGB to normalized NCHW exactly once."""
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB [N,H,W,3], got {tuple(rgb.shape)}.")
        if self.geometry == "square_resize" and rgb.shape[1] != rgb.shape[2]:
            raise ValueError("Nonsquare RGB requires an explicit geometry configuration.")
        if rgb.dtype == torch.uint8:
            byte_rgb = rgb.to(dtype=torch.float32)
        elif rgb.is_floating_point():
            if self.rgb_range == "uint8":
                raise TypeError("Float RGB requires explicit rgb_range='0_1' or '0_255'; normalized legacy RGB is unsupported.")
            upper = 1.0 if self.rgb_range == "0_1" else 255.0
            if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > upper).any():
                raise ValueError(f"RGB values violate declared range [0,{upper}].")
            byte_rgb = rgb.to(dtype=torch.float32) * (255.0 if self.rgb_range == "0_1" else 1.0)
        else:
            raise TypeError(f"RGB must be uint8 or explicitly declared floating point, got {rgb.dtype}.")
        pixels = byte_rgb.permute(0, 3, 1, 2).contiguous() * self.rescale_factor
        pixels = F.interpolate(pixels, size=(self.image_size, self.image_size), mode=self.interpolation,
                               align_corners=False, antialias=self.antialias)
        if self.training and not deterministic:
            shape = (pixels.shape[0], 1, 1, 1)
            if self.brightness:
                gain = 1 + torch.empty(shape, device=pixels.device).uniform_(-self.brightness, self.brightness)
                pixels = pixels * gain
            if self.contrast:
                gain = 1 + torch.empty(shape, device=pixels.device).uniform_(-self.contrast, self.contrast)
                center = pixels.mean(dim=(1, 2, 3), keepdim=True)
                pixels = center + gain * (pixels - center)
            if self.brightness or self.contrast:
                pixels = pixels.clamp(0.0, 255.0 * self.rescale_factor)
        return (pixels - self.image_mean) / self.image_std

    def forward(self, rgb: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        if not self._restore_ready:
            raise RuntimeError("DINO restore construction requires loading the complete policy state_dict before forward.")
        with torch.no_grad():
            output = self.backbone(pixel_values=self.preprocess(rgb, deterministic=deterministic))
            sequence = output.last_hidden_state
            expected = 1 + self.num_register_tokens + self.num_patches
            if sequence.shape != (rgb.shape[0], expected, self.hidden_size):
                raise RuntimeError(f"DINO output {tuple(sequence.shape)} violates expected {(rgb.shape[0], expected, self.hidden_size)}.")
            return sequence[:, 1 + self.num_register_tokens:, :]

    def export_config(self) -> dict:
        if self._injected_backbone:
            raise ValueError("An injected test backbone cannot be exported as a production DINO artifact.")
        return dict(load_mode="restore", model_id=self.model_id, config=copy.deepcopy(self.config),
                    processor_config=copy.deepcopy(self.processor_config), revision=self.revision,
                    weight_sha256=self.weight_sha256, rgb_range=self.rgb_range, image_size=self.image_size,
                    geometry=self.geometry, interpolation=self.interpolation, antialias=self.antialias,
                    brightness=self.brightness, contrast=self.contrast)
