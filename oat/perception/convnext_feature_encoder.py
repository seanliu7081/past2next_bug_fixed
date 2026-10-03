"""Frozen local ConvNeXt V2-Nano features with an offline restoration contract.

Only the feature extractor is frozen. Returned tensors are ordinary no-grad
tensors that a trainable projection can safely retain for its backward pass.
No construction path asks timm or the Hugging Face Hub to download weights.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


CONVNEXT_MODEL_NAME = "convnextv2_nano.fcmae_ft_in22k_in1k"
CONVNEXT_MODEL_ID = "timm/" + CONVNEXT_MODEL_NAME
TIMM_VERSION = "1.0.29"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
BACKBONE_CONFIG = dict(features_only=True, out_indices=[2, 3], in_chans=3,
                       output_stride=32, drop_path_rate=0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timm_source_fingerprints() -> dict:
    modules = ("timm.models.convnext", "timm.models._features", "timm.layers.grn",
               "timm.layers.norm", "timm.layers.mlp")
    return {name: _sha256(Path(importlib.import_module(name).__file__)) for name in modules}


def _resolve_weight_file(model_path: str) -> tuple[Path, dict]:
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local ConvNeXt weights do not exist: {path}")
    config = {}
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.is_file():
            with config_path.open() as stream:
                config = json.load(stream)
            if config.get("architecture", "convnextv2_nano") != "convnextv2_nano":
                raise ValueError("The local timm config is not convnextv2_nano.")
        # Hugging Face snapshots often contain both equivalent serialization
        # formats. Prefer the safe format, and record exactly which bytes load.
        if (path / "model.safetensors").is_file():
            path = path / "model.safetensors"
        elif (path / "pytorch_model.bin").is_file():
            path = path / "pytorch_model.bin"
        else:
            candidates = sorted(p for p in path.iterdir() if p.suffix in (".pt", ".pth", ".bin", ".safetensors"))
            if len(candidates) != 1:
                raise ValueError("Local ConvNeXt directory must contain model.safetensors, pytorch_model.bin, or one unambiguous weight file.")
            path = candidates[0]
    if not path.is_file() or path.suffix not in (".pt", ".pth", ".bin", ".safetensors"):
        raise ValueError("ConvNeXt weights must be a local .safetensors, .bin, .pt, or .pth file.")
    return path, config


def _read_local_state(path: Path) -> tuple[dict, str]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(str(path), device="cpu")
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    container_key = "root"
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict_ema", "model_ema", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                checkpoint, container_key = checkpoint[key], key
                break
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("ConvNeXt checkpoint must contain a nonempty tensor state_dict.")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
        raise ValueError("ConvNeXt state_dict must contain only named tensors.")
    return dict(checkpoint), container_key


def _load_feature_state(backbone: nn.Module, state: Mapping[str, torch.Tensor]) -> dict:
    """Validate every source key and all feature weights before a strict load.

    Accepted schemas are timm's full classifier, its features-only wrapper, and
    the official Facebook ConvNeXt V2 checkpoint. Only the four known final
    normalization/classifier tensors may be omitted from the feature extractor.
    """
    expected = backbone.state_dict()
    mapped, excluded, unexpected = {}, [], []
    head_shapes = {"head.norm.weight": (640,), "head.norm.bias": (640,),
                   "head.fc.weight": (1000, 640), "head.fc.bias": (1000,)}
    names = list(state)
    module_prefix = all(name.startswith("module.") for name in names)
    official = any(name.removeprefix("module.").startswith("downsample_layers.") for name in names)
    for source_key, tensor in state.items():
        key = source_key.removeprefix("module.") if module_prefix else source_key
        if official:
            key = key.replace("downsample_layers.0.", "stem.")
            key = re.sub(r"^stages\.(\d+)\.(\d+)\.", r"stages.\1.blocks.\2.", key)
            key = re.sub(r"^downsample_layers\.(\d+)\.(\d+)\.", r"stages.\1.downsample.\2.", key)
            key = key.replace(".dwconv.", ".conv_dw.")
            key = key.replace(".pwconv1.", ".mlp.fc1.").replace(".pwconv2.", ".mlp.fc2.")
            if key.endswith(".grn.gamma") or key.endswith(".grn.beta"):
                key = key.replace(".grn.gamma", ".mlp.grn.weight").replace(".grn.beta", ".mlp.grn.bias")
                if tensor.ndim != 4 or tuple(tensor.shape[:3]) != (1, 1, 1):
                    raise ValueError(f"Invalid official GRN tensor shape for {source_key}: {tuple(tensor.shape)}")
                tensor = tensor.reshape(-1)
            if key.startswith("norm."):
                key = "head.norm." + key[len("norm."):]
            elif key.startswith("head."):
                key = "head.fc." + key[len("head."):]
        if key in head_shapes:
            if tuple(tensor.shape) != head_shapes[key]:
                raise ValueError(f"Unexpected classification head shape for {source_key}: {tuple(tensor.shape)}")
            excluded.append(source_key)
            continue
        key = re.sub(r"^(stem|stages)\.(\d+)\.", r"\1_\2.", key)
        if key not in expected:
            unexpected.append(source_key)
            continue
        if key in mapped:
            raise ValueError(f"Multiple checkpoint keys map to the feature parameter {key}.")
        if official and key.endswith(("mlp.fc1.weight", "mlp.fc2.weight")) and tensor.ndim == 2:
            tensor = tensor[:, :, None, None]
        if tuple(tensor.shape) != tuple(expected[key].shape):
            raise ValueError(f"Feature shape mismatch for {source_key}: {tuple(tensor.shape)} versus {tuple(expected[key].shape)}")
        mapped[key] = tensor
    missing = sorted(set(expected) - set(mapped))
    if missing or unexpected:
        raise ValueError(f"Incomplete ConvNeXt feature weights: missing={missing}, unexpected={unexpected}")
    backbone.load_state_dict(mapped, strict=True)
    return dict(source_format="facebook" if official else "timm", excluded_head_keys=sorted(excluded),
                feature_state_keys=len(mapped))


class ConvNeXtFeatureEncoder(nn.Module):
    """Raw NHWC RGB to stride-16 and stride-32 frozen Nano feature maps.

    ``backbone`` is a deliberate CPU-test hook, never a production fallback.
    Production exports cannot be created from injected backbones.
    """

    def __init__(self, model_name: str = CONVNEXT_MODEL_NAME, model_path: Optional[str] = None,
                 revision: Optional[str] = None, construction_mode: str = "fresh", frozen: bool = True,
                 image_size: int = 224, feature_stages: Sequence[int] = (2, 3),
                 metadata: Optional[Mapping] = None, backbone_config: Optional[Mapping] = None,
                 input_range: str = "uint8_0_255", mean: Sequence[float] = IMAGE_MEAN,
                 std: Sequence[float] = IMAGE_STD, brightness: float = 0.1, contrast: float = 0.1,
                 interpolation: str = "bicubic", antialias: bool = True,
                 backbone: Optional[nn.Module] = None):
        super().__init__()
        if model_name != CONVNEXT_MODEL_NAME:
            raise ValueError(f"ConvNeXt visual backend requires {CONVNEXT_MODEL_NAME}.")
        if construction_mode not in ("fresh", "restore"):
            raise ValueError("construction_mode must be fresh or restore.")
        if frozen is not True:
            raise ValueError("ConvNeXt V2-Nano must remain frozen in this implementation.")
        if image_size != 224 or tuple(feature_stages) != (2, 3):
            raise ValueError("ConvNeXt V2-Nano requires image_size=224 and feature_stages=(2,3).")
        if input_range not in ("uint8_0_255", "float_0_1", "float_0_255"):
            raise ValueError("input_range must be uint8_0_255, float_0_1, or float_0_255.")
        if interpolation != "bicubic" or not isinstance(antialias, bool):
            raise ValueError("ConvNeXt preprocessing requires bicubic interpolation and an explicit bool antialias.")
        if not 0 <= brightness <= 1 or not 0 <= contrast <= 1:
            raise ValueError("brightness and contrast must lie in [0,1].")
        mean_tensor, std_tensor = torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
        if (mean_tensor.shape != (3,) or std_tensor.shape != (3,) or
                not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all() or (std_tensor <= 0).any()):
            raise ValueError("mean and std must contain three finite RGB values with positive std.")
        config = copy.deepcopy(dict(BACKBONE_CONFIG if backbone_config is None else backbone_config))
        if "out_indices" in config:
            config["out_indices"] = list(config["out_indices"])
        if config != BACKBONE_CONFIG:
            raise ValueError(f"ConvNeXt backbone_config must be exactly {BACKBONE_CONFIG}.")
        self.model_name, self.image_size = model_name, int(image_size)
        self.feature_stages, self.feature_channels, self.feature_reductions = (2, 3), (320, 640), (16, 32)
        self.feature_grids, self.grid_size = (14, 7), 14
        self.input_range, self.interpolation, self.antialias = input_range, interpolation, antialias
        self.mean, self.std = tuple(float(x) for x in mean), tuple(float(x) for x in std)
        self.brightness, self.contrast = float(brightness), float(contrast)
        self.frozen, self.backbone_config = True, config
        self.metadata = copy.deepcopy(dict(metadata or {}))
        self._injected_backbone = backbone is not None
        self._restore_ready = construction_mode == "fresh" or self._injected_backbone
        self._load_prefix = ""
        self.register_buffer("image_mean", mean_tensor.reshape(1, 3, 1, 1))
        self.register_buffer("image_std", std_tensor.reshape(1, 3, 1, 1))
        if self._injected_backbone:
            self.backbone = backbone
            self.revision = revision
        else:
            if self.mean != IMAGE_MEAN or self.std != IMAGE_STD:
                raise ValueError("Production ConvNeXt mean/std must match the fixed ImageNet pretrained weights.")
            weight_file, source_config = None, {}
            if construction_mode == "fresh":
                if model_path is None:
                    raise ValueError("Fresh ConvNeXt construction requires an explicit local model_path.")
                weight_file, source_config = _resolve_weight_file(model_path)
                revision = revision or source_config.get("_commit_hash")
                if revision is None and re.fullmatch(r"[0-9a-fA-F]{40}", weight_file.parent.name):
                    revision = weight_file.parent.name
            else:
                required = {"model_id", "revision", "weight_sha256", "timm_version", "source_sha256"}
                if required - set(self.metadata):
                    raise ValueError(f"Offline ConvNeXt restore requires provenance metadata: {sorted(required - set(self.metadata))}")
                revision = revision or self.metadata["revision"]
            if revision is None or not re.fullmatch(r"[0-9a-fA-F]{40}", str(revision)):
                raise ValueError("ConvNeXt requires a pinned 40-character weight revision.")
            if self.metadata.get("revision", revision) != revision:
                raise ValueError("ConvNeXt revision conflicts with saved metadata.")
            if self.metadata.get("model_id", CONVNEXT_MODEL_ID) != CONVNEXT_MODEL_ID:
                raise ValueError("ConvNeXt metadata model_id does not match the selected weights.")
            try:
                import timm
            except ImportError as exc:
                raise ImportError(f"ConvNeXt requires timm=={TIMM_VERSION}; see requirements-convnext-nano.txt.") from exc
            if timm.__version__ != TIMM_VERSION or self.metadata.get("timm_version", TIMM_VERSION) != TIMM_VERSION:
                raise ValueError(f"ConvNeXt requires pinned timm=={TIMM_VERSION}; installed {timm.__version__}.")
            source_fingerprints = _timm_source_fingerprints()
            if "source_sha256" in self.metadata and self.metadata["source_sha256"] != source_fingerprints:
                raise ValueError("Installed timm source fingerprints differ from the ConvNeXt artifact.")
            self.backbone = timm.create_model(model_name, pretrained=False, **config)
            if (tuple(self.backbone.feature_info.channels()) != self.feature_channels or
                    tuple(self.backbone.feature_info.reduction()) != self.feature_reductions):
                raise ValueError("timm ConvNeXt feature channels/reductions violate the Nano contract.")
            pretrained_cfg = self.backbone.pretrained_cfg
            if tuple(pretrained_cfg["mean"]) != self.mean or tuple(pretrained_cfg["std"]) != self.std:
                raise ValueError("Configured normalization differs from the selected timm pretrained mean/std.")
            if weight_file is not None:
                digest = _sha256(weight_file)
                if self.metadata.get("weight_sha256", digest) != digest:
                    raise ValueError("ConvNeXt local weight SHA256 does not match metadata.")
                state, container_key = _read_local_state(weight_file)
                loading = _load_feature_state(self.backbone, state)
                self.metadata.update(loading, weight_sha256=digest, source_file=weight_file.name,
                                     checkpoint_container=container_key)
            if not re.fullmatch(r"[0-9a-fA-F]{64}", str(self.metadata.get("weight_sha256", ""))):
                raise ValueError("ConvNeXt provenance requires a valid weight_sha256.")
            self.revision = str(revision)
            self.metadata.update(model_id=CONVNEXT_MODEL_ID, revision=self.revision,
                                 timm_version=TIMM_VERSION, source_sha256=source_fingerprints,
                                 feature_channels=list(self.feature_channels),
                                 feature_reductions=list(self.feature_reductions),
                                 feature_grids=list(self.feature_grids))
        self.maintain_frozen_backbone_mode()
        self.register_load_state_dict_pre_hook(self._before_load)
        self.register_load_state_dict_post_hook(self._after_load)

    def maintain_frozen_backbone_mode(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.maintain_frozen_backbone_mode()
        return self

    def _before_load(self, module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self._load_prefix = prefix
        # Check shapes here as well: PyTorch's post-hook does not expose shape
        # errors, and a failed restore must never unblock a random backbone.
        own_state = self.state_dict()
        self._restore_ready = all(prefix + key in state_dict and state_dict[prefix + key].shape == value.shape
                                  for key, value in own_state.items())

    def _after_load(self, module, incompatible_keys):
        self._restore_ready = self._restore_ready and not any(
            key.startswith(self._load_prefix) for key in incompatible_keys.missing_keys + incompatible_keys.unexpected_keys)
        self.maintain_frozen_backbone_mode()

    def preprocess(self, rgb: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        """FP32 full-field resize and exactly one declared RGB normalization."""
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.shape[0] < 1 or min(rgb.shape[1:3]) < 1:
            raise ValueError(f"Expected nonempty raw RGB [N,H,W,3], got {tuple(rgb.shape)}.")
        if self.input_range == "uint8_0_255":
            if rgb.dtype != torch.uint8:
                raise TypeError("uint8_0_255 RGB requires uint8 tensors; floats need an explicit float input_range.")
            scale = 1.0 / 255.0
        else:
            if not rgb.is_floating_point():
                raise TypeError("A float input_range requires floating point RGB tensors.")
            upper = 1.0 if self.input_range == "float_0_1" else 255.0
            if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > upper).any():
                raise ValueError(f"RGB values violate the declared input range [0,{upper}].")
            scale = 1.0 / upper
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            pixels = rgb.to(torch.float32).permute(0, 3, 1, 2).contiguous() * scale
            pixels = F.interpolate(pixels, size=(self.image_size, self.image_size), mode=self.interpolation,
                                   align_corners=False, antialias=self.antialias)
            if self.training and not deterministic:
                shape = (pixels.shape[0], 1, 1, 1)
                if self.brightness:
                    gain = 1 + torch.empty(shape, device=pixels.device, dtype=torch.float32).uniform_(-self.brightness, self.brightness)
                    pixels = pixels * gain
                if self.contrast:
                    gain = 1 + torch.empty(shape, device=pixels.device, dtype=torch.float32).uniform_(-self.contrast, self.contrast)
                    center = pixels.mean(dim=(1, 2, 3), keepdim=True)
                    pixels = center + gain * (pixels - center)
                if self.brightness or self.contrast:
                    pixels = pixels.clamp(0.0, 1.0)
            return (pixels - self.image_mean.float()) / self.image_std.float()

    def forward(self, rgb: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._restore_ready:
            raise RuntimeError("ConvNeXt restore requires loading the complete policy state_dict before forward.")
        self.maintain_frozen_backbone_mode()
        with torch.no_grad():
            features = self.backbone(self.preprocess(rgb, deterministic=deterministic))
        if not isinstance(features, (tuple, list)) or len(features) != 2:
            raise RuntimeError("ConvNeXt backbone must return exactly stride-16 and stride-32 feature maps.")
        for feature, channels, grid in zip(features, self.feature_channels, self.feature_grids):
            expected = (rgb.shape[0], channels, grid, grid)
            if tuple(feature.shape) != expected:
                raise RuntimeError(f"ConvNeXt feature shape {tuple(feature.shape)} violates {expected}.")
        return features[0], features[1]

    def export_config(self) -> dict:
        if self._injected_backbone:
            raise ValueError("An injected test backbone cannot be exported as a production ConvNeXt artifact.")
        return dict(model_name=self.model_name, construction_mode="restore", revision=self.revision,
                    frozen=True, image_size=self.image_size, feature_stages=list(self.feature_stages),
                    metadata=copy.deepcopy(self.metadata), backbone_config=copy.deepcopy(self.backbone_config),
                    input_range=self.input_range, mean=list(self.mean), std=list(self.std),
                    brightness=self.brightness, contrast=self.contrast,
                    interpolation=self.interpolation, antialias=self.antialias)
