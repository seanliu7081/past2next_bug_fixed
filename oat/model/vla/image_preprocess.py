"""Camera preprocessing for the PaliGemma/SigLIP prefix of P2N-VLA.

Input frames are byte-range RGB (uint8, or float 0..255 as the LIBERO runner delivers them), channels last,
``[B, To, H, W, 3]`` (the last frame is used) or ``[B, H, W, 3]``. LIBERO frames in the zarr and from
``LiberoEnv`` are already upright (both are vertically flipped from robosuite renders), so no flip or mirror
is applied here.

Pipeline per camera, all in fp32 with autocast disabled::

    bytes -> [0, 1] -> (train_aug) augment -> bilinear antialiased resize to S x S -> x * 2 - 1

Training augmentation follows openpi (``models/model.py`` ``preprocess_observation`` and
``models_pytorch/preprocessing_pytorch.py``), in this order:

- non-wrist cameras only: random crop of ``int(H*0.95) x int(W*0.95)`` at a uniform integer offset, bilinear
  resize back to ``H x W`` (``align_corners=False``), then rotation by ``U(-5, 5)`` degrees about the image
  centre (bilinear, zero padding);
- every camera: brightness ``x * U(0.7, 1.3)``, contrast ``(x - mean) * U(0.6, 1.4) + mean`` (mean over
  C, H, W), saturation ``gray + (x - gray) * U(0.5, 1.5)`` (gray = channel mean), then clamp to [0, 1].

Parameters are drawn independently per sample and per camera (openpi's JAX path draws per sample; its
PyTorch port draws one value per batch). Evaluation and self-past use the deterministic path.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


AUG_PARAM_NAMES = ("crop_top", "crop_left", "angle_deg", "brightness", "contrast", "saturation")


def _ports(value, name) -> List[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of observation keys")
    ports = list(value)
    if any(not isinstance(port, str) or not port for port in ports):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(ports)) != len(ports):
        raise ValueError(f"{name} must be distinct, got {ports}")
    return ports


def _fraction(value, name, low, high) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not low <= float(value) <= high:
        raise ValueError(f"{name} must be a number in [{low}, {high}], got {value!r}")
    return float(value)


class ImagePreprocessor:
    """Byte RGB observations -> ``[B, n_cams, 3, S, S]`` fp32 in [-1, 1] (cameras in ``rgb_ports`` order)."""

    def __init__(self, rgb_ports: Sequence[str], wrist_ports: Sequence[str] = (), image_size: int = 224, *,
                 crop_scale: float = 0.95, max_rotation_deg: float = 5.0, brightness: float = 0.3,
                 contrast: float = 0.4, saturation: float = 0.5):
        self.rgb_ports = _ports(rgb_ports, "rgb_ports")
        if not self.rgb_ports:
            raise ValueError("rgb_ports must not be empty")
        self.wrist_ports = _ports(wrist_ports, "wrist_ports")
        unknown = sorted(set(self.wrist_ports) - set(self.rgb_ports))
        if unknown:
            raise ValueError(f"wrist_ports must be a subset of rgb_ports; unknown: {unknown}")
        if isinstance(image_size, bool) or not isinstance(image_size, Integral) or image_size < 1:
            raise ValueError(f"image_size must be a positive integer, got {image_size!r}")
        self.image_size = int(image_size)
        self.crop_scale = _fraction(crop_scale, "crop_scale", 0.0, 1.0)
        if self.crop_scale <= 0:
            raise ValueError("crop_scale must be positive")
        self.max_rotation_deg = _fraction(max_rotation_deg, "max_rotation_deg", 0.0, 180.0)
        self.brightness = _fraction(brightness, "brightness", 0.0, 1.0)
        self.contrast = _fraction(contrast, "contrast", 0.0, 1.0)
        self.saturation = _fraction(saturation, "saturation", 0.0, 1.0)

    def __repr__(self) -> str:
        return (f"ImagePreprocessor(rgb_ports={self.rgb_ports}, wrist_ports={self.wrist_ports}, "
                f"image_size={self.image_size}, crop_scale={self.crop_scale}, "
                f"max_rotation_deg={self.max_rotation_deg}, brightness={self.brightness}, "
                f"contrast={self.contrast}, saturation={self.saturation})")

    # ------------------------------------------------------------------ inputs
    def frames(self, obs: Mapping[str, Tensor]) -> List[Tensor]:
        """Last frame of each camera as ``[B, 3, H, W]`` fp32 in [0, 1]."""
        if not isinstance(obs, Mapping):
            raise TypeError("Observations must be a mapping of tensors")
        frames, batch, device = [], None, None
        for port in self.rgb_ports:
            if port not in obs:
                raise KeyError(f"RGB port {port!r} missing from observations")
            value = obs[port]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"RGB observation {port!r} must be a torch.Tensor")
            if value.ndim == 5:
                if value.shape[1] < 1:
                    raise ValueError(f"RGB observation {port!r} has an empty time axis")
                value = value[:, -1]
            elif value.ndim != 4:
                raise ValueError(f"RGB observation {port!r} must have shape [B,To,H,W,3] or [B,H,W,3], "
                                 f"got {tuple(value.shape)}")
            if value.shape[-1] != 3 or value.shape[1] < 1 or value.shape[2] < 1:
                raise ValueError(f"RGB observation {port!r} must be channels-last RGB, got {tuple(value.shape)}")
            if batch is None:
                batch, device = value.shape[0], value.device
                if batch < 1:
                    raise ValueError("RGB observations must have a nonempty batch")
            elif value.shape[0] != batch:
                raise ValueError(f"RGB observation {port!r} has batch {value.shape[0]}, expected {batch}")
            elif value.device != device:
                raise ValueError(f"RGB observation {port!r} is on {value.device}, expected {device}")
            if value.dtype == torch.uint8:
                image = value.float()
            elif value.is_floating_point():
                image = value.float()
                if (~torch.isfinite(image) | (image < 0) | (image > 255)).any():
                    raise ValueError(f"Floating RGB observation {port!r} must be finite bytes in [0, 255]")
            else:
                raise TypeError(f"RGB observation {port!r} must be uint8 or float 0..255, got {value.dtype}")
            frames.append((image / 255.0).permute(0, 3, 1, 2).contiguous())
        return frames

    # ------------------------------------------------------------------ augmentation parameters
    def sample_params(self, batch: int, device: torch.device,
                      generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
        """Per-(camera, sample) augmentation parameters ``{name: [n_cams, B]}`` on ``device``.

        ``crop_top``/``crop_left`` are uniform fractions in [0, 1) mapped to integer offsets per image size.
        """
        if isinstance(batch, bool) or not isinstance(batch, Integral) or batch < 1:
            raise ValueError("batch must be a positive integer")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        draw_device = generator.device if generator is not None else torch.device(device)
        u = torch.rand((len(AUG_PARAM_NAMES), len(self.rgb_ports), int(batch)), generator=generator,
                       device=draw_device, dtype=torch.float32).to(device)
        params = dict(zip(AUG_PARAM_NAMES, u.unbind(0)))
        params["angle_deg"] = (params["angle_deg"] * 2 - 1) * self.max_rotation_deg
        for name, magnitude in (("brightness", self.brightness), ("contrast", self.contrast),
                                ("saturation", self.saturation)):
            params[name] = 1.0 + (params[name] * 2 - 1) * magnitude
        return params

    # ------------------------------------------------------------------ augmentation ops
    def crop_size(self, height: int, width: int):
        crop_h, crop_w = int(height * self.crop_scale), int(width * self.crop_scale)
        if crop_h < 1 or crop_w < 1:
            raise ValueError(f"crop_scale {self.crop_scale} leaves no pixels of a {height}x{width} image")
        return crop_h, crop_w

    def random_crop_resize(self, image: Tensor, crop_top: Tensor, crop_left: Tensor) -> Tensor:
        """openpi crop: ``image[:, :, top:top+ch, left:left+cw]`` resized back to ``H x W`` (per sample)."""
        batch, _, height, width = image.shape
        crop_h, crop_w = self.crop_size(height, width)
        if (crop_h, crop_w) == (height, width):
            return image
        top = torch.clamp((crop_top * (height - crop_h + 1)).floor().long(), 0, height - crop_h)
        left = torch.clamp((crop_left * (width - crop_w + 1)).floor().long(), 0, width - crop_w)
        rows = top[:, None] + torch.arange(crop_h, device=image.device)
        cols = left[:, None] + torch.arange(crop_w, device=image.device)
        index = torch.arange(batch, device=image.device)[:, None, None]
        crops = image.permute(0, 2, 3, 1)[index, rows[:, :, None], cols[:, None, :]]  # [B, ch, cw, C]
        return F.interpolate(crops.permute(0, 3, 1, 2), size=(height, width), mode="bilinear",
                             align_corners=False)

    @staticmethod
    def rotate(image: Tensor, angle_deg: Tensor) -> Tensor:
        """Rotate each image about its centre by ``angle_deg`` (bilinear, zero padding, pixel-aspect exact)."""
        batch, _, height, width = image.shape
        radians = angle_deg.to(image.dtype) * (math.pi / 180.0)
        cos, sin = torch.cos(radians), torch.sin(radians)
        zero = torch.zeros_like(cos)
        theta = torch.stack((torch.stack((cos, -sin * (height / width), zero), dim=-1),
                             torch.stack((sin * (width / height), cos, zero), dim=-1)), dim=1)
        grid = F.affine_grid(theta, [batch, image.shape[1], height, width], align_corners=False)
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    @staticmethod
    def color_jitter(image: Tensor, brightness: Tensor, contrast: Tensor, saturation: Tensor) -> Tensor:
        """openpi colour ops with per-sample factors, then clamp to [0, 1]."""
        view = (-1, 1, 1, 1)
        image = image * brightness.view(view)
        mean = image.mean(dim=(1, 2, 3), keepdim=True)
        image = (image - mean) * contrast.view(view) + mean
        gray = image.mean(dim=1, keepdim=True)
        image = gray + (image - gray) * saturation.view(view)
        return image.clamp(0.0, 1.0)

    def augment(self, image: Tensor, params: Mapping[str, Tensor], camera: int) -> Tensor:
        """Training augmentation of one camera ``[B, 3, H, W]`` in [0, 1] with ``params[name][camera]``."""
        if self.rgb_ports[camera] not in self.wrist_ports:
            image = self.random_crop_resize(image, params["crop_top"][camera], params["crop_left"][camera])
            if self.max_rotation_deg > 0:
                image = self.rotate(image, params["angle_deg"][camera])
        return self.color_jitter(image, params["brightness"][camera], params["contrast"][camera],
                                 params["saturation"][camera])

    def finalize(self, image: Tensor) -> Tensor:
        """Resize to ``S x S`` (bilinear, antialiased) and map [0, 1] -> [-1, 1]."""
        if tuple(image.shape[-2:]) != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear",
                                  align_corners=False, antialias=True)
        return image.clamp(0.0, 1.0) * 2.0 - 1.0

    # ------------------------------------------------------------------ entry point
    def __call__(self, obs: Mapping[str, Tensor], *, train_aug: bool,
                 generator: Optional[torch.Generator] = None) -> Tensor:
        if not isinstance(train_aug, bool):
            raise TypeError("train_aug must be a bool")
        frames = self.frames(obs)
        batch, device = frames[0].shape[0], frames[0].device
        with torch.autocast(device_type=device.type, enabled=False):
            if train_aug:
                params = self.sample_params(batch, device, generator)
                frames = [self.augment(image, params, camera) for camera, image in enumerate(frames)]
            return torch.stack([self.finalize(image) for image in frames], dim=1)
