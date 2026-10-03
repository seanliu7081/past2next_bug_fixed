"""Dual-scale Nano features and a narrow resampler for modern Past2Next.

Only the feature backbone is frozen. Raw camera bytes bypass the policy's state
normalizer; the returned state tokens use the original 768-dimensional path.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from oat.model.autoregressive.modern_transformer_cache import RMSNorm
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.convnext_feature_encoder import ConvNeXtFeatureEncoder
from oat.perception.visual_resampler import VisualResampler


ENCODER_TYPE = "convnextv2_tokens"
ENCODER_SCHEMA_VERSION = 1
EMBEDDING_LAYOUT = "batch_frame_camera_query"


def fusion_config(width=256):
    return dict(input_channels=[320, 640], width=int(width), grid_size=14,
                interpolation="bilinear", align_corners=False,
                scale="1/sqrt(2)", normalization="rmsnorm_channels", eps=1e-6)


class DualScaleFusion(nn.Module):
    """Project before resizing, sum with variance scaling, normalize channels."""

    def __init__(self, width=256):
        super().__init__()
        self.projection16 = nn.Conv2d(320, width, kernel_size=1)
        self.projection32 = nn.Conv2d(640, width, kernel_size=1)
        self.norm = RMSNorm(width, eps=1e-6)

    def forward(self, f16, f32):
        if f16.ndim != 4 or tuple(f16.shape[1:]) != (320, 14, 14):
            raise ValueError("F16 must be [N,320,14,14]")
        if f32.shape != (f16.shape[0], 640, 7, 7):
            raise ValueError("F32 must be [N,640,7,7]")
        p16 = self.projection16(f16)
        p32 = F.interpolate(self.projection32(f32), size=(14, 14),
                            mode="bilinear", align_corners=False)
        fused = (p16 + p32) / math.sqrt(2.0)
        # RMSNorm reduces its last dimension in FP32: channels, never space.
        return self.norm(fused.flatten(2).transpose(1, 2))


class ConvNeXtTokenObservationEncoder(BaseObservationEncoder):
    def __init__(self, shape_meta: Mapping, n_obs_steps=2, n_emb=768,
                 convnext: Optional[Mapping] = None,
                 convnext_encoder: Optional[nn.Module] = None,
                 visual_resampler_dim=256, visual_resampler_heads=4,
                 visual_resampler_ffn_dim=768, num_queries=64,
                 resampler_depth=2, dropout=0.1, activation_checkpointing=False,
                 encoder_type=ENCODER_TYPE, schema_version=ENCODER_SCHEMA_VERSION,
                 fusion=None, embedding_layout=EMBEDDING_LAYOUT):
        super().__init__()
        if encoder_type != ENCODER_TYPE or schema_version != ENCODER_SCHEMA_VERSION:
            raise ValueError("Unsupported ConvNeXt observation encoder type/schema")
        if embedding_layout != EMBEDDING_LAYOUT:
            raise ValueError("ConvNeXt tokens require batch/frame/camera/query order")
        if min(n_obs_steps, n_emb, visual_resampler_dim, visual_resampler_heads,
               visual_resampler_ffn_dim) < 1:
            raise ValueError("Observation and adapter dimensions must be positive")
        expected_fusion = fusion_config(visual_resampler_dim)
        if fusion is not None and dict(fusion) != expected_fusion:
            raise ValueError("ConvNeXt dual-scale fusion configuration does not match the fixed formula")
        self.encoder_type = encoder_type
        self.schema_version = schema_version
        self.shape_meta = copy.deepcopy(dict(shape_meta))
        self.n_obs_steps, self.n_emb = int(n_obs_steps), int(n_emb)
        self.rgb_ports, self.state_ports, self.state_shapes = [], [], {}
        for name, info in self.shape_meta["obs"].items():
            modality = info.get("type", "state")
            if modality == "rgb":
                if len(info["shape"]) != 3 or info["shape"][-1] != 3:
                    raise ValueError(f"RGB schema for {name} must be [H,W,3]")
                self.rgb_ports.append(name)
            elif modality in ("state", "low_dim"):
                if len(info["shape"]) != 1 or info["shape"][0] < 1:
                    raise ValueError(f"State schema for {name} must be a nonempty vector")
                self.state_ports.append(name)
                self.state_shapes[name] = int(info["shape"][0])
            else:
                raise ValueError(f"Unsupported observation modality {modality!r} for {name}")
        if not self.rgb_ports or not self.state_ports:
            raise ValueError("ConvNeXt tokens require RGB cameras and current state information")
        if convnext is not None and convnext_encoder is not None:
            raise ValueError("Supply either convnext configuration or an injected convnext_encoder")
        self.convnext_encoder = (convnext_encoder if convnext_encoder is not None
                                 else ConvNeXtFeatureEncoder(**dict(convnext or {})))
        self.fusion = DualScaleFusion(visual_resampler_dim)
        self.resampler = VisualResampler(
            n_emb=visual_resampler_dim, n_head=visual_resampler_heads,
            ffn_dim=visual_resampler_ffn_dim, num_queries=num_queries,
            depth=resampler_depth, grid_size=14, dropout=dropout,
            activation_checkpointing=activation_checkpointing)
        self.output_projection = nn.Linear(visual_resampler_dim, n_emb)
        self.state_projection = nn.Linear(sum(self.state_shapes.values()), n_emb)
        self.camera_embedding = nn.Parameter(torch.empty(len(self.rgb_ports), n_emb))
        self.frame_embedding = nn.Parameter(torch.empty(n_obs_steps, n_emb))
        for embedding in (self.camera_embedding, self.frame_embedding):
            nn.init.normal_(embedding, std=0.02)
        self._architecture = dict(
            n_obs_steps=n_obs_steps, n_emb=n_emb,
            visual_resampler_dim=visual_resampler_dim,
            visual_resampler_heads=visual_resampler_heads,
            visual_resampler_ffn_dim=visual_resampler_ffn_dim,
            num_queries=num_queries, resampler_depth=resampler_depth,
            dropout=dropout, activation_checkpointing=activation_checkpointing,
            fusion=expected_fusion, embedding_layout=embedding_layout)
        self.maintain_frozen_backbone_mode()

    @property
    def num_queries(self):
        return self.resampler.num_queries

    @property
    def num_visual_tokens(self):
        return self.n_obs_steps * len(self.rgb_ports) * self.num_queries

    def modalities(self):
        return ["rgb", "state"]

    def output_feature_dim(self):
        return self.n_emb

    def set_normalizer(self, normalizer):
        # Only the policy normalizes low-dimensional state. RGB stays raw.
        return None

    def maintain_frozen_backbone_mode(self):
        self.convnext_encoder.maintain_frozen_backbone_mode()

    def train(self, mode=True):
        super().train(mode)
        self.maintain_frozen_backbone_mode()
        return self

    def forward(self, obs_dict):
        missing = [key for key in self.rgb_ports + self.state_ports if key not in obs_dict]
        if missing:
            raise KeyError(f"Missing required observation ports: {missing}")
        cameras = [obs_dict[key] for key in self.rgb_ports]
        shape = cameras[0].shape
        if len(shape) != 5 or shape[1] != self.n_obs_steps or shape[-1] != 3:
            raise ValueError(f"RGB requires [B,{self.n_obs_steps},H,W,3], got {tuple(shape)}")
        if any(image.shape != shape for image in cameras):
            raise ValueError("All cameras must share the same batch/frame/image shape")
        if any(image.dtype != cameras[0].dtype or image.device != cameras[0].device
               for image in cameras):
            raise ValueError("Camera tensors must share dtype and device")
        batch, frames, height, width, channels = shape
        camera_count = len(cameras)
        # Camera axis follows frame, preserving the existing policy layout.
        images = torch.stack(cameras, dim=2).reshape(
            batch * frames * camera_count, height, width, channels)
        f16, f32 = self.convnext_encoder(images)
        # These operations must remain outside the frozen feature no_grad block.
        visual = self.output_projection(self.resampler(self.fusion(f16, f32)))
        visual = visual.reshape(batch, frames, camera_count, self.num_queries, self.n_emb)
        visual = visual + self.camera_embedding[None, None, :, None, :].to(visual.dtype)
        visual = visual + self.frame_embedding[None, :, None, None, :].to(visual.dtype)
        visual = visual.reshape(batch, self.num_visual_tokens, self.n_emb)
        states = []
        for key in self.state_ports:
            state = obs_dict[key]
            expected = (batch, frames, self.state_shapes[key])
            if state.shape != expected:
                raise ValueError(f"State {key} must be {expected}, got {tuple(state.shape)}")
            if not torch.isfinite(state).all():
                raise ValueError(f"Current state {key} contains nonfinite values")
            states.append(state.to(dtype=self.state_projection.weight.dtype))
        proprio = self.state_projection(torch.cat(states, dim=-1))
        proprio = proprio + self.frame_embedding[None, :, :].to(proprio.dtype)
        return visual, proprio

    def export_config(self):
        return dict(encoder_type=ENCODER_TYPE, schema_version=ENCODER_SCHEMA_VERSION,
                    shape_meta=copy.deepcopy(self.shape_meta),
                    **copy.deepcopy(self._architecture),
                    convnext=self.convnext_encoder.export_config())

    def parameter_counts(self):
        def count(module):
            return sum(p.numel() for p in module.parameters())
        adapter = sum(count(module) for module in
                      (self.fusion, self.resampler, self.output_projection))
        return dict(backbone=count(self.convnext_encoder.backbone),
                    fusion=count(self.fusion), resampler=count(self.resampler),
                    output_projection=count(self.output_projection),
                    fusion_resampler_output=adapter,
                    camera_frame_embeddings=self.camera_embedding.numel() + self.frame_embedding.numel(),
                    state_projection=count(self.state_projection),
                    total=count(self),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad))
