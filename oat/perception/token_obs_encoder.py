"""Multi-camera/frame observation tokens for p2n_new and p2n_state_gate_new.

The policy normalizes low-dimensional state before this module. Raw RGB bypasses
legacy vision normalizers and is processed only by DINOv3PatchEncoder.
"""
from __future__ import annotations

import copy
from typing import Mapping, Optional

import torch
from torch import nn

from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.dinov3_patch_encoder import DINOv3PatchEncoder
from oat.perception.visual_resampler import VisualResampler


class TokenObservationEncoder(BaseObservationEncoder):
    def __init__(self, shape_meta: Mapping, n_obs_steps: int = 2, n_emb: int = 768,
                 dino_encoder: Optional[nn.Module] = None, dino: Optional[Mapping] = None,
                 n_head: int = 12, ffn_dim: int = 2048, num_queries: int = 64,
                 resampler_depth: int = 2, dropout: float = 0.0,
                 activation_checkpointing: bool = False):
        super().__init__()
        if n_obs_steps < 1:
            raise ValueError("n_obs_steps must be positive.")
        self.shape_meta = copy.deepcopy(dict(shape_meta))
        self.n_obs_steps, self.n_emb = int(n_obs_steps), int(n_emb)
        self.rgb_ports, self.state_ports = [], []
        self.state_shapes = {}
        for name, info in shape_meta["obs"].items():
            modality = info.get("type", "state")
            if modality == "rgb":
                if len(info["shape"]) != 3 or info["shape"][-1] != 3:
                    raise ValueError(f"RGB schema for {name} must be [H,W,3].")
                self.rgb_ports.append(name)
            elif modality in ("state", "low_dim"):
                if len(info["shape"]) != 1 or info["shape"][0] < 1:
                    raise ValueError(f"State schema for {name} must be a nonempty vector.")
                self.state_ports.append(name)
                self.state_shapes[name] = int(info["shape"][0])
            else:
                raise ValueError(f"Unsupported observation modality {modality!r} for {name}; task IDs belong to state ports.")
        if not self.rgb_ports or not self.state_ports:
            raise ValueError("TokenObservationEncoder requires RGB cameras and current state/task information.")
        if dino_encoder is not None and dino is not None:
            raise ValueError("Supply either dino configuration or an injected dino_encoder.")
        self.dino_encoder = dino_encoder if dino_encoder is not None else DINOv3PatchEncoder(**dict(dino or {}))
        self.patch_projection = nn.Linear(self.dino_encoder.hidden_size, n_emb)
        self.resampler = VisualResampler(n_emb=n_emb, n_head=n_head, ffn_dim=ffn_dim,
                                        num_queries=num_queries, depth=resampler_depth,
                                        grid_size=self.dino_encoder.grid_size, dropout=dropout,
                                        activation_checkpointing=activation_checkpointing)
        self.state_projection = nn.Linear(sum(self.state_shapes.values()), n_emb)
        self.camera_embedding = nn.Parameter(torch.empty(len(self.rgb_ports), n_emb))
        self.frame_embedding = nn.Parameter(torch.empty(n_obs_steps, n_emb))
        for embedding in (self.camera_embedding, self.frame_embedding):
            nn.init.normal_(embedding, mean=0.0, std=0.02)
        self._architecture = dict(n_obs_steps=n_obs_steps, n_emb=n_emb, n_head=n_head,
                                  ffn_dim=ffn_dim, num_queries=num_queries,
                                  resampler_depth=resampler_depth, dropout=dropout,
                                  activation_checkpointing=activation_checkpointing)

    @property
    def num_queries(self) -> int:
        return self.resampler.num_queries

    @property
    def num_visual_tokens(self) -> int:
        return self.n_obs_steps * len(self.rgb_ports) * self.resampler.num_queries

    def modalities(self) -> list[str]:
        return ["rgb", "state"]

    def output_feature_dim(self) -> int:
        return self.n_emb

    def set_normalizer(self, normalizer):
        # Normalization deliberately belongs to the policy; RGB remains raw.
        return None

    def forward(self, obs_dict: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [port for port in self.rgb_ports + self.state_ports if port not in obs_dict]
        if missing:
            raise KeyError(f"Missing required observation ports: {missing}")
        cameras = [obs_dict[port] for port in self.rgb_ports]
        shape = cameras[0].shape
        if len(shape) != 5 or shape[1] != self.n_obs_steps or shape[-1] != 3:
            raise ValueError(f"RGB requires [B,{self.n_obs_steps},H,W,3], got {tuple(shape)}.")
        if any(image.shape != shape for image in cameras):
            raise ValueError("All required cameras must have the same batch/frame/image shape for batched DINO processing.")
        if any(image.dtype != cameras[0].dtype or image.device != cameras[0].device for image in cameras):
            raise ValueError("Camera tensors must share their explicitly declared dtype and device.")
        batch, frames, height, width, channels = shape
        n_cameras = len(cameras)
        # stack axis 2 is camera: reversible order is batch -> frame -> camera.
        image_batch = torch.stack(cameras, dim=2).reshape(batch * frames * n_cameras, height, width, channels)
        patches = self.dino_encoder(image_batch)
        projected = self.patch_projection(patches)
        visual = self.resampler(projected).reshape(batch, frames, n_cameras, self.resampler.num_queries, self.n_emb)
        visual = visual + self.camera_embedding[None, None, :, None, :].to(visual.dtype)
        visual = visual + self.frame_embedding[None, :, None, None, :].to(visual.dtype)
        visual = visual.reshape(batch, self.num_visual_tokens, self.n_emb)
        states = []
        for port in self.state_ports:
            state = obs_dict[port]
            if state.shape != (batch, frames, self.state_shapes[port]):
                raise ValueError(f"State {port} must be {(batch, frames, self.state_shapes[port])}, got {tuple(state.shape)}.")
            if not torch.isfinite(state).all():
                raise ValueError(f"Current state {port} contains nonfinite values.")
            states.append(state.to(dtype=self.state_projection.weight.dtype))
        proprio = self.state_projection(torch.cat(states, dim=-1))
        proprio = proprio + self.frame_embedding[None, :, :].to(proprio.dtype)
        return visual, proprio

    def export_config(self) -> dict:
        return dict(shape_meta=copy.deepcopy(self.shape_meta), **self._architecture,
                    dino=self.dino_encoder.export_config())
