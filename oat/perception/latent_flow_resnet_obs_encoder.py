"""Original trainable ResNet-18 observation features adapted to flow tokens.

The camera networks, GroupNorm, SpatialSoftmax, output ReLU, pixel normalizer,
and crop randomizers come directly from ``RobomimicRgbEncoder``. Only the small
64-to-DiTX projection and camera/frame embeddings adapt those features to the
latent-flow context. There is one visual token per camera and observed frame.

Crop realization is separate from feature extraction: EMA and student consume
the same detached, normalized crops and each run their own trainable backbone.
The policy normalizes state observations before calling ``forward``.
"""
from __future__ import annotations

import copy
from typing import Mapping

import torch
from torch import nn

from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.robomimic_vision_encoder import RobomimicRgbEncoder
from oat.perception.state_encoder import ProjectionStateEncoder


class FlowResNetObservationEncoder(BaseObservationEncoder):
    def __init__(self, shape_meta: Mapping, n_obs_steps: int = 2, n_emb: int = 768,
                 crop_shape=(112, 112), use_group_norm: bool = True,
                 share_rgb_model: bool = False, eval_fixed_crop: bool = True):
        super().__init__()
        if n_obs_steps < 1 or n_emb < 1:
            raise ValueError('n_obs_steps and n_emb must be positive')
        self.shape_meta = copy.deepcopy(dict(shape_meta))
        self.n_obs_steps, self.n_emb = int(n_obs_steps), int(n_emb)
        self.rgb_ports, self.state_ports, self.state_shapes = [], [], {}
        original_meta = copy.deepcopy(self.shape_meta)
        for port, info in self.shape_meta['obs'].items():
            modality = info.get('type', 'state')
            shape = tuple(info['shape'])
            if modality == 'rgb':
                if len(shape) != 3 or shape[-1] != 3:
                    raise ValueError(f'RGB schema for {port} must be [H,W,3]')
                self.rgb_ports.append(port)
            elif modality in ('state', 'low_dim'):
                if len(shape) != 1 or shape[0] < 1:
                    raise ValueError(f'State schema for {port} must be a nonempty vector')
                self.state_ports.append(port)
                self.state_shapes[port] = int(shape[0])
                original_meta['obs'][port]['type'] = 'state'
            else:
                raise ValueError(f'Unsupported observation modality {modality!r} for {port}')
        if not self.rgb_ports or not self.state_ports:
            raise ValueError('ResNet observation encoding requires RGB and current state ports')
        if crop_shape is not None:
            crop_shape = tuple(int(value) for value in crop_shape)
            if len(crop_shape) != 2 or min(crop_shape) < 1:
                raise ValueError('crop_shape must contain two positive sizes')
            for port in self.rgb_ports:
                height, width, _ = self.shape_meta['obs'][port]['shape']
                if crop_shape[0] >= height or crop_shape[1] >= width:
                    raise ValueError(f'Original random crop must be smaller than RGB shape for {port}')
        else:
            shapes = {tuple(self.shape_meta['obs'][port]['shape']) for port in self.rgb_ports}
            if len(shapes) != 1:
                raise ValueError('Uncropped cameras must share image dimensions')
        self.crop_shape = crop_shape
        # The original VisualCore uses ResNet18Conv(pretrained=False), so this
        # creates independent trainable camera backbones without any downloads.
        self.vision_encoder = RobomimicRgbEncoder(
            original_meta, crop_shape=crop_shape, use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model, eval_fixed_crop=eval_fixed_crop)
        self.state_encoder = ProjectionStateEncoder(original_meta, out_dim=n_emb)
        self.visual_projection = nn.Linear(64, n_emb)
        self.camera_embedding = nn.Parameter(torch.empty(len(self.rgb_ports), n_emb))
        self.frame_embedding = nn.Parameter(torch.empty(n_obs_steps, n_emb))
        for projection in (self.visual_projection, self.state_encoder.state_proj):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        for embedding in (self.camera_embedding, self.frame_embedding):
            nn.init.normal_(embedding, std=0.02)

        # Match the original real-robot dataset's fixed byte-range [-1,1]
        # normalization even for a freshly constructed policy's dummy inputs.
        normalizer = LinearNormalizer()
        for port in self.rgb_ports:
            normalizer[port] = SingleFieldLinearNormalizer.create_fit(
                torch.tensor([[0., 0., 0.], [255., 255., 255.]]), mode='limits')
        self.set_normalizer(normalizer)
        self._architecture = dict(
            n_obs_steps=n_obs_steps, n_emb=n_emb, crop_shape=crop_shape,
            use_group_norm=use_group_norm, share_rgb_model=share_rgb_model,
            eval_fixed_crop=eval_fixed_crop)

    @property
    def num_queries(self) -> int:
        return 1

    @property
    def num_visual_tokens(self) -> int:
        return self.n_obs_steps * len(self.rgb_ports)

    @property
    def state_projection(self):
        return self.state_encoder.state_proj

    def modalities(self):
        return ['rgb', 'state']

    def output_feature_dim(self) -> int:
        return self.n_emb

    def set_normalizer(self, normalizer):
        missing = [port for port in self.rgb_ports if port not in normalizer.params_dict]
        if missing:
            raise KeyError(f'Missing original RGB normalizer fields: {missing}')
        pixels = LinearNormalizer()
        for port in self.rgb_ports:
            pixels[port] = normalizer[port]
        self.vision_encoder.set_normalizer(pixels)
        self.vision_encoder.normalizer.requires_grad_(False)

    @torch.no_grad()
    def prepare_conditioning(self, obs):
        """Return detached [batch,frame,camera,3,height,width] normalized crops."""
        missing = [port for port in self.rgb_ports if port not in obs]
        if missing:
            raise KeyError(f'Missing required observation ports: {missing}')
        first = obs[self.rgb_ports[0]]
        if first.ndim != 5 or first.shape[1] != self.n_obs_steps:
            raise ValueError(f'RGB must be [B,{self.n_obs_steps},H,W,3]')
        batch = first.shape[0]
        for port in self.rgb_ports:
            value = obs[port]
            expected = (batch, self.n_obs_steps, *self.shape_meta['obs'][port]['shape'])
            if tuple(value.shape) != expected:
                raise ValueError(f'RGB {port} must have shape {expected}')
            if value.dtype != first.dtype or value.device != first.device:
                raise ValueError('All cameras must share dtype and device')
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite RGB observation: {port}')
        normalized = self.vision_encoder._normalize_obs_dict(obs)
        crops = []
        for port in self.rgb_ports:
            value = normalized[port]
            height, width, channels = value.shape[-3:]
            image = value.reshape(batch * self.n_obs_steps, height, width, channels).permute(0, 3, 1, 2)
            randomizer = self.vision_encoder.encoder.obs_randomizers[port]
            if randomizer is not None:
                image = randomizer.forward_in(image)
            crops.append(image.reshape(batch, self.n_obs_steps, *image.shape[1:]))
        return torch.stack(crops, dim=2).detach()

    def state_features(self, obs):
        """Concatenate already normalized states in the original port order."""
        missing = [port for port in self.state_ports if port not in obs]
        if missing:
            raise KeyError(f'Missing required observation ports: {missing}')
        batch = obs[self.state_ports[0]].shape[0]
        states = []
        for port in self.state_ports:
            value = obs[port]
            if tuple(value.shape) != (batch, self.n_obs_steps, self.state_shapes[port]):
                raise ValueError(f'Invalid current-state shape for {port}')
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite current state: {port}')
            states.append(value.float())
        return torch.cat(states, dim=-1)

    def _encode_prepared_visual(self, prepared_visual):
        """Run the original per-camera net, ReLU, and output randomizer once."""
        encoder = self.vision_encoder.encoder
        batch, frames = prepared_visual.shape[:2]
        features = []
        for camera, port in enumerate(self.rgb_ports):
            images = prepared_visual[:, :, camera].reshape(-1, *prepared_visual.shape[-3:])
            feature = encoder.obs_nets[port](images)
            if encoder.activation is not None:
                feature = encoder.activation(feature)
            randomizer = encoder.obs_randomizers[port]
            if randomizer is not None:
                feature = randomizer.forward_out(feature)
            features.append(feature.reshape(batch, frames, 64))
        return torch.stack(features, dim=2)

    def forward(self, obs, prepared_visual=None):
        states = self.state_features(obs)
        if prepared_visual is None:
            prepared_visual = self.prepare_conditioning(obs)
        size = self.crop_shape or tuple(self.shape_meta['obs'][self.rgb_ports[0]]['shape'][:2])
        expected = (states.shape[0], self.n_obs_steps, len(self.rgb_ports), 3, *size)
        if tuple(prepared_visual.shape) != expected or prepared_visual.requires_grad:
            raise ValueError(f'Prepared ResNet crops must have shape {expected} and no gradient')
        if not prepared_visual.is_floating_point() or not torch.isfinite(prepared_visual).all():
            raise ValueError('Prepared ResNet crops must be finite floating-point tensors')
        visual = self.visual_projection(self._encode_prepared_visual(prepared_visual))
        visual = visual + self.camera_embedding[None, None].to(visual.dtype)
        visual = visual + self.frame_embedding[None, :, None].to(visual.dtype)
        visual = visual.reshape(states.shape[0], self.num_visual_tokens, self.n_emb)
        # Use the original state encoder's linear head, bypassing its input
        # normalization because the flow policy has already normalized states.
        proprio = self.state_projection(states.to(self.state_projection.weight.dtype))
        proprio = proprio + self.frame_embedding[None].to(proprio.dtype)
        return visual, proprio

    def export_config(self):
        return dict(shape_meta=copy.deepcopy(self.shape_meta), **copy.deepcopy(self._architecture))
