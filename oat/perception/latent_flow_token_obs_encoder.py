"""Shared frozen patch realization, independently trainable EMA/student adapters."""
import torch
from torch import nn
from oat.perception.token_obs_encoder import TokenObservationEncoder

class FlowTokenObservationEncoder(TokenObservationEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize only the new adapters, never the pretrained DINO backbone.
        for parent in (self.patch_projection, self.state_projection, self.resampler):
            for module in parent.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    @torch.no_grad()
    def extract_frozen_patches(self, obs):
        cameras = [obs[key] for key in self.rgb_ports]
        shape = cameras[0].shape
        if len(shape) != 5 or shape[1] != self.n_obs_steps or shape[-1] != 3:
            raise ValueError('RGB must be [B,n_obs_steps,H,W,3]')
        if any(x.shape != shape or x.dtype != cameras[0].dtype or x.device != cameras[0].device for x in cameras):
            raise ValueError('All cameras must share shape, dtype, and device')
        batch, frames, height, width, channels = shape
        images = torch.stack(cameras, dim=2).reshape(-1, height, width, channels)
        patches = self.dino_encoder(images)
        return patches.detach().reshape(batch, frames, len(cameras), *patches.shape[1:])

    def state_features(self, obs):
        values = []
        batch = obs[self.state_ports[0]].shape[0]
        for key in self.state_ports:
            value = obs[key]
            if value.shape != (batch, self.n_obs_steps, self.state_shapes[key]):
                raise ValueError(f'Invalid current-state shape for {key}')
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite current state: {key}')
            values.append(value.float())
        return torch.cat(values, dim=-1)

    def adapt_frozen_patches(self, obs, patches):
        batch = obs[self.state_ports[0]].shape[0]
        expected = (batch, self.n_obs_steps, len(self.rgb_ports),
                    self.dino_encoder.num_patches, self.dino_encoder.hidden_size)
        if tuple(patches.shape) != expected or patches.requires_grad:
            raise ValueError(f'Frozen patches must have shape {expected} and no gradient')
        flat = patches.reshape(-1, *patches.shape[-2:])
        visual = self.resampler(self.patch_projection(flat)).reshape(
            batch, self.n_obs_steps, len(self.rgb_ports), self.num_queries, self.n_emb)
        visual = visual + self.camera_embedding[None, None, :, None].to(visual.dtype)
        visual = visual + self.frame_embedding[None, :, None, None].to(visual.dtype)
        visual = visual.reshape(batch, self.num_visual_tokens, self.n_emb)
        proprio = self.state_projection(self.state_features(obs))
        proprio = proprio + self.frame_embedding[None].to(proprio.dtype)
        return visual, proprio

    def forward(self, obs_dict, frozen_patches=None):
        if frozen_patches is None:
            frozen_patches = self.extract_frozen_patches(obs_dict)
        return self.adapt_frozen_patches(obs_dict, frozen_patches)
