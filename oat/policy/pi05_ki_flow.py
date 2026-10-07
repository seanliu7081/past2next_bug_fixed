"""Same-stack PI0.5 baseline: pretrained flow-matching expert + OAT knowledge insulation on the VLM.

Identical VLM, LoRA, prompt, KI loss and stop-gradient as the P2N-VLA policies;
only the action expert differs (PI0.5's unfolded adaRMS flow head, no past
conditioning). Actions use the policy's dataset normalizer, padded 7 -> 32.
"""
from __future__ import annotations

import contextlib

import torch
from torch import Tensor
import torch.nn.functional as F

from oat.model.vla.flow_head import FLOW_ACTION_DIM, FlowHeads, flow_velocity, sample_actions
from oat.policy.p2n_vla_common import P2NVLACommonPolicy


class PI05KIFlowPolicy(P2NVLACommonPolicy):
    VARIANT = "pi05_ki_flow"
    VARIANT_CODE = 12
    NORM_MODE = "adarms"
    LOADER_MODE = "flow"
    HAS_AR_HEAD = False
    supports_generated_history_validation = False

    def __init__(self, *args, flow_num_steps=10, flow_time_beta=(1.5, 1.0), **kwargs):
        if kwargs.get("use_past", False):
            raise ValueError("The PI0.5 flow baseline has no past conditioning")
        kwargs["use_past"] = False
        kwargs["self_past_p"] = 0.0
        if isinstance(flow_num_steps, bool) or not isinstance(flow_num_steps, int) or flow_num_steps < 1:
            raise ValueError("flow_num_steps must be a positive integer")
        self.flow_num_steps = int(flow_num_steps)
        self.flow_time_beta = tuple(float(x) for x in flow_time_beta)
        super().__init__(*args, **kwargs)
        if self.action_dim > FLOW_ACTION_DIM:
            raise ValueError("Action dimension exceeds the PI0.5 padded action width")
        self._construction.update(flow_num_steps=self.flow_num_steps, flow_time_beta=list(self.flow_time_beta))

    def _make_flow_heads(self, width):
        return FlowHeads(width)

    def _pad(self, actions: Tensor) -> Tensor:
        return F.pad(actions, (0, FLOW_ACTION_DIM - actions.shape[-1]))

    def forward(self, batch, history_mode=None):
        self._check_ready()
        targets = self.encode_targets(batch["action"])
        actions = self._pad(self.action_normalizer["action"].normalize(batch["action"].float()).float())
        batch_size = actions.shape[0]
        beta = torch.distributions.Beta(*[torch.tensor(x, device=actions.device) for x in self.flow_time_beta])
        t = beta.sample((batch_size,)) * 0.999 + 0.001
        noise = torch.randn_like(actions)
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        velocity_target = noise - actions
        use_ki = self.lambda_ki > 0
        with self._autocast():
            prefix = self.build_prefix(batch["obs"], train_aug=self.training and self.train_image_aug)
            with contextlib.nullcontext() if use_ki else torch.no_grad():
                prefix_out, layout, ki_logits = self.vlm_pass(prefix, targets if use_ki else None)
            velocity = flow_velocity(self.joint, self.flow_heads, self._detached_kv(prefix_out), prefix_out.pos0,
                                     layout.nonki_valid, x_t, t)
        loss_flow = F.mse_loss(velocity.float()[..., :self.action_dim], velocity_target[..., :self.action_dim])
        loss, loss_ki = loss_flow, None
        if use_ki:
            loss_ki = F.cross_entropy(ki_logits.reshape(-1, ki_logits.shape[-1]).float(), targets.reshape(-1))
            loss = loss_flow + self.lambda_ki * loss_ki
        self.last_loss_components = dict(
            loss=loss.detach().float().item(), loss_flow=loss_flow.detach().float().item(),
            loss_ki=None if loss_ki is None else loss_ki.detach().float().item(),
            ki_token_acc=None if ki_logits is None else (ki_logits.argmax(-1) == targets).float().mean().item(),
            self_past_p=0.0, self_past_rows=0.0, gate_mean=None, gate_min=None, gate_max=None,
            hist_attention_mass=None, ar_token_acc=None, loss_ar=None)
        return loss

    def _generate_actions(self, obs, past, valid, n_tokens, temperature, topk):
        self._check_ready()
        with torch.no_grad(), self._autocast():
            prefix = self.build_prefix(obs, train_aug=False)
            prefix_out, layout, _ = self.vlm_pass(prefix, None)
            noise = torch.randn(prefix.embeds.shape[0], self.horizon, FLOW_ACTION_DIM, device=prefix.embeds.device)
            sampled = sample_actions(self.joint, self.flow_heads, self._detached_kv(prefix_out), prefix_out.pos0,
                                     layout.nonki_valid, noise, self.flow_num_steps)
        with torch.autocast(device_type=sampled.device.type, enabled=False):
            actions = self.action_normalizer["action"].unnormalize(sampled[..., :self.action_dim].float())
        with torch.inference_mode(False):
            return actions.float().detach().clone()
