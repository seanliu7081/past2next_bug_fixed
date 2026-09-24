"""Direct-flow policies import, train and predict with OAT imports unavailable."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = r'''
import importlib.abc
import sys

class RejectTokenizers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "oat.tokenizer" or fullname.startswith("oat.tokenizer."):
            raise ModuleNotFoundError("Direct action flow cannot import tokenizer code: " + fullname)
        return None

sys.meta_path.insert(0, RejectTokenizers())

import copy
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.perception.dinov3_patch_encoder import DINOv3PatchEncoder
from oat.perception.latent_flow_token_obs_encoder import FlowTokenObservationEncoder
from oat.policy.p2n_action_flow import P2NActionFlowPolicy
from oat.policy.p2n_state_gate_action_flow import P2NStateGateActionFlowPolicy

torch.set_num_threads(1)
torch.manual_seed(123)

class TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 16)
    def forward(self, pixel_values):
        patches = self.projection(F.avg_pool2d(pixel_values, 16).flatten(2).transpose(1, 2))
        return SimpleNamespace(last_hidden_state=torch.cat((patches[:, :1], patches), dim=1))

meta = {"obs": {
    "camera_a": {"shape": [8, 8, 3], "type": "rgb"},
    "camera_b": {"shape": [8, 8, 3], "type": "rgb"},
    "robot0_eef_pos": {"shape": [3], "type": "state"},
    "robot0_eef_rot6d": {"shape": [6], "type": "state"},
    "robot0_gripper_qpos": {"shape": [1], "type": "state"},
    "task_uid": {"shape": [1], "type": "state"},
}, "action": {"shape": [7]}}
dino = DINOv3PatchEncoder(backbone=TinyDINO(), image_size=32,
    config={"patch_size": 16, "hidden_size": 16, "num_register_tokens": 0},
    processor_config={"do_rescale": True, "rescale_factor": 1 / 255., "do_normalize": True,
                      "image_mean": [.5, .5, .5], "image_std": [.5, .5, .5]})
encoder = FlowTokenObservationEncoder(meta, n_obs_steps=2, n_emb=16, n_head=2,
    ffn_dim=32, num_queries=2, resampler_depth=1, dino_encoder=dino)
gate = sys.argv[1] == "gate"
kwargs = {"shape_meta": meta, "obs_encoder": encoder, "embed_dim": 16, "n_layers": 2,
          "n_heads": 2, "resampler_ffn_dim": 32, "flow": {"ffn_hidden_dim": 64}, "self_past_p": 0.}
if gate:
    kwargs.update(history_embed_dim=16, history_n_heads=2, history_n_layers=1, history_gate_hidden_dim=16)
policy = (P2NStateGateActionFlowPolicy if gate else P2NActionFlowPolicy)(**kwargs)
normalizer = LinearNormalizer()
for key in ["action", *encoder.state_ports]:
    normalizer[key] = SingleFieldLinearNormalizer.create_identity()
policy.set_normalizer(normalizer)
policy.train()
obs = policy.create_dummy_observation(4)
batch = {"obs": obs, "action": torch.randn(4, 16, 7), "past_action": torch.zeros(4, 7, 7),
         "past_action_valid": torch.zeros(4, 7, dtype=torch.bool)}
teacher = copy.deepcopy(policy).requires_grad_(False).eval()
prepared = policy.prepare_training_batch(batch, teacher=teacher, generator=torch.Generator().manual_seed(7))
loss = policy(prepared)
assert torch.isfinite(loss)
loss.backward()
assert any(parameter.grad is not None for parameter in policy.parameters())
policy.eval()
result = policy.predict_action(obs, num_flow_steps=2)
assert result["action_pred"].shape == (4, 16, 7)
assert torch.isfinite(result["action_pred"]).all()
policy.record_executed_actions(torch.zeros(4, 8, 7), executed_lengths=[8, 4, 0, 1])
assert policy._past_valid_buffer.sum(dim=1).tolist() == [7, 4, 0, 1]
assert not any(name == "oat.tokenizer" or name.startswith("oat.tokenizer.") for name in sys.modules)
assert not any("tokenizer" in name or "quantizer" in name for name in policy.state_dict())
print("codec-independent " + ("gate" if gate else "plain") + " forward/backward/prediction passed")
'''


@pytest.mark.parametrize("variant", ["plain", "gate"])
def test_policies_construct_train_and_predict_without_tokenizer_imports(variant):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", SCRIPT, variant], cwd=root,
                            env={**os.environ, "OMP_NUM_THREADS": "1"},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "forward/backward/prediction passed" in result.stdout
