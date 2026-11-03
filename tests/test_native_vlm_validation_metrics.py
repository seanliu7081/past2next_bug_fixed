"""Validation consumes emitted free-running IDs without doing a second policy decode."""
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace


class _ValidationPolicy(torch.nn.Module):
    supports_explicit_past_actions = True
    supports_explicit_past_action_valid = True
    supports_generated_history_validation = True
    requires_state_history = False
    n_action_steps = 8

    def __init__(self, emit_tokens=True):
        super().__init__()
        self.emit_tokens = emit_tokens
        self.decode_calls = 0
        self.was_reset = False
        self.last_loss_components = {}
        self.action_normalizer = {"action": SimpleNamespace(normalize=lambda x: x / 2)}

    def forward(self, batch, history_mode=None):
        return torch.tensor(0.0)

    def predict_action(self, obs, past_actions=None, past_action_valid=None):
        assert past_actions is not None and past_action_valid is not None
        assert not torch.is_grad_enabled() and not self.training
        self.decode_calls += 1
        result = {"action_pred": torch.full((2, 16, 7), 2.0)}
        if self.emit_tokens:
            result["action_tokens"] = torch.zeros((2, 8), dtype=torch.long)
            result["action_tokens"][1, -1] = 1
        return result

    def encode_targets(self, actions):
        assert torch.count_nonzero(actions) == 0
        return torch.zeros((2, 8), dtype=torch.long)

    def reset(self):
        self.was_reset = True


@pytest.mark.parametrize("emit_tokens", (True, False))
def test_validation_reports_native_free_running_metrics_and_preserves_legacy(emit_tokens):
    policy = _ValidationPolicy(emit_tokens).train()
    batch = {"action": torch.zeros((2, 16, 7)), "obs": {}, "prev_obs": {}}
    for prefix in ("", "prev_"):
        batch[prefix + "past_action"] = torch.zeros((2, 7, 7))
        batch[prefix + "past_action_valid"] = torch.ones((2, 7), dtype=torch.bool)
    accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1, process_index=0,
                                  reduce=lambda tensor, reduction: tensor)
    run = SimpleNamespace(policy=policy, val_loader=[batch], accelerator=accelerator, ema=None)
    cfg = OmegaConf.create({"training": {"seed": 42, "validate_generated_history": True,
                                        "max_reconst_steps": 1}, "n_action_steps": 8})
    workspace = SimpleNamespace(cfg=cfg, _memory=lambda device: {})
    result = TrainP2NVLAWorkspace._validate(workspace, run)
    assert policy.decode_calls == 1 and policy.was_reset and policy.training
    assert result["val/reconstruction_mse_executed"] == pytest.approx(4.0)
    names = ("val/free_running_token_acc", "val/full_sequence_acc", "val/normalized_executed_action_mse")
    if emit_tokens:
        assert [result[key] for key in names] == pytest.approx([15 / 16, 0.5, 1.0])
    else:
        assert not set(names) & result.keys()
