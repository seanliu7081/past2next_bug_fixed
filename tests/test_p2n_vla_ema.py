"""TrainableEMA (P2N-VLA M4): math, swap-in/out of weights and module modes, persistence."""
import copy

import pytest
import torch
from torch import nn

from oat.model.common.trainable_ema import TrainableEMA
from test_p2n_vla_workspace import StubVLAPolicy


class Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.5))
        self.norm = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 2)
        self.frozen = nn.Linear(2, 2).requires_grad_(False)

    def forward(self, x):
        return self.frozen(self.head(self.norm(self.encoder(x))))


def trainables(module):
    return [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]


def perturb(module, scale=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, parameter in trainables(module):
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)


def test_step_is_the_contract_formula_and_counts_updates():
    torch.manual_seed(0)
    module = Mixed()
    ema = TrainableEMA(trainables(module), decay=0.9)
    assert ema.names == ("encoder.0.weight", "encoder.0.bias", "norm.weight", "norm.bias",
                         "head.weight", "head.bias")
    assert all(shadow.dtype == torch.float32 for shadow in ema.shadow.values())
    previous = {name: shadow.clone() for name, shadow in ema.shadow.items()}
    perturb(module)
    assert ema.step(trainables(module)) == 0.9 and ema.updates == 1
    live = dict(module.named_parameters())
    for name in ema.names:
        expected = (previous[name] * 0.9).add(live[name].detach(), alpha=1 - 0.9)
        torch.testing.assert_close(ema.shadow[name], expected, rtol=0, atol=0)
    # Parameters are never touched by an update.
    for name in ema.names:
        assert not torch.equal(ema.shadow[name], live[name].detach())


def test_warmup_power_copies_first_then_caps_at_decay():
    module = Mixed()
    ema = TrainableEMA(trainables(module), decay=0.999, warmup_power=0.75)
    perturb(module)
    assert ema.current_decay() == 0.0
    ema.step(trainables(module))
    for name, parameter in trainables(module):
        torch.testing.assert_close(ema.shadow[name], parameter.detach(), rtol=0, atol=0)
    assert ema.current_decay() == pytest.approx(1 - 2 ** -0.75)
    ema.updates = 10 ** 6
    assert ema.current_decay() == 0.999


def test_tracks_only_trainable_parameters_once_and_validates_inputs():
    module = Mixed()
    named = trainables(module) + [("tied_alias", module.head.weight)]
    ema = TrainableEMA(named)
    assert "tied_alias" not in ema.shadow and len(ema.names) == 6
    assert ema.num_elements() == sum(p.numel() for _, p in trainables(module))
    with pytest.raises(ValueError, match="frozen"):
        TrainableEMA([("frozen.weight", module.frozen.weight)])
    with pytest.raises(ValueError, match="at least one"):
        TrainableEMA([])
    with pytest.raises(ValueError, match="decay"):
        TrainableEMA(trainables(module), decay=1.0)
    with pytest.raises(ValueError, match="two different"):
        TrainableEMA([("x", module.head.weight), ("x", module.head.bias)])
    with pytest.raises(TypeError, match="pairs"):
        TrainableEMA([module.head.weight])
    with pytest.raises(ValueError, match="parameter set changed"):
        ema.step(trainables(module)[:-1])
    other = Mixed()
    other.head = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="shape mismatch"):
        ema.step(trainables(other))


def test_swap_in_exposes_shadow_then_restores_values_and_mixed_modes():
    torch.manual_seed(1)
    module = Mixed()
    ema = TrainableEMA(trainables(module), decay=0.5)
    perturb(module)
    ema.step(trainables(module))
    perturb(module, seed=3)
    module.train()
    module.encoder[1].eval()      # intentionally mixed modes
    module.frozen.eval()
    modes = {name: sub.training for name, sub in module.named_modules()}
    live = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    storage = {name: parameter.data_ptr() for name, parameter in module.named_parameters()}
    with ema.swap_in(module) as swapped:
        assert swapped is module
        module.eval()
        for name in ema.names:
            torch.testing.assert_close(dict(module.named_parameters())[name].detach(), ema.shadow[name],
                                       rtol=0, atol=0)
        torch.testing.assert_close(module.frozen.weight.detach(), live["frozen.weight"], rtol=0, atol=0)
        module(torch.randn(5, 3))
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter.detach(), live[name], rtol=0, atol=0)
        assert parameter.data_ptr() == storage[name]
    assert {name: sub.training for name, sub in module.named_modules()} == modes
    # Restored even when the body raises.
    with pytest.raises(RuntimeError, match="boom"):
        with ema.swap_in(module):
            module.eval()
            raise RuntimeError("boom")
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter.detach(), live[name], rtol=0, atol=0)
    assert {name: sub.training for name, sub in module.named_modules()} == modes


def test_swap_in_copy_path_and_dtype_mismatch():
    module = Mixed()
    ema = TrainableEMA(trainables(module), decay=0.0)
    perturb(module)
    ema.step(trainables(module))  # decay 0: shadow == params
    perturb(module, seed=5)
    live = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    with ema.swap_in(module, zero_copy=False):
        for name in ema.names:
            torch.testing.assert_close(dict(module.named_parameters())[name].detach(), ema.shadow[name])
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter.detach(), live[name], rtol=0, atol=0)
    half = nn.Linear(2, 2).to(torch.bfloat16)
    ema_half = TrainableEMA(trainables(half))
    assert ema_half.shadow["weight"].dtype == torch.float32
    before = half.weight.detach().clone()
    with torch.no_grad():
        ema_half.shadow["weight"].add_(1.0)
    with ema_half.swap_in(half):
        assert half.weight.dtype == torch.bfloat16
        torch.testing.assert_close(half.weight.detach(), (before.float() + 1).to(torch.bfloat16))
    torch.testing.assert_close(half.weight.detach(), before, rtol=0, atol=0)


def test_swap_in_guards_inference_mode_nesting_and_updates():
    module = Mixed()
    ema = TrainableEMA(trainables(module))
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="inference_mode"):
            with ema.swap_in(module):
                pass
    with ema.swap_in(module):
        with pytest.raises(RuntimeError, match="already swapped"):
            with ema.swap_in(module):
                pass
        with pytest.raises(RuntimeError, match="swapped in"):
            ema.step(trainables(module))
    wrapper = nn.Module()
    wrapper.module = module
    with pytest.raises(KeyError, match="unwrapped"):
        with ema.swap_in(wrapper):
            pass


def test_training_backward_works_after_swapped_validation():
    module = Mixed()
    optimizer = torch.optim.AdamW([p for _, p in trainables(module)], lr=0.1)
    ema = TrainableEMA(trainables(module), decay=0.5)
    for _ in range(2):
        module.train()
        loss = module(torch.randn(6, 3)).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        ema.step(trainables(module))
        with ema.swap_in(module), torch.no_grad():
            module.eval()
            module(torch.randn(6, 3))
    assert module.training and ema.updates == 2


def test_state_dict_round_trip_and_strict_loading():
    torch.manual_seed(2)
    module = Mixed()
    ema = TrainableEMA(trainables(module), decay=0.8)
    for seed in range(3):
        perturb(module, seed=seed)
        ema.step(trainables(module))
    state = copy.deepcopy(ema.state_dict())
    assert set(state) == {"decay", "updates", "warmup_power", "shadow"} and state["updates"] == 3
    restored = TrainableEMA(trainables(module), decay=0.8)
    restored.load_state_dict(state)
    assert restored.updates == 3
    for name in ema.names:
        torch.testing.assert_close(restored.shadow[name], ema.shadow[name], rtol=0, atol=0)
    perturb(module, seed=9)
    ema.step(trainables(module))
    restored.step(trainables(module))
    for name in ema.names:
        torch.testing.assert_close(restored.shadow[name], ema.shadow[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="decay"):
        TrainableEMA(trainables(module), decay=0.9).load_state_dict(state)
    with pytest.raises(ValueError, match="warmup_power"):
        TrainableEMA(trainables(module), decay=0.8, warmup_power=0.75).load_state_dict(state)
    broken = copy.deepcopy(state)
    broken["shadow"].pop("head.bias")
    with pytest.raises(ValueError, match="names differ"):
        restored.load_state_dict(broken)
    broken = copy.deepcopy(state)
    broken["shadow"]["head.bias"] = torch.zeros(5)
    with pytest.raises(ValueError, match="shape"):
        restored.load_state_dict(broken)
    broken = copy.deepcopy(state)
    broken["shadow"]["head.bias"] = torch.full((2,), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        restored.load_state_dict(broken)


def test_trainable_override_feeds_the_policy_artifact():
    policy = StubVLAPolicy()
    policy.set_normalizer({"action_scale": torch.full((7,), 2.0)})
    ema = TrainableEMA(policy.trainable_named_parameters(), decay=0.5)
    perturb(policy)
    ema.step(policy.trainable_named_parameters())
    artifact = policy.artifact_state_dict(trainable_override=ema.trainable_override())
    assert "backbone.weight" not in artifact
    for name in ema.names:
        torch.testing.assert_close(artifact[name], ema.shadow[name], rtol=0, atol=0)
    torch.testing.assert_close(artifact["action_scale"], torch.full((7,), 2.0))


@pytest.mark.gpu
def test_cuda_ema_with_fused_adamw_swap_and_restore():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available (gpu marker)")
    device = torch.device("cuda")
    module = Mixed().to(device)
    optimizer = torch.optim.AdamW([p for _, p in trainables(module)], lr=0.05, fused=True)
    ema = TrainableEMA(trainables(module), decay=0.9)
    assert all(shadow.device.type == "cuda" for shadow in ema.shadow.values())
    for _ in range(3):
        module(torch.randn(8, 3, device=device)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        ema.step(trainables(module))
    live = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    with ema.swap_in(module), torch.no_grad():
        module.eval()
        for name in ema.names:
            assert torch.equal(dict(module.named_parameters())[name], ema.shadow[name])
    for name, parameter in module.named_parameters():
        assert torch.equal(parameter.detach(), live[name])
    assert module.training
