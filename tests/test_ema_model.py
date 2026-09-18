"""Regression tests for EMA updates of tied and independent parameters."""
import copy

import pytest
import torch
from torch import nn

from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from oat.model.diffusion.ema_model import EMAModel


def make_ema(model):
    averaged_model = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in averaged_model.parameters():
            parameter.zero_()
    ema = EMAModel(averaged_model, power=1.0, max_value=0.5)
    # Skip the zero-decay warmup to expose repeated averaging.
    ema.optimization_step = 2
    return ema


@pytest.mark.parametrize("tied", [True, False])
def test_each_parameter_is_averaged_once_per_step(tied):
    model = AutoregressiveModel(
        vocab_size=9, max_seq_len=3, max_cond_len=2, cond_dim=8,
        n_layer=1, n_head=2, n_emb=8, p_drop_emb=0, p_drop_attn=0,
    )
    if not tied:
        model.head.weight = nn.Parameter(model.tok_emb.weight.detach().clone())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(10)
    ema = make_ema(model)

    # Equal-valued independent parameters must still receive their own update.
    # Repeating the step also checks that deduplication resets on each call.
    for step, expected in enumerate((5.0, 7.5), start=3):
        ema.step(model)

        assert ema.decay == 0.5
        assert ema.optimization_step == step
        assert (model.head.weight is model.tok_emb.weight) == tied
        assert (ema.averaged_model.head.weight is ema.averaged_model.tok_emb.weight) == tied
        for name, parameter in ema.averaged_model.named_parameters():
            torch.testing.assert_close(
                parameter, torch.full_like(parameter, expected),
                rtol=0, atol=0, msg=name,
            )
        for parameter in model.parameters():
            torch.testing.assert_close(
                parameter, torch.full_like(parameter, 10), rtol=0, atol=0,
            )


def test_frozen_and_batchnorm_parameters_are_copied():
    model = nn.ModuleDict({
        "trainable": nn.Linear(2, 2),
        "frozen": nn.Linear(2, 2),
        "batch_norm": nn.BatchNorm1d(2),
    })
    model["frozen"].requires_grad_(False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(10)
    ema = make_ema(model)

    ema.step(model)

    for name, module in ema.averaged_model.items():
        expected = 5 if name == "trainable" else 10
        for parameter in module.parameters():
            torch.testing.assert_close(
                parameter, torch.full_like(parameter, expected),
                rtol=0, atol=0,
            )
    assert ema.optimization_step == 3
