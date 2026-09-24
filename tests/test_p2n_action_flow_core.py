"""CPU action-space mathematics, conditioning masks and DiT-X contracts."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from oat.common.action_flow_batch import PreparedActionFlowBatch
from oat.model.common.action_flow_context import ActionFlowContextBatch
from oat.model.common.context_batch import Segment
from oat.model.flow.action_euler_sampler import euler_sample
from oat.model.flow.consistency_flow import (
    consistency_velocity_target, fm_velocity_target, interpolate_latents,
    packed_flow_loss, sample_flow_schedule,
)
from oat.model.flow.ditx_action import DiTXActionVectorField


torch.set_num_threads(1)


def context(batch=2, width=24, gate=False):
    segments = [Segment.VISUAL] * 4 + [Segment.RAW_ACTION, Segment.ACTION_DIFF]
    if gate:
        segments += [Segment.HISTORY_SUMMARY] * 4
    metadata = {} if not gate else {
        "observation_summary": torch.randn(batch, width),
        "history_summary_pool": torch.randn(batch, width),
        "history_valid_fraction": torch.ones(batch, 1),
        "history_log_gate": torch.full((batch, 1), float("-inf")),
    }
    return ActionFlowContextBatch(
        memory=torch.randn(batch, len(segments), width),
        valid_mask=torch.ones(batch, len(segments), dtype=torch.bool),
        segment_ids=torch.tensor(segments, dtype=torch.int64), **metadata,
    )


def tiny_model(checkpointing=False, gate=False, nonzero=False):
    model = DiTXActionVectorField(
        embed_dim=24, n_layers=2, n_heads=3, ffn_hidden_dim=32,
        time_hidden_dim=32, time_embed_dim=16, activation_checkpointing=checkpointing,
        variant="p2n_state_gate_action_flow" if gate else "p2n_action_flow",
    )
    if nonzero:
        with torch.no_grad():
            nn.init.normal_(model.output[-1].weight, std=0.2)
            for block in model.blocks:
                nn.init.normal_(block.modulation[-1].weight, std=0.1)
    return model


def model_args(batch=2, gate=False):
    return {"time": torch.full((batch,), 0.3), "step_size": torch.full((batch,), 0.125),
            "context": context(batch=batch, gate=gate)}


def test_default_architecture_is_ln_gelu_time_only_without_codec():
    with torch.device("meta"):
        model = DiTXActionVectorField()
    assert model.horizon == 16 and model.action_dim == 7
    assert len(model.blocks) == 16 and model.embed_dim == 768
    block = model.blocks[0]
    for norm in (block.norm_sa, block.norm_ca, block.norm_ff):
        assert isinstance(norm, nn.LayerNorm) and not norm.elementwise_affine and norm.eps == 1e-6
    assert isinstance(block.self_attention.q_norm, nn.Identity)
    assert isinstance(block.self_attention.k_norm, nn.Identity)
    assert isinstance(block.cross_attention.q_norm, nn.LayerNorm)
    assert block.cross_attention.q_norm.elementwise_affine
    assert block.cross_attention.q_norm.normalized_shape == (64,)
    assert block.cross_attention.k.bias is not None and block.cross_attention.v.bias is not None
    assert block.ffn[0].out_features == 3072 and isinstance(block.ffn[1], nn.GELU)
    assert block.ffn[1].approximate == "tanh" and block.ffn[2].bias is not None
    assert isinstance(model.time_embedding.mlp[1], nn.Mish)
    assert model.time_embedding.mlp[0].in_features == 128
    assert model.time_embedding.mlp[0].out_features == 512
    assert model.global_projection.in_features == 2 * 768
    assert not any(any(word in name for word in ("current_state", "tokenizer", "codec", "quantizer", "history")) for name, _ in model.named_parameters())


def test_action_context_rejects_cross_family_and_extra_plain_modules():
    context().validate_variant("p2n_action_flow", 0)
    context(gate=True).validate_variant("p2n_state_gate_action_flow", 4)
    for invalid in ("p2n_new", "p2n_latent_flow", "continuous_action_flow"):
        with pytest.raises(ValueError, match="variant"):
            context().validate_variant(invalid)
    with pytest.raises(ValueError, match="must not contain"):
        context(gate=True).validate_variant("p2n_action_flow")
    with pytest.raises(ValueError, match="requires summaries"):
        context().validate_variant("p2n_state_gate_action_flow")


def test_zero_initialization_and_multistep_gradients():
    torch.manual_seed(5)
    model = tiny_model(checkpointing=True).train()
    actions, args = torch.randn(2, 16, 7), model_args()
    assert torch.equal(model(actions, **args), torch.zeros_like(actions))
    for block in model.blocks:
        assert torch.count_nonzero(block.modulation[-1].weight) == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    target, losses = torch.randn_like(actions), []
    for _ in range(5):
        optimizer.zero_grad()
        loss = (model(actions, **args) - target).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        losses.append(loss.item())
        optimizer.step()
    assert losses[-1] < losses[0]
    assert model.blocks[0].cross_attention.k.weight.grad.abs().sum() > 0
    assert model.blocks[0].self_attention.k.weight.grad.abs().sum() > 0
    assert model.step_embedding.mlp[0].weight.grad.abs().sum() > 0


def test_invalid_and_closed_memory_cannot_leak_even_nan_inf():
    torch.manual_seed(11)
    model = tiny_model(gate=True, nonzero=True).eval()
    actions, args = torch.randn(2, 16, 7), model_args(gate=True)
    ctx = args["context"]
    ctx.valid_mask[:, 1] = False
    with torch.no_grad():
        expected = model(actions, **args)
        ctx.memory[:, 1] = float("nan")
        ctx.memory[:, 6:] = float("inf")
        # These pooled fields must never reach AdaLN or any other bypass.
        ctx.observation_summary.fill_(float("nan"))
        ctx.history_summary_pool.fill_(float("inf"))
        actual = model(actions, **args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.isfinite(ctx.attention_bias()[:, :, :, :6][..., 0]).all()
        ctx.memory[:, 4] += 4  # Raw commands remain readable with a closed gate.
        assert not torch.allclose(model(actions, **args), expected)


def test_open_gate_gradient_and_masked_memory_backward_are_finite():
    torch.manual_seed(19)
    model = tiny_model(gate=True, nonzero=True).train()
    args = model_args(gate=True)
    ctx = args["context"]
    ctx.valid_mask[:, 1] = False
    ctx.memory[:, 1] = float("nan")
    ctx.memory.requires_grad_(True)
    gate_logits = nn.Parameter(torch.zeros(2, 1))
    ctx.history_log_gate = torch.nn.functional.logsigmoid(gate_logits)
    model(torch.randn(2, 16, 7), **args).square().sum().backward()
    assert torch.isfinite(ctx.memory.grad).all()
    assert torch.count_nonzero(ctx.memory.grad[:, 1]) == 0
    assert torch.isfinite(gate_logits.grad).all() and gate_logits.grad.abs().sum() > 0


def test_action_attention_is_bidirectional():
    torch.manual_seed(7)
    model = tiny_model(nonzero=True).eval()
    actions, args = torch.randn(2, 16, 7), model_args()
    with torch.no_grad():
        baseline = model(actions, **args)
        actions[:, -1] += 3
        changed = model(actions, **args)
    assert not torch.allclose(changed[:, 0], baseline[:, 0])


def test_cached_and_uncached_euler_agree_and_self_kv_recomputes():
    torch.manual_seed(9)
    model = tiny_model(nonzero=True).eval()
    ctx, noise = context(batch=1), torch.randn(1, 16, 7)
    calls = {"self": 0, "cross": 0}
    handles = []
    for block in model.blocks:
        for label, attn in (("self", block.self_attention), ("cross", block.cross_attention)):
            def record(module, inputs, output, label=label):
                calls[label] += 1
            handles.append(attn.k.register_forward_hook(record))
    with torch.no_grad():
        cached = euler_sample(model, context=ctx, initial_noise=noise, num_steps=3)
        assert calls == {"self": 6, "cross": 2}
        uncached = euler_sample(model, context=ctx, initial_noise=noise, num_steps=3, use_kv_cache=False)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-6)
    with pytest.raises(RuntimeError, match="gradients disabled"):
        model.build_kv_cache(ctx)
    with torch.no_grad():
        cache = model.build_kv_cache(ctx)
        args = {"time": torch.zeros(1), "step_size": torch.ones(1), "kv_cache": cache}
        with pytest.raises(ValueError, match="another model or context"):
            model(noise, context=context(batch=1), **args)
        ctx.valid_mask[:, -1] = False
        with pytest.raises(ValueError, match="mutated"):
            model(noise, context=ctx, **args)


def test_checkpointed_backward_matches_direct_including_context():
    torch.manual_seed(13)
    direct = tiny_model(nonzero=True).train()
    checkpointed = deepcopy(direct)
    checkpointed.activation_checkpointing = True
    args = model_args()
    args["context"].memory.requires_grad_(True)
    copied_args = deepcopy(args)
    actions = torch.randn(2, 16, 7)
    first, second = direct(actions, **args), checkpointed(actions, **copied_args)
    first.square().sum().backward()
    second.square().sum().backward()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(args["context"].memory.grad, copied_args["context"].memory.grad)
    for p1, p2 in zip(direct.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad)


def test_euler_known_velocity_remains_continuous_without_clipping():
    class ConstantVelocity(nn.Module):
        horizon, action_dim = 16, 7

        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, actions, **kwargs):
            self.seen.append(actions.clone())
            return torch.full_like(actions, .37)

    model = ConstantVelocity().eval()
    noise = torch.full((1, 16, 7), 1.125)
    endpoint = euler_sample(model, context=context(batch=1), num_steps=8,
                            initial_noise=noise, use_kv_cache=False)
    torch.testing.assert_close(endpoint, noise + .37)
    assert len(model.seen) == 8 and model.seen[1].min() > 1
    assert endpoint.dtype == torch.float32 and torch.equal(noise, torch.full_like(noise, 1.125))


def test_bf16_cached_generation_accumulates_fp32_and_preserves_rng_state():
    model = tiny_model(nonzero=True).eval()
    ctx = context(batch=1)
    generator = torch.Generator().manual_seed(81)
    state = generator.get_state()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        first = euler_sample(model, context=ctx, num_steps=2, generator=generator)
        generator.set_state(state)
        second = euler_sample(model, context=ctx, num_steps=2, generator=generator)
    assert first.dtype == torch.float32 and torch.isfinite(first).all()
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_fm_ct_action_targets_independent_reference_and_teacher_endpoints():
    target, noise = torch.randn(3, 16, 7), torch.randn(3, 16, 7)
    time, dt = torch.tensor([0., .4, .9]), torch.tensor([.3, .7, .2])
    teacher_parameter = nn.Parameter(torch.tensor(2.0))
    calls = []

    def teacher(next_actions, next_time, original_dt, rows):
        assert not torch.is_grad_enabled()
        calls.append((next_actions.clone(), next_time.clone(), original_dt.clone(), rows.clone()))
        return torch.ones_like(next_actions) * teacher_parameter

    velocity = consistency_velocity_target(target, noise, time, dt, teacher)
    tn = (time + dt).clamp(max=1)
    xt = (1 - time[:, None, None]) * noise + time[:, None, None] * target
    xn = (1 - tn[:, None, None]) * noise + tn[:, None, None] * target
    reference = (xn + (1 - tn[:, None, None]) * 2 - xt) / (1 - time[:, None, None])
    torch.testing.assert_close(velocity, reference)
    assert not velocity.requires_grad and calls[0][-1].tolist() == [0]
    assert torch.equal(calls[0][2], dt[:1])
    assert torch.equal(fm_velocity_target(target, noise), target - noise)
    assert torch.equal(interpolate_latents(target, noise, torch.zeros(3)), noise)
    assert torch.equal(interpolate_latents(target, noise, torch.ones(3)), target)
    terminal = consistency_velocity_target(target, noise, torch.full((3,), .9), torch.full((3,), .8),
                                           lambda *args: pytest.fail("Terminal teacher must be skipped"))
    torch.testing.assert_close(terminal, target - noise, atol=3e-6, rtol=3e-6)


def test_training_partition_fp32_loss_and_all_16_action_positions():
    generator = torch.Generator().manual_seed(4)
    schedule = sample_flow_schedule(4, device="cpu", generator=generator)
    assert schedule.fm_indices.numel() == 3 and schedule.ct_indices.numel() == 1
    assert torch.equal(torch.cat((schedule.fm_indices, schedule.ct_indices)).sort().values, torch.arange(4))
    assert torch.equal(schedule.step_size[schedule.fm_indices], torch.zeros(3))
    assert ((schedule.time[schedule.ct_indices] * 10) % 1 == 0).all()
    prediction = torch.zeros(4, 16, 7, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, -1] = 2  # Edge-repeated final positions participate in training.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = packed_flow_loss(prediction, target, schedule.fm_indices, schedule.ct_indices)
    assert loss["loss"].dtype == torch.float32
    torch.testing.assert_close(loss["loss"], torch.tensor(.5))
    loss["loss"].backward()
    assert (prediction.grad[:, -1] != 0).all()


def prepared_batch():
    schedule = sample_flow_schedule(4, device="cpu", generator=torch.Generator().manual_seed(4))
    return PreparedActionFlowBatch(
        noisy_actions=torch.randn(4, 16, 7), time=schedule.time, step_size=schedule.step_size,
        velocity_targets=torch.randn(4, 16, 7), fm_indices=schedule.fm_indices, ct_indices=schedule.ct_indices,
        obs={"robot_state": torch.randn(4, 2, 10)}, past_actions=torch.randn(4, 7, 7),
        past_action_valid=torch.ones(4, 7, dtype=torch.bool), frozen_patches=torch.randn(4, 2, 2, 196, 8),
    )


def test_prepared_batch_is_detached_and_partitions_all_samples():
    prepared_batch().validate()
    bad = prepared_batch()
    bad.ct_indices = bad.fm_indices[:1]
    with pytest.raises(ValueError, match="partition"):
        bad.validate()
    bad = prepared_batch()
    bad.obs["robot_state"].requires_grad_(True)
    with pytest.raises(ValueError, match="detached"):
        bad.validate()
    bad = prepared_batch()
    with torch.inference_mode():
        bad.frozen_patches = bad.frozen_patches.clone()
    with pytest.raises(ValueError, match="ordinary tensors"):
        bad.validate()


@pytest.mark.parametrize("field,value", [("time", torch.tensor([1.1, .2])), ("step_size", torch.tensor([.1, -.1])), ("time", torch.tensor([.1, .2]).bfloat16())])
def test_time_contract_rejects_bad_range_or_precision(field, value):
    args = model_args()
    args[field] = value
    with pytest.raises(ValueError, match=field):
        tiny_model()(torch.randn(2, 16, 7), **args)
