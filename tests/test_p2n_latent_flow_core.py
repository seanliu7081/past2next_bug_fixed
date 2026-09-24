"""CPU-only checks of the actual FSQ grid, objectives and DiT-X contracts."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from oat.model.common.context_batch import ContextBatch, Segment
from oat.model.flow.consistency_flow import (
    consistency_velocity_target, fm_velocity_target, interpolate_latents,
    packed_flow_loss, sample_flow_schedule, sample_fm_times,
)
from oat.model.flow.ditx_latent import DiTXLatentFlow
from oat.model.flow.euler_sampler import euler_sample
from oat.tokenizer.oat.latent_adapter import FrozenOATLatentAdapter
from oat.tokenizer.oat.quantizer.fsq import FSQ


torch.set_num_threads(1)


class TinyOAT(nn.Module):
    """Track raw input semantics while using the repository's real FSQ."""
    def __init__(self):
        super().__init__()
        self.quantizer = FSQ([8, 5, 5, 5, 5], packed_call=False)
        self.decoder = nn.Identity()
        self.decoder.latent_horizon = 8
        self.decoder.sample_horizon = 16
        self.decoder.sample_dim = 7
        self.latent_horizon = 8
        self.offset = nn.Parameter(torch.tensor(3.0))
        self.last_raw = None

    def encode(self, raw):
        self.last_raw = raw.clone()
        return self.quantizer(raw[:, :8, :5] - self.offset)

    def decode(self, codes):
        out = torch.zeros(codes.shape[0], 16, 7, device=codes.device)
        out[:, :8, :5] = codes
        return out + self.offset

    def detokenize(self, indices):
        return self.decode(self.quantizer.indices_to_embedding(indices))


def adapter():
    return FrozenOATLatentAdapter(TinyOAT())


def context(batch=2, width=24, with_summary=False):
    count = 6 if with_summary else 4
    segments = torch.tensor([int(Segment.VISUAL)] * 4 + ([int(Segment.HISTORY_SUMMARY)] * 2 if with_summary else []))
    return ContextBatch(memory=torch.randn(batch, count, width), valid_mask=torch.ones(batch, count, dtype=torch.bool),
                        segment_ids=segments, history_log_gate=torch.full((batch, 1), float('-inf')) if with_summary else None)


def tiny_model(checkpointing=False):
    return DiTXLatentFlow(current_state_dim=6, embed_dim=24, n_layers=2, n_heads=3,
                         ffn_dim=32, time_hidden_dim=32, time_embed_dim=16,
                         activation_checkpointing=checkpointing)


def nonzero_model(checkpointing=False):
    model = tiny_model(checkpointing)
    with torch.no_grad():
        nn.init.normal_(model.output[-1].weight, std=0.2)
        for block in model.blocks:
            nn.init.normal_(block.modulation[-1].weight, std=0.1)
    return model


def model_args(batch=2, with_summary=False):
    return dict(time=torch.full((batch,), 0.3), step_size=torch.full((batch,), 0.125),
                context=context(batch, with_summary=with_summary), current_state=torch.randn(batch, 6))


def test_actual_fsq_all_5000_codes_roundtrip_and_decode():
    codec = adapter()
    ids = torch.arange(5000)
    codes = codec.tokenizer.quantizer.indices_to_embedding(ids)
    grid = codec.snap_codes(codes)
    assert torch.equal(grid, codes)
    assert torch.equal(codec.tokenizer.quantizer.codes_to_indices(grid).long(), ids)
    ids = ids[:16].reshape(2, 8)
    grid = codec.tokenizer.quantizer.indices_to_embedding(ids)
    assert torch.equal(codec.decode_grid_codes(grid), codec.tokenizer.detokenize(ids))


def test_grid_bounds_ties_and_nonfinite_rejection():
    codec = adapter()
    extremes = codec.snap_codes(torch.tensor([[1e10] * 5, [-1e10] * 5]))
    assert torch.equal(extremes[0], torch.tensor([0.75, 1, 1, 1, 1]))
    assert torch.equal(extremes[1], torch.full((5,), -1.0))
    assert codec.snap_codes(torch.tensor([[0.125, 0.25, 0.75, -0.25, -0.75]])).tolist() == [[0., 0., 1., 0., -1.]]
    for bad in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError, match='NaN or Inf'):
            codec.snap_codes(torch.full((1, 8, 5), bad))
    with pytest.raises(ValueError, match='grid'):
        codec.decode_grid_codes(torch.full((1, 8, 5), .11))


def test_adapter_frozen_normalizes_once_and_returns_ordinary_targets():
    codec = adapter().train()
    raw = torch.randn(2, 16, 7)
    before = deepcopy(codec.tokenizer.state_dict())
    with torch.inference_mode():
        target = codec.encode_actions(raw)
    assert torch.equal(codec.tokenizer.last_raw, raw)
    assert target.codes.shape == (2, 8, 5) and target.codes.dtype == torch.float32
    assert target.indices.dtype == torch.int64
    assert not torch.is_inference(target.codes) and not target.codes.requires_grad
    assert not codec.tokenizer.training and not any(p.requires_grad for p in codec.parameters())
    nn.Linear(5, 2)(target.codes).sum().backward()
    assert all(torch.equal(value, codec.tokenizer.state_dict()[key]) for key, value in before.items())


def test_adapter_checks_actual_schema():
    tok = TinyOAT()
    tok.decoder.sample_horizon = 15
    with pytest.raises(ValueError, match='sample_horizon'):
        FrozenOATLatentAdapter(tok)
    with pytest.raises(ValueError, match='levels'):
        FrozenOATLatentAdapter(TinyOAT(), levels=[5] * 5)


def test_fm_sign_interpolation_and_singleton_axis():
    z1, noise = torch.randn(1, 8, 5), torch.randn(1, 8, 5)
    assert torch.equal(interpolate_latents(z1, noise, torch.zeros(1)), noise)
    assert torch.equal(interpolate_latents(z1, noise, torch.ones(1)), z1)
    assert torch.equal(fm_velocity_target(z1, noise), z1 - noise)
    with pytest.raises(ValueError, match='shape'):
        interpolate_latents(z1, noise, torch.tensor(0.5))


def test_schedule_partition_relative_dt_and_explicit_generator_resume():
    rng = torch.Generator().manual_seed(8)
    saved = rng.get_state()
    a = sample_flow_schedule(4, device='cpu', generator=rng)
    rng.set_state(saved)
    b = sample_flow_schedule(4, device='cpu', generator=rng)
    for field in ('time', 'step_size', 'fm_indices', 'ct_indices'):
        assert torch.equal(getattr(a, field), getattr(b, field))
    assert a.fm_indices.numel() == 3 and a.ct_indices.numel() == 1
    assert torch.equal(torch.cat((a.fm_indices, a.ct_indices)).sort().values, torch.arange(4))
    assert (a.step_size[a.fm_indices] == 0).all()
    assert (a.time[a.ct_indices] * 10 == torch.round(a.time[a.ct_indices] * 10)).all()
    assert (a.time < 1).all() and (a.step_size < 1).all()
    for batch in (0, 1, 2, 3, 5, 6, 7):
        with pytest.raises(ValueError, match='multiple of four'):
            sample_flow_schedule(batch, device='cpu')


def test_beta_times_use_scaled_distribution():
    rng = torch.Generator().manual_seed(9)
    values = sample_fm_times(100000, device='cpu', generator=rng)
    assert values.min() >= 0 and values.max() < .999
    assert values.mean().item() == pytest.approx(.999 / 2.5, abs=.002)


def test_ct_independent_reference_and_teacher_endpoints():
    z1, noise = torch.randn(3, 8, 5), torch.randn(3, 8, 5)
    t, dt = torch.tensor([0., .4, .9]), torch.tensor([.3, .7, .2])
    teacher_parameter = nn.Parameter(torch.tensor(2.0))
    calls = []
    def teacher(z_next, t_next, original_dt, indices):
        assert not torch.is_grad_enabled()
        calls.append((z_next.clone(), t_next.clone(), original_dt.clone(), indices.clone()))
        return torch.ones_like(z_next) * teacher_parameter
    target = consistency_velocity_target(z1, noise, t, dt, teacher)
    tn = (t + dt).clamp(max=1)
    zt = (1 - t[:, None, None]) * noise + t[:, None, None] * z1
    zn = (1 - tn[:, None, None]) * noise + tn[:, None, None] * z1
    endpoint = zn + (1 - tn[:, None, None]) * 2
    reference = (endpoint - zt) / (1 - t[:, None, None])
    torch.testing.assert_close(target, reference)
    assert not target.requires_grad
    assert len(calls) == 1 and calls[0][3].tolist() == [0]
    assert torch.equal(calls[0][2], dt[:1])
    torch.testing.assert_close(calls[0][0], zn[:1])
    # Every row terminal: even a deliberately broken teacher must not run.
    target = consistency_velocity_target(z1, noise, torch.full((3,), .9), torch.full((3,), .8),
                                         lambda *args: pytest.fail('terminal teacher must be skipped'))
    torch.testing.assert_close(target, z1 - noise, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError):
        consistency_velocity_target(z1, noise, torch.ones(3), dt, teacher)


def test_fp32_arithmetic_and_packed_loss_gradients_match_independent_means():
    z1, noise = torch.randn(4, 8, 5).bfloat16(), torch.randn(4, 8, 5).bfloat16()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        target = fm_velocity_target(z1, noise)
        interp = interpolate_latents(z1, noise, torch.full((4,), .3))
    assert target.dtype == interp.dtype == torch.float32
    assert torch.equal(target, z1.float() - noise.float())
    prediction = torch.randn(4, 8, 5, requires_grad=True)
    ref_prediction = prediction.detach().clone().requires_grad_(True)
    fm, ct = torch.tensor([2, 0, 1]), torch.tensor([3])
    loss = packed_flow_loss(prediction, target, fm, ct)['loss']
    reference = (ref_prediction[fm] - target[fm]).square().mean() + (ref_prediction[ct] - target[ct]).square().mean()
    loss.backward()
    reference.backward()
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(prediction.grad, ref_prediction.grad)
    with pytest.raises(ValueError, match='partition'):
        packed_flow_loss(prediction, target, torch.tensor([0, 1, 2]), torch.tensor([2]))


def test_euler_known_velocity_is_continuous_until_endpoint():
    class ConstantVelocity(nn.Module):
        num_slots = 8
        code_dim = 5
        def __init__(self):
            super().__init__()
            self.seen = []
        def forward(self, z, **kwargs):
            self.seen.append(z.clone())
            return torch.full_like(z, .37)
    model = ConstantVelocity().eval()
    ctx = context(batch=1)
    noise = torch.full((1, 8, 5), 1.125)
    endpoint = euler_sample(model, context=ctx, current_state=torch.zeros(1, 6), num_steps=8,
                            initial_noise=noise, use_kv_cache=False)
    torch.testing.assert_close(endpoint, noise + .37)
    assert endpoint.dtype == torch.float32
    assert len(model.seen) == 8 and model.seen[1].max() > 1
    assert model.seen[1][0, 0, 0] != adapter().snap_codes(model.seen[1])[0, 0, 0]


def test_ditx_zero_initialization_and_finite_multistep_gradients():
    torch.manual_seed(5)
    model = tiny_model(checkpointing=True).train()
    z, args = torch.randn(2, 8, 5), model_args()
    assert torch.equal(model(z, **args), torch.zeros_like(z))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    target = torch.randn_like(z)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = (model(z, **args) - target).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        losses.append(loss.item())
        optimizer.step()
    assert losses[-1] < losses[0]
    assert model.blocks[0].cross_attention.k.weight.grad.abs().sum() > 0
    assert model.blocks[0].self_attention.k.weight.grad.abs().sum() > 0
    assert model.step_embedding.mlp[0].weight.grad.abs().sum() > 0


def test_context_padding_closed_summary_and_bidirectional_attention():
    torch.manual_seed(11)
    model = nonzero_model().eval()
    z, args = torch.randn(2, 8, 5), model_args(with_summary=True)
    ctx = args['context']
    ctx.valid_mask[:, 1] = False
    with torch.no_grad():
        baseline = model(z, **args)
        ctx.memory[:, 1] = float('nan')
        ctx.memory[:, 4:] = float('inf')
        masked = model(z, **args)
        torch.testing.assert_close(masked, baseline, rtol=0, atol=0)
        changed = z.clone()
        changed[:, -1] += 3
        output = model(changed, **args)
        assert not torch.allclose(output[:, 0], baseline[:, 0])


def test_cached_and_uncached_euler_agree_without_self_kv_reuse():
    torch.manual_seed(7)
    model = nonzero_model().eval()
    args = model_args(batch=1)
    ctx, state, noise = args['context'], args['current_state'], torch.randn(1, 8, 5)
    with torch.no_grad():
        cached = euler_sample(model, context=ctx, current_state=state, initial_noise=noise)
        uncached = euler_sample(model, context=ctx, current_state=state, initial_noise=noise, use_kv_cache=False)
    torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-6)
    with pytest.raises(RuntimeError, match='gradients disabled'):
        model.build_kv_cache(ctx)
    with torch.no_grad():
        cache = model.build_kv_cache(ctx)
        with pytest.raises(ValueError, match='another model or context'):
            model(noise, **{**args, 'context': context(batch=1)}, kv_cache=cache)


def test_checkpointed_backward_matches_direct_backward_including_context():
    torch.manual_seed(13)
    direct = nonzero_model(False).train()
    checkpointed = deepcopy(direct)
    checkpointed.activation_checkpointing = True
    args = model_args()
    args['context'].memory.requires_grad_(True)
    args2 = deepcopy(args)
    z = torch.randn(2, 8, 5)
    first, second = direct(z, **args), checkpointed(z, **args2)
    first.square().sum().backward()
    second.square().sum().backward()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(args['context'].memory.grad, args2['context'].memory.grad)
    for p1, p2 in zip(direct.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad)


def test_bf16_cached_generation_still_accumulates_fp32():
    model = nonzero_model().eval()
    args = model_args(batch=1)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        value = euler_sample(model, context=args['context'], current_state=args['current_state'], num_steps=2)
    assert value.dtype == torch.float32 and torch.isfinite(value).all()
