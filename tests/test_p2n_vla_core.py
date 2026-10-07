"""Core P2N-VLA backbone tests: layouts, Gemma layers, two-pass equivalence, caching, gate, LoRA."""
import math

import pytest
import torch
import torch.nn.functional as F

from oat.model.vla.flow_head import FlowHeads, flow_velocity, sample_actions
from oat.model.vla.gemma_joint import (ExpertCache, GemmaJoint, SegmentRMSNorm, apply_rope,
                                       rope_cos_sin)
from oat.model.vla.layout import (AR_POSITION_OFFSET, HIST_POSITION_OFFSET, NEG, SEG_AR, SEG_HIST,
                                  SEG_RAWDIFF, build_flow_suffix_layout, build_prefix_layout,
                                  build_suffix_layout, decode_view, prefill_view)
from oat.model.vla.lora import LoRALinear, inject_lora, lora_parameters
from oat.model.vla.specs import TINY_EXPERT, TINY_SIGLIP, TINY_VLM, GemmaSpec
from oat.model.vla.siglip import SiglipEncoder

torch.set_default_dtype(torch.float32)


def _randomize(module, seed=0, scale=0.05):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale
                            + (0 if parameter.ndim > 1 else 0.0))


def _joint(norm_mode="segment", seed=0, checkpointing=False):
    torch.manual_seed(seed)
    joint = GemmaJoint(TINY_VLM, TINY_EXPERT, norm_mode, frozen_dtype=torch.float32,
                       activation_checkpointing=checkpointing)
    _randomize(joint, seed)
    with torch.no_grad():  # keep residual gates away from zero so every path matters
        for module in joint.modules():
            if isinstance(module, SegmentRMSNorm):
                module.modulation[:, 2 * module.dim:] += 1.0
            if hasattr(module, "dense") and module.__class__.__name__ == "AdaRMSNorm":
                module.dense.bias[2 * module.dim:] += 1.0
    return joint


def _prefix_inputs(batch=2, n_img=4, prompt_len=6, ki_len=0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    img_valid = torch.ones(batch, n_img, dtype=torch.bool)
    prompt_valid = torch.ones(batch, prompt_len, dtype=torch.bool)
    prompt_valid[0, 4:] = False  # padded sample
    layout = build_prefix_layout(img_valid, prompt_valid, ki_len)
    embeds = torch.randn(batch, n_img + prompt_len + ki_len, TINY_VLM.width, generator=generator)
    return embeds, layout


# ---------------------------------------------------------------- layout rules
def test_prefix_layout_rules():
    _, layout = _prefix_inputs(ki_len=3)
    bias = layout.bias[:, 0]
    nonki, total = layout.nonki_len, layout.nonki_len + layout.ki_len
    valid = layout.valid
    for b in range(bias.shape[0]):
        for q in range(total):
            visible = bias[b, q] == 0
            assert visible[q], "every row keeps its diagonal"
            if q < nonki:
                assert not visible[nonki:].any(), "non-KI rows never see KI"
                expected = valid[b, :nonki].clone()
                expected[q] = True
                assert torch.equal(visible[:nonki], expected)
            else:
                assert torch.equal(visible[:nonki], valid[b, :nonki])
                assert torch.equal(visible[nonki:], torch.arange(layout.ki_len) <= q - nonki)
    # positions: pads do not advance; KI continues from pos0
    assert layout.pos0.tolist() == [8, 10]
    assert layout.positions[0, :nonki].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]
    assert layout.positions[0, nonki:].tolist() == [8, 9, 10]
    assert layout.positions[1, nonki:].tolist() == [10, 11, 12]


def test_suffix_layout_rules_and_positions():
    pos0 = torch.tensor([8, 10])
    prefix_valid = torch.ones(2, 10, dtype=torch.bool)
    prefix_valid[0, 8:] = False
    rawdiff_valid = torch.ones(2, 9, dtype=torch.bool)
    rawdiff_valid[0, :5] = False
    log_gate = torch.tensor([[-0.1], [-2.0]])
    layout = build_suffix_layout(pos0, prefix_valid, rawdiff_valid, 4, 8, log_gate=log_gate)
    assert layout.seg_ids.tolist() == [SEG_RAWDIFF] * 9 + [SEG_HIST] * 4 + [SEG_AR] * 8
    expected_pos = list(range(9)) + [HIST_POSITION_OFFSET] * 4 + [AR_POSITION_OFFSET + k for k in range(8)]
    assert (layout.positions - pos0[:, None]).tolist() == [expected_pos, expected_pos]
    bias = layout.bias[:, 0]
    nonki = 10
    assert (bias.max(dim=-1).values == 0).all(), "no row is fully masked"
    for row in range(13):  # condition rows see only themselves
        visible = (bias[:, row] == 0)
        assert visible.sum(dim=-1).tolist() == [1, 1] and visible[:, nonki + row].all()
    ar_row = bias[:, 13 + 3]
    assert torch.equal(ar_row[:, :nonki] == 0, prefix_valid)
    assert torch.equal(ar_row[:, nonki:nonki + 9] == 0, rawdiff_valid)
    assert torch.allclose(ar_row[:, nonki + 9:nonki + 13], log_gate.expand(2, 4))
    assert (ar_row[:, nonki + 13:nonki + 13 + 4] == 0).all() and (ar_row[:, nonki + 13 + 4:] == NEG).all()
    # variant/validity independence of AR positions
    plain = build_suffix_layout(pos0, prefix_valid, None, 0, 8)
    assert torch.equal(plain.positions, layout.positions[:, 13:])
    closed = build_suffix_layout(pos0, prefix_valid, rawdiff_valid, 4, 8, hist_closed=True)
    assert (closed.bias[:, 0, 13:, nonki + 9:nonki + 13] == NEG).all()
    with pytest.raises(ValueError):
        build_suffix_layout(pos0, prefix_valid, rawdiff_valid, 4, 8, log_gate=torch.tensor([[0.1], [0.0]]))


def test_decode_view_matches_teacher_forcing_rows():
    pos0 = torch.tensor([3, 5])
    prefix_valid = torch.ones(2, 6, dtype=torch.bool)
    layout = build_suffix_layout(pos0, prefix_valid, torch.ones(2, 9, dtype=torch.bool), 4, 8,
                                 log_gate=torch.full((2, 1), -0.3))
    for k in range(8):
        bias, position = decode_view(layout, k)
        row = layout.n_cond + k
        assert torch.equal(bias, layout.bias[:, :, row:row + 1, :layout.nonki_len + row + 1])
        assert (layout.bias[:, :, row, layout.nonki_len + row + 1:] == NEG).all()
        assert torch.equal(position, layout.positions[:, row:row + 1])
    bias, positions, seg = prefill_view(layout, layout.n_cond + 1)
    assert bias.shape[-1] == layout.nonki_len + layout.n_cond + 1 and seg[-1] == SEG_AR


def test_flow_suffix_layout():
    layout = build_flow_suffix_layout(torch.tensor([4]), torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool), 3)
    assert layout.positions.tolist() == [[4, 5, 6]]
    assert (layout.bias[0, 0, :, 4] == NEG).all() and (layout.bias[0, 0, :, 5:] == 0).all()


# ---------------------------------------------------------------- Gemma pieces
def _hf_gemma_config():
    from transformers import GemmaConfig
    config = GemmaConfig(vocab_size=128, hidden_size=TINY_VLM.width, intermediate_size=TINY_VLM.mlp_dim,
                         num_hidden_layers=1, num_attention_heads=TINY_VLM.num_heads,
                         num_key_value_heads=TINY_VLM.num_kv_heads, head_dim=TINY_VLM.head_dim,
                         hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6, rope_theta=10000.0)
    config._attn_implementation = "eager"
    return config


def test_rope_matches_transformers():
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding, apply_rotary_pos_emb
    config = _hf_gemma_config()
    positions = torch.tensor([[0, 1, 2, 5, 9, 40], [3, 3, 7, 8, 100, 101]])
    q = torch.randn(2, 8, 6, TINY_VLM.head_dim)
    k = torch.randn(2, 1, 6, TINY_VLM.head_dim)
    cos_hf, sin_hf = GemmaRotaryEmbedding(config)(q, positions)
    q_ref, k_ref = apply_rotary_pos_emb(q, k, cos_hf, sin_hf)
    cos, sin = rope_cos_sin(positions, TINY_VLM.head_dim, 10000.0)
    assert torch.allclose(apply_rope(q, cos, sin), q_ref, atol=1e-5)
    assert torch.allclose(apply_rope(k, cos, sin), k_ref, atol=1e-5)


def test_vlm_layer_matches_transformers_decoder_layer():
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaRotaryEmbedding
    config = _hf_gemma_config()
    reference = GemmaDecoderLayer(config, layer_idx=0)
    _randomize(reference, 3, 0.1)
    joint = _joint()
    ours = joint.vlm.layers[0]
    ours.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 7, TINY_VLM.width)
    positions = torch.arange(7)[None].expand(2, 7)
    mask = torch.zeros(2, 1, 7, 7)
    mask[1, :, :, 5:] = NEG
    mask[1, :, 5:, 5:] = torch.where(torch.eye(2, dtype=torch.bool), 0.0, NEG)
    cos_hf, sin_hf = GemmaRotaryEmbedding(config)(x, positions)
    expected = reference(x, attention_mask=mask, position_ids=positions, position_embeddings=(cos_hf, sin_hf))
    cos, sin = rope_cos_sin(positions, TINY_VLM.head_dim, 10000.0)
    actual, _, _ = joint._vlm_layer(ours, x, cos, sin, mask)
    assert torch.allclose(actual, expected, atol=1e-5)


# ---------------------------------------------------------------- two pass == joint pass
def _reference_joint_pass(joint, prefix_embeds, prefix_layout, suffix_x, suffix_layout, seg_ids):
    """Single joint pass over [prefix | suffix] with per-stream weights and one softmax per layer."""
    vlm_spec = joint.vlm_spec
    p_len, s_len = prefix_embeds.shape[1], suffix_x.shape[1]
    nonki = prefix_layout.nonki_len
    bias = torch.full((prefix_embeds.shape[0], 1, p_len + s_len, p_len + s_len), NEG)
    bias[:, :, :p_len, :p_len] = prefix_layout.bias
    bias[:, :, p_len:, :nonki] = suffix_layout.bias[:, :, :, :nonki]
    bias[:, :, p_len:, p_len:] = suffix_layout.bias[:, :, :, nonki:]
    positions = torch.cat((prefix_layout.positions, suffix_layout.positions), dim=1)
    cos, sin = rope_cos_sin(positions, vlm_spec.head_dim, vlm_spec.rope_theta)
    xp, xs = prefix_embeds, suffix_x
    for vlm_layer, expert_layer in zip(joint.vlm.layers, joint.expert.layers):
        hp = vlm_layer.input_layernorm(xp)
        hs, gate_attn = expert_layer.input_layernorm(xs, seg_ids)
        qs, ks, vs = [], [], []
        for layer, h in ((vlm_layer, hp), (expert_layer, hs)):
            length = h.shape[1]
            qs.append(layer.self_attn.q_proj(h).view(h.shape[0], length, 8, -1).transpose(1, 2))
            ks.append(layer.self_attn.k_proj(h).view(h.shape[0], length, 1, -1).transpose(1, 2))
            vs.append(layer.self_attn.v_proj(h).view(h.shape[0], length, 1, -1).transpose(1, 2))
        q, k, v = torch.cat(qs, 2), torch.cat(ks, 2), torch.cat(vs, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        probs = ((q * vlm_spec.head_dim ** -0.5) @ k.transpose(-1, -2) + bias).softmax(-1)
        out = (probs @ v).transpose(1, 2).reshape(q.shape[0], p_len + s_len, -1)
        xp = xp + vlm_layer.self_attn.o_proj(out[:, :p_len])
        xs = xs + expert_layer.self_attn.o_proj(out[:, p_len:]) * gate_attn
        xp = xp + vlm_layer.mlp(vlm_layer.post_attention_layernorm(xp))
        hs, gate_mlp = expert_layer.post_attention_layernorm(xs, seg_ids)
        xs = xs + expert_layer.mlp(hs) * gate_mlp
    return joint.expert.norm(xs, seg_ids)[0]


def test_two_pass_equals_joint_pass_with_ki():
    joint = _joint().eval()
    embeds, prefix_layout = _prefix_inputs(ki_len=3)
    out = joint.prefix_forward(embeds, prefix_layout)
    rawdiff_valid = torch.ones(2, 9, dtype=torch.bool)
    rawdiff_valid[1, :3] = False
    suffix = build_suffix_layout(out.pos0, prefix_layout.nonki_valid, rawdiff_valid, 4, 8,
                                 log_gate=torch.tensor([[-0.2], [-1.5]]))
    x = torch.randn(2, suffix.positions.shape[1], TINY_EXPERT.width)
    two_pass = joint.expert_forward(x, out.kv, suffix.bias, suffix.positions, seg_ids=suffix.seg_ids).hidden
    reference = _reference_joint_pass(joint, embeds, prefix_layout, x, suffix, suffix.seg_ids)
    assert torch.allclose(two_pass, reference, atol=2e-5), (two_pass - reference).abs().max()


def test_ki_tokens_do_not_change_prefix_kv_or_expert():
    joint = _joint().eval()
    embeds, without = _prefix_inputs(ki_len=0)
    ki = torch.randn(2, 3, TINY_VLM.width)
    with_ki = build_prefix_layout(without.valid[:, :4], without.valid[:, 4:], 3)
    a = joint.prefix_forward(embeds, without)
    b = joint.prefix_forward(torch.cat((embeds, ki), 1), with_ki)
    assert torch.equal(a.pos0, b.pos0)
    for (ka, va), (kb, vb) in zip(a.kv, b.kv):
        assert torch.allclose(ka, kb, atol=1e-6) and torch.allclose(va, vb, atol=1e-6)
    assert torch.allclose(a.hidden, b.hidden[:, :without.nonki_len], atol=1e-6)


# ---------------------------------------------------------------- caching and gate
def _cond_and_ar(batch=2, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, 13, TINY_EXPERT.width, generator=generator),
            torch.randn(batch, 8, TINY_EXPERT.width, generator=generator))


def test_cached_decode_matches_teacher_forcing():
    joint = _joint().eval()
    embeds, prefix_layout = _prefix_inputs()
    out = joint.prefix_forward(embeds, prefix_layout)
    rawdiff_valid = torch.ones(2, 9, dtype=torch.bool)
    rawdiff_valid[0, :6] = False
    layout = build_suffix_layout(out.pos0, prefix_layout.nonki_valid, rawdiff_valid, 4, 8,
                                 log_gate=torch.tensor([[-0.4], [-0.05]]))
    cond, ar = _cond_and_ar()
    full = joint.expert_forward(torch.cat((cond, ar), 1), out.kv, layout.bias, layout.positions,
                                seg_ids=layout.seg_ids).hidden[:, 13:]
    bias, positions, seg = prefill_view(layout, 14)
    step = joint.expert_forward(torch.cat((cond, ar[:, :1]), 1), out.kv, bias, positions, seg_ids=seg,
                                use_cache=True)
    decoded = [step.hidden[:, -1:]]
    cache = step.cache
    assert isinstance(cache, ExpertCache) and cache.length == 14
    for k in range(1, 8):
        bias, position = decode_view(layout, k)
        step = joint.expert_forward(ar[:, k:k + 1], out.kv, bias, position,
                                    seg_ids=torch.tensor([SEG_AR]), cache=cache, use_cache=True)
        assert step.cache.length == cache.length + 1 and cache.length == 14 + k - 1  # never mutated
        cache = step.cache
        decoded.append(step.hidden)
    assert torch.allclose(torch.cat(decoded, 1), full, atol=2e-5)


def test_gate_open_closed_semantics():
    joint = _joint().eval()
    embeds, prefix_layout = _prefix_inputs()
    out = joint.prefix_forward(embeds, prefix_layout)
    rawdiff_valid = torch.ones(2, 9, dtype=torch.bool)
    cond, ar = _cond_and_ar()

    def run(hist_tokens, **kwargs):
        n_hist = 0 if hist_tokens is None else 4
        layout = build_suffix_layout(out.pos0, prefix_layout.nonki_valid, rawdiff_valid, n_hist, 8, **kwargs)
        parts = [cond[:, :9]] + ([] if hist_tokens is None else [hist_tokens]) + [ar]
        hidden = joint.expert_forward(torch.cat(parts, 1), out.kv, layout.bias, layout.positions,
                                      seg_ids=layout.seg_ids).hidden
        return hidden[:, -8:]

    hist = cond[:, 9:]
    assert torch.equal(run(hist, log_gate=torch.zeros(2, 1)), run(hist))  # open == no bias, bitwise
    closed = run(hist, hist_closed=True)
    assert torch.equal(closed, run(torch.randn_like(hist) * 10, hist_closed=True))
    assert torch.allclose(closed, run(None), atol=1e-6)  # identical AR positions without HIST
    assert not torch.allclose(run(hist, log_gate=torch.full((2, 1), -0.5)), run(hist), atol=1e-6)


def test_gate_bias_receives_gradient_and_probe():
    joint = _joint().eval()
    embeds, prefix_layout = _prefix_inputs()
    out = joint.prefix_forward(embeds, prefix_layout)
    log_gate = torch.full((2, 1), -0.3, requires_grad=True)
    layout = build_suffix_layout(out.pos0, prefix_layout.nonki_valid, torch.ones(2, 9, dtype=torch.bool), 4, 8,
                                 log_gate=log_gate)
    cond, ar = _cond_and_ar()
    result = joint.expert_forward(torch.cat((cond, ar), 1), out.kv, layout.bias, layout.positions,
                                  seg_ids=layout.seg_ids, probe_cols=layout.hist_columns)
    result.hidden.sum().backward()
    assert log_gate.grad is not None and log_gate.grad.abs().sum() > 0
    assert result.probe_mass.shape == (TINY_EXPERT.depth,)
    assert ((result.probe_mass >= 0) & (result.probe_mass <= 1)).all()


# ---------------------------------------------------------------- checkpointing and LoRA
def test_activation_checkpointing_equivalence():
    losses, grads = [], []
    for checkpointing in (False, True):
        joint = _joint(checkpointing=checkpointing)
        inject_lora(joint.vlm.layers, rank=4, alpha=4.0)
        torch.manual_seed(5)
        for parameter in lora_parameters(joint):
            parameter.data.normal_(0, 0.02)
        joint.train()
        embeds, layout = _prefix_inputs(ki_len=2)
        out = joint.prefix_forward(embeds, layout)
        loss = out.hidden.pow(2).mean() + sum(k.pow(2).mean() for k, _ in out.kv)
        loss.backward()
        losses.append(loss.detach())
        grads.append(torch.cat([p.grad.flatten() for p in lora_parameters(joint)]))
    assert torch.allclose(losses[0], losses[1]) and torch.allclose(grads[0], grads[1], atol=1e-7)


def test_lora_zero_init_identity_and_gradients():
    base = torch.nn.Linear(16, 12, bias=False)
    reference = base.weight.detach().clone()
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 16)
    assert torch.equal(lora(x), x @ reference.T)
    lora.lora_B.data.normal_()
    lora(x).sum().backward()
    assert base.weight.grad is None and lora.lora_A.grad is not None and lora.lora_B.grad is not None
    assert torch.allclose(lora.merged_weight(), reference + lora.lora_B @ lora.lora_A * 2.0)
    joint = _joint()
    replaced = inject_lora(joint.vlm.layers, rank=2, alpha=2.0)
    assert len(replaced) == 7 * TINY_VLM.depth
    assert all(p.requires_grad for p in lora_parameters(joint))
    assert not any(p.requires_grad for n, p in joint.vlm.named_parameters() if "lora_" not in n)


# ---------------------------------------------------------------- SigLIP and flow head
def test_tiny_siglip_encoder():
    encoder = SiglipEncoder(TINY_SIGLIP, TINY_VLM.width, frozen_dtype=torch.float32)
    assert not any(p.requires_grad for p in encoder.parameters())
    tokens = encoder(torch.rand(3, 3, 28, 28) * 2 - 1)
    assert tokens.shape == (3, TINY_SIGLIP.num_tokens, TINY_VLM.width)
    encoder.train()
    assert not encoder.training


def test_flow_head_velocity_and_sampling_shapes():
    joint = _joint("adarms").eval()
    heads = FlowHeads(TINY_EXPERT.width)
    embeds, layout = _prefix_inputs()
    out = joint.prefix_forward(embeds, layout)
    x_t = torch.randn(2, 5, 32)
    v = flow_velocity(joint, heads, out.kv, out.pos0, layout.nonki_valid, x_t, torch.tensor([0.3, 0.8]))
    assert v.shape == (2, 5, 32) and torch.isfinite(v).all()
    actions = sample_actions(joint, heads, out.kv, out.pos0, layout.nonki_valid, torch.randn(2, 5, 32), 4)
    assert actions.shape == (2, 5, 32)


def test_expert_validates_inputs():
    joint = _joint().eval()
    embeds, layout = _prefix_inputs()
    out = joint.prefix_forward(embeds, layout)
    suffix = build_suffix_layout(out.pos0, layout.nonki_valid, None, 0, 2)
    x = torch.randn(2, 2, TINY_EXPERT.width)
    with pytest.raises(ValueError):
        joint.expert_forward(x, out.kv, suffix.bias, suffix.positions)  # missing seg_ids
    with pytest.raises(ValueError):
        joint.expert_forward(x, out.kv, suffix.bias[..., :-1], suffix.positions, seg_ids=suffix.seg_ids)
