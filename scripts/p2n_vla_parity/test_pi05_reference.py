"""Checks for the LeRobot PI0.5 fp32 parity reference (scripts/p2n_vla_parity).

Run under /venv/oat, from the repository root:

    /venv/oat/bin/python -m pytest scripts/p2n_vla_parity -q

- The fast tests cover:
  - the deterministic inputs and SentencePiece ids;
  - the schema validator and the comparison helper, on a tiny synthetic payload;
  - the M1-parity threshold logic;
  - the dump script's interpreter guard.
- ``requires_pi05`` tests check the dumped ``output/parity/pi05_reference_fp32.pt`` against the raw
  ``model.safetensors``, independently of LeRobot. They cover:
  - text-embedding scaling;
  - post-RoPE layer-0 K/V;
  - logits from the tied head;
  - adaRMS time conditioning.

  ``slow`` + ``requires_pi05`` runs the M1 exit check, ``check_m1_parity.py --device cpu``, in a
  subprocess. The repo's ``oat.model.vla`` port must match LeRobot at fp32 tolerance.

  These tests skip loudly when the weights or the dump are missing.
"""
import copy
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import pi05_reference_spec as spec

HERE = Path(__file__).resolve().parent
DUMP_SCRIPT = HERE / "dump_pi05_reference.py"
DUMP_COMMAND = ("CUDA_VISIBLE_DEVICES=1 HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 "
                "/venv/lerobot_ref/bin/python scripts/p2n_vla_parity/dump_pi05_reference.py")
VLM = "paligemma_with_expert.paligemma.model.language_model."
LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"


def _require_spm():
    if not spec.DEFAULT_SPM.is_file():
        pytest.skip(f"SKIPPED LOUDLY: PaliGemma SentencePiece model missing at {spec.DEFAULT_SPM} "
                    "(run scripts/fetch_p2n_vla_assets.py)")


@pytest.fixture(scope="module")
def reference():
    if not spec.PI05_WEIGHTS.is_file():
        pytest.skip(f"SKIPPED LOUDLY: pi05_base weights missing at {spec.PI05_WEIGHTS}")
    if not spec.DEFAULT_OUTPUT.is_file():
        pytest.skip(f"SKIPPED LOUDLY: reference dump missing at {spec.DEFAULT_OUTPUT}; produce it with: {DUMP_COMMAND}")
    return torch.load(spec.DEFAULT_OUTPUT, map_location="cpu", weights_only=True)


def _read(names):
    from safetensors import safe_open

    with safe_open(str(spec.PI05_WEIGHTS), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name).float() for name in names}


def _read_rows(name, ids):
    """Rows ``ids`` of a [V, W] checkpoint tensor without loading all of it."""
    from safetensors import safe_open

    with safe_open(str(spec.PI05_WEIGHTS), framework="pt", device="cpu") as handle:
        sliced = handle.get_slice(name)
        return torch.stack([sliced[int(i):int(i) + 1][0] for i in ids]).float()


def _gemma_rms(x, weight, eps=1e-6):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight.float())


def _rope(x, positions, inv_freq):
    freqs = positions.to(torch.float32)[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[:, None], emb.sin()[:, None]
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


# ----------------------------------------------------------------------------- fast: inputs
def test_reference_inputs_are_deterministic_and_well_formed():
    _require_spm()
    first, second = spec.build_reference_inputs(), spec.build_reference_inputs()
    assert sorted(first) == ["images01", "lang_masks", "lang_tokens", "noise", "t", "x_t"]
    for key in first:
        assert torch.equal(first[key], second[key]), key
    batch, lang, chunk = spec.BATCH, spec.LANG_MAX_LEN, spec.CHUNK_SIZE
    assert first["images01"].shape == (batch, spec.N_REAL_CAMERAS, 3, spec.IMAGE_SIZE, spec.IMAGE_SIZE)
    assert first["images01"].dtype == torch.float32
    assert 0.0 <= float(first["images01"].min()) and float(first["images01"].max()) <= 1.0
    assert first["noise"].shape == first["x_t"].shape == (batch, chunk, spec.ACTION_DIM)
    assert torch.equal(first["t"], torch.tensor([0.3, 0.8], dtype=torch.float32))
    assert first["lang_tokens"].dtype == torch.long and first["lang_masks"].dtype == torch.bool
    assert first["lang_tokens"].shape == first["lang_masks"].shape == (batch, lang)
    for row, expected in enumerate(spec.EXPECTED_PROMPT_IDS):
        valid = first["lang_masks"][row]
        assert tuple(first["lang_tokens"][row][valid].tolist()) == expected
        assert torch.equal(valid, torch.arange(lang) < len(expected))       # right-padded
        assert bool((first["lang_tokens"][row][~valid] == spec.PAD_ID).all())
    assert [len(ids) for ids in spec.EXPECTED_PROMPT_IDS] == [51, 35]
    other = spec.build_reference_inputs(seed=spec.SEED + 1)
    assert not torch.equal(other["images01"], first["images01"])
    assert torch.equal(other["lang_tokens"], first["lang_tokens"])


def test_model_images_follow_lerobot_empty_camera_convention():
    generator = torch.Generator().manual_seed(0)
    images01 = spec.make_camera_images01(2, 2, 28, generator)
    images, masks = spec.to_model_images(images01, 3)
    assert images.shape == (2, 3, 3, 28, 28) and masks.shape == (2, 3)
    assert torch.equal(images[:, :2], images01 * 2.0 - 1.0)
    assert bool((images[:, 2] == -1.0).all())
    assert masks.tolist() == [[True, True, False], [True, True, False]]
    with pytest.raises(ValueError, match="n_slots"):
        spec.to_model_images(images01, 1)


def test_prompt_longer_than_max_len_raises():
    _require_spm()
    # The requested length 48 cannot hold prompt 0 (51 tokens), hence LANG_MAX_LEN = 64.
    with pytest.raises(ValueError, match="needs 51 tokens, which exceeds max_len=48"):
        spec.tokenize_prompts(max_len=48)
    tokens, masks = spec.tokenize_prompts(max_len=51)
    assert masks.sum(dim=1).tolist() == [51, 35]
    with pytest.raises(ValueError, match="max_len"):
        spec.tokenize_prompts(max_len=0)


# ----------------------------------------------------------------------------- fast: schema validator
def _synthetic_reference():
    """A tiny, self-consistent payload in the reference layout (shapes from its own metadata)."""
    generator = torch.Generator().manual_seed(0)
    batch, slots, per_image, lang, chunk, action, steps = 2, 3, 4, 8, 5, 3, 2
    width, depth, head_dim, vocab, expert_width = 8, 2, 4, 16, 6
    images01 = torch.rand((batch, 2, 3, spec.IMAGE_SIZE, spec.IMAGE_SIZE), generator=generator)
    images, img_masks = spec.to_model_images(images01, slots)
    lengths = torch.tensor([6, 4])
    masks = torch.arange(lang)[None] < lengths[:, None]
    tokens = torch.randint(3, vocab, (batch, lang), generator=generator) * masks
    tokens[:, 0] = spec.BOS_ID
    n_img = slots * per_image
    prefix = n_img + lang
    token_valid = img_masks.repeat_interleave(per_image, dim=1)
    pad = torch.cat((token_valid, masks), dim=1)
    pos0 = pad.sum(dim=1)
    prefix_embeddings = torch.randn((batch, prefix, width), generator=generator)
    noise = torch.randn((batch, chunk, action), generator=generator)
    velocities = torch.randn((steps, batch, chunk, action), generator=generator)
    trajectory = [noise]
    for step in range(steps):
        trajectory.append(trajectory[-1] + (-1.0 / steps) * velocities[step])
    prompt_index = lengths[:, None] - spec.N_LOGIT_POSITIONS + torch.arange(spec.N_LOGIT_POSITIONS)[None]
    rand = lambda *shape: torch.randn(shape, generator=generator)  # noqa: E731
    metadata = {key: "x" for key in spec.REQUIRED_METADATA_KEYS}
    metadata.update({
        "prompts": ["a", "b"], "camera_slots": ["c0", "c1", "c2"], "camera_slot_valid": [True, True, False],
        "lang_max_len": lang, "chunk_size": chunk, "max_action_dim": action, "n_tokens_per_image": per_image,
        "n_image_tokens": n_img, "prefix_len": prefix, "num_inference_steps": steps,
        "lang_valid_lengths": lengths.tolist(), "prefix_valid_counts": pos0.tolist(), "t_values": [0.3, 0.8],
        "keys_post_rope": True, "scaling": {"text_embed_scale": math.sqrt(width)},
        "geometry": {"vlm_width": width, "depth": depth, "head_dim": head_dim, "vocab_size": vocab,
                     "expert_width": expert_width},
    })
    return {
        "format": spec.REFERENCE_FORMAT,
        "metadata": metadata,
        "inputs": {"images": images, "img_masks": img_masks, "lang_tokens": tokens, "lang_masks": masks,
                   "noise": noise, "x_t": rand(batch, chunk, action), "t": torch.tensor([0.3, 0.8])},
        "image_embeddings": prefix_embeddings[:, :n_img].clone(),
        "image_token_valid": token_valid,
        "prefix_embeddings": prefix_embeddings,
        "prefix_pad_masks": pad,
        "prefix_position_ids": torch.cumsum(pad.long(), dim=1) - 1,
        "prefix_att_2d_masks": pad[:, None, :] & pad[:, :, None],
        "prefix_hidden": rand(batch, prefix, width),
        "prefix_keys": [rand(batch, 1, prefix, head_dim) for _ in range(depth)],
        "prefix_values": [rand(batch, 1, prefix, head_dim) for _ in range(depth)],
        "vlm_logits": rand(batch, spec.N_LOGIT_POSITIONS, vocab),
        "vlm_logits_prompt_index": prompt_index,
        "vlm_logits_prefix_index": prompt_index + n_img,
        "vlm_logits_input_ids": torch.gather(tokens, 1, prompt_index),
        "time_embedding": rand(batch, expert_width),
        "adarms_cond": rand(batch, expert_width),
        "suffix_position_ids": pos0[:, None] + torch.arange(chunk)[None],
        "suffix_att_2d_masks": torch.cat((pad[:, None, :].expand(batch, chunk, prefix),
                                          torch.ones((batch, chunk, chunk), dtype=torch.bool)), dim=2),
        "v_t": rand(batch, chunk, action),
        "v_t_joint": rand(batch, chunk, action),
        "sample_actions": trajectory[-1].clone(),
        "sample_trajectory": torch.stack(trajectory),
        "sample_velocities": velocities,
        "sample_times": torch.tensor([1.0 + k * (-1.0 / steps) for k in range(steps)], dtype=torch.float32),
        "rope_inv_freq": rand(head_dim // 2),
    }


def test_validator_accepts_consistent_payload():
    shapes = spec.validate_reference(_synthetic_reference())
    assert shapes["prefix_keys"] == (2, 2, 1, 3 * 4 + 8, 4)
    assert shapes["inputs.images"] == (2, 3, 3, spec.IMAGE_SIZE, spec.IMAGE_SIZE)


@pytest.mark.parametrize("corruption, message", [
    (lambda r: r.__setitem__("format", "other"), "format must be"),
    (lambda r: r.__setitem__("v_t", r["v_t"][:, :2]), "v_t has shape"),
    (lambda r: r["prefix_hidden"].__setitem__((0, 0, 0), float("nan")), "prefix_hidden contains non-finite"),
    (lambda r: r.__setitem__("vlm_logits", r["vlm_logits"].double()), "vlm_logits has dtype"),
    (lambda r: r["prefix_keys"].pop(), "prefix_keys must be a list"),
    (lambda r: r.__setitem__("prefix_position_ids", r["prefix_position_ids"] + 1), "cumsum"),
    (lambda r: r["inputs"]["img_masks"].__setitem__((0, 2), True), "img_masks disagree"),
    (lambda r: r["inputs"]["lang_tokens"].__setitem__((1, 7), 5), "pad id 0"),
    (lambda r: r["image_embeddings"].mul_(2.0), "no extra scaling"),
    (lambda r: r["sample_velocities"].add_(1.0), r"sample_trajectory step 0 is not x \+ dt\*v"),
    (lambda r: r["metadata"].__setitem__("keys_post_rope", False), "post_rope"),
    (lambda r: r["metadata"].pop("geometry"), "missing keys"),
    (lambda r: r["metadata"].__setitem__("torch_version", torch.__version__), "plain"),   # TorchVersion
    (lambda r: r["metadata"]["scaling"].__setitem__("bad", torch.Size([1])), "plain"),
])
def test_validator_rejects_corruption(corruption, message):
    payload = copy.deepcopy(_synthetic_reference())
    corruption(payload)
    with pytest.raises(ValueError, match=message):
        spec.validate_reference(payload)


def test_compare_references_masks_pad_rows_and_pad_keys():
    reference = _synthetic_reference()
    stats = spec.compare_references(copy.deepcopy(reference), reference)
    assert stats and all(row["max_abs"] == 0.0 for row in stats.values())
    pad = reference["prefix_pad_masks"]
    pad_row = int((~pad[0]).nonzero()[0])           # the first token of the empty camera slot
    valid_row = int(pad[0].nonzero()[0])
    candidate = copy.deepcopy(reference)
    candidate["prefix_hidden"][0, pad_row] += 5.0
    candidate["prefix_keys"][1][0, 0, pad_row] += 5.0
    close = lambda value, target: math.isclose(value, target, rel_tol=1e-5)  # noqa: E731 - fp32 offsets
    assert spec.compare_references(candidate, reference)["prefix_hidden"]["max_abs"] == 0.0
    assert spec.compare_references(candidate, reference)["prefix_keys[1]"]["max_abs"] == 0.0
    assert close(spec.compare_references(candidate, reference, valid_only=False)["prefix_hidden"]["max_abs"], 5.0)
    candidate["prefix_keys"][0][0, 0, pad_row] += 3.0             # layer 0 is always compared in full
    candidate["prefix_hidden"][0, valid_row] += 2.0
    candidate["v_t"][1, 2, 0] -= 1.5
    stats = spec.compare_references(candidate, reference)
    assert close(stats["prefix_keys[0]"]["max_abs"], 3.0)
    assert close(stats["prefix_hidden"]["max_abs"], 2.0)
    assert close(stats["v_t"]["max_abs"], 1.5)
    assert close(stats["v_t"]["max_rel_to_absmax"], stats["v_t"]["max_abs"] / float(reference["v_t"].abs().max()))


def test_compare_cli_reports_every_tensor(tmp_path, capsys):
    import compare_pi05_references

    reference = _synthetic_reference()
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(reference, first)
    candidate = copy.deepcopy(reference)
    candidate["v_t"] += 0.25
    torch.save(candidate, second)
    out_json = tmp_path / "stats.json"
    assert compare_pi05_references.main([str(second), str(first), "--json", str(out_json)]) == 0
    printed = capsys.readouterr().out
    assert "v_t" in printed and "prefix_keys[1]" in printed and "sample_actions" in printed
    import json

    stats = json.loads(out_json.read_text())
    assert math.isclose(stats["v_t"]["max_abs"], 0.25, rel_tol=1e-6) and stats["prefix_hidden"]["max_abs"] == 0.0
    candidate["inputs"]["x_t"] += 1.0
    torch.save(candidate, second)
    with pytest.raises(SystemExit, match="inputs.x_t differ"):
        compare_pi05_references.main([str(second), str(first)])


# ----------------------------------------------------------------------------- fast: M1 parity thresholds
def _stats(value, metric):
    names = ["image_embeddings", "prefix_embeddings", "prefix_hidden", "vlm_logits", "time_embedding",
             "adarms_cond", "v_t", "sample_actions"] + [f"prefix_keys[{i}]" for i in range(3)] + \
            [f"prefix_values[{i}]" for i in range(3)]
    return {name: {metric: value, "max_abs": value} for name in names}


@pytest.mark.parametrize("precision, metric", [("fp32", "max_rel_to_absmax"), ("bf16", "rel_fro")])
def test_m1_parity_thresholds_pass_and_fail(precision, metric):
    import check_m1_parity as check

    assert check.THRESHOLDS[precision][0] == metric
    assert set(check.THRESHOLDS[precision][1]) == set(check.GROUPS)
    rows = check.evaluate(_stats(0.0, metric), precision)
    assert [group for group, *_ in rows] == list(check.THRESHOLDS[precision][1])
    assert all(ok for *_, ok in rows)
    stats = _stats(0.0, metric)
    stats["prefix_keys[2]"][metric] = 1.0                            # a single bad layer fails the group
    failed = {group for group, _, _, ok in check.evaluate(stats, precision) if not ok}
    assert failed == {"prefix_keys"}
    assert check.group_metric(stats, metric)["prefix_keys"] == 1.0
    stats.pop("v_t")
    with pytest.raises(ValueError, match="missing groups \\['v_t'\\]"):
        check.evaluate(stats, precision)
    custom = check.evaluate(_stats(0.5, metric), precision, thresholds={"v_t": 0.6, "vlm_logits": 0.4})
    assert [(group, ok) for group, _, _, ok in custom] == [("v_t", True), ("vlm_logits", False)]
    with pytest.raises(ValueError, match="precision"):
        check.evaluate(_stats(0.0, metric), "fp16")


def test_m1_parity_fp32_thresholds_cover_the_reference_noise_floor():
    """fp32 thresholds must clear the measured CPU-vs-CUDA spread of the LeRobot reference itself."""
    import json
    import check_m1_parity as check

    floor_path = spec.DEFAULT_OUTPUT.with_name("cpu_vs_cuda_fp32_noise_floor.json")
    if not floor_path.is_file():
        pytest.skip(f"SKIPPED LOUDLY: noise-floor stats missing at {floor_path}")
    floor = check.group_metric(json.loads(floor_path.read_text()), "max_rel_to_absmax")
    for group, limit in check.THRESHOLDS["fp32"][1].items():
        assert floor[group] * 2 <= limit or floor[group] == 0.0, (group, floor[group], limit)


@pytest.mark.slow
@pytest.mark.requires_pi05
def test_m1_stack_matches_lerobot_reference_fp32_cpu(tmp_path):
    """M1 exit criterion, run on CPU in a subprocess (about 40 s and about 15 GB RAM): the repo's port matches LeRobot."""
    import json

    if not spec.PI05_WEIGHTS.is_file() or not spec.DEFAULT_OUTPUT.is_file():
        pytest.skip(f"SKIPPED LOUDLY: needs {spec.PI05_WEIGHTS} and {spec.DEFAULT_OUTPUT} ({DUMP_COMMAND})")
    out = tmp_path / "parity.json"
    env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, str(HERE / "check_m1_parity.py"), "--device", "cpu", "--precision", "fp32",
                           "--json", str(out)], capture_output=True, text=True, timeout=1800, env=env,
                          cwd=str(spec.REPO_ROOT))
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    result = json.loads(out.read_text())
    assert result["passed"] is True
    assert result["exact"]["positions_equal"] and result["exact"]["pos0_equal"]
    assert result["exact"]["text_embeddings_bitwise"] is True
    assert result["exact"]["top5_after_action"] == result["exact"]["top5_after_action_reference"]


# ----------------------------------------------------------------------------- fast: interpreter guard
def test_dump_script_refuses_an_incompatible_interpreter():
    if sys.version_info >= (3, 12):
        pytest.skip("this check documents the guard from /venv/oat (Python 3.10, transformers 5.2)")
    proc = subprocess.run([sys.executable, str(DUMP_SCRIPT), "--check-env"], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 2, proc.stderr
    assert "/venv/lerobot_ref" in proc.stderr
    assert "transformers" in proc.stderr and "45x" in proc.stderr


def test_dump_script_accepts_the_lerobot_ref_interpreter():
    python = spec.LEROBOT_VENV / "bin" / "python"
    if not python.is_file():
        pytest.skip(f"SKIPPED LOUDLY: throwaway venv missing at {spec.LEROBOT_VENV}")
    proc = subprocess.run([str(python), str(DUMP_SCRIPT), "--check-env"], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "environment OK" in proc.stdout


# ----------------------------------------------------------------------------- real reference (pi05 weights)
@pytest.mark.requires_pi05
def test_reference_schema_provenance_and_inputs(reference):
    shapes = spec.validate_reference(reference)
    meta = reference["metadata"]
    assert meta["lerobot_commit"] == spec.LEROBOT_COMMIT
    assert meta["transformers_version"].startswith(("5.4.", "5.5."))
    assert meta["pi05_revision"] == spec.PI05_REVISION and meta["pi05_sha256_verified"] is True
    assert meta["pi05_sha256"] == spec.PI05_SHA256 and meta["spm_sha256"] == spec.SPM_SHA256
    assert meta["weights_check"]["bitwise_equal"] is True and meta["weights_check"]["tensors_compared"] == 813
    assert meta["dtype"] == "float32" and meta["autocast"] is False
    assert all(value == "ieee" for key, value in meta["fp32_precision"].items()
               if key in ("global", "cuda_matmul", "cudnn_conv"))
    assert meta["attn_implementation"]["vlm_prefix"] == "eager" and meta["attn_implementation"]["expert"] == "eager"
    assert meta["camera_slots"] == ["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb",
                                    "observation.images.right_wrist_0_rgb"]
    assert meta["camera_slot_valid"] == [True, True, False]
    assert (meta["chunk_size"], meta["num_inference_steps"], meta["lang_max_len"]) == (50, 10, 64)
    assert (meta["n_image_tokens"], meta["prefix_len"]) == (768, 832)
    assert meta["lang_valid_lengths"] == [51, 35] and meta["prefix_valid_counts"] == [563, 547]
    assert shapes["prefix_keys"] == (18, 2, 1, 832, 256)
    assert shapes["vlm_logits"] == (2, 4, 257152)
    diagnostics = meta["diagnostics"]
    assert diagnostics["cache_unchanged_after_denoise"] is True
    assert diagnostics["sample_actions_loop_bitwise_equal_official"] is True
    assert diagnostics["reproducible_bitwise"] is True
    assert diagnostics["v_t_joint_vs_cached_max_abs_diff"] <= 1e-4 * max(1.0, diagnostics["v_t_abs_max"])
    # The saved inputs are exactly the regenerated deterministic inputs.
    inputs = spec.build_reference_inputs(seed=meta["seed"], max_len=meta["lang_max_len"])
    images, img_masks = spec.to_model_images(inputs["images01"], len(meta["camera_slots"]))
    assert torch.equal(reference["inputs"]["images"], images)
    assert torch.equal(reference["inputs"]["img_masks"], img_masks)
    for key in ("lang_tokens", "lang_masks", "noise", "x_t", "t"):
        assert torch.equal(reference["inputs"][key], inputs[key]), key
    # Sanity: the flow integrates away from the noise, and the two time values give different velocities.
    assert float((reference["sample_actions"] - reference["inputs"]["noise"]).abs().mean()) > 0.1
    assert not torch.allclose(reference["v_t"][0], reference["v_t"][1])


@pytest.mark.requires_pi05
def test_text_embeddings_are_checkpoint_rows_times_sqrt_width(reference):
    tokens = reference["inputs"]["lang_tokens"]
    ids = sorted(set(tokens.flatten().tolist()))
    rows = _read_rows(LM_HEAD, ids)
    lookup = torch.full((max(ids) + 1,), -1, dtype=torch.long)
    lookup[torch.tensor(ids)] = torch.arange(len(ids))
    expected = rows[lookup[tokens]] * torch.tensor(math.sqrt(spec.VLM_WIDTH), dtype=torch.float32)
    n_img = reference["metadata"]["n_image_tokens"]
    assert torch.equal(reference["prefix_embeddings"][:, n_img:], expected)
    assert reference["metadata"]["scaling"]["text_embed_scale"] == float(torch.tensor(math.sqrt(2048.0)))


@pytest.mark.requires_pi05
def test_layer0_cache_keys_are_post_rope(reference):
    weights = _read([VLM + "layers.0.input_layernorm.weight", VLM + "layers.0.self_attn.k_proj.weight",
                     VLM + "layers.0.self_attn.v_proj.weight"])
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, spec.HEAD_DIM, 2, dtype=torch.int64).float() / spec.HEAD_DIM))
    assert torch.allclose(reference["rope_inv_freq"], inv_freq, rtol=1e-6, atol=0.0)
    x = reference["prefix_embeddings"]
    batch, length, _ = x.shape
    normed = _gemma_rms(x, weights[VLM + "layers.0.input_layernorm.weight"])
    k = F.linear(normed, weights[VLM + "layers.0.self_attn.k_proj.weight"]).view(batch, length, 1, -1).transpose(1, 2)
    v = F.linear(normed, weights[VLM + "layers.0.self_attn.v_proj.weight"]).view(batch, length, 1, -1).transpose(1, 2)
    k_rot = _rope(k, reference["prefix_position_ids"], reference["rope_inv_freq"])
    cached_k, cached_v = reference["prefix_keys"][0], reference["prefix_values"][0]
    scale = float(cached_k.abs().max())
    # Layer 0 depends only on the embeddings, so pad positions match too.
    assert float((k_rot - cached_k).abs().max()) <= 1e-5 * scale
    assert float((k - cached_k).abs().max()) > 1e-2 * scale                  # pre-RoPE keys do not match
    assert float((v - cached_v).abs().max()) <= 1e-5 * float(cached_v.abs().max())


@pytest.mark.requires_pi05
def test_vlm_logits_come_from_final_hidden_and_tied_head(reference):
    logits, hidden = reference["vlm_logits"], reference["prefix_hidden"]
    index = reference["vlm_logits_prefix_index"]
    picked = hidden[torch.arange(hidden.shape[0])[:, None], index]                   # [B, 4, 2048]
    vocab_ids = sorted(set(torch.topk(logits, k=3, dim=-1).indices.flatten().tolist())
                       | {0, 1, 2, 108, 4022, 235248, 256000, 257151})
    rows = _read_rows(LM_HEAD, vocab_ids)
    expected = picked @ rows.T
    got = logits[..., torch.tensor(vocab_ids)]
    assert float((expected - got).abs().max()) <= 1e-4 * max(1.0, float(got.abs().max()))
    # The logits are aligned with the last valid prompt token: there, "Action: " is followed by pi05_base's
    # FAST pre-training prior, i.e. ids in openpi's FAST range 257023 - t, t in [0, 2048).
    assert reference["vlm_logits_input_ids"][:, -1].tolist() == [235248, 235248]       # the trailing '▁'
    top5 = torch.topk(logits[:, -1], k=5, dim=-1).indices
    assert bool(((top5 >= 257023 - 2047) & (top5 <= 257023)).all()), top5
    assert top5.tolist() == reference["metadata"]["top5_next_token_ids"]


@pytest.mark.requires_pi05
def test_time_conditioning_matches_checkpoint_time_mlp(reference):
    meta = reference["metadata"]
    t = reference["inputs"]["t"]
    fraction = torch.linspace(0.0, 1.0, spec.EXPERT_WIDTH // 2, dtype=torch.float64)
    period = meta["min_period"] * (meta["max_period"] / meta["min_period"]) ** fraction
    angle = t.double()[:, None] * (1.0 / period * 2 * math.pi)
    sincos = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1).float()
    assert float((sincos - reference["time_embedding"]).abs().max()) <= 1e-6
    weights = _read(["time_mlp_in.weight", "time_mlp_in.bias", "time_mlp_out.weight", "time_mlp_out.bias"])
    hidden = F.silu(F.linear(sincos, weights["time_mlp_in.weight"], weights["time_mlp_in.bias"]))
    cond = F.silu(F.linear(hidden, weights["time_mlp_out.weight"], weights["time_mlp_out.bias"]))
    assert float((cond - reference["adarms_cond"]).abs().max()) <= 1e-5 * max(1.0, float(cond.abs().max()))
