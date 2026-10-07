"""P2N-VLA M2: prompt format, prompt-state transforms, OAT-KI table and image preprocessing (CPU)."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from oat.model.vla.image_preprocess import ImagePreprocessor
from oat.model.vla.oat_ki import OATKITable
from oat.model.vla.paligemma_prompt import (
    DEFAULT_MAX_PROMPT_LEN, DEFAULT_SPM_PATH, INSTRUCTION_SOURCES, LIBERO10_INSTRUCTIONS,
    LIBERO10_INSTRUCTIONS_SOURCE, LIBERO_PROMPT_STATE, PaliGemmaTokenizer, PromptBuilder, PromptStateSpec,
    _LIBERO10_FALLBACK, check_constant_instruction, derive_libero10_instructions, discretize_state, format_prompt,
    uid_from_obs)
from oat.model.vla.state_transforms import (matrix_to_axis_angle, quat_xyzw_to_axis_angle, rot6d_to_axis_angle,
                                            rot6d_to_matrix)

LIBERO_ZARR = Path("/workspace/past_action/data/libero/libero10_N500.zarr")


# ---------------------------------------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def tokenizer():
    if not DEFAULT_SPM_PATH.is_file():
        pytest.skip(f"PaliGemma SentencePiece model missing at {DEFAULT_SPM_PATH}; "
                    "run scripts/fetch_p2n_vla_assets.py (M0)")
    return PaliGemmaTokenizer(str(DEFAULT_SPM_PATH))


def openpi_pi05_tokenize(sp_tokenizer, prompt, state, max_len):
    """Verbatim semantics of openpi PaligemmaTokenizer.tokenize (pi05 branch, state given)."""
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
    tokens = sp_tokenizer.encode(full_prompt, add_bos=True)
    assert len(tokens) <= max_len
    mask = [True] * len(tokens) + [False] * (max_len - len(tokens))
    tokens = tokens + [0] * (max_len - len(tokens))
    return np.asarray(tokens), np.asarray(mask)


def rodrigues(axis_angle):
    """float64 axis-angle [..., 3] -> rotation matrix [..., 3, 3] (exponential map)."""
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe = np.where(theta > 0, theta, 1.0)
    k = axis_angle / safe
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    cross = np.stack([np.stack([zero, -kz, ky], -1), np.stack([kz, zero, -kx], -1),
                      np.stack([-ky, kx, zero], -1)], -2)
    theta = theta[..., None]
    eye = np.broadcast_to(np.eye(3), cross.shape)
    return eye + np.sin(theta) * cross + (1 - np.cos(theta)) * (cross @ cross)


def to_rot6d(rotation, layout):
    if layout == "rows":
        return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)
    return np.swapaxes(rotation[..., :, :2], -1, -2).reshape(*rotation.shape[:-2], 6)


def random_axis_angles(rng, n, max_angle=math.pi * 0.999):
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    return axis * rng.uniform(0.0, max_angle, size=(n, 1))


# ---------------------------------------------------------------------------------------------- tokenizer
def test_tokenizer_special_ids_bos_and_no_eos(tokenizer):
    assert (tokenizer.pad_id, tokenizer.eos_id, tokenizer.bos_id) == (0, 1, 2)
    assert tokenizer.vocab_size == 257152
    assert [tokenizer.id_to_piece(i) for i in range(3)] == ["<pad>", "<eos>", "<bos>"]
    ids = tokenizer.encode("Task: x, State: 12 7;\nAction: ")
    assert ids[0] == 2 and 1 not in ids and ids.count(2) == 1
    assert tokenizer.encode("Task: x", add_bos=False)[0] != 2
    # The PaliGemma SentencePiece model splits digits; the prompt ends with the "Action: " space piece.
    assert [tokenizer.id_to_piece(i) for i in ids[-10:]] == ["▁", "1", "2", "▁", "7", ";", "\n", "Action", ":", "▁"]


def test_format_prompt_is_pi05_text():
    assert format_prompt("  put_the\nbowl  ", [0, 255, 7]) == "Task: put the bowl, State: 0 255 7;\nAction: "
    assert format_prompt("Pick Up", []) == "Task: Pick Up, State: ;\nAction: "  # no lowercasing (pi05)
    for bad in ([256], [-1], [1.0], [True]):
        with pytest.raises(ValueError):
            format_prompt("x", bad)


def test_prompt_ids_match_openpi_pi05_tokenizer(tokenizer):
    rng = np.random.default_rng(0)
    instructions = ["put both moka pots on the stove", "  pick_up the\nbook  ", LIBERO10_INSTRUCTIONS[34]]
    for instruction in instructions:
        builder = PromptBuilder(tokenizer, instruction, max_len=DEFAULT_MAX_PROMPT_LEN)
        states = rng.uniform(-1, 1, size=(5, 8)).astype(np.float32)
        states[0] = [-1, 1, 0, -0.9921875, 0.9921875, 1 / 128, -1 / 128, 0.5]  # exact bin edges
        bins = discretize_state(torch.from_numpy(states))
        ids, valid = builder.build(None, bins)
        for row, state in enumerate(states):
            ref_ids, ref_mask = openpi_pi05_tokenize(tokenizer, instruction, state.astype(np.float64),
                                                     DEFAULT_MAX_PROMPT_LEN)
            np.testing.assert_array_equal(ids[row].numpy(), ref_ids)
            np.testing.assert_array_equal(valid[row].numpy(), ref_mask)


# ---------------------------------------------------------------------------------------------- discretization
def test_discretize_state_matches_numpy_digitize():
    rng = np.random.default_rng(1)
    edges = np.linspace(-1, 1, 257)
    values = np.concatenate([rng.uniform(-1.5, 1.5, 5000), edges, edges + 1e-7, edges - 1e-7,
                             [-1.0, 1.0, -1e9, 1e9, 0.0, -0.0]])
    for dtype in (np.float64, np.float32):
        x = values.astype(dtype)
        expected = np.digitize(np.clip(x.astype(np.float64), -1, 1), np.linspace(-1, 1, 257)[:-1]) - 1
        got = discretize_state(torch.from_numpy(x).reshape(-1, 1))
        assert got.dtype == torch.int64 and got.shape == (len(values), 1)
        np.testing.assert_array_equal(got.squeeze(1).numpy(), expected)
    assert discretize_state(torch.tensor([[-1.0, 1.0]])).tolist() == [[0, 255]]
    bf16 = torch.linspace(-1.2, 1.2, 101).to(torch.bfloat16)[None]
    expected = np.digitize(np.clip(bf16.float().numpy(), -1, 1), np.linspace(-1, 1, 257)[:-1]) - 1
    np.testing.assert_array_equal(discretize_state(bf16).numpy(), expected)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert discretize_state(torch.zeros(2, 3)).eq(128).all()
    for bad in (torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]])):
        with pytest.raises(ValueError):
            discretize_state(bad)
    with pytest.raises(TypeError):
        discretize_state(torch.zeros(2, 3, dtype=torch.long))


# ---------------------------------------------------------------------------------------------- prompt builder
def test_all_libero10_prompts_fit_with_worst_case_bins(tokenizer):
    builder = PromptBuilder(tokenizer, "libero10")
    assert builder.max_len == DEFAULT_MAX_PROMPT_LEN == 96
    lengths = builder.worst_case_lengths(8)
    assert sorted(lengths) == list(range(30, 40))
    assert max(lengths.values()) <= 96, lengths
    uids = torch.arange(30, 40)
    ids, valid = builder.build(uids, torch.full((10, 8), 255))
    assert valid.sum(1).tolist() == [lengths[u] for u in range(30, 40)]
    # Three-digit bins are the worst case: random bins never produce a longer prompt.
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):
        bins = torch.randint(0, 256, (10, 8), generator=generator)
        _, valid = builder.build(uids, bins)
        assert (valid.sum(1) <= torch.tensor([lengths[u] for u in range(30, 40)])).all()


def test_prompt_builder_layout_padding_and_mask(tokenizer):
    builder = PromptBuilder(tokenizer, "libero10")
    uids = torch.tensor([38, 34, 30])
    bins = torch.tensor([[0] * 8, [255] * 8, list(range(8))])
    ids, valid = builder.build(uids, bins)
    assert ids.shape == valid.shape == (3, 96)
    assert ids.dtype == torch.int64 and valid.dtype == torch.bool
    assert ids.device.type == valid.device.type == "cpu"
    for row, (uid, row_bins) in enumerate(zip(uids.tolist(), bins.tolist())):
        expected = tokenizer.encode(format_prompt(LIBERO10_INSTRUCTIONS[uid], row_bins), add_bos=True)
        n = len(expected)
        assert valid[row, :n].all() and not valid[row, n:].any()  # right padding
        assert ids[row, :n].tolist() == expected
        assert (ids[row, n:] == 0).all()
        assert ids[row, 0] == 2 and (ids[row, :n] != 1).all()  # BOS, never EOS
        assert tokenizer.decode(ids[row, 1:n].tolist()).startswith(f"Task: {LIBERO10_INSTRUCTIONS[uid]}, State: ")
    # Rollout-style float uids [B,To,1] go through uid_from_obs.
    obs = {"task_uid": torch.tensor([[[38.0]], [[34.0]], [[30.0]]])}
    torch.testing.assert_close(builder.build(uid_from_obs(obs), bins)[0], ids)


def test_prompt_builder_raises_instead_of_truncating(tokenizer):
    builder = PromptBuilder(tokenizer, "libero10", max_len=20)
    with pytest.raises(ValueError, match="max_len"):
        builder.build(torch.tensor([30]), torch.zeros(1, 8, dtype=torch.long))


def test_instruction_sources(tokenizer):
    bins = torch.zeros(2, 8, dtype=torch.long)
    constant = PromptBuilder(tokenizer, "remove the nut and the washer from the rod and place them on the tray")
    assert not constant.requires_uids
    ids_none, _ = constant.build(None, bins)
    ids_uids, _ = constant.build(torch.tensor([0, 0]), bins)  # uids are ignored
    torch.testing.assert_close(ids_none, ids_uids)
    libero = PromptBuilder(tokenizer, "libero10")
    assert libero.requires_uids
    with pytest.raises(ValueError, match="uids"):
        libero.build(None, bins)
    with pytest.raises(ValueError, match="Unknown task uid"):
        libero.build(torch.tensor([30, 29]), bins)
    with pytest.raises(ValueError, match="shape"):
        libero.build(torch.tensor([30, 31, 32]), bins)
    with pytest.raises(TypeError):
        libero.build(torch.tensor([30.0, 31.0]), bins)
    mapping = PromptBuilder(tokenizer, OmegaConf.create({0: "pick up the cube", "7": "open the drawer"}))
    assert mapping.instruction(0) == "pick up the cube" and mapping.instruction(7) == "open the drawer"
    ids, valid = mapping.build(torch.tensor([7, 0]), bins)
    assert tokenizer.decode(ids[0, 1:valid[0].sum()].tolist()).startswith("Task: open the drawer, State: ")
    with pytest.raises(ValueError, match="Unknown task uid"):
        mapping.build(torch.tensor([1, 0]), bins)
    for bad in ({}, {"x": "a"}, {0: ""}, "   ", 3, ["a"]):
        with pytest.raises((ValueError, TypeError)):
            PromptBuilder(tokenizer, bad)
    with pytest.raises(ValueError):
        PromptBuilder(tokenizer, "libero10", max_len=1)
    with pytest.raises(ValueError):
        libero.build(torch.tensor([30, 31]), torch.full((2, 8), 256))
    with pytest.raises(TypeError):
        libero.build(torch.tensor([30, 31]), torch.zeros(2, 8))


def test_misspelled_instruction_sources_are_rejected_not_used_as_prompts(tokenizer):
    """Any string other than a source name becomes every sample's instruction, so near-miss source names and
    single words must raise instead of silently training all tasks on one meaningless prompt."""
    for bad in ("LIBERO10", "Libero10", "libero_10", "libero-10", " libero10 ", "libero 10", "libero90",
                "libero_spatial", "libero10\n", "pick", "  stack_  "):
        with pytest.raises(ValueError, match="libero10|two words"):
            PromptBuilder(tokenizer, bad)
        with pytest.raises(ValueError):
            check_constant_instruction(bad)
    assert PromptBuilder(tokenizer, "libero10").requires_uids  # the exact source name still selects the table
    # openpi cleaning turns underscored task names into words, so they remain valid constant instructions.
    underscored = PromptBuilder(tokenizer, "pick_up_the_cube")
    assert not underscored.requires_uids and underscored.instruction(None) == "pick_up_the_cube"
    ids, valid = underscored.build(None, torch.zeros(1, 8, dtype=torch.long))
    assert tokenizer.decode(ids[0, 1:valid[0].sum()].tolist()).startswith("Task: pick up the cube, State: ")
    assert INSTRUCTION_SOURCES == ("libero10",)


def test_uid_from_obs_training_and_rollout_dtypes():
    expected = torch.tensor([30, 39])
    cases = [
        torch.tensor([[[30]], [[39]]], dtype=torch.int64),                 # training [B, To=1, 1]
        torch.tensor([[[31], [30]], [[38], [39]]], dtype=torch.int64),     # [B, To=2, 1]: last frame
        torch.tensor([[[30.0]], [[39.0]]], dtype=torch.float32),           # rollout float
        torch.tensor([[30.0004], [38.9996]], dtype=torch.float32),         # [B, 1], rounded
        torch.tensor([30, 39], dtype=torch.uint8),                         # [B]
        torch.tensor([[30.0], [39.0]], dtype=torch.bfloat16),
    ]
    for value in cases:
        uid = uid_from_obs({"task_uid": value})
        assert uid.dtype == torch.int64 and uid.shape == (2,)
        torch.testing.assert_close(uid, expected)
    for bad, error in ((torch.tensor([[30.4]]), ValueError), (torch.tensor([[float("nan")]]), ValueError),
                       (torch.tensor([True]), TypeError), (torch.zeros(2, 1, 2), ValueError),
                       (torch.zeros(1, 1, 1, 1), ValueError)):
        with pytest.raises(error):
            uid_from_obs({"task_uid": bad})
    with pytest.raises(KeyError):
        uid_from_obs({})


# ---------------------------------------------------------------------------------------------- LIBERO language
def test_libero10_instruction_table_matches_libero():
    assert sorted(LIBERO10_INSTRUCTIONS) == list(range(30, 40))
    assert LIBERO10_INSTRUCTIONS == _LIBERO10_FALLBACK
    try:
        from libero.libero import benchmark
        from libero.libero.benchmark.libero_suite_task_map import libero_task_map
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"libero is not importable here ({exc!r}); the fallback table is in use")
    assert LIBERO10_INSTRUCTIONS_SOURCE == "libero"
    assert derive_libero10_instructions() == _LIBERO10_FALLBACK
    # Independent re-derivation with oat/env/libero/env.py's uid rule.
    uid, table = 0, {}
    for suite_name, task_names in libero_task_map.items():
        for local_id, _ in enumerate(task_names):
            if suite_name == "libero_10":
                table[uid] = benchmark.get_benchmark_dict()[suite_name]().get_task(local_id).language
            uid += 1
    assert table == LIBERO10_INSTRUCTIONS
    # uids 30 and 37 share their first 31 characters: the zarr <U31 'prompt' field cannot tell them apart.
    assert LIBERO10_INSTRUCTIONS[30][:31] == LIBERO10_INSTRUCTIONS[37][:31]


# ---------------------------------------------------------------------------------------------- state transforms
def test_quat_to_axis_angle_matches_robosuite():
    robosuite_t = pytest.importorskip("robosuite.utils.transform_utils")
    rng = np.random.default_rng(2)
    quats = rng.normal(size=(2000, 4))
    quats[:1500] /= np.linalg.norm(quats[:1500], axis=-1, keepdims=True)  # also a few non-unit inputs
    edge = np.array([[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 1 + 1e-7], [0.3, 0, 0, -1.2], [0, 0, 0, 0],
                     [1, 0, 0, 0], [0.999, 0.0, -0.0447, -0.0019], [1e-9, 0, 0, 1 - 5e-19]], dtype=np.float64)
    quats = np.concatenate([quats, edge])
    expected = np.stack([robosuite_t.quat2axisangle(q.copy()) for q in quats])
    got = quat_xyzw_to_axis_angle(torch.from_numpy(quats))
    assert got.dtype == torch.float64
    np.testing.assert_allclose(got.numpy(), expected, rtol=0, atol=1e-12)
    q32 = quats[:1500].astype(np.float32)
    expected32 = np.stack([robosuite_t.quat2axisangle(q.astype(np.float64)) for q in q32])
    got32 = quat_xyzw_to_axis_angle(torch.from_numpy(q32))
    assert got32.dtype == torch.float32
    np.testing.assert_allclose(got32.numpy(), expected32, rtol=0, atol=2e-6)
    batched = quat_xyzw_to_axis_angle(torch.from_numpy(quats[:12]).reshape(3, 4, 4))
    np.testing.assert_allclose(batched.reshape(12, 3).numpy(), expected[:12], atol=1e-12)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert quat_xyzw_to_axis_angle(torch.from_numpy(q32)).dtype == torch.float32
    for bad in (torch.zeros(3, 3), torch.zeros(2, 4, dtype=torch.long), torch.full((1, 4), float("nan"))):
        with pytest.raises((ValueError, TypeError)):
            quat_xyzw_to_axis_angle(bad)


@pytest.mark.requires_data
def test_quat_to_axis_angle_matches_robosuite_on_libero_quaternions():
    robosuite_t = pytest.importorskip("robosuite.utils.transform_utils")
    if not LIBERO_ZARR.exists():
        pytest.skip(f"LIBERO zarr missing: {LIBERO_ZARR}")
    import zarr
    quats = zarr.open(str(LIBERO_ZARR), "r")["data"]["robot0_eef_quat"][:]
    rows = np.random.default_rng(3).choice(len(quats), size=5000, replace=False)
    sample = quats[rows]
    assert sample.dtype == np.float32
    expected = np.stack([robosuite_t.quat2axisangle(q.astype(np.float64)) for q in sample])
    got = quat_xyzw_to_axis_angle(torch.from_numpy(sample)).numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=2e-6)


@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_rot6d_round_trips_and_matches_history_encoder(layout):
    from oat.policy.past2next_state_history_gate_real_robot import Rotation6DStateActionHistoryEncoder
    rng = np.random.default_rng(4)
    near_pi = rng.normal(size=(50, 3))
    near_pi *= (math.pi - 1e-4) / np.linalg.norm(near_pi, axis=-1, keepdims=True)
    axis_angles = np.concatenate([random_axis_angles(rng, 500), random_axis_angles(rng, 50, 1e-6), near_pi,
                                  np.zeros((1, 3))])
    rotations = rodrigues(axis_angles)
    six = to_rot6d(rotations, layout)
    torch.testing.assert_close(rot6d_to_matrix(torch.from_numpy(six), layout), torch.from_numpy(rotations),
                               rtol=0, atol=1e-12)
    back = rot6d_to_axis_angle(torch.from_numpy(six), layout)
    assert back.dtype == torch.float64
    np.testing.assert_allclose(back.numpy(), axis_angles, rtol=0, atol=1e-9)
    # Same Gram-Schmidt convention as the real-robot history encoder, also for noisy (non-orthonormal) 6D.
    noisy = six + rng.normal(scale=0.05, size=six.shape)
    encoder = object.__new__(Rotation6DStateActionHistoryEncoder)
    encoder.rotation_6d_layout = layout
    torch.testing.assert_close(rot6d_to_matrix(torch.from_numpy(noisy), layout),
                               encoder._matrix_from_6d(torch.from_numpy(noisy)), rtol=0, atol=1e-12)
    log = rot6d_to_axis_angle(torch.from_numpy(noisy), layout).numpy()
    np.testing.assert_allclose(rodrigues(log), rot6d_to_matrix(torch.from_numpy(noisy), layout).numpy(),
                               atol=1e-9)
    assert rot6d_to_axis_angle(torch.from_numpy(six).float(), layout).dtype == torch.float32


def test_rotation_log_map_near_pi_and_robosuite_consistency():
    robosuite_t = pytest.importorskip("robosuite.utils.transform_utils")
    rng = np.random.default_rng(5)
    for angle in (math.pi, math.pi - 1e-7, math.pi - 1e-3):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = rodrigues(axis * angle)
        got = matrix_to_axis_angle(torch.from_numpy(rotation)).numpy()
        assert abs(np.linalg.norm(got) - angle) < 1e-7
        np.testing.assert_allclose(rodrigues(got), rotation, atol=1e-9)
    # For angles below pi the canonical log map equals robosuite mat2quat (w >= 0) -> quat2axisangle.
    axis_angles = random_axis_angles(rng, 200, 3.0)
    rotations = rodrigues(axis_angles)
    expected = np.stack([robosuite_t.quat2axisangle(robosuite_t.mat2quat(r).astype(np.float64)) for r in rotations])
    np.testing.assert_allclose(matrix_to_axis_angle(torch.from_numpy(rotations)).numpy(), expected, atol=5e-6)
    with pytest.raises(ValueError, match="degenerate"):
        rot6d_to_axis_angle(torch.zeros(1, 6), "rows")
    with pytest.raises(ValueError, match="collinear"):
        rot6d_to_axis_angle(torch.tensor([[1.0, 0, 0, 2.0, 0, 0]]), "columns")
    with pytest.raises(ValueError):
        rot6d_to_axis_angle(torch.zeros(1, 6), "diagonal")


def test_prompt_state_spec_extracts_last_frame_in_key_order():
    spec = PromptStateSpec(list(LIBERO_PROMPT_STATE["keys"]), dict(LIBERO_PROMPT_STATE["transforms"]))
    assert spec.keys == ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    shape_meta = {"robot0_eef_pos": [3], "robot0_eef_quat": [4], "robot0_gripper_qpos": [2], "task_uid": [1]}
    assert spec.output_dim(shape_meta) == 8
    generator = torch.Generator().manual_seed(0)
    pos = torch.randn(4, 2, 3, generator=generator)
    quat = F.normalize(torch.randn(4, 2, 4, generator=generator), dim=-1)
    grip = torch.randn(4, 2, generator=generator)  # [B, D] input: used as-is
    state = spec.extract({"robot0_eef_pos": pos, "robot0_eef_quat": quat, "robot0_gripper_qpos": grip,
                          "agentview_rgb": torch.zeros(4, 2, 8, 8, 3, dtype=torch.uint8)})
    assert state.shape == (4, 8) and state.dtype == torch.float32
    expected = torch.cat((pos[:, -1], quat_xyzw_to_axis_angle(quat[:, -1]), grip), dim=-1)
    torch.testing.assert_close(state, expected, rtol=0, atol=1e-6)
    # Earlier frames never matter.
    pos2, quat2 = pos.clone(), quat.clone()
    pos2[:, 0], quat2[:, 0] = 100.0, 0.5
    torch.testing.assert_close(spec.extract({"robot0_eef_pos": pos2, "robot0_eef_quat": quat2,
                                             "robot0_gripper_qpos": grip}), state)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert spec.extract({"robot0_eef_pos": pos, "robot0_eef_quat": quat,
                             "robot0_gripper_qpos": grip}).dtype == torch.float32
    # Config round trip (OmegaConf input, implicit identity transforms).
    implicit = PromptStateSpec.from_config(OmegaConf.create(
        {"keys": spec.keys, "transforms": {"robot0_eef_quat": "quat_to_axis_angle"}}))
    assert implicit == spec and implicit.to_dict() == LIBERO_PROMPT_STATE
    real_robot = PromptStateSpec(["robot0_eef_pos", "robot0_eef_rot6d", "robot0_gripper_qpos"],
                                 {"robot0_eef_rot6d": "rot6d_rows_to_axis_angle"})
    assert real_robot.output_dim({"robot0_eef_pos": [3], "robot0_eef_rot6d": [6], "robot0_gripper_qpos": [1]}) == 7


def test_prompt_state_spec_validation():
    good = dict(LIBERO_PROMPT_STATE)
    obs = {"robot0_eef_pos": torch.zeros(2, 1, 3), "robot0_eef_quat": torch.tensor([[[0.0, 0, 0, 1]]] * 2),
           "robot0_gripper_qpos": torch.zeros(2, 1, 2)}
    spec = PromptStateSpec.from_config(good)
    torch.testing.assert_close(spec.extract(obs), torch.zeros(2, 8))
    for keys, transforms in (([], {}), (["a", "a"], {}), (["a"], {"b": "identity"}), (["a"], {"a": "nope"}),
                             ("abc", {})):
        with pytest.raises((ValueError, TypeError)):
            PromptStateSpec(keys, transforms)
    with pytest.raises(ValueError):
        PromptStateSpec.from_config({"keys": ["a"], "extra": 1})
    bad_cases = [
        {**obs, "robot0_eef_quat": torch.zeros(2, 1, 3)},                  # wrong quaternion width
        {**obs, "robot0_eef_pos": torch.zeros(3, 1, 3)},                   # batch mismatch
        {**obs, "robot0_eef_pos": torch.zeros(2, 1, 1, 3)},                # wrong rank
        {**obs, "robot0_eef_pos": torch.full((2, 1, 3), float("nan"))},    # non-finite
        {**obs, "robot0_eef_pos": torch.zeros(2, 1, 3, dtype=torch.bool)},
    ]
    for case in bad_cases:
        with pytest.raises((ValueError, TypeError)):
            spec.extract(case)
    with pytest.raises(KeyError):
        spec.extract({k: v for k, v in obs.items() if k != "robot0_gripper_qpos"})


# ---------------------------------------------------------------------------------------------- KI table
def test_ki_table_copies_raw_paligemma_rows_and_bos():
    vocab, width = 257152, 8
    embed = torch.randn(vocab, width).to(torch.bfloat16)
    table = OATKITable(width)
    assert table.rows.shape == (5001, width) and table.rows.dtype == torch.float32
    assert table.rows.requires_grad and isinstance(table.rows, torch.nn.Parameter)
    table.init_from_embedding(embed)
    for t in (0, 1, 2500, 4999):
        torch.testing.assert_close(table.rows[t], embed[255999 - t].float(), rtol=0, atol=0)
    assert table.source_indices(vocab)[[0, -1]].tolist() == [255999, 251000]
    torch.testing.assert_close(table.rows[5000], embed[2].float(), rtol=0, atol=0)
    assert table.rows.requires_grad and table.rows.grad is None
    small = OATKITable(4, n_codes=10, skip=3)
    small_embed = torch.arange(40 * 4, dtype=torch.float32).reshape(40, 4)
    small.init_from_embedding(small_embed)
    torch.testing.assert_close(small.rows[:10], small_embed[[36 - t for t in range(10)]])  # V - 1 - skip - t
    torch.testing.assert_close(small.rows[10], small_embed[2])
    with pytest.raises(ValueError):
        OATKITable(4, n_codes=10, skip=30).init_from_embedding(small_embed)  # would reach the special rows
    with pytest.raises(ValueError):
        small.init_from_embedding(torch.zeros(40, 5))


def test_ki_rows_keep_paligemma_loc_and_seg_tokens_intact(tokenizer):
    index = OATKITable(2048).source_indices(tokenizer.vocab_size)
    assert (index.max().item(), index.min().item()) == (255999, 251000)
    assert tokenizer.id_to_piece(255999) == "<start_of_image>" and tokenizer.id_to_piece(256000) == "<loc0000>"
    pieces = [tokenizer.id_to_piece(i) for i in index.tolist()]
    assert not any(p.startswith(("<loc", "<seg")) for p in pieces)
    assert tokenizer.id_to_piece(2) == "<bos>"  # KI_BOS source row


def test_ki_block_embedding_and_tied_fp32_logits():
    torch.manual_seed(0)
    width = 16
    table = OATKITable(width)
    targets = torch.randint(0, 5000, (3, 8))
    ids = table.block_ids(targets)
    assert ids[:, 0].eq(5000).all() and torch.equal(ids[:, 1:], targets[:, :7])
    embeds = table.embed_block(targets)
    assert embeds.shape == (3, 8, width) and embeds.dtype == torch.float32
    torch.testing.assert_close(embeds, table.rows.detach()[ids] * math.sqrt(width), rtol=0, atol=0)
    hidden = torch.randn(3, 8, width).to(torch.bfloat16)
    logits = table.logits(hidden)
    assert logits.shape == (3, 8, 5000) and logits.dtype == torch.float32
    torch.testing.assert_close(logits, hidden.float() @ table.rows.detach()[:5000].T)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_logits = table.logits(hidden)
        autocast_embeds = table.embed_block(targets)
    assert autocast_logits.dtype == torch.float32 and autocast_embeds.dtype == torch.float32
    torch.testing.assert_close(autocast_logits, logits)
    loss = F.cross_entropy(table.logits(embeds).flatten(0, 1), targets.flatten())
    loss.backward()
    used = torch.unique(torch.cat((ids.flatten(), targets.flatten())))
    assert table.rows.grad[used].abs().sum(-1).gt(0).all()
    for bad in (torch.full((2, 8), 5000), torch.full((2, 8), -1), torch.zeros(2, 8), torch.zeros(8, dtype=torch.long)):
        with pytest.raises((ValueError, TypeError)):
            table.embed_block(bad)
    with pytest.raises(ValueError):
        table.logits(torch.zeros(2, 8, width + 1))


def test_ki_block_is_causal_self_contained_and_pad_invariant():
    """KI logits at block position k see the prompt and z_<k only, and never depend on the padding length."""
    from oat.model.vla.gemma_joint import GemmaJoint
    from oat.model.vla.layout import build_prefix_layout
    from oat.model.vla.specs import TINY_EXPERT, TINY_VLM

    torch.manual_seed(0)
    joint = GemmaJoint(TINY_VLM, TINY_EXPERT, "segment", frozen_dtype=torch.float32,
                       activation_checkpointing=False).eval()
    table = OATKITable(TINY_VLM.width)
    table.init_from_embedding(joint.vlm.embed_tokens.weight)
    batch, n_img = 2, 4
    images = torch.randn(batch, n_img, TINY_VLM.width)
    prompts = [torch.randint(3, 250000, (11,)), torch.randint(3, 250000, (17,))]
    prompts[0][0] = prompts[1][0] = 2

    def ki_logits(targets, prompt_len, prompts=prompts):
        ids = torch.zeros(batch, prompt_len, dtype=torch.long)
        valid = torch.zeros(batch, prompt_len, dtype=torch.bool)
        for row, prompt in enumerate(prompts):
            ids[row, :len(prompt)], valid[row, :len(prompt)] = prompt, True
        embeds = torch.cat((images, joint.embed_text(ids), table.embed_block(targets)), dim=1)
        layout = build_prefix_layout(torch.ones(batch, n_img, dtype=torch.bool), valid, 8)
        with torch.no_grad():
            return table.logits(joint.prefix_forward(embeds, layout).hidden[:, -8:])

    targets = torch.randint(0, 5000, (batch, 8))
    base = ki_logits(targets, 24)
    torch.testing.assert_close(ki_logits(targets, 40), base, rtol=0, atol=2e-5)  # padding length
    for k in range(8):
        perturbed = targets.clone()
        perturbed[:, k:] = (perturbed[:, k:] + 1 + torch.arange(8 - k)) % 5000
        out = ki_logits(perturbed, 24)
        torch.testing.assert_close(out[:, :k + 1], base[:, :k + 1], rtol=0, atol=1e-6)
        if k < 7:
            assert not torch.allclose(out[:, k + 1:], base[:, k + 1:])  # positive control
    # z0 (from KI_BOS) is predicted from the prompt, so it is supervised.
    other = [prompts[0].clone(), prompts[1].clone()]
    other[0][5] += 1
    assert not torch.allclose(ki_logits(targets, 24, other)[0, 0], base[0, 0])


# ---------------------------------------------------------------------------------------------- images
def _uint8_images(batch, steps=1, size=128, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, steps, size, size, 3), generator=generator, dtype=torch.uint8)


def _reference_eval(image_bhwc, size=224):
    x = image_bhwc.float().permute(0, 3, 1, 2) / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return x.clamp(0, 1) * 2 - 1


def test_eval_path_is_a_plain_resize_upright_and_deterministic():
    pre = ImagePreprocessor(["agentview_rgb", "robot0_eye_in_hand_rgb"], ["robot0_eye_in_hand_rgb"], 224)
    obs = {"agentview_rgb": _uint8_images(3, 2, seed=0), "robot0_eye_in_hand_rgb": _uint8_images(3, 2, seed=1)}
    state = torch.get_rng_state()
    out = pre(obs, train_aug=False)
    assert torch.equal(state, torch.get_rng_state())  # no randomness consumed
    assert out.shape == (3, 2, 3, 224, 224) and out.dtype == torch.float32
    assert out.min() >= -1 and out.max() <= 1
    torch.testing.assert_close(out[:, 0], _reference_eval(obs["agentview_rgb"][:, -1]), rtol=0, atol=1e-6)
    torch.testing.assert_close(out[:, 1], _reference_eval(obs["robot0_eye_in_hand_rgb"][:, -1]), rtol=0, atol=1e-6)
    torch.testing.assert_close(pre(obs, train_aug=False), out, rtol=0, atol=0)
    # Last frame only; [B, H, W, 3] input and float 0..255 input give the same result.
    changed = {k: v.clone() for k, v in obs.items()}
    changed["agentview_rgb"][:, 0] = 0
    torch.testing.assert_close(pre(changed, train_aug=False), out, rtol=0, atol=0)
    last = {k: v[:, -1].float() for k, v in obs.items()}
    torch.testing.assert_close(pre(last, train_aug=False), out, rtol=0, atol=0)
    # A marker in the top-left corner stays top-left: no flip, no mirror.
    marker = torch.zeros(1, 128, 128, 3, dtype=torch.uint8)
    marker[:, :32, :16] = 255
    image = pre({"agentview_rgb": marker, "robot0_eye_in_hand_rgb": marker}, train_aug=False)[0, 0, 0]
    bright = (image > 0).nonzero()
    assert bright[:, 0].max() < 60 and bright[:, 1].max() < 32
    assert image[:50, :24].min() > 0.99 and image[60:].max() < -0.99 and image[:, 32:].max() < -0.99


def test_train_aug_is_seeded_bounded_and_random():
    pre = ImagePreprocessor(["agentview_rgb", "robot0_eye_in_hand_rgb"], ["robot0_eye_in_hand_rgb"], 224)
    obs = {"agentview_rgb": _uint8_images(4, seed=2), "robot0_eye_in_hand_rgb": _uint8_images(4, seed=3)}
    a = pre(obs, train_aug=True, generator=torch.Generator().manual_seed(7))
    b = pre(obs, train_aug=True, generator=torch.Generator().manual_seed(7))
    c = pre(obs, train_aug=True, generator=torch.Generator().manual_seed(8))
    assert a.shape == (4, 2, 3, 224, 224) and a.dtype == torch.float32
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(a, c)
    assert a.min() >= -1 and a.max() <= 1
    assert not torch.allclose(a, pre(obs, train_aug=False))
    params = pre.sample_params(4, torch.device("cpu"), torch.Generator().manual_seed(7))
    assert set(params) == {"crop_top", "crop_left", "angle_deg", "brightness", "contrast", "saturation"}
    assert all(v.shape == (2, 4) for v in params.values())
    assert params["angle_deg"].abs().max() <= 5 and (params["brightness"] - 1).abs().max() <= 0.3
    assert (params["contrast"] - 1).abs().max() <= 0.4 and (params["saturation"] - 1).abs().max() <= 0.5
    assert not torch.allclose(params["brightness"][0], params["brightness"][1])  # independent per camera
    with pytest.raises(TypeError):
        pre(obs, train_aug=1)


def test_wrist_cameras_get_colour_jitter_only():
    obs = {"agentview_rgb": _uint8_images(3, seed=4), "robot0_eye_in_hand_rgb": _uint8_images(3, seed=5)}
    no_colour = ImagePreprocessor(["agentview_rgb", "robot0_eye_in_hand_rgb"], ["robot0_eye_in_hand_rgb"], 224,
                                  brightness=0.0, contrast=0.0, saturation=0.0)
    reference = no_colour(obs, train_aug=False)
    augmented = no_colour(obs, train_aug=True, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(augmented[:, 1], reference[:, 1], rtol=0, atol=1e-6)  # wrist untouched
    assert not torch.allclose(augmented[:, 0], reference[:, 0], atol=1e-3)            # crop + rotate
    identity = ImagePreprocessor(["agentview_rgb"], [], 224, crop_scale=1.0, max_rotation_deg=0.0,
                                 brightness=0.0, contrast=0.0, saturation=0.0)
    torch.testing.assert_close(identity(obs, train_aug=True), identity(obs, train_aug=False), rtol=0, atol=1e-6)


def test_augmentation_ops_match_openpi():
    pre = ImagePreprocessor(["agentview_rgb"], [], 224)
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(3, 3, 128, 128, generator=generator)
    assert pre.crop_size(128, 128) == (121, 121) and pre.crop_size(224, 224) == (212, 212)
    top_frac, left_frac = torch.tensor([0.0, 0.5, 0.999]), torch.tensor([0.999, 0.25, 0.0])
    cropped = pre.random_crop_resize(image, top_frac, left_frac)
    for row, (top, left) in enumerate([(0, 7), (4, 2), (7, 0)]):
        expected = F.interpolate(image[row:row + 1, :, top:top + 121, left:left + 121], size=(128, 128),
                                 mode="bilinear", align_corners=False)
        torch.testing.assert_close(cropped[row:row + 1], expected, rtol=0, atol=1e-6)
    # Rotation: 0 deg is the identity; +90 deg on a square image is a proper rotation (rot90), never a mirror.
    torch.testing.assert_close(pre.rotate(image, torch.zeros(3)), image, rtol=0, atol=1e-5)
    torch.testing.assert_close(pre.rotate(image, torch.full((3,), 90.0)), torch.rot90(image, 1, dims=(-2, -1)),
                               rtol=0, atol=1e-4)
    # Colour ops: openpi preprocessing_pytorch.py arithmetic ([B,H,W,C] layout) with per-sample factors.
    b, c, s = torch.tensor([0.7, 1.0, 1.3]), torch.tensor([0.6, 1.2, 1.4]), torch.tensor([0.5, 1.0, 1.5])
    got = pre.color_jitter(image, b, c, s)
    for row in range(3):
        x = image[row:row + 1].permute(0, 2, 3, 1)
        x = x * b[row]
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        x = (x - mean) * c[row] + mean
        gray = x.mean(dim=-1, keepdim=True)
        x = torch.clamp(gray + (x - gray) * s[row], 0, 1)
        torch.testing.assert_close(got[row:row + 1], x.permute(0, 3, 1, 2), rtol=0, atol=1e-6)


def test_image_preprocessor_validation():
    with pytest.raises(ValueError):
        ImagePreprocessor(["a"], ["b"])
    with pytest.raises(ValueError):
        ImagePreprocessor([])
    with pytest.raises(ValueError):
        ImagePreprocessor(["a", "a"])
    with pytest.raises(ValueError):
        ImagePreprocessor(["a"], image_size=0)
    pre = ImagePreprocessor(["a", "b"], ["b"], 32)
    good = {"a": _uint8_images(2, size=16), "b": _uint8_images(2, size=16)}
    assert pre(good, train_aug=True).shape == (2, 2, 3, 32, 32)
    bad_cases = [
        ({"a": good["a"]}, KeyError),
        ({**good, "b": good["b"].long()}, TypeError),
        ({**good, "b": good["b"].float() * 2}, ValueError),
        ({**good, "b": torch.full((2, 1, 16, 16, 3), float("nan"))}, ValueError),
        ({**good, "b": _uint8_images(3, size=16)}, ValueError),
        ({**good, "b": good["b"].permute(0, 1, 4, 2, 3)}, ValueError),
        ({**good, "b": good["b"][0, 0]}, ValueError),
    ]
    for obs, error in bad_cases:
        with pytest.raises(error):
            pre(obs, train_aug=False)


@pytest.mark.gpu
def test_preprocessing_and_ki_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda")
    pre = ImagePreprocessor(["agentview_rgb", "robot0_eye_in_hand_rgb"], ["robot0_eye_in_hand_rgb"], 224)
    obs = {"agentview_rgb": _uint8_images(2, seed=6), "robot0_eye_in_hand_rgb": _uint8_images(2, seed=7)}
    cuda_obs = {k: v.to(device).float() for k, v in obs.items()}  # runner-style float bytes on the GPU
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pre(cuda_obs, train_aug=False)
        aug = pre(cuda_obs, train_aug=True)
        aug_cpu_gen = pre(cuda_obs, train_aug=True, generator=torch.Generator().manual_seed(0))
    assert out.device.type == "cuda" and out.dtype == aug.dtype == aug_cpu_gen.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), pre(obs, train_aug=False), rtol=0, atol=5e-5)  # CPU vs CUDA rounding
    reference = pre(obs, train_aug=True, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(aug_cpu_gen.cpu(), reference, rtol=0, atol=1e-4)
    table = OATKITable(16).to(device)
    targets = torch.randint(0, 5000, (2, 8), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = table.logits(table.embed_block(targets).to(torch.bfloat16))
    assert logits.dtype == torch.float32 and logits.device.type == "cuda"
    spec = PromptStateSpec.from_config(LIBERO_PROMPT_STATE)
    state = spec.extract({"robot0_eef_pos": torch.zeros(2, 1, 3, device=device),
                          "robot0_eef_quat": torch.tensor([[[1.0, 0, 0, 0]]] * 2, device=device),
                          "robot0_gripper_qpos": torch.zeros(2, 1, 2, device=device)})
    assert state.device.type == "cuda"
    torch.testing.assert_close(state[:, 3].cpu(), torch.full((2,), math.pi))
    bins = discretize_state(state.clamp(-1, 1))
    assert bins.device.type == "cuda" and bins.dtype == torch.int64
    uids = uid_from_obs({"task_uid": torch.full((2, 1, 1), 33.0, device=device)})
    assert uids.device.type == "cuda" and uids.tolist() == [33, 33]
    if DEFAULT_SPM_PATH.is_file():
        ids, valid = PromptBuilder(PaliGemmaTokenizer(str(DEFAULT_SPM_PATH)), "libero10").build(uids, bins)
        assert ids.device.type == valid.device.type == "cpu"
