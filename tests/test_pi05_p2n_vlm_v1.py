"""Native-VLM/OAT integration contracts using tiny models and a synthetic OAT.

Run with /venv/oat/bin/python -m pytest tests/test_pi05_p2n_vlm_v1.py -q.
The random tokenizer is a test fixture, never a training or evaluation asset.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

import dill
from omegaconf import OmegaConf
import pytest
import torch
import torch.nn.functional as F

from oat.policy.pi05_p2n_vlm_v1 import Pi05P2NVLMV1Policy
from p2n_vla_vision_action_fixtures import make_test_tokenizer_checkpoint
from test_pi05_past import (REPO, SHAPE_META, SPM, _normalizer, _records, cpu_accelerate,  # noqa: F401
                           make_batch, make_libero_like_zarr)


@pytest.fixture(scope="module")
def assets():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    try:
        with tempfile.TemporaryDirectory(prefix="pi05-p2n-vlm-tests-") as directory:
            root = Path(directory)
            yield root, make_test_tokenizer_checkpoint(root / "oat.ckpt")
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture
def build(assets):
    def construct(seed=0, **overrides):
        torch.manual_seed(seed)
        settings = dict(
            shape_meta=SHAPE_META, model_size="tiny", tokenizer_checkpoint=str(assets[1]),
            spm_path=str(SPM), activation_checkpointing=False,
            vision_activation_checkpointing=False, lora_rank=4, lora_alpha=4.0,
            cond_dropout=0.0, train_image_aug=False, self_past_warmup_steps=0,
            self_past_ramp_steps=0)
        settings.update(overrides)
        policy = Pi05P2NVLMV1Policy(**settings)
        policy.set_normalizer(_normalizer())
        return policy
    return construct


def test_native_scope_and_optimizer_cover_all_trainables(build):
    from oat.workspace.train_p2n_vla import validate_parameter_partition

    policy = build()
    assert policy.VARIANT_CODE == 15
    assert policy.tok_emb.weight.shape == (5001, 64)
    assert not hasattr(policy.joint, "expert")
    assert not hasattr(policy, "flow_heads")
    assert not hasattr(policy, "ki_table")
    trainable = dict(policy.trainable_named_parameters())
    assert all("expert" not in name and "flow_head" not in name for name in trainable)
    last = len(policy.joint.vlm.layers) - 1
    for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        for suffix in ("lora_A", "lora_B"):
            assert f"joint.vlm.layers.{last}.{path}.{suffix}" in trainable
    assert any(name.startswith("siglip.") for name in trainable)
    assert not policy.action_tokenizer.training
    assert not any(parameter.requires_grad for parameter in policy.action_tokenizer.parameters())
    optimizer = policy.get_optimizer()
    named, clip = validate_parameter_partition(policy, optimizer)
    assert {name for name, _ in named} == set(trainable)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(trainable)
    assert sum(len(parameters) for parameters in clip.values()) == len(trainable)
    assert not any(name.startswith("siglip.") for name in policy.frozen_base_keys())


def test_history_slots_require_valid_derivative_inputs_and_ignore_padding(build):
    policy = build().eval()
    batch = make_batch(batch=5, episode_step=(0, 1, 2, 3, 7))
    past, valid = batch["past_action"], batch["past_action_valid"]
    tokens, slots = policy.history_tokens(past, valid)
    expected = torch.cat((valid[:, -2:].all(1, keepdim=True),
                          valid[:, -3:].all(1, keepdim=True), valid), dim=1)
    assert tokens.shape == (5, 9, policy.vlm_width)
    torch.testing.assert_close(slots, expected)
    assert torch.isfinite(tokens).all()
    assert torch.equal(tokens[~slots], torch.zeros_like(tokens[~slots]))
    contaminated = past.clone()
    contaminated[~valid] = float("nan")
    clean_tokens, clean_slots = policy.history_tokens(contaminated, valid)
    torch.testing.assert_close(clean_tokens, tokens, rtol=0, atol=0)
    torch.testing.assert_close(clean_slots, slots)
    contaminated[4, -1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        policy.history_tokens(contaminated, valid)


def test_history_is_a_condition_and_padded_history_cannot_change_logits(build):
    policy = build().eval()
    batch = make_batch(batch=2, episode_step=(0, 7))
    targets = torch.randint(0, policy.N_CODES, (2, 8), generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        context = policy.build_native_context(batch["obs"], batch["past_action"], batch["past_action_valid"])
        original = policy.ar_logits(*context, targets)
        changed = batch["past_action"].clone()
        changed[0] = float("nan")
        changed[1] = changed[1].roll(1, 0) + 0.2
        new_context = policy.build_native_context(batch["obs"], changed, batch["past_action_valid"])
        altered = policy.ar_logits(*new_context, targets)
    torch.testing.assert_close(altered[0], original[0], rtol=0, atol=0)
    assert (altered[1] - original[1]).abs().max() > 1e-6


def test_native_cache_matches_forced_teacher_tokens_and_has_no_future_leak(build, monkeypatch):
    policy = build().eval()
    batch = make_batch(batch=3, episode_step=(0, 2, 9))
    forced = torch.randint(0, policy.N_CODES, (3, 8), generator=torch.Generator().manual_seed(39))
    observed = []

    def force_token(logits, temperature, topk, bos_id):
        assert bos_id == 5000
        observed.append(logits.clone())
        return forced[:, len(observed) - 1]

    monkeypatch.setattr(policy, "_sample", force_token)
    with torch.no_grad():
        context = policy.build_native_context(batch["obs"], batch["past_action"], batch["past_action_valid"])
        teacher = policy.ar_logits(*context, forced)
        generated = policy.generate_tokens(*context, 8, 0.0, 10)
        altered_targets = forced.clone()
        altered_targets[:, 4:] = (altered_targets[:, 4:] + 123) % policy.N_CODES
        altered = policy.ar_logits(*context, altered_targets)
    assert teacher.shape == (3, 8, 5001)
    torch.testing.assert_close(generated, forced)
    torch.testing.assert_close(torch.stack(observed, dim=1), teacher, rtol=2e-5, atol=2e-5)
    # Target 4 is input only at position 5; its own prediction cannot read it.
    torch.testing.assert_close(altered[:, :5], teacher[:, :5], rtol=0, atol=0)
    assert (altered[:, 5:] - teacher[:, 5:]).abs().max() > 1e-6


@pytest.mark.parametrize("steps", [(12, 3, 9), (0, 0, 0)])
def test_action_ce_reaches_vision_history_and_every_vlm_layer(build, steps):
    policy = build().train()
    batch = make_batch(episode_step=steps)
    loss = policy(batch, history_mode="expert")
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    trainable = dict(policy.trainable_named_parameters())
    missing = [name for name, parameter in trainable.items() if parameter.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable.values())
    assert not any(parameter.grad is not None for parameter in policy.parameters() if not parameter.requires_grad)
    assert trainable["tok_emb.weight"].grad.abs().sum() > 0
    assert sum(parameter.grad.abs().sum() for name, parameter in trainable.items()
               if name.startswith("siglip.")) > 0
    for index in range(len(policy.joint.vlm.layers)):
        assert sum(parameter.grad.abs().sum() for name, parameter in trainable.items()
                   if name.startswith(f"joint.vlm.layers.{index}.") and name.endswith("lora_B")) > 0
    if any(steps):
        for prefix in ("acc_proj.", "jerk_proj.", "raw_proj."):
            assert sum(parameter.grad.abs().sum() for name, parameter in trainable.items()
                       if name.startswith(prefix)) > 0
    assert policy.last_loss_components["loss_ar"] == pytest.approx(loss.item())
    assert policy.last_loss_components["loss_flow"] is None
    assert policy.last_loss_components["loss_ki"] is None
    assert policy.training and policy.siglip.training and not policy.action_tokenizer.training


def test_full_ce_contains_bos_class_but_sampler_never_returns_bos(build):
    policy = build().eval()
    batch = make_batch(batch=1, episode_step=(7,))
    with torch.no_grad():
        targets = policy.encode_targets(batch["action"])
        context = policy.build_native_context(batch["obs"], batch["past_action"], batch["past_action_valid"])
        logits = policy.ar_logits(*context, targets)
        expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        actual = policy(batch, history_mode="expert")
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(logits[..., policy.BOS_ID]).all()
    artificial = torch.zeros(2, 5001)
    artificial[:, 5000] = 100
    artificial[:, 42] = 20
    torch.testing.assert_close(policy._sample(artificial, 0.0, 10, 5000), torch.tensor([42, 42]))


def test_greedy_inference_prefix_decode_and_acknowledged_history(build):
    policy = build().train()
    batch = make_batch(batch=2, episode_step=(2, 9))
    kwargs = dict(past_actions=batch["past_action"], past_action_valid=batch["past_action_valid"])
    torch.manual_seed(1)
    first = policy.predict_action(batch["obs"], **kwargs)
    first_tokens = first["action_tokens"].clone()
    torch.manual_seed(999)
    second = policy.predict_action(batch["obs"], **kwargs)
    torch.testing.assert_close(first["action_pred"], second["action_pred"], rtol=0, atol=0)
    torch.testing.assert_close(first["action_tokens"], first_tokens)
    torch.testing.assert_close(second["action_tokens"], first_tokens)
    assert first_tokens.shape == (2, 8) and ((first_tokens >= 0) & (first_tokens < 5000)).all()
    assert first["action"].shape == (2, 8, 7)
    assert first["action_pred"].shape == (2, 16, 7)
    assert policy._pending_execution_steps is None
    short = policy.predict_action(batch["obs"], use_k_tokens=3, **kwargs)
    assert short["action_pred"].shape == (2, 16, 7)
    torch.testing.assert_close(short["action_tokens"], first_tokens[:, :3])
    prediction = policy.predict_action(batch["obs"])
    with pytest.raises(RuntimeError, match="feedback is pending"):
        policy.predict_action(batch["obs"])
    actual = prediction["action"].clone() + 0.125
    actual[0, 3:] = float("nan")  # unexecuted padded entries are irrelevant
    policy.record_executed_actions(actual, executed_lengths=torch.tensor([3, 8]))
    torch.testing.assert_close(policy._past_buffer[0, -3:], actual[0, :3])
    torch.testing.assert_close(policy._past_buffer[1], actual[1, -7:])
    assert policy._past_valid_buffer[0].tolist() == [False] * 4 + [True] * 3
    assert policy._past_valid_buffer[1].all()
    assert policy.training and not policy.action_tokenizer.training
    assert policy._pending_execution_steps is None
    policy.reset()
    assert policy._past_buffer is None and policy._past_valid_buffer is None


def test_self_past_uses_executed_prefix_and_restores_training_modes(build, monkeypatch):
    policy = build(self_past_chunk_size=1, self_past_temperature=0.7, self_past_topk=3).train()
    batch = make_batch()
    calls = []

    def generate(obs, past, valid, n_tokens, temperature, topk):
        calls.append((n_tokens, temperature, topk))
        assert not policy.training and not policy.action_tokenizer.training
        assert not torch.is_grad_enabled()
        return torch.arange(16, dtype=past.dtype).reshape(1, 16, 1).expand(past.shape[0], 16, 7)

    monkeypatch.setattr(policy, "_generate_actions", generate)
    history, probability = policy._resolve_history(batch, "generated")
    expected = torch.arange(1, 8, dtype=history.dtype)[:, None].expand(7, 7)
    torch.testing.assert_close(history[0], expected)
    torch.testing.assert_close(history[2], expected)
    torch.testing.assert_close(history[1], batch["past_action"][1])
    assert probability == 1 and policy._last_self_past_rows == 2
    assert calls == [(8, 0.7, 3)] * 2
    assert policy._last_self_past_mse > 0
    assert policy.training and policy.siglip.training and not policy.action_tokenizer.training


def test_optimizer_step_schedule_and_history_free_control(build):
    policy = build(self_past_p=0.6, self_past_warmup_steps=2, self_past_ramp_steps=3)
    for step, expected in ((0, 0), (2, 0), (3, 0.2), (5, 0.6)):
        policy.set_self_past_step(step)
        assert policy.self_past_probability() == pytest.approx(expected)
    policy.on_optimizer_step()
    assert policy.self_past_step == 6
    policy.eval().on_optimizer_step()
    assert policy.self_past_step == 6
    control = build(use_past=False)
    assert not any(name.startswith(("acc_proj", "jerk_proj", "raw_proj", "cond_pos_emb"))
                   for name, _ in control.named_parameters())
    batch = make_batch(batch=1, episode_step=(7,))
    history, valid = control.history_tokens(batch["past_action"], batch["past_action_valid"])
    assert history.shape == (1, 9, control.vlm_width)
    assert not valid.any() and not history.any()
    assert control.self_past_probability() == 0
    control(batch, history_mode="expert").backward()
    assert all(parameter.grad is not None for _, parameter in control.trainable_named_parameters())


def test_compact_artifact_restores_trained_vision_history_and_action_head(build, assets):
    policy = build()
    batch = make_batch(batch=2, episode_step=(3, 9))
    optimizer = policy.get_optimizer()
    policy(batch, history_mode="expert").backward()
    optimizer.step()
    policy.set_self_past_step(23)
    policy.eval()
    kwargs = dict(past_actions=batch["past_action"], past_action_valid=batch["past_action_valid"])
    reference = policy.predict_action(batch["obs"], **kwargs)["action_pred"]
    artifact = policy.artifact_state_dict()
    assert "tok_emb.weight" in artifact and "cond_pos_emb" in artifact
    assert any(name.startswith("siglip.") for name in artifact)
    assert "joint.vlm.embed_tokens.weight" not in artifact
    metadata = policy.artifact_metadata()
    assert metadata[policy.VARIANT]["head"] == "native_vlm_ar_oat8"
    checkpoint = assets[0] / "policy.ckpt"
    torch.save(dict(policy_config=policy.export_config(), metadata=metadata,
                    state_dicts={"model": artifact}), checkpoint, pickle_module=dill)
    torch.manual_seed(9876)  # frozen tiny base reconstruction must not depend on caller RNG
    restored = Pi05P2NVLMV1Policy.from_checkpoint(checkpoint, weights="model")
    actual = restored.predict_action(batch["obs"], **kwargs)["action_pred"]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert restored.self_past_step == 23
    assert not restored.training and not restored.action_tokenizer.training
    assert {name for name, _ in restored.trainable_named_parameters()} == {
        name for name, _ in policy.trainable_named_parameters()}
    for name, tensor in restored.artifact_state_dict().items():
        torch.testing.assert_close(tensor, artifact[name], rtol=0, atol=0)
    invalid = dict(artifact, _variant_code=torch.tensor(13))
    with pytest.raises(ValueError, match="_variant_code"):
        restored.load_artifact_state(invalid)
    missing = dict(artifact)
    del missing["tok_emb.weight"]
    with pytest.raises(KeyError, match="Artifact key mismatch"):
        restored.load_artifact_state(missing)


def _tiny_run_config(data, tokenizer, *overrides):
    import scripts.train_p2n_vla as launcher

    return launcher.compose_config("pi05_p2n_vlm_v1", "libero", [
        "policy.model_size=tiny", "policy.pi05_weights=null",
        f"policy.tokenizer_checkpoint={tokenizer}", f"policy.spm_path={SPM}",
        "policy.activation_checkpointing=false", "policy.vision_activation_checkpointing=false",
        "policy.lora_rank=4", "policy.lora_alpha=4.0", "policy.self_past_p=1.0",
        "policy.self_past_warmup_steps=0", f"task.policy.dataset.zarr_path={data}",
        "training.num_demo=3", "task.policy.lazy_eval=true", "training.keep_snapshots=all",
        "training.num_epochs=2", "training.gradient_accumulate_every=1", "training.max_train_steps=2",
        "training.max_optimizer_steps=10", "training.lr_warmup_steps=1", "training.snapshot_every=1",
        "training.val_every=1", "training.checkpoint_every=1", "training.max_val_steps=1",
        "training.max_reconst_steps=1", "dataloader.batch_size=2", "dataloader.num_workers=0",
        "dataloader.persistent_workers=false", "dataloader.pin_memory=false", "val_dataloader.batch_size=2",
        "val_dataloader.num_workers=0", "val_dataloader.persistent_workers=false", "val_dataloader.pin_memory=false",
        "logging.mode=disabled", *overrides])


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_nested_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for item, value in zip(actual, expected):
            _assert_nested_equal(item, value)
    else:
        assert actual == expected


@pytest.mark.slow
def test_workspace_resume_matches_uninterrupted_weights_optimizer_rng_and_ema(assets, cpu_accelerate):
    from accelerate.state import AcceleratorState
    from oat.workspace.train_p2n_vla import TrainP2NVLAWorkspace

    with tempfile.TemporaryDirectory(prefix="pi05-p2n-vlm-resume-") as directory:
        root = Path(directory)
        data = make_libero_like_zarr(root / "data.zarr")

        def run(output, *overrides):
            AcceleratorState._reset_state(True)
            cfg = _tiny_run_config(data, assets[1], "training.use_ema=true", *overrides)
            workspace = TrainP2NVLAWorkspace(cfg, output_dir=str(output))
            workspace.run()
            assert workspace.model.self_past_step == workspace.completed_optimizer_steps
            return workspace

        full = run(root / "full")
        run(root / "first", "training.num_epochs=1")
        resumed = run(root / "resumed", "training.resume=true",
                      f"training.resume_checkpoint={root / 'first/checkpoints/latest.ckpt'}")
        assert full.completed_optimizer_steps == resumed.completed_optimizer_steps == 4
        assert full.ema.updates == resumed.ema.updates == 4
        for attribute in ("model", "optimizer", "lr_scheduler"):
            original, restored = getattr(full, attribute), getattr(resumed, attribute)
            extract = lambda obj: obj.artifact_state_dict() if attribute == "model" else obj.state_dict()
            _assert_nested_equal(extract(restored), extract(original))
        _assert_nested_equal(resumed.ema.shadow, full.ema.shadow)
        payloads = [torch.load(root / name / "checkpoints/latest.ckpt", map_location="cpu", weights_only=False)
                    for name in ("full", "resumed")]
        _assert_nested_equal(payloads[1]["training"]["counters"], payloads[0]["training"]["counters"])
        torch.testing.assert_close(payloads[1]["training"]["rng_states"][0]["torch"],
                                   payloads[0]["training"]["rng_states"][0]["torch"], rtol=0, atol=0)
        full_steps = _records(root / "full/logs.jsonl", "train_step")[2:]
        resumed_steps = _records(root / "resumed/logs.jsonl", "train_step")
        assert [row["train/loss"] for row in resumed_steps] == [row["train/loss"] for row in full_steps]
        assert all(row["train/loss_ar"] is not None and row.get("train/loss_flow") is None
                   for row in resumed_steps)
        epochs = _records(root / "resumed/logs.jsonl", "epoch")
        assert epochs and epochs[-1]["val_loss_generated_history"] is not None
        assert 0 <= epochs[-1]["val/free_running_token_acc"] <= 1
        assert 0 <= epochs[-1]["val/full_sequence_acc"] <= 1
        assert epochs[-1]["val/normalized_executed_action_mse"] >= 0
        checkpoint = root / "resumed/checkpoints/latest.ckpt"
        restored = Pi05P2NVLMV1Policy.from_checkpoint(checkpoint, weights="ema")
        assert restored.self_past_step == 4 and not restored.training


@pytest.mark.slow
def test_two_rank_gloo_workspace_has_all_gradients_and_equal_replicas(assets):
    with tempfile.TemporaryDirectory(prefix="pi05-p2n-vlm-gloo-") as directory:
        root = Path(directory)
        data = make_libero_like_zarr(root / "data.zarr")
        cfg = _tiny_run_config(data, assets[1])
        config_path = root / "worker.yaml"
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), config_path)
        output = root / "run"
        env = dict(os.environ, ACCELERATE_USE_CPU="true", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                   str(REPO / "scripts/train_p2n_vla.py"), "--worker-config", str(config_path), "--output", str(output)]
        result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
        report = json.loads((output / "training_report.json").read_text())
        assert report["world_size"] == 2
        assert report["ddp"]["wrapper"] == "DistributedDataParallel"
        assert report["ddp"]["find_unused_parameters"] is False
        assert report["ddp"]["gradient_as_bucket_view"] is True
        epochs = _records(output / "logs.jsonl", "epoch")
        assert len(epochs) == 2 and all(row["replica_fingerprint"] is not None for row in epochs)
        steps = _records(output / "logs.jsonl", "train_step")
        assert len(steps) == 4 and all(row["train/loss_ar"] is not None for row in steps)
        assert any(row.get("train/self_past_hist_mse") is not None for row in steps)
