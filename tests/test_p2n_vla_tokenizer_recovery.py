"""Recover embedded OAT weights without silently replacing existing artifacts."""
import dill
from omegaconf import OmegaConf
import pytest
import torch

from oat.model.vla.pi05_checkpoint import file_sha256
from p2n_vla_vision_action_fixtures import make_test_tokenizer_checkpoint
from scripts.recover_p2n_vla_tokenizer import recover_tokenizer


def embedded_tokenizer(tmp_path, *, include_weights=True):
    original_path = make_test_tokenizer_checkpoint(tmp_path / "original_oat.ckpt")
    original = torch.load(original_path, map_location="cpu", pickle_module=dill, weights_only=False)
    original_metadata = {"source": str(original_path), "sha256": file_sha256(original_path), "weights": "ema",
                         "training_task": {"tokenizer": {"name": "unit_test_oat"}}}
    state = {"unrelated_action_parameter": torch.ones(2)}
    if include_weights:
        state.update({"latent_adapter.tokenizer." + key: value
                      for key, value in original["state_dicts"]["ema_model"].items()})
    snapshot = tmp_path / "p2n_vla.ckpt"
    torch.save({"policy_config": {"tokenizer_config": OmegaConf.to_container(original["cfg"].tokenizer),
                                 "tokenizer_metadata": original_metadata},
                "state_dicts": {"ema_model": state}}, snapshot)
    return snapshot, original, original_metadata


def test_recovers_exact_tensors_config_and_provenance(tmp_path):
    snapshot, original, metadata = embedded_tokenizer(tmp_path)
    output = tmp_path / "recovered" / "oat.ckpt"
    summary = recover_tokenizer(snapshot, output)
    recovered = torch.load(output, map_location="cpu", pickle_module=dill, weights_only=False)
    actual = recovered["state_dicts"]["ema_model"]
    expected = original["state_dicts"]["ema_model"]
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key].dtype == expected[key].dtype
        assert torch.equal(actual[key], expected[key]), key
    assert recovered["cfg"].tokenizer == original["cfg"].tokenizer
    assert OmegaConf.to_container(recovered["cfg"].task) == metadata["training_task"]
    assert recovered["metadata"]["original_tokenizer"] == metadata
    assert recovered["metadata"]["recovery"] == "embedded_frozen_oat_from_p2n_vla"
    assert summary["checkpoint_sha256"] == file_sha256(snapshot)
    assert summary["sha256"] == file_sha256(output)
    assert summary["tensor_count"] == len(expected)


def test_existing_output_is_never_overwritten(tmp_path):
    snapshot, _, _ = embedded_tokenizer(tmp_path)
    output = tmp_path / "do_not_overwrite.ckpt"
    output.write_bytes(b"existing user file")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        recover_tokenizer(snapshot, output)
    assert output.read_bytes() == b"existing user file"


def test_missing_embedded_oat_fails_without_creating_output(tmp_path):
    snapshot, _, _ = embedded_tokenizer(tmp_path, include_weights=False)
    output = tmp_path / "absent" / "oat.ckpt"
    with pytest.raises(ValueError, match="No embedded OAT tensors"):
        recover_tokenizer(snapshot, output)
    assert not output.exists()
