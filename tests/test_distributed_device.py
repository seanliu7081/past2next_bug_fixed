"""Catch CUDA rank collisions before the NCCL process group is initialized."""

import pytest
import torch

from oat.workspace.train_policy import TrainPolicyWorkspace


@pytest.fixture
def cuda_devices(monkeypatch):
    monkeypatch.delenv("ACCELERATE_TORCH_DEVICE", raising=False)
    monkeypatch.delenv("ACCELERATE_USE_CPU", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    return selected


@pytest.mark.parametrize("rank", [0, 1])
def test_torchrun_selects_each_ranks_gpu_before_nccl(monkeypatch, cuda_devices, rank):
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    selected = TrainPolicyWorkspace._bind_distributed_cuda_device()
    assert selected == torch.device("cuda", rank)
    assert cuda_devices == [selected]


def test_unindexed_cuda_override_uses_local_rank(monkeypatch, cuda_devices):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cuda")
    TrainPolicyWorkspace._bind_distributed_cuda_device()
    assert cuda_devices == [torch.device("cuda", 1)]


def test_fixed_cuda_override_cannot_pin_both_ranks_to_gpu_zero(monkeypatch, cuda_devices):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cuda:0")
    with pytest.raises(RuntimeError, match="Unset ACCELERATE_TORCH_DEVICE"):
        TrainPolicyWorkspace._bind_distributed_cuda_device()
    assert not cuda_devices


def test_too_few_visible_gpus_fails_before_nccl(monkeypatch, cuda_devices):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="only 1 CUDA devices are visible"):
        TrainPolicyWorkspace._bind_distributed_cuda_device()
    assert not cuda_devices


@pytest.mark.parametrize("mode", ["cpu_flag", "cpu_device", "no_cuda", "single_process"])
def test_cpu_and_non_distributed_runs_do_not_select_cuda(monkeypatch, cuda_devices, mode):
    monkeypatch.setenv("LOCAL_RANK", "0")
    if mode == "cpu_flag":
        monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    elif mode == "cpu_device":
        monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cpu")
    elif mode == "no_cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    else:
        monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert TrainPolicyWorkspace._bind_distributed_cuda_device() is None
    assert not cuda_devices
