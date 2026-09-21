"""Two CPU DDP ranks exercise both offline state-history training paths."""

from datetime import timedelta
import time
from types import SimpleNamespace

import pytest
import torch
from torch import distributed as dist, multiprocessing as mp, nn
from torch.nn.parallel import DistributedDataParallel

from oat.model.common.normalizer import LinearNormalizer
from oat.policy.past2next_state_history import Past2NextStateHistoryPolicy


_STATE_SHAPES = {"robot0_eef_pos": (3,), "robot0_eef_quat": (4,),
                 "robot0_gripper_qpos": (2,)}


class _PhysicalObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(9, 8)

    def modalities(self):
        return ["state"]

    def output_feature_dim(self):
        return 8

    def set_normalizer(self, normalizer):
        pass

    def forward(self, obs):
        return self.projection(torch.cat([obs[key] for key in _STATE_SHAPES], dim=-1))


class _DiscreteTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(1))
        self.quantizer = SimpleNamespace(codebook_size=8)
        self.latent_horizon = 2
        self.decode_calls = 0

    def tokenize(self, actions):
        # Data-dependent labels ensure the two ranks train on distinct examples.
        return (actions[:, :2, 0].abs().mul(5).long() % 8)

    def detokenize(self, tokens):
        self.decode_calls += 1
        return tokens.float().mean(dim=1)[:, None, None].expand(-1, 16, 7) / 8


def _make_policy():
    policy = Past2NextStateHistoryPolicy(
        shape_meta={"action": {"shape": [7]}, "obs": {
            key: {"shape": list(shape), "type": "state"}
            for key, shape in _STATE_SHAPES.items()
        }},
        obs_encoder=_PhysicalObservationEncoder(), action_tokenizer=_DiscreteTokenizer(),
        n_action_steps=8, n_obs_steps=2, past_n=7,
        embed_dim=8, n_layers=1, n_heads=2, dropout=0, temperature=0,
        self_past_p=1, self_past_warmup_steps=0, self_past_ramp_steps=0,
        self_past_schedule="optimizer_step", state_history_steps=8,
        history_embed_dim=8, history_n_heads=2, history_n_layers=1,
        history_summary_tokens=2, history_dropout=0,
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.stack((-torch.ones(7), torch.ones(7))),
                    **{key: torch.stack((-torch.ones(*shape), torch.ones(*shape)))
                       for key, shape in _STATE_SHAPES.items()}})
    policy.set_normalizer(normalizer)
    return policy.train()


def _make_batch(rank):
    generator = torch.Generator().manual_seed(100 + rank)

    def observation():
        obs = {}
        for key, shape in _STATE_SHAPES.items():
            history = torch.randn(2, 8, *shape, generator=generator) / 5
            if key.endswith("_quat"):
                history.zero_()
                history[..., -1] = 1
            obs["state_history__" + key] = history
            obs[key] = history[:, -2:]
        obs["state_history_valid"] = torch.ones(2, 8, dtype=torch.bool)
        # Different padding across ranks exercises local masks under DDP.
        obs["state_history_valid"][0, :rank + 2] = False
        return obs

    return {
        "obs": observation(), "prev_obs": observation(),
        "action": torch.randn(2, 16, 7, generator=generator),
        "past_action": torch.randn(2, 7, 7, generator=generator),
        "prev_past_action": torch.randn(2, 7, 7, generator=generator),
        "past_action_valid": torch.ones(2, 7, dtype=torch.bool),
        "prev_window_valid": torch.tensor([bool(rank), True]),
    }


def _run_rank(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(7 + rank)
        policy = _make_policy()
        ddp = DistributedDataParallel(policy, find_unused_parameters=False)
        optimizer = policy.get_optimizer(policy_lr=1e-3, obs_enc_lr=1e-3,
                                          weight_decay=0, betas=(0.9, 0.95))
        batch = _make_batch(rank)
        tracked = {
            "projection": policy.history_encoder.input_projection[0].weight,
            "queries": policy.history_encoder.summary_queries,
        }
        for step, mode in enumerate(("expert", "configured"), start=1):
            before = {key: parameter.detach().clone() for key, parameter in tracked.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = ddp(batch, history_mode=mode)
            assert torch.isfinite(loss), (rank, mode)
            loss.backward()
            # Explicit checks also catch unused parameters in the final step,
            # which DDP otherwise detects only at the next forward call.
            for name, parameter in policy.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None, (rank, mode, name)
                    assert torch.isfinite(parameter.grad).all(), (rank, mode, name)
            optimizer.step()
            policy.on_optimizer_step()
            assert policy.self_past_step == step
            for key, parameter in tracked.items():
                assert not torch.equal(parameter.detach(), before[key]), (rank, mode, key)
            synchronized = [torch.empty_like(tracked["projection"]) for _ in range(2)]
            dist.all_gather(synchronized, tracked["projection"].detach())
            torch.testing.assert_close(synchronized[0], synchronized[1], atol=0, rtol=0)
        assert policy.action_tokenizer.decode_calls == 1
        assert not policy.action_tokenizer.training
        assert all(not parameter.requires_grad for parameter in policy.action_tokenizer.parameters())
        assert policy._past_buffer is None and policy._pending_execution_steps is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason="CPU gloo distributed backend is unavailable")
def test_state_history_policy_two_rank_ddp_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    rendezvous = (tmp_path / "gloo_rendezvous").as_uri()
    context = mp.spawn(_run_rank, args=(rendezvous,), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Two-rank CPU state-history training did not finish within 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
