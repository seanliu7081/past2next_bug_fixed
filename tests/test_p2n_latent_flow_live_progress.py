"""CPU workspace checks for live progress and W&B event ordering.

Tracking is recorded in memory; no W&B service or GPU is used.
"""
import json
from types import SimpleNamespace

from accelerate import Accelerator
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

import oat.policy.p2n_latent_flow_common as policy_module
import oat.workspace.train_p2n_latent_flow as workspace_module
from oat.workspace.train_p2n_latent_flow import TrainP2NLatentFlowWorkspace


class ProgressStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 1)
        self.register_buffer("curriculum", torch.tensor(0))

    @property
    def self_past_step(self):
        return int(self.curriculum)

    def on_optimizer_step(self):
        self.curriculum.add_(1)

    def set_self_past_step(self, step):
        self.curriculum.fill_(step)

    def forward(self, values):
        loss = self.projection(values).square().mean()
        self._last_flow_losses = {"fm_loss": loss.detach(), "ct_loss": loss.detach() * 0}
        return loss

    def get_optimizer(self, **_):
        return torch.optim.AdamW(self.parameters(), lr=.001)

    def set_normalizer(self, _):
        pass


class ProgressDataset(torch.utils.data.Dataset):
    dataset_identity = "workspace-live-progress-test"
    pad_before = 0

    def __init__(self, validation=False):
        self.validation = validation
        self.train_mask = np.asarray([not validation, validation])
        ids = np.arange(len(self))
        self.seq_sampler = SimpleNamespace(indices=np.stack((ids, ids + 1, ids * 0), axis=1))

    def __len__(self):
        return 5 if self.validation else 12

    def __getitem__(self, index):
        return {"action": torch.tensor([1., index / 10.]), "sample_id": torch.tensor(index)}

    def get_validation_dataset(self):
        return ProgressDataset(True)

    def get_normalizer(self):
        return None


def progress_config(interval):
    return OmegaConf.create(dict(
        policy_family="oat_latent_flow", variant="p2n_latent_flow", task_type="real_robot",
        policy=dict(_target_="progress.Student", variant="p2n_latent_flow", flow=dict(inference_steps=8)),
        training=dict(use_ema=True, seed=42, gradient_accumulate_every=2,
                      lr_scheduler="cosine", lr_warmup_steps=None, lr_warmup_ratio=.05,
                      resume=False, allow_bf16=False, num_epochs=1, offline_validation_enabled=True,
                      max_grad_norm=1., checkpoint_every=1, snapshot_every=0,
                      validate_generated_history=False, tqdm_interval_sec=interval),
        ema=dict(power=.75), optimizer=dict(policy_lr=.001),
        dataloader=dict(batch_size=4, drop_last=True, num_workers=0, shuffle=False),
        val_dataloader=dict(batch_size=4, drop_last=False, num_workers=0, shuffle=False),
        task=dict(policy=dict(lazy_eval=True, dataset=dict(_target_="progress.Dataset", seed=42))),
        logging=dict(mode="online", project="fake-progress-project"),
        checkpoint=dict(save_last_ckpt=True, save_last_snapshot=False),
    ))


@pytest.mark.parametrize("interval", [0., 3600.])
def test_real_cpu_loop_logs_first_loss_before_validation_and_checkpoint(monkeypatch, tmp_path, interval):
    timeline = []
    definitions = []
    accelerators = []

    class Tracker:
        def define_metric(self, *args, **kwargs):
            definitions.append((args, kwargs))

    tracker = Tracker()

    class RecordingAccelerator(Accelerator):
        def __init__(self, **kwargs):
            assert kwargs.pop("log_with") == "wandb"
            super().__init__(cpu=True, log_with=None, **kwargs)
            accelerators.append(self)

        def init_trackers(self, project, *, config, init_kwargs):
            assert project == "fake-progress-project"
            assert init_kwargs["wandb"]["mode"] == "online"
            timeline.append(("tracker_init",))

        def get_tracker(self, name, unwrap=False):
            assert name == "wandb" and unwrap
            return tracker

        def log(self, values, **kwargs):
            # Reusing global_step as W&B's event step drops validation events.
            assert kwargs == {}
            timeline.append(("log", dict(values)))

    def validation_metrics(self, batch, *, sample_seeds, history_mode, compute_decoded):
        ids = batch["sample_id"].tolist()
        assert len(sample_seeds) == len(ids)
        assert history_mode == "expert"
        assert batch["action"].device.type == "cpu"
        timeline.append(("validate", ids))
        return {"fm_loss": dict(sum=batch["action"].square().sum(), count=batch["action"].numel())}

    monkeypatch.setattr(ProgressStudent, "validation_metrics", validation_metrics, raising=False)
    original_instantiate = workspace_module.hydra.utils.instantiate

    def instantiate(config, *args, **kwargs):
        if config.get("_target_") == "progress.Student":
            return ProgressStudent()
        if config.get("_target_") == "progress.Dataset":
            return ProgressDataset()
        return original_instantiate(config, *args, **kwargs)

    monkeypatch.setattr(workspace_module.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(workspace_module, "Accelerator", RecordingAccelerator)
    monkeypatch.setattr(policy_module, "prepare_flow_training_batch", lambda batch, **_: batch["action"])

    def checkpoint(self, *, tag):
        assert tag == "latest"
        assert self.completed_optimizer_steps == 2
        timeline.append(("checkpoint",))
        return tmp_path / "recorded-checkpoint.ckpt"

    monkeypatch.setattr(TrainP2NLatentFlowWorkspace, "save_checkpoint", checkpoint)
    workspace = TrainP2NLatentFlowWorkspace(progress_config(interval), output_dir=str(tmp_path))
    workspace.run()
    assert accelerators[0].device.type == "cpu"
    assert workspace.global_step == 3
    assert workspace.completed_optimizer_steps == workspace.ema_state["optimization_step"] == 2
    assert workspace.model.self_past_step == workspace.ema_model.self_past_step == 2
    assert workspace.lr_scheduler_state["last_epoch"] == 2

    reports = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    by_phase = {phase: [row for row in reports if row["phase"] == phase]
                for phase in ("setup", "train", "validation", "checkpoint")}
    assert [row["completed"] for row in by_phase["setup"]] == list(range(6))
    train = by_phase["train"]
    expected_completed = [0, 1, 2, 3] if interval == 0 else [0, 1, 3]
    assert [row["completed"] for row in train] == expected_completed
    assert [row["global_step"] for row in train] == expected_completed
    expected_updates = [0, 0, 1, 2] if interval == 0 else [0, 0, 2]
    assert [row["successful_optimizer_updates"] for row in train] == expected_updates
    assert all(row["skipped_optimizer_updates"] == 0 for row in train)
    assert all(row["loss"] >= 0 and row["batch_loss"] >= 0 for row in train[1:])
    assert all(row["ct_loss"] == 0 for row in train[1:])
    assert train[-1]["eta_seconds"] == 0
    assert by_phase["validation"][0]["completed"] == 0
    assert by_phase["validation"][-1]["completed"] == by_phase["validation"][-1]["total"] == 5
    assert by_phase["validation"][-1]["unit"] == "samples"
    assert by_phase["validation"][-1]["global_step"] == 3
    assert [row["completed"] for row in by_phase["checkpoint"]] == [0, 1]
    assert all(row["epoch"] == 0 for row in reports)
    assert all(row["rank"] == 0 for row in reports)

    first_loss = next(index for index, event in enumerate(timeline)
                      if event[0] == "log" and "train/loss" in event[1])
    first_validation = next(index for index, event in enumerate(timeline) if event[0] == "validate")
    saving = next(index for index, event in enumerate(timeline) if event[0] == "checkpoint")
    assert first_loss < first_validation < saving
    assert timeline[first_loss][1]["global_step"] == 1
    assert [event[1] for event in timeline if event[0] == "validate"] == [[0, 1, 2, 3], [4]]
    logged = [event[1] for event in timeline if event[0] == "log"]
    val_logged = [event for event in logged if "validation/completed" in event]
    assert len(val_logged) >= 2 and all(event["global_step"] == 3 for event in val_logged)
    summary = logged[-1]
    assert summary["global_step"] == 3
    assert summary["validation_samples"] == 5
    assert summary["train_loss"] == pytest.approx(train[-1]["loss"])
    assert definitions == [(("global_step",), {}), (("*",), {"step_metric": "global_step"})]


def test_non_main_rank_progress_has_no_file_or_tracker_effects(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("non-main progress must not print or log")

    accelerator = SimpleNamespace(is_main_process=False, print=forbidden, log=forbidden)
    workspace = TrainP2NLatentFlowWorkspace(progress_config(0), output_dir=str(tmp_path / "rank1"))
    workspace._progress_tracking_enabled = True
    progress = workspace._progress(accelerator, "validation", 0, unit="samples")
    assert not progress.update(0, metrics=forbidden, force=True)
    workspace._emit_progress(accelerator, {"phase": "validation"})
    assert not (tmp_path / "rank1").exists()
