"""CPU integration coverage for the standalone real-robot diffusion policy."""

import copy
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from oat.common.hydra_util import register_new_resolvers
from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.policy.diffpolicy import DiffusionTransformerPolicy
from oat.workspace.base_workspace import BaseWorkspace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "oat/config"), version_base=None):
        cfg = compose(config_name="train_diffpolicy_real_robot")
    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_real_robot_config_resolves_without_a_tokenizer(config):
    cfg = config
    assert cfg._target_ == "oat.workspace.train_policy.TrainPolicyWorkspace"
    assert cfg.policy._target_ == "oat.policy.diffpolicy.DiffusionTransformerPolicy"
    assert "action_tokenizer" not in cfg.policy
    assert "tokenizer" not in cfg
    assert (cfg.n_obs_steps, cfg.horizon, cfg.n_action_steps) == (2, 16, 8)
    assert cfg.shape_meta.action.shape == [7]
    task = cfg.task.policy
    assert task.lazy_eval is True
    assert "_target_" not in task.env_runner
    assert task.dataset._target_ == "oat.dataset.real_robot_dataset.RealRobotZarrDataset"
    assert task.dataset.n_obs_steps == 2
    assert task.dataset.n_action_steps == 16
    assert task.dataset.val_ratio == 0.1
    assert task.dataset.seed == cfg.seed == 42
    assert task.dataset.max_train_episodes is None
    assert "past_n" not in task.dataset
    for name in ("agentview_rgb", "robot0_eye_in_hand_rgb"):
        assert cfg.shape_meta.obs[name].shape == [128, 128, 3]
    assert cfg.shape_meta.obs.robot0_eef_rot6d.shape == [6]
    assert cfg.shape_meta.obs.robot0_gripper_qpos.shape == [1]
    encoder = cfg.policy.obs_encoder.vision_encoder
    assert encoder.crop_shape == [112, 112]
    assert encoder.eval_fixed_crop is True
    assert cfg.training.resume is False
    assert cfg.training.use_ema is True
    assert cfg.training.offline_validation_enabled is True
    assert cfg.val_dataloader.drop_last is False
    assert cfg.checkpoint.topk.monitor_key == "test_reconst_mse"
    assert cfg.checkpoint.topk.mode == "min"


def test_original_diffusion_hyperparameters_are_preserved(config):
    policy = config.policy
    assert (policy.embed_dim, policy.n_layers, policy.n_heads) == (256, 4, 4)
    assert policy.dropout == 0.1
    assert policy.num_inference_steps == 10
    scheduler = policy.noise_scheduler
    assert scheduler._target_ == "diffusers.schedulers.scheduling_ddim.DDIMScheduler"
    assert scheduler.num_train_timesteps == 100
    assert scheduler.prediction_type == "epsilon"
    assert scheduler.beta_schedule == "squaredcos_cap_v2"
    assert scheduler.clip_sample is True
    assert config.optimizer.policy_lr == 5e-5
    assert config.optimizer.obs_enc_lr == 1e-5


class TinyObservationEncoder(BaseObservationEncoder):
    """Keep real observation ports while making checkpoint tests inexpensive."""

    def __init__(self, shape_meta):
        super().__init__()
        self.obs_meta = shape_meta["obs"]
        input_dim = sum(3 if meta["type"] == "rgb" else meta["shape"][0]
                        for meta in self.obs_meta.values())
        self.projection = nn.Linear(input_dim, 16)
        self.normalizer = LinearNormalizer()

    def modalities(self):
        return ["rgb", "state"]

    def output_feature_dim(self):
        return 16

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, obs_dict):
        features = []
        for key, meta in self.obs_meta.items():
            value = self.normalizer[key].normalize(obs_dict[key])
            if meta["type"] == "rgb":
                value = value.mean(dim=(-3, -2))
            features.append(value)
        return self.projection(torch.cat(features, dim=-1))


def synthetic_batch_and_normalizer(config, batch_size=2):
    observations = {}
    fit_data = {}
    for key, meta in config.shape_meta.obs.items():
        shape = (batch_size, config.n_obs_steps, *meta.shape)
        if meta.type == "rgb":
            observations[key] = torch.randint(0, 256, shape, dtype=torch.uint8)
            fit_data[key] = torch.tensor([[0., 0., 0.], [255., 255., 255.]])
        else:
            observations[key] = torch.randn(shape)
            fit_data[key] = observations[key]
    actions = torch.randn(batch_size, config.horizon, config.shape_meta.action.shape[0])
    fit_data["action"] = actions
    normalizer = LinearNormalizer()
    normalizer.fit(fit_data)
    return {"obs": observations, "action": actions}, normalizer


def tiny_policy(config):
    return DiffusionTransformerPolicy(
        shape_meta=config.shape_meta,
        noise_scheduler=hydra.utils.instantiate(config.policy.noise_scheduler),
        obs_encoder=TinyObservationEncoder(config.shape_meta),
        horizon=config.horizon,
        n_action_steps=config.n_action_steps,
        n_obs_steps=config.n_obs_steps,
        embed_dim=16,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        num_inference_steps=config.policy.num_inference_steps,
    )


def test_cpu_training_prediction_and_checkpoint_roundtrip(config, cpu_threads, tmp_path):
    torch.manual_seed(42)
    batch, normalizer = synthetic_batch_and_normalizer(config)
    policy = tiny_policy(config)
    policy.set_normalizer(normalizer)
    optimizer = policy.get_optimizer(**config.optimizer)
    before = policy.model.head.weight.detach().clone()
    loss = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for module in (policy.obs_encoder, policy.model):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        assert any(torch.count_nonzero(grad).item() for grad in gradients)
    optimizer.step()
    assert not torch.equal(before, policy.model.head.weight)
    policy.eval()
    with torch.no_grad():
        torch.manual_seed(123)
        prediction = policy.predict_action(batch["obs"])
    assert prediction["action"].shape == (2, 8, 7)
    assert prediction["action_pred"].shape == (2, 16, 7)
    assert torch.isfinite(prediction["action_pred"]).all()
    torch.testing.assert_close(prediction["action"], prediction["action_pred"][:, :8])

    workspace = BaseWorkspace(config, output_dir=str(tmp_path))
    workspace.model = policy
    workspace.ema_model = copy.deepcopy(policy)
    workspace.optimizer = optimizer
    checkpoint = workspace.save_checkpoint(use_thread=False)
    restored = BaseWorkspace(config, output_dir=str(tmp_path))
    restored.model = tiny_policy(config)
    # Training initializes normalizer parameters before constructing AdamW.
    restored.model.set_normalizer(normalizer)
    restored.ema_model = copy.deepcopy(restored.model)
    restored.optimizer = restored.model.get_optimizer(**config.optimizer)
    restored.load_checkpoint(checkpoint, map_location="cpu")
    assert restored.optimizer.state
    for key, value in workspace.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(restored.ema_model.state_dict()[key], value, rtol=0, atol=0)
    restored.model.eval()
    with torch.no_grad():
        torch.manual_seed(123)
        actual = restored.model.predict_action(batch["obs"])
    torch.testing.assert_close(actual["action_pred"], prediction["action_pred"], rtol=0, atol=0)


def test_full_configured_vision_policy_accepts_real_robot_batch(config, cpu_threads):
    torch.manual_seed(42)
    batch, normalizer = synthetic_batch_and_normalizer(config, batch_size=1)
    policy = hydra.utils.instantiate(config.policy)
    policy.set_normalizer(normalizer)
    policy.eval()
    with torch.no_grad():
        loss = policy(batch)
        prediction = policy.predict_action(batch["obs"])
    assert torch.isfinite(loss)
    assert prediction["action"].shape == (1, 8, 7)
    assert prediction["action_pred"].shape == (1, 16, 7)
    assert torch.isfinite(prediction["action_pred"]).all()
