"""CPU-only regressions for random training and center inference crops."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from hydra import compose, initialize_config_dir
import robomimic.models.base_nets as rmbn
import torch
from torch import nn

from oat.perception.crop_randomizer import CropRandomizer
from oat.perception.fused_obs_encoder import FusedObservationEncoder
from oat.perception.robomimic_vision_encoder import RobomimicRgbEncoder
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy


class EvaluationCropTests(unittest.TestCase):
    def setUp(self):
        # Two examples, each with two observation frames, as flattened by the
        # vision encoder. Pixel offsets reveal accidental mixing of images.
        self.images = torch.arange(72, dtype=torch.float32).reshape(1, 1, 8, 9)
        self.images = self.images + 1000 * torch.arange(4).reshape(4, 1, 1, 1)
        self.expected = torch.tensor([
            [20, 21, 22, 23], [29, 30, 31, 32], [38, 39, 40, 41],
        ], dtype=torch.float32)

    def make_randomizer(self, num_crops=1):
        return CropRandomizer((1, 8, 9), crop_height=3, crop_width=4,
                              num_crops=num_crops).eval()

    def test_existing_center_mode_and_repeated_center_are_unchanged(self):
        randomizer = self.make_randomizer()
        expected = self.expected.reshape(1, 1, 3, 4)
        expected = expected + 1000 * torch.arange(4).reshape(4, 1, 1, 1)
        torch.testing.assert_close(randomizer.forward_in(self.images), expected)
        repeated = self.make_randomizer(num_crops=3)
        output = repeated.forward_in(self.images).reshape(4, 3, 1, 3, 4)
        torch.testing.assert_close(output, expected[:, None].expand(-1, 3, -1, -1, -1))

    def test_center_is_deterministic_and_does_not_consume_rng(self):
        randomizer = self.make_randomizer()
        torch.manual_seed(17)
        before = torch.random.get_rng_state().clone()
        first = randomizer.forward_in(self.images)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        torch.manual_seed(999)
        torch.testing.assert_close(randomizer.forward_in(self.images), first)

    def test_feature_average_does_not_mix_examples_or_frames(self):
        randomizer = self.make_randomizer(num_crops=3)
        features = torch.arange(24, dtype=torch.float32).reshape(12, 2)
        pooled = randomizer.forward_out(features).reshape(2, 2, 2)
        expected = torch.tensor([[[2, 3], [8, 9]], [[14, 15], [20, 21]]], dtype=torch.float32)
        torch.testing.assert_close(pooled, expected)

    def test_random_training_matches_legacy_crops_and_rng(self):
        images = torch.arange(4 * 3 * 128 * 128, dtype=torch.float32).reshape(4, 3, 128, 128)
        for num_crops in (1, 3):
            options = dict(input_shape=(3, 128, 128), crop_height=112,
                           crop_width=112, num_crops=num_crops)
            legacy = rmbn.CropRandomizer(**options).train()
            current = CropRandomizer(**options).train()
            outputs = []
            for seed in (17, 999):
                torch.manual_seed(seed)
                expected = legacy.forward_in(images)
                expected_rng = torch.random.get_rng_state().clone()
                torch.manual_seed(seed)
                actual = current.forward_in(images)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(torch.random.get_rng_state(), expected_rng)
                outputs.append(actual)
            self.assertFalse(torch.equal(*outputs))

    def test_cli_preserves_native_crop_size_and_reports_actual_crops(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_candidate.py"
        spec = importlib.util.spec_from_file_location("evaluate_candidate_crops", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = ["--checkpoint", str(script), "--output-dir", "/tmp/new_eval"]
        args = module.parser().parse_args(base)
        module.validate_args(args)
        self.assertEqual(args.crop_mode, "center")
        self.assertEqual(module.checkpoint_policy_overrides(args), {
            "obs_encoder": {"vision_encoder": {"eval_fixed_crop": True}}})
        legacy = module.parser().parse_args(base + ["--crop-mode", "checkpoint"])
        self.assertEqual(module.checkpoint_policy_overrides(legacy), {})
        center = module.parser().parse_args(base + ["--crop-mode", "center"])
        self.assertEqual(module.checkpoint_policy_overrides(center), {
            "obs_encoder": {"vision_encoder": {"eval_fixed_crop": True}}})
        crops = SimpleNamespace(named_modules=lambda: iter([
            ("camera0", SimpleNamespace(crop_height=112, crop_width=112, num_crops=1)),
            ("camera1", SimpleNamespace(crop_height=112, crop_width=112, num_crops=1)),
        ]))
        settings = module.crop_inference_settings(crops, center)
        self.assertEqual(settings["crop_mode"], "center")
        self.assertEqual(len(settings["crop_randomizers"]), 2)
        self.assertTrue(all(crop["crop_size"] == [112, 112]
                            for crop in settings["crop_randomizers"]))


class TinyTokenizer(nn.Module):
    """Only the frozen tokenizer interface needed to construct a real policy."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.quantizer = SimpleNamespace(codebook_size=8)
        self.latent_horizon = 2


class PolicyCropModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera_names = ("agentview_rgb", "robot0_eye_in_hand_rgb")
        cls.shape_meta = {
            "action": {"shape": [7]},
            "obs": {key: {"shape": [128, 128, 3], "type": "rgb"}
                    for key in cls.camera_names},
        }
        # Exercise the real fused encoder, both ResNet camera paths and policy
        # mode transitions. Only the tokenizer and AR dimensions are reduced.
        encoder = FusedObservationEncoder(cls.shape_meta, vision_encoder={
            "_target_": "oat.perception.robomimic_vision_encoder.RobomimicRgbEncoder",
            "crop_shape": [112, 112],
        })
        cls.policy = Past2NextSelfPastPolicy(
            shape_meta=cls.shape_meta, obs_encoder=encoder,
            action_tokenizer=TinyTokenizer(), n_action_steps=8, n_obs_steps=2,
            past_n=7, embed_dim=8, n_layers=1, n_heads=2, dropout=0,
        )
        cls.images = torch.arange(4 * 3 * 128 * 128, dtype=torch.float32).reshape(4, 3, 128, 128)
        cls.expected = cls.images[:, :, 8:120, 8:120]

    def setUp(self):
        self.policy.train()

    def crops(self):
        randomizers = self.policy.obs_encoder.vision_encoder.encoder.obs_randomizers
        self.assertEqual(set(randomizers), set(self.camera_names))
        for crop in randomizers.values():
            self.assertIsInstance(crop, CropRandomizer)
        return list(randomizers.values())

    def assert_center_crops(self):
        before = torch.random.get_rng_state().clone()
        for crop in self.crops():
            self.assertFalse(crop.training)
            torch.testing.assert_close(crop.forward_in(self.images), self.expected, rtol=0, atol=0)
            torch.testing.assert_close(crop.forward_in(self.images), self.expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), before)

    def assert_random_crops(self):
        for crop in self.crops():
            self.assertTrue(crop.training)
            torch.manual_seed(17)
            before = torch.random.get_rng_state().clone()
            first = crop.forward_in(self.images)
            self.assertFalse(torch.equal(torch.random.get_rng_state(), before))
            torch.manual_seed(17)
            torch.testing.assert_close(crop.forward_in(self.images), first, rtol=0, atol=0)
            torch.manual_seed(999)
            self.assertFalse(torch.equal(crop.forward_in(self.images), first))
            self.assertFalse(torch.equal(first, self.expected))

    def test_policy_train_eval_train_switches_both_camera_crops(self):
        self.assert_random_crops()
        self.policy.eval()
        self.assert_center_crops()
        self.policy.train()
        self.assert_random_crops()

    def test_generated_history_uses_center_and_restores_training_crops(self):
        self.assert_random_crops()
        with self.policy._rollout_mode():
            self.assert_center_crops()
        self.assert_random_crops()
        # Restore modes even if a generation fails, and preserve eval mode if
        # history is generated as part of validation.
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with self.policy._rollout_mode():
                self.assert_center_crops()
                raise RuntimeError("generation failed")
        self.assert_random_crops()
        self.policy.eval()
        with self.policy._rollout_mode():
            self.assert_center_crops()
        self.assert_center_crops()

    def test_real_encoder_features_are_deterministic_in_eval(self):
        self.policy.eval()
        observations = {
            key: (self.images / self.images.max()).permute(0, 2, 3, 1).reshape(2, 2, 128, 128, 3)
            for key in self.camera_names
        }
        with torch.no_grad():
            torch.manual_seed(17)
            before = torch.random.get_rng_state().clone()
            first = self.policy.obs_encoder(observations)
            torch.testing.assert_close(torch.random.get_rng_state(), before)
            torch.manual_seed(999)
            before = torch.random.get_rng_state().clone()
            second = self.policy.obs_encoder(observations)
            torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertEqual(first.shape, (2, 2, 128))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_legacy_encoder_weights_load_strictly_with_center_crop_default(self):
        legacy = RobomimicRgbEncoder(self.shape_meta, crop_shape=[112, 112], eval_fixed_crop=False)
        self.assertTrue(all(isinstance(crop, rmbn.CropRandomizer)
                            for crop in legacy.encoder.obs_randomizers.values()))
        current = self.policy.obs_encoder.vision_encoder
        result = current.load_state_dict(legacy.state_dict(), strict=True)
        self.assertFalse(result.missing_keys)
        self.assertFalse(result.unexpected_keys)
        self.policy.eval()
        self.assert_center_crops()

    def test_all_policy_configs_explicitly_select_fixed_eval_crop(self):
        directory = Path(__file__).resolve().parents[1] / "oat/config"
        configs = sorted(directory.glob("train_past2next*.yaml"))
        self.assertEqual(len(configs), 5)
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            for path in configs:
                with self.subTest(config=path.stem):
                    config = compose(config_name=path.stem)
                    self.assertTrue(config.policy.obs_encoder.vision_encoder.eval_fixed_crop)
                    crop_size = 112 if "scratch" in path.stem else 76
                    self.assertEqual(list(config.policy.obs_encoder.vision_encoder.crop_shape),
                                     [crop_size, crop_size])


if __name__ == "__main__":
    unittest.main()
