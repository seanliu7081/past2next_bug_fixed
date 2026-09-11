"""CPU-only regressions for native center crops used by 015, 043 and 046."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from oat.perception.crop_randomizer import CropRandomizer


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

    def test_cli_preserves_native_crop_size_and_reports_actual_crops(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_candidate.py"
        spec = importlib.util.spec_from_file_location("evaluate_candidate_crops", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = ["--checkpoint", str(script), "--output-dir", "/tmp/new_eval"]
        args = module.parser().parse_args(base)
        module.validate_args(args)
        self.assertEqual(module.checkpoint_policy_overrides(args), {})
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


if __name__ == "__main__":
    unittest.main()
