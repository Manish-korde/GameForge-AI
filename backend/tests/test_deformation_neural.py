"""
GameForge AI — Neural Deformation & Dataset Pipeline Tests
Stage 3 regression tests covering synthetic dataset generation, zero split identity leakage,
FlowSTN model forward pass, checkpoint loading, graceful fallback, and metric evaluation.
"""

import os
import unittest
import numpy as np
from PIL import Image

from deformation.dataset import SyntheticDatasetGenerator
from deformation.neural.model import build_flow_stn_model, BilinearGridWarp, FlowScalingLayer
from deformation.neural.infer import NeuralDeformationAdapter, extract_pose_delta
from deformation.eval.benchmark import compute_photometric_metrics, compute_silhouette_metrics
from deformation.skeleton import get_default_skeleton, get_template_by_name


class TestDeformationNeural(unittest.TestCase):
    """Test suite for synthetic dataset generation, neural STN, and evaluation benchmark."""

    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDatasetGenerator(seed=42)
        cls.manifest_path = os.path.join(cls.generator.output_dir, "manifest.json")

    def test_partition_zero_identity_leakage(self):
        """Train, val, and test splits must have strictly disjoint character identities."""
        characters = self.generator.load_available_characters()
        self.assertGreaterEqual(len(characters), 10)

        splits = self.generator.partition_character_splits(characters, train_ratio=0.6, val_ratio=0.2)
        train_ids = {c["character_id"] for c in splits["train"]}
        val_ids = {c["character_id"] for c in splits["val"]}
        test_ids = {c["character_id"] for c in splits["test"]}

        # Check disjointness
        self.assertEqual(len(train_ids.intersection(val_ids)), 0, "Leakage between train and val splits!")
        self.assertEqual(len(train_ids.intersection(test_ids)), 0, "Leakage between train and test splits!")
        self.assertEqual(len(val_ids.intersection(test_ids)), 0, "Leakage between val and test splits!")

        # Check total count matches
        total_partitioned = len(train_ids) + len(val_ids) + len(test_ids)
        self.assertEqual(total_partitioned, len(characters))

    def test_partition_deterministic_seeding(self):
        """Same random seed must produce identical partition splits."""
        chars = self.generator.load_available_characters()
        gen1 = SyntheticDatasetGenerator(seed=123)
        gen2 = SyntheticDatasetGenerator(seed=123)

        s1 = gen1.partition_character_splits(chars)
        s2 = gen2.partition_character_splits(chars)

        ids1 = [c["character_id"] for c in s1["train"]]
        ids2 = [c["character_id"] for c in s2["train"]]
        self.assertEqual(ids1, ids2)

    def test_manifest_file_exists_and_valid(self):
        """Dataset manifest must exist and contain summary and valid sample entries."""
        self.assertTrue(os.path.exists(self.manifest_path), "manifest.json does not exist!")

        import json
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("summary", data)
        self.assertIn("samples", data)

        summary = data["summary"]
        self.assertEqual(summary["total_samples"], 175)
        self.assertEqual(summary["split_counts"]["train"], 105)
        self.assertEqual(summary["split_counts"]["val"], 35)
        self.assertEqual(summary["split_counts"]["test"], 35)
        self.assertEqual(summary["failures"], 0)

        # Inspect first sample
        sample0 = data["samples"][0]
        self.assertIn("source_image_path", sample0)
        self.assertIn("target_image_path", sample0)
        self.assertIn("source_skeleton", sample0)
        self.assertIn("target_skeleton", sample0)

    def test_neural_stn_model_architecture(self):
        """FlowSTN model must accept (1, 128, 128, 4) and (1, 32) and produce correct output shapes."""
        import tensorflow as tf
        model = build_flow_stn_model(image_shape=(128, 128, 4), pose_dim=32, max_displacement=0.25)

        dummy_img = tf.zeros((1, 128, 128, 4), dtype=tf.float32)
        dummy_delta = tf.zeros((1, 32), dtype=tf.float32)

        out_img, flow_field = model([dummy_img, dummy_delta], training=False)

        self.assertEqual(out_img.shape, (1, 128, 128, 4))
        self.assertEqual(flow_field.shape, (1, 128, 128, 2))

    def test_neural_adapter_loading_and_inference(self):
        """NeuralDeformationAdapter must load checkpoint and deform sprite without raising exceptions."""
        adapter = NeuralDeformationAdapter()
        if adapter.is_available():
            loaded = adapter.load_model()
            self.assertTrue(loaded)

            dummy_img = Image.new("RGBA", (128, 128), (100, 150, 200, 255))
            skel_a = get_default_skeleton()
            skel_b = get_template_by_name("walk_1").skeleton

            out = adapter.deform(dummy_img, skel_a, skel_b)
            self.assertIsInstance(out, Image.Image)
            self.assertEqual(out.size, (128, 128))

    def test_neural_adapter_graceful_fallback(self):
        """If neural checkpoint is nonexistent, adapter must gracefully fallback to geometric deformer."""
        adapter = NeuralDeformationAdapter(checkpoint_path="nonexistent_checkpoint.keras")
        self.assertFalse(adapter.is_available())

        dummy_img = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
        skel_a = get_default_skeleton()
        skel_b = get_template_by_name("idle").skeleton

        # Should not raise; should return fallback image
        out = adapter.deform(dummy_img, skel_a, skel_b)
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (128, 128))

    def test_photometric_and_silhouette_metrics(self):
        """Metric calculations must accurately report identical vs perturbed arrays."""
        img1 = np.ones((64, 64, 4), dtype=np.uint8) * 200
        img2 = np.copy(img1)

        # Identical arrays
        p_metrics = compute_photometric_metrics(img1, img2)
        s_metrics = compute_silhouette_metrics(img1, img2)

        self.assertEqual(p_metrics["l1_error"], 0.0)
        self.assertEqual(p_metrics["mse"], 0.0)
        self.assertEqual(p_metrics["psnr_db"], 100.0)
        self.assertEqual(s_metrics["target_overlap_iou"], 1.0)
        self.assertEqual(s_metrics["alpha_integrity_pct"], 100.0)

        # Perturbed array
        img3 = np.copy(img1)
        img3[:, :, 3] = 0  # Erase alpha
        s_metrics2 = compute_silhouette_metrics(img3, img1)
        self.assertEqual(s_metrics2["target_overlap_iou"], 0.0)
        self.assertEqual(s_metrics2["alpha_integrity_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
