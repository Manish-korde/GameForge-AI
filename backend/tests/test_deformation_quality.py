"""
Tests for Deformation Quality Assurance & Distortion Metrics
Validates IoU silhouette overlap, alpha mass integrity, canvas border clipping detection,
and maximum bone stretch ratio warnings.
"""

import os
import unittest
import numpy as np

import sys
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from deformation.quality import evaluate_quality_metrics


class TestDeformationQuality(unittest.TestCase):

    def test_identical_arrays_pass_cleanly(self):
        """Identical source and transformed arrays must achieve 1.0 IoU and 100% alpha integrity with no clipping."""
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        # Inner 32x32 square
        arr[16:48, 16:48, 3] = 255

        result = evaluate_quality_metrics(
            src_array=arr,
            out_array=arr,
            max_stretch_ratio=1.0,
            warnings=[],
            width=64,
            height=64
        )

        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.silhouette_overlap_iou, 1.0, places=3)
        self.assertAlmostEqual(result.alpha_integrity_pct, 100.0, places=1)
        self.assertFalse(result.clipping_detected)
        self.assertEqual(len(result.warnings), 0)

    def test_clipping_detection(self):
        """When transformed pixels contact the image perimeter, clipping must be detected."""
        src_arr = np.zeros((64, 64, 4), dtype=np.uint8)
        src_arr[16:48, 16:48, 3] = 255

        out_arr = np.zeros((64, 64, 4), dtype=np.uint8)
        # Shifted so it touches top edge (y=0)
        out_arr[0:32, 16:48, 3] = 255

        result = evaluate_quality_metrics(
            src_array=src_arr,
            out_array=out_arr,
            max_stretch_ratio=1.0,
            warnings=[],
            width=64,
            height=64
        )

        self.assertTrue(result.clipping_detected)
        self.assertTrue(any("clipping" in w.lower() for w in result.warnings))

    def test_alpha_loss_warning(self):
        """Significant loss of non-transparent pixel mass must trigger a low alpha integrity warning."""
        src_arr = np.zeros((64, 64, 4), dtype=np.uint8)
        src_arr[10:50, 10:50, 3] = 255  # 1600 pixels

        out_arr = np.zeros((64, 64, 4), dtype=np.uint8)
        out_arr[20:30, 20:30, 3] = 255  # 100 pixels (~6.25% retention)

        result = evaluate_quality_metrics(
            src_array=src_arr,
            out_array=out_arr,
            max_stretch_ratio=1.0,
            warnings=[],
            width=64,
            height=64
        )

        self.assertLess(result.alpha_integrity_pct, 50.0)
        self.assertTrue(any("alpha integrity" in w.lower() for w in result.warnings))


if __name__ == "__main__":
    unittest.main()
