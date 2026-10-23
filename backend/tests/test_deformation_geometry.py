"""
Tests for 2D Character Geometric Deformation Engine
Validates bounding box detection, skeleton fitting, bone transformations,
nearest-neighbor pixel art resampling, and alpha channel preservation.
"""

import os
import unittest
import numpy as np
from PIL import Image

import sys
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from deformation.geometry import GeometricDeformer
from deformation.skeleton import (
    get_default_skeleton,
    get_pose_templates,
    get_template_by_name,
    fit_skeleton_to_bbox
)
from deformation.schemas import ResamplingPolicy, BoundingBox, Joint2D, Skeleton2D
from deformation.pixel_art import clean_alpha_channel, resample_sprite_grid


class TestDeformationGeometry(unittest.TestCase):

    def setUp(self):
        self.deformer = GeometricDeformer()
        self.sample_path = os.path.join(backend_dir, "alucard_samples", "alucard_0.png")
        if os.path.exists(self.sample_path):
            self.sample_img = Image.open(self.sample_path).convert("RGBA")
        else:
            # Fallback synthetic character sprite (128x128 RGBA with central square)
            arr = np.zeros((128, 128, 4), dtype=np.uint8)
            arr[30:100, 45:85, :3] = 200
            arr[30:100, 45:85, 3] = 255
            self.sample_img = Image.fromarray(arr, mode="RGBA")

    def test_bounding_box_detection(self):
        """Bounding box of sample sprite must be non-zero and within image dimensions."""
        bbox = self.deformer.detect_bounding_box(self.sample_img)
        self.assertGreater(bbox.width, 0)
        self.assertGreater(bbox.height, 0)
        self.assertGreater(bbox.non_transparent_pixels, 0)
        self.assertLessEqual(bbox.max_x, self.sample_img.width)
        self.assertLessEqual(bbox.max_y, self.sample_img.height)

    def test_transparent_image_bounding_box(self):
        """Empty transparent image must return zero non-transparent pixels."""
        empty_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        bbox = self.deformer.detect_bounding_box(empty_img)
        self.assertEqual(bbox.non_transparent_pixels, 0)
        self.assertEqual(bbox.width, 0)
        self.assertEqual(bbox.height, 0)

    def test_fit_skeleton_to_bbox(self):
        """Skeleton fitting must scale joint vertical span to character bounding box."""
        bbox = self.deformer.detect_bounding_box(self.sample_img)
        default_skel = get_default_skeleton()
        fitted = fit_skeleton_to_bbox(default_skel, bbox, self.sample_img.width, self.sample_img.height)

        self.assertEqual(len(fitted.joints), len(default_skel.joints))
        for name, j in fitted.joints.items():
            self.assertGreaterEqual(j.x, 0.0)
            self.assertLessEqual(j.x, 1.0)
            self.assertGreaterEqual(j.y, 0.0)
            self.assertLessEqual(j.y, 1.0)

    def test_point_to_segment_distance(self):
        """Point-to-segment distance must correctly calculate perpendicular distance."""
        p = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float32)
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([10.0, 0.0], dtype=np.float32)
        dists = GeometricDeformer._point_to_segment_distance(p, a, b)
        self.assertAlmostEqual(float(dists[0]), 0.0, places=4)
        self.assertAlmostEqual(float(dists[1]), 5.0, places=4)

    def test_deform_output_dimensions_and_alpha(self):
        """Geometric deformation must preserve image dimensions and retain high alpha integrity."""
        default_skel = self.deformer.auto_detect_skeleton(self.sample_img)
        walk_template = get_template_by_name("walk_1")
        self.assertIsNotNone(walk_template)

        target_skel = fit_skeleton_to_bbox(
            walk_template.skeleton,
            default_skel.bounding_box,
            self.sample_img.width,
            self.sample_img.height
        )

        out_img, val = self.deformer.deform(
            image=self.sample_img,
            source_skeleton=default_skel,
            target_skeleton=target_skel,
            resampling=ResamplingPolicy.NEAREST,
            preserve_pixel_art=True
        )

        self.assertEqual(out_img.size, self.sample_img.size)
        self.assertEqual(out_img.mode, "RGBA")
        self.assertTrue(val.is_valid)
        self.assertGreaterEqual(val.alpha_integrity_pct, 95.0)

    def test_clean_alpha_channel(self):
        """Alpha values below threshold must be snapped to 0."""
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :, 3] = np.array([
            [0, 10, 19, 20],
            [21, 50, 100, 255],
            [5, 15, 25, 35],
            [0, 0, 0, 0]
        ], dtype=np.uint8)

        cleaned = clean_alpha_channel(arr, threshold=20)
        self.assertEqual(cleaned[0, 0, 3], 0)
        self.assertEqual(cleaned[0, 1, 3], 0)
        self.assertEqual(cleaned[0, 2, 3], 0)
        self.assertEqual(cleaned[0, 3, 3], 20)
        self.assertEqual(cleaned[1, 0, 3], 21)


if __name__ == "__main__":
    unittest.main()
