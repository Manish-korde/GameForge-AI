"""
GameForge AI — Animation & Spritesheet Regression Tests
Validates keyframe interpolation, preset generation, multi-frame rendering,
transparency integrity, spritesheet packaging, and FastAPI endpoint contracts.
"""

import os
import unittest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from deformation.schemas import (
    Skeleton2D,
    Joint2D,
    BoundingBox,
    AnimationRequest,
    SpritesheetRequest
)
from deformation.skeleton import (
    get_default_skeleton,
    get_template_by_name,
    JOINT_NAMES
)
from deformation.animation import (
    interpolate_skeletons,
    get_animation_presets,
    get_animation_preset_by_id,
    calculate_spritesheet_grid,
    assemble_spritesheet,
    animation_service
)
from app import app


class TestDeformationAnimation(unittest.TestCase):
    """Test suite for character animation generation, interpolation, and spritesheets."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "alucard_samples", "alucard_0.png")
        )

    def test_interpolate_identical_poses(self):
        """Interpolating between two identical poses must yield the exact same joint coordinates."""
        skel = get_default_skeleton()
        interp = interpolate_skeletons(skel, skel, t=0.5)

        for name in JOINT_NAMES:
            orig = skel.get_joint(name)
            res = interp.get_joint(name)
            self.assertIsNotNone(res)
            self.assertAlmostEqual(orig.x, res.x, places=3)
            self.assertAlmostEqual(orig.y, res.y, places=3)

    def test_interpolate_distinct_poses_midpoint(self):
        """Interpolating at t=0.5 with linear factor must yield the algebraic midpoint."""
        skel_a = get_default_skeleton()
        # Create skel_b with head shifted from (0.50, 0.18) to (0.60, 0.28)
        skel_b = skel_a.with_updated_joint("head", 0.60, 0.28)

        # Non-smoothstep (linear) interpolation at t=0.5
        interp = interpolate_skeletons(skel_a, skel_b, t=0.5, use_smoothstep=False)
        head = interp.get_joint("head")
        self.assertAlmostEqual(head.x, 0.55, places=3)
        self.assertAlmostEqual(head.y, 0.23, places=3)

    def test_interpolation_clamps_coordinates(self):
        """Joints must remain strictly within [0.01, 0.99] bounds during extreme interpolation."""
        skel_a = get_default_skeleton()
        skel_b = skel_a.with_updated_joint("head", 0.9999, 0.0001)

        interp = interpolate_skeletons(skel_a, skel_b, t=1.0)
        head = interp.get_joint("head")
        self.assertLessEqual(head.x, 0.99)
        self.assertGreaterEqual(head.y, 0.01)

    def test_presets_catalog_completeness(self):
        """All 5 canonical presets (idle, walk, run, jump, attack) must be defined with valid metadata."""
        presets = get_animation_presets()
        preset_ids = {p.id for p in presets}
        expected = {"idle", "walk", "run", "jump", "attack"}
        self.assertTrue(expected.issubset(preset_ids))

        for p in presets:
            self.assertGreaterEqual(p.default_frame_count, 2)
            self.assertGreaterEqual(p.default_fps, 1)
            self.assertGreaterEqual(len(p.keyframe_template_ids), 2)

    def test_invalid_frame_count_rejected(self):
        """Requests with frame_count < 2 or > 32 must be rejected by Pydantic validation."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            AnimationRequest(
                image_url="alucard_0.png",
                preset_id="walk",
                frame_count=1
            )

        # Also verify API returns HTTP 422 Unprocessable Entity
        res = self.client.post("/deformation/animate", json={
            "image_url": "alucard_0.png",
            "preset_id": "walk",
            "frame_count": 1
        })
        self.assertEqual(res.status_code, 422)

    def test_generate_animation_walk_cycle(self):
        """Generating a 4-frame walk cycle must return 4 valid frames with uniform dimensions and high alpha."""
        req = AnimationRequest(
            image_url="alucard_0.png",
            preset_id="walk",
            frame_count=4,
            fps=8,
            is_looping=True
        )
        res = animation_service.generate_animation(req)

        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.frames), 4)
        self.assertEqual(res.frame_count, 4)
        self.assertEqual(res.fps, 8)
        self.assertGreater(res.duration_ms, 0)
        self.assertGreaterEqual(res.overall_validation.alpha_integrity_pct, 95.0)

        # Verify dimension uniformity
        first_w, first_h = res.frames[0].width, res.frames[0].height
        for f in res.frames:
            self.assertEqual(f.width, first_w)
            self.assertEqual(f.height, first_h)
            self.assertTrue(f.image_b64.startswith("data:image/png;base64,"))

    def test_spritesheet_grid_calculation(self):
        """Spritesheet grid calculation must produce expected columns and rows."""
        # <= 4 frames: single row
        c, r = calculate_spritesheet_grid(4)
        self.assertEqual((c, r), (4, 1))

        # 8 frames: 4 cols x 2 rows
        c, r = calculate_spritesheet_grid(8)
        self.assertEqual((c, r), (4, 2))

        # 5 frames: 4 cols x 2 rows (incomplete final row)
        c, r = calculate_spritesheet_grid(5)
        self.assertEqual((c, r), (4, 2))

        # Explicit columns override
        c, r = calculate_spritesheet_grid(6, columns=3)
        self.assertEqual((c, r), (3, 2))

    def test_spritesheet_assembly_and_transparency(self):
        """Assembling frames into a spritesheet must preserve transparent padding in incomplete rows."""
        # Create 5 simple 32x32 RGBA frames
        frames = [
            Image.new("RGBA", (32, 32), (255, 0, 0, 255))
            for _ in range(5)
        ]

        sheet, meta = assemble_spritesheet(frames, columns=4, fps=12)

        self.assertEqual(meta.columns, 4)
        self.assertEqual(meta.rows, 2)
        self.assertEqual(meta.total_width, 4 * 32)
        self.assertEqual(meta.total_height, 2 * 32)
        self.assertEqual(meta.frame_count, 5)

        # Incomplete row check: cell (col 3, row 1) must be completely transparent (alpha = 0)
        arr = np.array(sheet)
        # Pixel at x = 3.5 * 32 = 112, y = 1.5 * 32 = 48
        empty_pixel_alpha = arr[48, 112, 3]
        self.assertEqual(empty_pixel_alpha, 0)

    def test_api_animation_presets_endpoint(self):
        """GET /deformation/animation/presets must return 200 with 5 presets."""
        res = self.client.get("/deformation/animation/presets")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertGreaterEqual(data["count"], 5)

    def test_api_animate_endpoint(self):
        """POST /deformation/animate must generate animation and return 200."""
        payload = {
            "image_url": "alucard_0.png",
            "preset_id": "idle",
            "frame_count": 3,
            "fps": 6
        }
        res = self.client.post("/deformation/animate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["frame_count"], 3)
        self.assertEqual(len(data["frames"]), 3)

    def test_api_spritesheet_endpoint(self):
        """POST /deformation/spritesheet must assemble spritesheet and return 200 with layout metadata."""
        payload = {
            "image_url": "alucard_0.png",
            "preset_id": "walk",
            "frame_count": 4,
            "columns": 2
        }
        res = self.client.post("/deformation/spritesheet", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["spritesheet_image_b64"].startswith("data:image/png;base64,"))
        meta = data["metadata"]
        self.assertEqual(meta["columns"], 2)
        self.assertEqual(meta["rows"], 2)
        self.assertEqual(meta["frame_count"], 4)


if __name__ == "__main__":
    unittest.main()
