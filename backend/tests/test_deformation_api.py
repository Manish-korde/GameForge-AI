"""
Integration Tests for FastAPI Character Pose Deformation Endpoints
Validates /deformation/templates, /deformation/detect_skeleton, and /deformation/pose
including contract serialization and error handling.
"""

import os
import unittest
from fastapi.testclient import TestClient

import sys
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app


class TestDeformationAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_url = "/alucard_samples/alucard_0.png"

    def test_get_templates_endpoint(self):
        """GET /deformation/templates must return HTTP 200 and 7 tested templates."""
        res = self.client.get("/deformation/templates")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("count"), 7)
        self.assertEqual(len(data.get("templates", [])), 7)

    def test_detect_skeleton_endpoint(self):
        """POST /deformation/detect_skeleton must compute bounding box and align skeleton."""
        res = self.client.post("/deformation/detect_skeleton", json={"image_url": self.sample_url})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("bounding_box", data)
        self.assertIn("skeleton", data)
        # Verify JSON dictionary contract for joints
        self.assertIsInstance(data["skeleton"]["joints"], dict)
        self.assertIn("head", data["skeleton"]["joints"])

    def test_deform_pose_named_template(self):
        """POST /deformation/pose with valid template must return deformed PNG data URI and QA audit."""
        payload = {
            "image_url": self.sample_url,
            "target_pose": "walk_1",
            "method": "geometric",
            "intensity": 1.0,
            "preserve_pixel_art": True
        }
        res = self.client.post("/deformation/pose", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "success")
        self.assertTrue(data.get("output_image_b64", "").startswith("data:image/png;base64,"))
        self.assertIn("validation", data)
        self.assertTrue(data["validation"].get("is_valid"))
        self.assertGreaterEqual(data["validation"].get("alpha_integrity_pct", 0), 95.0)

        # Critical contract check: joints must be serialized as a Dict
        self.assertIsInstance(data["target_skeleton"]["joints"], dict)
        self.assertIsInstance(data["source_skeleton"]["joints"], dict)

    def test_deform_pose_unknown_template_error(self):
        """POST /deformation/pose with unknown template must return HTTP 400 error."""
        payload = {
            "image_url": self.sample_url,
            "target_pose": "non_existent_pose_xyz"
        }
        res = self.client.post("/deformation/pose", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unknown target pose template", res.json().get("detail", ""))

    def test_deform_pose_missing_image_payload(self):
        """POST /deformation/pose without image_url or image_b64 must return HTTP 400."""
        res = self.client.post("/deformation/pose", json={"target_pose": "idle"})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
