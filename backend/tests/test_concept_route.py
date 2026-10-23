"""
Regression Tests for Authoritative /generate_concept Route
Verifies single authoritative route handling both JSON payload and Form data,
input validation, and ethical guardrail boundaries.
"""

import os
import unittest
from fastapi.testclient import TestClient

import sys
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app


class TestConceptRoute(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_json_payload_success(self):
        """Authoritative endpoint must accept JSON payload and return structured concept spec."""
        payload = {"prompt": "A retro pixel art knight with an enchanted sword"}
        res = self.client.post("/generate_concept", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("genre", data)
        self.assertIn("environment", data)
        self.assertIn("main_character", data)

    def test_form_data_success(self):
        """Authoritative endpoint must also accept legacy Form data transparently."""
        res = self.client.post("/generate_concept", data={"prompt": "Cyberpunk space explorer with plasma blaster"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("genre", data)
        self.assertIn("environment", data)

    def test_empty_prompt_error(self):
        """Empty prompt must return HTTP 400 validation error."""
        res = self.client.post("/generate_concept", json={"prompt": ""})
        self.assertEqual(res.status_code, 400)
        self.assertIn("required", res.json().get("detail", "").lower())

    def test_safety_guardrails_blocked(self):
        """Harmful or prohibited terms must be blocked by safety audit with HTTP 400."""
        res = self.client.post("/generate_concept", json={"prompt": "A gory explicit violent battle scene with gore"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("restricted or unsafe content", res.json().get("detail", ""))


if __name__ == "__main__":
    unittest.main()
