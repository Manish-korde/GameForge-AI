import sys
import os
import unittest

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from transformer_service import TransformerService

class TestEthicalGuardrails(unittest.TestCase):
    def setUp(self):
        self.service = TransformerService()

    def test_safe_prompt(self):
        prompt = "Retro 16-bit dark fantasy RPG with a knight fighting a dragon"
        is_safe, msg = self.service.validate_prompt_safety(prompt)
        self.assertTrue(is_safe)
        self.assertEqual(msg, "Passed Safety Audit")

    def test_unsafe_nsfw_prompt(self):
        prompt = "NSFW character with explicit gore and decapitation"
        is_safe, msg = self.service.validate_prompt_safety(prompt)
        self.assertFalse(is_safe)
        self.assertIn("restricted or unsafe content", msg)

    def test_protected_ip_warning(self):
        prompt = "Create a Mario pixel sprite in Pokemon style"
        is_safe, msg = self.service.validate_prompt_safety(prompt)
        # Should pass safety audit but generate IP warnings in logs
        self.assertTrue(is_safe)

    def test_concept_generation_with_guardrails(self):
        prompt = "Chibi mage casting ice spells in a crystal cave"
        spec = self.service.generate_concept_spec(prompt)
        self.assertIn("ethical_guardrails", spec)
        self.assertEqual(spec["ethical_guardrails"]["license"], "CC-BY-NC-SA 4.0")
        self.assertEqual(spec["ethical_guardrails"]["content_safety"], "Passed Safety Audit")

if __name__ == "__main__":
    unittest.main()
