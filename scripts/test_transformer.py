import sys
import os
import json
import time

# Add backend directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, BACKEND_DIR)

from transformer_service import transformer_service

print("=== TRANSFORMER SEMANTIC PLANNING VERIFICATION ===")

# Test 1: Unit Verification of TransformerService
print("\n1. Testing TransformerService model loading and inference...")
start_time = time.time()
transformer_service.load_model()
elapsed = time.time() - start_time
print(f"   Model load status: is_loaded={transformer_service.is_loaded} (Time: {elapsed:.2f}s)")

test_prompts = [
    "Retro 16-bit dark fantasy RPG about a rogue exploring a sunken temple with acid traps and giant serpent bosses",
    "Cyberpunk action platformer featuring a neon hacker battling rogue AI drones in a rain-slicked metropolis",
    "Chibi dungeon crawler with a wizard casting flame spells against slime monsters in crystal catacombs"
]

print("\n2. Testing natural language prompt parsing to structured JSON specs...")
for i, prompt in enumerate(test_prompts, 1):
    t0 = time.time()
    spec = transformer_service.generate_concept_spec(prompt)
    t_gen = time.time() - t0
    
    print(f"\n--- Test Prompt #{i} ({t_gen*1000:.1f}ms) ---")
    print(f"Prompt: '{prompt}'")
    print(f"Parsed Title: {spec.get('game_title')}")
    print(f"Genre / Style: {spec.get('genre')} | {spec.get('art_style')}")
    print(f"Main Character: {spec.get('main_character')}")
    print(f"Environment: {spec.get('environment')}")
    print(f"Recommended Tags: {spec.get('recommended_asset_tags')}")
    
    # Assertions
    assert "game_title" in spec, "Missing game_title"
    assert "genre" in spec, "Missing genre"
    assert "main_character" in spec, "Missing main_character"
    assert "recommended_asset_tags" in spec, "Missing recommended_asset_tags"
    assert len(spec["recommended_asset_tags"]) >= 3, "Expected at least 3 recommended asset tags"

print("\nVerification Successful! All Transformer tests passed cleanly.")
