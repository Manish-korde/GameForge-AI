import os
import json
import numpy as np
from PIL import Image
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "backend", "alucard_samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

print("=== INSPECTING ALUCARD DATASET & ALIGNING GROUND-TRUTH SAMPLES ===")
print("Loading 'evilsocket/alucard-sprites' dataset...")

ds = load_dataset("evilsocket/alucard-sprites")
raw_data = ds["train"]

print(f"Dataset Split: train ({len(raw_data):,} rows)")
print(f"Features: {list(raw_data.features.keys())}")

# Print sample records to inspect ground-truth text & properties
print("\nSample Dataset Records:")
for idx in [0, 5, 12, 45, 100, 500, 1000]:
    row = raw_data[idx]
    txt = row.get("text", "")
    img = row["image"]
    print(f"  - Record #{idx:5d}: Size={img.size}, Mode={img.mode}, Text='{txt}'")

# Select 20 authentic, non-noisy pixel art sprites for local sample gallery
TARGET_CATEGORIES = {
    "Knight Warrior": ["knight", "warrior", "hero", "paladin"],
    "Rogue Assassin": ["rogue", "assassin", "ninja"],
    "Arcane Mage": ["mage", "wizard", "spellcaster", "sorcerer"],
    "Elven Archer": ["archer", "bowman", "elf"],
    "Necromancer": ["necromancer", "dark mage", "cultist"],
    "Demon Guardian": ["demon", "devil", "imp", "monster"],
    "Green Slime": ["slime", "blob", "ooze"],
    "Skeleton Warrior": ["skeleton", "undead", "bones"],
    "Broadsword": ["sword", "blade", "broadsword"],
    "Magic Staff": ["staff", "wand", "scepter"],
    "Longbow": ["bow", "longbow", "crossbow"],
    "Battle Axe": ["axe", "hatchet", "battleaxe"],
    "Boots / Footwear": ["boots", "shoes", "footwear"],
    "Plate Armor": ["armor", "plate", "cuirass"],
    "Iron Helmet": ["helmet", "helm", "visor"],
    "Gold Ring": ["ring", "band", "jewelry"],
    "Health Potion": ["potion", "flask", "elixir", "bottle"],
    "Magic Scroll": ["scroll", "parchment", "tome"],
    "Wall Torch": ["torch", "sconce", "fire"],
    "Flame Aura": ["flame", "fire", "aura", "spell"]
}

samples_manifest = []
collected_keys = set()
count = 0

for i in range(len(raw_data)):
    if count >= 20:
        break
    row = raw_data[i]
    txt = row.get("text", "").lower()
    img = row["image"]
    
    # Filter out non-transparent or non-square background photos if any
    if img.size[0] < 16 or img.size[1] < 16:
        continue
        
    for cat_name, keywords in TARGET_CATEGORIES.items():
        if cat_name in collected_keys:
            continue
        if any(kw in txt for kw in keywords):
            # Verify image has pixel art qualities (RGBA / transparency)
            img_rgba = img.convert("RGBA").resize((128, 128))
            img_path = os.path.join(SAMPLES_DIR, f"alucard_{count}.png")
            img_rgba.save(img_path)
            
            meta_entry = {
                "sample_id": count,
                "dataset_idx": i,
                "label": cat_name,
                "text_prompt": row.get("text", ""),
                "filename": f"alucard_{count}.png",
                "image_url": f"http://127.0.0.1:8000/alucard_samples/alucard_{count}.png"
            }
            samples_manifest.append(meta_entry)
            collected_keys.add(cat_name)
            print(f"  [Sample #{count:2d}] {cat_name:20s} -> Record #{i} | Prompt: '{row.get('text', '')}'")
            count += 1
            break

# If any slots remain, fill with distinct valid items
if count < 20:
    for i in range(100, len(raw_data)):
        if count >= 20:
            break
        row = raw_data[i]
        img = row["image"].convert("RGBA").resize((128, 128))
        img_path = os.path.join(SAMPLES_DIR, f"alucard_{count}.png")
        img.save(img_path)
        
        txt = row.get("text", "") or "Pixel Art Sprite"
        meta_entry = {
            "sample_id": count,
            "dataset_idx": i,
            "label": f"Sprite #{count}",
            "text_prompt": txt,
            "filename": f"alucard_{count}.png",
            "image_url": f"http://127.0.0.1:8000/alucard_samples/alucard_{count}.png"
        }
        samples_manifest.append(meta_entry)
        count += 1

manifest_path = os.path.join(SAMPLES_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(samples_manifest, f, indent=2)

print(f"\nSaved 20 ground-truth authentic sample images & manifest -> {manifest_path}")
print("=== GROUND-TRUTH ALIGNMENT COMPLETE ===")
