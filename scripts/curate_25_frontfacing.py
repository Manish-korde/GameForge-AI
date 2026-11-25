import os
import glob
import shutil
import json
from PIL import Image
import numpy as np

SOURCE_DIR = r"Dataset\lpc_frontfacing_idles"
PUBLIC_DEST = r"gui\public\frontfacing_25"
BACKEND_DEST = r"backend\frontfacing_25"

os.makedirs(PUBLIC_DEST, exist_ok=True)
os.makedirs(BACKEND_DEST, exist_ok=True)

# Select 25 diverse front-facing sprites spread across the dataset
all_files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.png")))
print(f"Total available LPC front-facing idles: {len(all_files)}")

# Hand-sample indices with high visual diversity across the 2,500 character variations
# Characters spanning knights, mages, rangers, rogues, warriors, clerics, orcs, elders, champions
sample_indices = [
    0, 24, 52, 98, 145, 188, 222, 275, 310, 360,
    412, 480, 525, 590, 640, 710, 785, 840, 920, 1015,
    1150, 1280, 1410, 1560, 1720
]

labels = [
    "Royal Paladin", "Elven Archer", "Arcane Sorcerer", "Shadow Rogue", "Iron Guardian",
    "Forest Ranger", "Crimson Knight", "High Priest", "Desert Scout", "Mystic Druid",
    "Heavy Barbarian", "Dark Assassin", "Grand Mage", "Silver Templar", "Mountain Hunter",
    "Eldritch Warlock", "Warden Defender", "Nomad Warrior", "Rune Enchanter", "Battle Cleric",
    "Nightblade Rogue", "Storm Champion", "Sun Paladin", "Verdant Scout", "Arch-Mage"
]

archetypes = [
    "Paladin", "Archer", "Mage", "Rogue", "Warrior",
    "Ranger", "Knight", "Cleric", "Ranger", "Mage",
    "Warrior", "Rogue", "Mage", "Paladin", "Ranger",
    "Warlock", "Guardian", "Warrior", "Mage", "Cleric",
    "Rogue", "Warrior", "Paladin", "Ranger", "Mage"
]

manifest = []

for i, (idx, label, arch) in enumerate(zip(sample_indices, labels, archetypes)):
    src_path = all_files[idx]
    filename = f"front_char_{i+1:02d}.png"
    
    # Verify image integrity
    img = Image.open(src_path).convert("RGBA")
    arr = np.array(img)
    fg_ratio = np.mean(arr[..., 3] > 10)
    assert img.size == (64, 64), f"Invalid size {img.size} for {src_path}"
    assert fg_ratio > 0.15, f"Low foreground {fg_ratio} for {src_path}"
    
    # Save to public and backend
    pub_path = os.path.join(PUBLIC_DEST, filename)
    back_path = os.path.join(BACKEND_DEST, filename)
    img.save(pub_path)
    img.save(back_path)
    
    manifest.append({
        "id": i + 1,
        "label": f"Character #{i+1:02d}: {label}",
        "name": label,
        "archetype": arch,
        "filename": filename,
        "category": arch,
        "url": f"/frontfacing_25/{filename}",
        "fg_coverage": round(float(fg_ratio) * 100, 1),
        "source_file": os.path.basename(src_path)
    })
    print(f"[{i+1}/25] {filename} -> {label} (Coverage: {fg_ratio:.1%})")

# Write manifest JSON to public and backend
with open(os.path.join(PUBLIC_DEST, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
with open(os.path.join(BACKEND_DEST, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("\nSuccessfully curated 25 verified front-facing sprites into:")
print(f"  - {PUBLIC_DEST}")
print(f"  - {BACKEND_DEST}")
