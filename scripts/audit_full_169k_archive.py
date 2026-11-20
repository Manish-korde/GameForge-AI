"""
Audit Full 169,094 Archive Entries for GameForge AI DCGAN Forensic Review
Audits all 169,094 verified character entries from evilsocket/alucard-sprites.
"""

import os
import sys
import json
import time
import io
import re
import hashlib
from collections import Counter, defaultdict
import pyarrow.ipc as ipc
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("STARTING COMPLETE AUDIT OF 169,094 ARCHIVE ENTRIES")
print("=" * 80)

# Paths
CACHE_DIR = os.path.expanduser('~/.cache/huggingface/datasets/evilsocket___alucard-sprites/default/0.0.0/c6edd10779c218dda9eb4aa2f1398d570219ff72')
INDICES_PATH = 'models/dcgan_all_characters_50ep_results/config/all_character_indices.json'
OUTPUT_DIR = 'phase_dcgan_improvement/forensic_review'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the 169,094 target indices
with open(INDICES_PATH, 'r') as f:
    target_indices = json.load(f)
target_set = set(target_indices)
print(f"Loaded target index list: {len(target_indices):,} indices")

# Regex Patterns
RE_FRONT = re.compile(r"\b(front view|facing front|front facing|canonical south)\b", re.IGNORECASE)
RE_BACK = re.compile(r"\b(back view|rear|behind|facing away|facing north)\b", re.IGNORECASE)
RE_SIDE = re.compile(r"\b(side view|profile|facing east|facing west|walk_east|walk_west)\b", re.IGNORECASE)
RE_34 = re.compile(r"\b(three-quarter|3/4|quarter view)\b", re.IGNORECASE)

# Archetypes
RE_WARRIOR = re.compile(r"\b(warrior|fighter|soldier|adventurer|hero)\b", re.IGNORECASE)
RE_KNIGHT = re.compile(r"\b(knight|paladin|armor|armored|plate)\b", re.IGNORECASE)
RE_MAGE = re.compile(r"\b(wizard|mage|spellcaster|sorcerer|necromancer|warlock|witch|cleric|priest)\b", re.IGNORECASE)
RE_ROGUE = re.compile(r"\b(rogue|thief|ninja|assassin|archer|ranger)\b", re.IGNORECASE)
RE_BRAWLER = re.compile(r"\b(boxer|brawler|monk|martial|bare-chested|shirtless)\b", re.IGNORECASE)
RE_MONSTER = re.compile(r"\b(skeleton|zombie|goblin|orc|demon|vampire|monster|beast|slime|ghost|creature)\b", re.IGNORECASE)

# Accessories
RE_SHIELD = re.compile(r"\b(shield|buckler|aegis)\b", re.IGNORECASE)
RE_SWORD = re.compile(r"\b(sword|blade|dagger|rapier|saber)\b", re.IGNORECASE)
RE_STAFF = re.compile(r"\b(staff|wand|rod|scepter|spear|halberd|polearm)\b", re.IGNORECASE)
RE_BOW = re.compile(r"\b(bow|crossbow|arrow)\b", re.IGNORECASE)
RE_CLOAK = re.compile(r"\b(cape|cloak|mantle|robe)\b", re.IGNORECASE)
RE_HELMET = re.compile(r"\b(helmet|helm|hood|hat|crown)\b", re.IGNORECASE)

def compute_bilateral_symmetry(arr_rgba):
    alpha = arr_rgba[:, :, 3] > 30
    if np.sum(alpha) < 150:
        return 0.0
    flipped_alpha = np.fliplr(alpha)
    intersection = np.logical_and(alpha, flipped_alpha).sum()
    union = np.logical_or(alpha, flipped_alpha).sum()
    return float(intersection) / float(union) if union > 0 else 0.0

def verify_front_face_presence(arr_rgba):
    head = arr_rgba[18:30, 24:40]
    head_a = head[:, :, 3] > 30
    if np.sum(head_a) < 20:
        return False, 0.0, "no_head"
    lum = 0.299 * head[:, :, 0] + 0.587 * head[:, :, 1] + 0.114 * head[:, :, 2]
    lum_vals = lum[head_a]
    contrast = float(lum_vals.max() - lum_vals.min())
    if contrast < 15.0:
        return False, contrast, "no_face_contrast"
    return True, contrast, "valid_face"

# Metrics accumulators
total_scanned = 0
orientation_counts = Counter()
archetype_counts = Counter()
accessory_counts = Counter()
invalid_images = 0
duplicates_count = 0
seen_hashes = set()

# Ordering bins (blocks of 20,000 entries across 169K)
bin_size = 20000
ordering_bins = defaultdict(lambda: Counter())

# Curation tracking
strictly_front_facing_old_filter = []     # (sym >= 0.78, no weapon tokens)
front_facing_improved_filter = []         # (sym >= 0.70, face verified, accessories allowed)
rejected_diagnostic_samples = []          # for contact sheet
accepted_diagnostic_samples = []          # for contact sheet

arrow_files = [
    'alucard-sprites-train-00000-of-00003.arrow',
    'alucard-sprites-train-00001-of-00003.arrow',
    'alucard-sprites-train-00002-of-00003.arrow'
]

global_idx = 0
t0 = time.time()

print("Reading and auditing archive stream...")

for fname in arrow_files:
    fpath = os.path.join(CACHE_DIR, fname)
    with open(fpath, 'rb') as f:
        reader = ipc.open_stream(f)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
                
            batch_texts = batch['text'].to_pylist()
            batch_imgs = batch['image']
            
            for local_i, cap_raw in enumerate(batch_texts):
                current_global_idx = global_idx
                global_idx += 1
                
                # Check if this index is one of the 169,094 target entries
                if current_global_idx not in target_set:
                    continue
                    
                total_scanned += 1
                cap = str(cap_raw).lower()
                
                # Bin for ordering analysis
                bin_idx = (total_scanned - 1) // bin_size
                bin_name = f"[{bin_idx * bin_size:,} - {min(total_scanned, (bin_idx + 1) * bin_size):,}]"
                
                # 1. Orientation Categorization
                if RE_FRONT.search(cap):
                    ori = "front_facing"
                elif RE_34.search(cap):
                    ori = "three_quarter"
                elif RE_SIDE.search(cap):
                    ori = "side_facing"
                elif RE_BACK.search(cap):
                    ori = "rear_facing"
                else:
                    ori = "ambiguous"
                orientation_counts[ori] += 1
                ordering_bins[bin_name][ori] += 1
                
                # 2. Archetype Categorization
                arch_found = False
                if RE_KNIGHT.search(cap):
                    archetype_counts["knight_armored"] += 1
                    ordering_bins[bin_name]["knight_armored"] += 1
                    arch_found = True
                if RE_WARRIOR.search(cap):
                    archetype_counts["warrior_adventurer"] += 1
                    ordering_bins[bin_name]["warrior_adventurer"] += 1
                    arch_found = True
                if RE_MAGE.search(cap):
                    archetype_counts["mage_spellcaster"] += 1
                    ordering_bins[bin_name]["mage_spellcaster"] += 1
                    arch_found = True
                if RE_ROGUE.search(cap):
                    archetype_counts["rogue_ranger"] += 1
                    ordering_bins[bin_name]["rogue_ranger"] += 1
                    arch_found = True
                if RE_BRAWLER.search(cap):
                    archetype_counts["brawler_monk"] += 1
                    ordering_bins[bin_name]["brawler_monk"] += 1
                    arch_found = True
                if RE_MONSTER.search(cap):
                    archetype_counts["monster_creature"] += 1
                    ordering_bins[bin_name]["monster_creature"] += 1
                    arch_found = True
                if not arch_found:
                    archetype_counts["other_general"] += 1
                    ordering_bins[bin_name]["other_general"] += 1
                    
                # 3. Accessory Categorization
                if RE_SHIELD.search(cap): accessory_counts["shield"] += 1
                if RE_SWORD.search(cap): accessory_counts["sword"] += 1
                if RE_STAFF.search(cap): accessory_counts["staff_spear"] += 1
                if RE_BOW.search(cap): accessory_counts["bow"] += 1
                if RE_CLOAK.search(cap): accessory_counts["cloak_cape"] += 1
                if RE_HELMET.search(cap): accessory_counts["helmet"] += 1
                
                # 4. Image Physical Checks for Front-Facing Candidates
                # To keep execution under 2 minutes, decode image bytes for front candidates + sample of others
                if ori == "front_facing" or (len(rejected_diagnostic_samples) < 50 and ori in ("rear_facing", "side_facing")):
                    try:
                        struct_val = batch_imgs[local_i].as_py()
                        img_bytes = struct_val['bytes']
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                        if img.size != (64, 64):
                            img = img.resize((64, 64), Image.Resampling.NEAREST)
                        arr = np.asarray(img, dtype=np.uint8)
                        
                        # Hash deduplication
                        h = hashlib.md5(arr.tobytes()).hexdigest()
                        if h in seen_hashes:
                            duplicates_count += 1
                            continue
                        seen_hashes.add(h)
                        
                        alpha = arr[:, :, 3]
                        # Transparency check
                        c1, c2, c3, c4 = np.mean(alpha[:8, :8]), np.mean(alpha[:8, -8:]), np.mean(alpha[-8:, :8]), np.mean(alpha[-8:, -8:])
                        if max(c1, c2, c3, c4) > 15.0 or np.mean(alpha[16:48, 16:48]) < 25.0:
                            invalid_images += 1
                            if len(rejected_diagnostic_samples) < 36:
                                rejected_diagnostic_samples.append((img, "Solid background", 0.0))
                            continue
                            
                        # Centroid check
                        y_idx, x_idx = np.where(alpha > 30)
                        if len(x_idx) == 0:
                            invalid_images += 1
                            continue
                        x_com = float(np.mean(x_idx))
                        if not (25.0 <= x_com <= 39.0):
                            if len(rejected_diagnostic_samples) < 36:
                                rejected_diagnostic_samples.append((img, f"Off-center (x={x_com:.1f})", 0.0))
                            continue
                            
                        sym = compute_bilateral_symmetry(arr)
                        face_ok, contrast, face_reason = verify_front_face_presence(arr)
                        
                        # Old strict filter check (sym >= 0.78, no weapon token)
                        has_weapon_token = bool(re.search(r"\b(weapon|sword|staff|spear|bow|dagger|shield)\b", cap))
                        if sym >= 0.78 and face_ok and not has_weapon_token:
                            strictly_front_facing_old_filter.append(current_global_idx)
                            
                        # Improved filter check (sym >= 0.70, face contrast verified, accessories allowed!)
                        if sym >= 0.70 and face_ok:
                            front_facing_improved_filter.append(current_global_idx)
                            if len(accepted_diagnostic_samples) < 36:
                                accepted_diagnostic_samples.append((img, f"Sym:{sym:.2f} ({cap[:15]})", sym))
                        else:
                            if len(rejected_diagnostic_samples) < 36:
                                if not face_ok:
                                    rejected_diagnostic_samples.append((img, f"Back-view ({face_reason[:10]})", sym))
                                else:
                                    rejected_diagnostic_samples.append((img, f"Low Sym ({sym:.2f})", sym))
                                    
                    except Exception as e:
                        invalid_images += 1
                        
                if total_scanned % 25000 == 0:
                    print(f"  Scanned {total_scanned:,}/169,094 entries ({time.time()-t0:.1f}s) | Front-facing found: {orientation_counts['front_facing']:,} | Improved yield: {len(front_facing_improved_filter):,}")

elapsed = time.time() - t0
print("=" * 80)
print(f"FULL ARCHIVE AUDIT COMPLETE IN {elapsed:.1f}s")
print(f"Total entries audited: {total_scanned:,}")
print("=" * 80)

# Build summary results dictionary
results = {
    "total_archive_entries": total_scanned,
    "orientation_distribution": dict(orientation_counts),
    "archetype_distribution": dict(archetype_counts),
    "accessory_distribution": dict(accessory_counts),
    "invalid_images_count": invalid_images,
    "near_duplicate_count": duplicates_count,
    "old_strict_curation_yield": len(strictly_front_facing_old_filter),
    "improved_curation_yield": len(front_facing_improved_filter),
    "ordering_distribution_by_bin": {k: dict(v) for k, v in ordering_bins.items()}
}

print("\n--- RESULTS SUMMARY ---")
print(f"Total Archive Entries Audited: {total_scanned:,}")
print(f"Orientation Counts: {dict(orientation_counts)}")
print(f"Archetype Counts: {dict(archetype_counts)}")
print(f"Accessory Counts: {dict(accessory_counts)}")
print(f"Old Strict Curation Total Yield across Full Archive: {len(strictly_front_facing_old_filter):,}")
print(f"Improved Curation Total Yield across Full Archive  : {len(front_facing_improved_filter):,}")

# Save metrics JSON
with open(os.path.join(OUTPUT_DIR, 'full_archive_audit_metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Generate Accepted Contact Sheet
print("Rendering Accepted Samples Contact Sheet...")
fig, axes = plt.subplots(6, 6, figsize=(12, 12), facecolor="#0f172a")
for idx, ax in enumerate(axes.flatten()):
    if idx < len(accepted_diagnostic_samples):
        img, label, sym = accepted_diagnostic_samples[idx]
        ax.imshow(img)
        ax.set_title(label, color="#34d399", fontsize=7, pad=2)
    ax.axis("off")
plt.suptitle(f"Full Archive Audit: Accepted Front-Facing Sprites (N={len(front_facing_improved_filter):,})", color="#f8fafc", fontsize=14, y=0.98, fontweight="bold")
plt.tight_layout()
accepted_sheet_path = os.path.join(OUTPUT_DIR, 'contact_sheet_accepted_full_archive.png')
plt.savefig(accepted_sheet_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved: {accepted_sheet_path}")

# Generate Rejected Contact Sheet
print("Rendering Rejected Samples Contact Sheet...")
fig, axes = plt.subplots(6, 6, figsize=(12, 12), facecolor="#0f172a")
for idx, ax in enumerate(axes.flatten()):
    if idx < len(rejected_diagnostic_samples):
        img, label, sym = rejected_diagnostic_samples[idx]
        ax.imshow(img)
        ax.set_title(label, color="#f87171", fontsize=7, pad=2)
    ax.axis("off")
plt.suptitle("Full Archive Audit: Rejected Samples (Orientation, High-Sym Back-Views, Accessories)", color="#f8fafc", fontsize=14, y=0.98, fontweight="bold")
plt.tight_layout()
rejected_sheet_path = os.path.join(OUTPUT_DIR, 'contact_sheet_rejected_full_archive.png')
plt.savefig(rejected_sheet_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved: {rejected_sheet_path}")

print("AUDIT EXECUTION COMPLETE!")

