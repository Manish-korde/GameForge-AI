import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

OUTPUT_DIR = "research_paper_assets/06_sprite_deformation_and_pose_animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONTACT_SHEET_PATH = os.path.join(OUTPUT_DIR, "rung0_dataset_alignment_audit.png")

print("Initializing Rung 0: Streaming 20 characters from carlosuperb/lpc-action-pixel-art-diffusion...")
ds = load_dataset("carlosuperb/lpc-action-pixel-art-diffusion", split="train", streaming=True)

# Grid Layout: 576 x 768 -> 9 columns x 12 rows (each tile = 64x64)
# Row 2: Walk South (Front) -> Col 0: Idle / Stance, Col 2: Mid-stride Walk
# Row 5: Slash South (Front) -> Col 2: Slash swing apex
COLS = 9
ROWS = 12
TILE_W = 64
TILE_H = 64

characters_data = []
count = 0

for item in ds:
    sheet = item['image'].convert('RGBA')
    w, h = sheet.size
    
    # Verify dimensions
    cw = w // COLS
    ch = h // ROWS
    if cw != 64 or ch != 64:
        # Resize to standard if needed
        sheet = sheet.resize((576, 768), Image.Resampling.NEAREST)
        cw, ch = 64, 64
        
    # Crop Frame 0 (Walk South Col 0 - Idle Ref)
    ref_crop = sheet.crop((0 * cw, 2 * ch, 1 * cw, 3 * ch))
    # Crop Frame 2 (Walk South Col 2 - Walk Stride)
    walk_crop = sheet.crop((2 * cw, 2 * ch, 3 * cw, 3 * ch))
    # Crop Frame 2 of Slash South (Row 5, Col 2 - Slash Swing)
    slash_crop = sheet.crop((2 * cw, 5 * ch, 3 * cw, 6 * ch))
    
    ref_arr = np.array(ref_crop)
    walk_arr = np.array(walk_crop)
    slash_arr = np.array(slash_crop)
    
    ref_alpha = ref_arr[:, :, 3]
    walk_alpha = walk_arr[:, :, 3]
    slash_alpha = slash_arr[:, :, 3]
    
    # Check if sprite is empty
    if ref_alpha.sum() < 100 or walk_alpha.sum() < 100:
        continue
        
    # Metric 1: Footline Delta (Vertical)
    ref_foot_y = np.max(np.where(ref_alpha > 50)[0]) if np.any(ref_alpha > 50) else 0
    walk_foot_y = np.max(np.where(walk_alpha > 50)[0]) if np.any(walk_alpha > 50) else 0
    slash_foot_y = np.max(np.where(slash_alpha > 50)[0]) if np.any(slash_alpha > 50) else 0
    foot_delta_walk = abs(int(walk_foot_y) - int(ref_foot_y))
    
    # Metric 2: Horizontal Mass Centroid Delta
    def get_centroid_x(alpha):
        xs = np.where(alpha > 50)[1]
        return np.mean(xs) if len(xs) > 0 else 32.0
    
    ref_cx = get_centroid_x(ref_alpha)
    walk_cx = get_centroid_x(walk_alpha)
    centroid_delta_walk = abs(walk_cx - ref_cx)
    
    # Metric 3: Silhouette IoU between Ref and Walk
    ref_mask = (ref_alpha > 50)
    walk_mask = (walk_alpha > 50)
    intersection = np.logical_and(ref_mask, walk_mask).sum()
    union = np.logical_or(ref_mask, walk_mask).sum()
    sil_iou = float(intersection) / float(union) if union > 0 else 0.0
    
    # Metric 4: Torso Color Match (Upper torso: y in [18, 32], x in [24, 40])
    torso_ref = ref_arr[18:32, 24:40, :3].mean(axis=(0, 1))
    torso_walk = walk_arr[18:32, 24:40, :3].mean(axis=(0, 1))
    torso_color_dist = float(np.linalg.norm(torso_ref - torso_walk))
    
    characters_data.append({
        'idx': count,
        'ref': ref_crop,
        'walk': walk_crop,
        'slash': slash_crop,
        'foot_delta': foot_delta_walk,
        'centroid_delta': centroid_delta_walk,
        'sil_iou': sil_iou,
        'color_dist': torso_color_dist
    })
    
    count += 1
    if count >= 20:
        break

print(f"Audit completed across {len(characters_data)} characters. Generating visual grid...")

# Build Visual Grid: 20 rows x 3 columns [Reference Idle | Walk Stride | Slash Swing]
fig, axes = plt.subplots(10, 6, figsize=(14, 20), dpi=150)
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for i in range(20):
    row_block = i // 2
    col_block = (i % 2) * 3
    
    c = characters_data[i]
    ax_ref = axes[row_block, col_block]
    ax_walk = axes[row_block, col_block + 1]
    ax_slash = axes[row_block, col_block + 2]
    
    ax_ref.imshow(c['ref'])
    ax_ref.set_title(f"C{i} Ref (Idle)", fontsize=7)
    ax_ref.axis('off')
    
    ax_walk.imshow(c['walk'])
    ax_walk.set_title(f"Walk (Δy:{c['foot_delta']} IoU:{c['sil_iou']:.2f})", fontsize=7)
    ax_walk.axis('off')
    
    ax_slash.imshow(c['slash'])
    ax_slash.set_title("Slash", fontsize=7)
    ax_slash.axis('off')

plt.suptitle("Rung 0: Dataset Correspondence & 5-Way Spatial Alignment Audit (20 Characters)", fontsize=13, y=0.99)
plt.savefig(CONTACT_SHEET_PATH, bbox_inches='tight')
plt.close()

# Print summary table (ASCII safe)
print("=" * 70)
print(f"{'Char':<6} | {'Foot Dy (px)':<14} | {'Centroid Dx (px)':<18} | {'Sil IoU':<10} | {'Torso Color D':<12}")
print("-" * 70)
foot_deltas = []
centroid_deltas = []
sil_ious = []
color_dists = []

for c in characters_data:
    print(f"C_{c['idx']:<4} | {c['foot_delta']:<14} | {c['centroid_delta']:<18.2f} | {c['sil_iou']:<10.2f} | {c['color_dist']:<12.2f}")
    foot_deltas.append(c['foot_delta'])
    centroid_deltas.append(c['centroid_delta'])
    sil_ious.append(c['sil_iou'])
    color_dists.append(c['color_dist'])

print("=" * 70)
print(f"Mean Footline Dy:       {np.mean(foot_deltas):.2f} px (Rule: <= 3 px)")
print(f"Mean Centroid Dx:       {np.mean(centroid_deltas):.2f} px (Rule: <= 4 px)")
print(f"Mean Silhouette IoU:    {np.mean(sil_ious):.2f} (Rule: >= 0.45)")
print(f"Mean Torso Color D:     {np.mean(color_dists):.2f} (Rule: Low indicates same identity)")
print(f"Audit sheet saved to: {CONTACT_SHEET_PATH}")

