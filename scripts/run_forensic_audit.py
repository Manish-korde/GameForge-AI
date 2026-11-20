import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

OUTPUT_DIR = "research_paper_assets/06_sprite_deformation_and_pose_animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "rung0_forensic_metrics_report.csv")
CONTACT_SHEET_5FRAME = os.path.join(OUTPUT_DIR, "rung0_walk_5frame_verification.png")
CONTACT_SHEET_MASTER = os.path.join(OUTPUT_DIR, "rung0_corrected_master_audit.png")

print("Initializing Rung 0 Forensic Audit on 20 characters...")
ds = load_dataset("carlosuperb/lpc-action-pixel-art-diffusion", split="train", streaming=True)

# Grid Layout: 576 x 768 -> 9 columns x 12 rows (64x64 tiles)
# Row 2: Walk South (Front)
#   - Col 0: Idle Stance
#   - Col 1: Walk Prep
#   - Col 2: Walk Contact Step (Subtle stride)
#   - Col 3: Walk Full Stride Apex (Maximum leg separation)
# Row 6: Slash South (Front)
#   - Col 0: Windup Stance
#   - Col 3: Arm Extension Apex
#   - Col 4: Full Weapon Strike / Swing Arc

records = []
characters_data = []
count = 0

for item in ds:
    sheet = item['image'].convert('RGBA')
    w, h = sheet.size
    if w != 576 or h != 768:
        sheet = sheet.resize((576, 768), Image.Resampling.NEAREST)
        
    # Extract frames
    idle_im = sheet.crop((0 * 64, 2 * 64, 1 * 64, 3 * 64))
    walk_f0 = sheet.crop((0 * 64, 2 * 64, 1 * 64, 3 * 64))
    walk_f1 = sheet.crop((1 * 64, 2 * 64, 2 * 64, 3 * 64))
    walk_f2 = sheet.crop((2 * 64, 2 * 64, 3 * 64, 3 * 64))
    walk_f3 = sheet.crop((3 * 64, 2 * 64, 4 * 64, 3 * 64))
    slash_south = sheet.crop((3 * 64, 6 * 64, 4 * 64, 7 * 64)) # Row 6 Col 3: Front Slash Apex
    
    # Check emptiness
    if np.array(idle_im)[:, :, 3].sum() < 500:
        continue
        
    idle_arr = np.array(idle_im)
    w2_arr = np.array(walk_f2)
    w3_arr = np.array(walk_f3)
    slash_arr = np.array(slash_south)
    
    idle_alpha = idle_arr[:, :, 3]
    w2_alpha = w2_arr[:, :, 3]
    w3_alpha = w3_arr[:, :, 3]
    slash_alpha = slash_arr[:, :, 3]
    
    # Total Foreground Pixels in Idle
    fg_pixels = int(np.sum(idle_alpha > 20))
    
    # Helper to calculate alpha-aware metrics
    def compute_motion_stats(ref_arr, tgt_arr):
        ref_a = ref_arr[:, :, 3]
        tgt_a = tgt_arr[:, :, 3]
        
        # Valid union mask where sprite exists in either
        union_fg = (ref_a > 20) | (tgt_a > 20)
        
        # Changed pixel definition:
        # Either opacity changed significantly, OR within overlapping foreground RGB changed
        rgb_diff = np.abs(tgt_arr[:, :, :3].astype(int) - ref_arr[:, :, :3].astype(int)).max(axis=2)
        alpha_diff = np.abs(tgt_a.astype(int) - ref_a.astype(int))
        
        changed_mask = union_fg & ((rgb_diff > 30) | (alpha_diff > 40))
        changed_count = int(np.sum(changed_mask))
        changed_pct = float(changed_count) / float(fg_pixels) * 100.0 if fg_pixels > 0 else 0.0
        
        # Alpha-mask IoU
        ref_bin = (ref_a > 40)
        tgt_bin = (tgt_a > 40)
        intersection = np.sum(ref_bin & tgt_bin)
        union = np.sum(ref_bin | tgt_bin)
        alpha_iou = float(intersection) / float(union) if union > 0 else 0.0
        
        # Leg regions: y in [42, 63]
        # Left leg: x in [16, 31], Right leg: x in [32, 47]
        left_leg_changed = int(np.sum(changed_mask[42:64, 16:32]))
        right_leg_changed = int(np.sum(changed_mask[42:64, 32:48]))
        
        # Bounding box displacement
        ref_bbox = Image.fromarray(ref_arr).getbbox() or (0, 0, 0, 0)
        tgt_bbox = Image.fromarray(tgt_arr).getbbox() or (0, 0, 0, 0)
        dx_min = tgt_bbox[0] - ref_bbox[0]
        dy_min = tgt_bbox[1] - ref_bbox[1]
        dx_max = tgt_bbox[2] - ref_bbox[2]
        dy_max = tgt_bbox[3] - ref_bbox[3]
        bbox_disp = f"({dx_min:+d},{dy_min:+d},{dx_max:+d},{dy_max:+d})"
        
        return {
            'changed_count': changed_count,
            'changed_pct': round(changed_pct, 1),
            'alpha_iou': round(alpha_iou, 3),
            'left_leg_changed': left_leg_changed,
            'right_leg_changed': right_leg_changed,
            'bbox_disp': bbox_disp
        }
        
    stats_w2 = compute_motion_stats(idle_arr, w2_arr)
    stats_w3 = compute_motion_stats(idle_arr, w3_arr)
    stats_slash = compute_motion_stats(idle_arr, slash_arr)
    
    records.append({
        'character_id': f"C_{count:02d}",
        'fg_pixels': fg_pixels,
        # Walk Frame 2 (The subtle contact step originally extracted)
        'w2_changed_px': stats_w2['changed_count'],
        'w2_changed_pct': stats_w2['changed_pct'],
        'w2_alpha_iou': stats_w2['alpha_iou'],
        'w2_l_leg_px': stats_w2['left_leg_changed'],
        'w2_r_leg_px': stats_w2['right_leg_changed'],
        'w2_bbox_disp': stats_w2['bbox_disp'],
        # Walk Frame 3 (The apex stride frame with maximum leg movement)
        'w3_changed_px': stats_w3['changed_count'],
        'w3_changed_pct': stats_w3['changed_pct'],
        'w3_alpha_iou': stats_w3['alpha_iou'],
        'w3_l_leg_px': stats_w3['left_leg_changed'],
        'w3_r_leg_px': stats_w3['right_leg_changed'],
        'w3_bbox_disp': stats_w3['bbox_disp'],
        # Slash Frame 3 (Verified Row 6 South Front Slash)
        'slash_changed_px': stats_slash['changed_count'],
        'slash_changed_pct': stats_slash['changed_pct'],
        'slash_alpha_iou': stats_slash['alpha_iou'],
        'slash_bbox_disp': stats_slash['bbox_disp']
    })
    
    characters_data.append({
        'id': f"C_{count:02d}",
        'idle': idle_im,
        'f0': walk_f0,
        'f1': walk_f1,
        'f2': walk_f2,
        'f3': walk_f3,
        'slash': slash_south,
        'stats_w2': stats_w2,
        'stats_w3': stats_w3,
        'stats_slash': stats_slash
    })
    
    count += 1
    if count >= 20:
        break

# Export CSV
df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)
print(f"Per-character metrics exported to: {CSV_PATH}")

# 1. Generate 5-Frame Walk Cycle Contact Sheet: [Idle | Frame 0 | Frame 1 | Frame 2 | Frame 3]
fig, axes = plt.subplots(20, 5, figsize=(10, 36), dpi=150)
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for i in range(20):
    c = characters_data[i]
    frames = [c['idle'], c['f0'], c['f1'], c['f2'], c['f3']]
    titles = [
        f"{c['id']} Idle",
        "Walk F0 (Stance)",
        f"Walk F1 (Δ:{np.sum(np.abs(np.array(c['f1'])[:,:,:3]-np.array(c['idle'])[:,:,:3])>30)})",
        f"Walk F2 ({c['stats_w2']['changed_pct']}%)",
        f"Walk F3 Apex ({c['stats_w3']['changed_pct']}%)"
    ]
    for col in range(5):
        axes[i, col].imshow(frames[col])
        axes[i, col].set_title(titles[col], fontsize=7)
        axes[i, col].axis('off')

plt.suptitle("LPC Walk Cycle Frame Analysis (Idle vs F0-F3) Across 20 Characters", fontsize=12, y=0.995)
plt.savefig(CONTACT_SHEET_5FRAME, bbox_inches='tight')
plt.close()
print(f"5-frame contact sheet saved to: {CONTACT_SHEET_5FRAME}")

# 2. Generate Corrected Master Contact Sheet: [Idle Ref | Walk Apex (F3) | South Slash (R6 C3)]
fig, axes = plt.subplots(10, 6, figsize=(14, 20), dpi=150)
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for i in range(20):
    rb = i // 2
    cb = (i % 2) * 3
    c = characters_data[i]
    
    axes[rb, cb].imshow(c['idle'])
    axes[rb, cb].set_title(f"{c['id']} Ref (South Idle)", fontsize=8)
    axes[rb, cb].axis('off')
    
    axes[rb, cb + 1].imshow(c['f3'])
    axes[rb, cb + 1].set_title(f"Walk Apex (Δ:{c['stats_w3']['changed_pct']}% IoU:{c['stats_w3']['alpha_iou']})", fontsize=8)
    axes[rb, cb + 1].axis('off')
    
    axes[rb, cb + 2].imshow(c['slash'])
    axes[rb, cb + 2].set_title(f"South Slash (Δ:{c['stats_slash']['changed_pct']}%)", fontsize=8)
    axes[rb, cb + 2].axis('off')

plt.suptitle("Corrected Master Audit: South Idle vs South Walk Apex vs South Slash", fontsize=13, y=0.99)
plt.savefig(CONTACT_SHEET_MASTER, bbox_inches='tight')
plt.close()
print(f"Corrected master audit contact sheet saved to: {CONTACT_SHEET_MASTER}")

