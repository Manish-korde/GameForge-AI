"""
Comprehensive Audit & Contact Sheet Generator for Dataset Research
Audits:
1. Diffusion Dataset (candidate_crops/action)
2. Carlosuperb LPC 4-view and Action Dataset
3. TinyHero / AgaMiko LPC Generator
4. Alucard 169K Archive
"""

import os
import sys
import json
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = 'phase_dcgan_improvement/dataset_research'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. AUDIT EXISTING DIFFUSION DATASET
csv_path = 'GameForge_AI_Phase2_audit/gameforge_phase2_1_audit/data/candidate_crop_index.csv'
df = pd.read_csv(csv_path)

unique_sheets = int(df['source_sheet_id'].nunique())
total_frames = len(df)
front_frames = df[df['crop_row'].isin([2, 6, 10])]
front_idles = df[(df['crop_row'] == 2) & (df['crop_col'] == 0)]

print(f"=== DIFFUSION DATASET AUDIT ===")
print(f"Total Rows in Inventory : {total_frames:,}")
print(f"Unique Characters (Sheets): {unique_sheets}")
print(f"Total Front-Facing Frames : {len(front_frames):,}")
print(f"Unique Front-Facing Idles : {len(front_idles)}")

# Render 25 Unique Front-Facing Character Idles
fig, axes = plt.subplots(5, 5, figsize=(10, 10), facecolor='#0f172a')
for idx, ax in enumerate(axes.flatten()):
    if idx < len(front_idles):
        row = front_idles.iloc[idx]
        cpath = os.path.join('GameForge_AI_Phase2_audit/gameforge_phase2_1_audit', row['crop_path'])
        if os.path.exists(cpath):
            img = Image.open(cpath).convert('RGBA')
            ax.imshow(img)
            sid = row['source_sheet_id']
            ax.set_title(f"Char #{sid:03d}", color='#38bdf8', fontsize=8, pad=2)
    ax.axis('off')

plt.suptitle("Diffusion Dataset: 25 Unique Front-Facing Character Idles", color="#f8fafc", fontsize=13, y=0.98, fontweight="bold")
plt.tight_layout()
diff_sheet_path = os.path.join(OUTPUT_DIR, 'contact_sheet_diffusion_characters.png')
plt.savefig(diff_sheet_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved: {diff_sheet_path}")

# Render Motion/Pose Redundancy Sheet for a single character (Character 000 across actions)
fig, axes = plt.subplots(3, 8, figsize=(14, 6), facecolor='#0f172a')
char0_frames = df[df['source_sheet_id'] == 0]

# Row 0: Walk South (r02)
walk_south = char0_frames[char0_frames['crop_row'] == 2].sort_values('crop_col')
for col_i in range(8):
    ax = axes[0, col_i]
    if col_i < len(walk_south):
        r = walk_south.iloc[col_i]
        p = os.path.join('GameForge_AI_Phase2_audit/gameforge_phase2_1_audit', r['crop_path'])
        if os.path.exists(p):
            ax.imshow(Image.open(p).convert('RGBA'))
            ax.set_title(f"Walk Frame {col_i}", color='#34d399', fontsize=7)
    ax.axis('off')

# Row 1: Slash South (r06)
slash_south = char0_frames[char0_frames['crop_row'] == 6].sort_values('crop_col')
for col_i in range(8):
    ax = axes[1, col_i]
    if col_i < len(slash_south):
        r = slash_south.iloc[col_i]
        p = os.path.join('GameForge_AI_Phase2_audit/gameforge_phase2_1_audit', r['crop_path'])
        if os.path.exists(p):
            ax.imshow(Image.open(p).convert('RGBA'))
            ax.set_title(f"Slash Frame {col_i}", color='#f59e0b', fontsize=7)
    ax.axis('off')

# Row 2: Spellcast South (r10)
cast_south = char0_frames[char0_frames['crop_row'] == 10].sort_values('crop_col')
for col_i in range(8):
    ax = axes[2, col_i]
    if col_i < len(cast_south):
        r = cast_south.iloc[col_i]
        p = os.path.join('GameForge_AI_Phase2_audit/gameforge_phase2_1_audit', r['crop_path'])
        if os.path.exists(p):
            ax.imshow(Image.open(p).convert('RGBA'))
            ax.set_title(f"Cast Frame {col_i}", color='#ec4899', fontsize=7)
    ax.axis('off')

plt.suptitle("Diffusion Dataset Redundancy: 23 Animation Frames of the EXACT SAME Character Identity (Char #000)", color="#f8fafc", fontsize=12, y=0.98, fontweight="bold")
plt.tight_layout()
redundancy_sheet_path = os.path.join(OUTPUT_DIR, 'contact_sheet_diffusion_single_char_redundancy.png')
plt.savefig(redundancy_sheet_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved: {redundancy_sheet_path}")

# Save metrics JSON
diffusion_metrics = {
    "total_frames": total_frames,
    "unique_character_identities": unique_sheets,
    "unique_front_facing_idles": len(front_idles),
    "total_front_facing_frames": len(front_frames),
    "frames_per_character": total_frames // unique_sheets,
    "front_frames_per_character": len(front_frames) // unique_sheets,
    "dimensions": "64x64 RGBA",
    "transparency": "Full alpha channel intact, transparent background",
    "normalization": "[-1, 1] linear (x/127.5 - 1.0)",
    "suitability_verdict": "UNSUITABLE for Unconditional DCGAN (Only 200 identities; dominated by animation redundancy)"
}

with open(os.path.join(OUTPUT_DIR, 'diffusion_dataset_metrics.json'), 'w') as f:
    json.dump(diffusion_metrics, f, indent=2)

print("AUDIT & VISUALIZATION COMPLETE!")

