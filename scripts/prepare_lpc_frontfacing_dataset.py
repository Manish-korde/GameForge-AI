import os
import shutil
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def main():
    print("=" * 80)
    print("PHASE 1: LPC FRONT-FACING IDLE DATASET PREPARATION")
    print("=" * 80)
    
    # 1. Source directories
    index_csv = r"GameForge_AI_Phase2_audit\gameforge_phase2_1_audit\data\candidate_crop_index.csv"
    crop_dir = r"GameForge_AI_Phase2_audit\gameforge_phase2_1_audit\data\candidate_crops\action"
    
    out_dir = r"Dataset\lpc_frontfacing_idles"
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Read index and filter strictly for Row 2 (South Walk cycle), Col 0 (Canonical Idle Stance)
    df = pd.read_csv(index_csv)
    print(f"Total raw indexed crops: {len(df):,}")
    
    # Filter strictly for Front-Facing Canonical Idle (r02_c00)
    idle_df = df[(df['crop_row'] == 2) & (df['crop_col'] == 0)].copy().reset_index(drop=True)
    print(f"Filtered for Canonical Front-Facing Idle (crop_row=2, crop_col=0): {len(idle_df)} candidates")
    
    manifest_records = []
    
    for idx, row in idle_df.iterrows():
        src_fname = os.path.basename(row['crop_path'])
        src_path = os.path.join(crop_dir, src_fname)
        
        if not os.path.exists(src_path):
            continue
            
        # Target path
        char_id = row['source_sheet_id']
        char_num = int(char_id.split(':')[-1]) if ':' in char_id else int(char_id)
        dst_fname = f"lpc_char_{char_num:05d}_front_idle.png"
        dst_path = os.path.join(out_dir, dst_fname)
        
        # Open and verify
        im = Image.open(src_path)
        if im.size != (64, 64) or im.mode != "RGBA":
            im = im.convert("RGBA").resize((64, 64), Image.NEAREST)
        
        # Copy / save
        im.save(dst_path)
        
        # Compute basic quality stats
        arr = np.array(im)
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
        fg_mask = alpha > 20
        fg_pixels = rgb[fg_mask]
        
        # 8-bit packed colors
        if len(fg_pixels) > 0:
            packed = (fg_pixels[:, 0].astype(np.uint32) << 16) | (fg_pixels[:, 1].astype(np.uint32) << 8) | fg_pixels[:, 2].astype(np.uint32)
            n_colors = len(np.unique(packed))
        else:
            n_colors = 0
            
        binary_alpha = 100.0 * np.mean((alpha < 10) | (alpha > 245))
        
        manifest_records.append({
            'character_id': char_num,
            'source_sheet_id': char_id,
            'filename': dst_fname,
            'relative_path': os.path.join("Dataset", "lpc_frontfacing_idles", dst_fname).replace('\\', '/'),
            'width': 64,
            'height': 64,
            'channels': 4,
            'view': 'front',
            'action': 'idle',
            'crop_row': 2,
            'crop_col': 0,
            'fg_pixel_count': int(np.sum(fg_mask)),
            'unique_8bit_colors': n_colors,
            'alpha_binary_purity': float(binary_alpha)
        })
        
    out_manifest_df = pd.DataFrame(manifest_records)
    manifest_csv = r"Dataset\lpc_frontfacing_manifest.csv"
    out_manifest_df.to_csv(manifest_csv, index=False)
    print(f"\nManifest saved: {manifest_csv}")
    print(f"Total verified front-facing idle sprites saved: {len(out_manifest_df)}")
    print(f"Unique character identities: {out_manifest_df['character_id'].nunique()}")
    print(f"Mean unique 8-bit colors per sprite: {out_manifest_df['unique_8bit_colors'].mean():.1f}")
    print(f"Mean alpha binary purity: {out_manifest_df['alpha_binary_purity'].mean():.2f}%")
    
    # 3. Generate Contact Sheet for Visual Confirmation (10x10 = 100 Characters)
    print("\nGenerating 100-character visual confirmation contact sheet...")
    fig, axes = plt.subplots(10, 10, figsize=(14, 14), facecolor='#0f172a')
    axes = axes.flatten()
    
    for i in range(100):
        ax = axes[i]
        if i < len(out_manifest_df):
            row = out_manifest_df.iloc[i]
            img_p = os.path.join(out_dir, row['filename'])
            im = Image.open(img_p)
            ax.imshow(im)
            ax.set_title(f"#{row['character_id']:03d}", color="#38bdf8", fontsize=7, pad=2)
        ax.axis('off')
        
    plt.suptitle("GameForge AI — LPC Character Dataset: 100 Verified Front-Facing Canonical Idles\n(Extracted from carlosuperb/lpc-action-pixel-art-diffusion [Row 2, Col 0])", 
                 color="#f8fafc", fontsize=13, y=0.98, fontweight="bold")
    plt.tight_layout()
    
    contact_p = r"phase_dcgan_improvement\lpc_investigation\lpc_frontfacing_idles_contact_sheet.png"
    plt.savefig(contact_p, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Contact sheet saved to: {contact_p}")

if __name__ == "__main__":
    main()

