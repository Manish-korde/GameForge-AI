import os
import sys
import glob
import json
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def main():
    lpc_results_dir = r"C:\Users\manis\Downloads\dcgan_lpc_frontfacing_50ep_results"
    history_csv = os.path.join(lpc_results_dir, "logs", "training_history.csv")
    df = pd.read_csv(history_csv)
    
    print("=" * 80)
    print("LPC 50-EPOCH TRAINING RUN AUDIT & TRAJECTORY ANALYSIS")
    print("=" * 80)
    print(f"Total Epochs Recorded : {len(df)}")
    print(f"Mean Epoch Duration   : {df['epoch_time_sec'].mean():.2f}s")
    print(f"Total Training Time   : {df['epoch_time_sec'].sum() / 60:.1f} minutes")
    
    print("\n--- Key Epoch Milestones ---")
    milestone_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 41, 42, 43, 45, 50]
    print(f"{'Epoch':<6} | {'G_Loss':<9} | {'D_Loss':<9} | {'Adv_Loss':<9} | {'FM_Loss':<9}")
    print("-" * 55)
    for ep in milestone_epochs:
        row = df[df['epoch'] == ep].iloc[0]
        print(f"{int(ep):<6} | {row['g_loss']:<9.4f} | {row['d_loss']:<9.4f} | {row['adv_loss']:<9.4f} | {row['fm_loss']:<9.4f}")
        
    print("\n--- Loss Extremes & Dynamics ---")
    min_d = df.loc[df['d_loss'].idxmin()]
    max_d = df.loc[df['d_loss'].idxmax()]
    min_fm = df.loc[df['fm_loss'].idxmin()]
    max_fm = df.loc[df['fm_loss'].idxmax()]
    
    print(f"Minimum D_Loss: {min_d['d_loss']:.4f} at Epoch {int(min_d['epoch'])}")
    print(f"Maximum D_Loss: {max_d['d_loss']:.4f} at Epoch {int(max_d['epoch'])}")
    print(f"Minimum FM_Loss: {min_fm['fm_loss']:.4f} at Epoch {int(min_fm['epoch'])}")
    print(f"Maximum FM_Loss: {max_fm['fm_loss']:.4f} at Epoch {int(max_fm['epoch'])}")
    
    # -------------------------------------------------------------------------
    # 2. Inspect Extracted Training Data vs Generated Outputs
    # -------------------------------------------------------------------------
    extracted_idles = glob.glob(os.path.join(lpc_results_dir, "hf_extracted_idles", "*.png"))
    print(f"\nExtracted Training Idles in Archive: {len(extracted_idles):,} images")
    
    # Analyze 50 real training idles
    real_colors = []
    real_alphas = []
    real_coherence = []
    
    for rp in extracted_idles[:100]:
        rim = Image.open(rp).convert("RGBA")
        rarr = np.array(rim)
        ralpha = rarr[:, :, 3]
        rrgb = rarr[:, :, :3]
        
        r_fg = ralpha > 20
        r_fg_px = rrgb[r_fg]
        if len(r_fg_px) > 10:
            packed = (r_fg_px[:, 0].astype(np.uint32) << 16) | (r_fg_px[:, 1].astype(np.uint32) << 8) | r_fg_px[:, 2].astype(np.uint32)
            real_colors.append(len(np.unique(packed)))
            
        real_alphas.append(100.0 * np.mean((ralpha < 10) | (ralpha > 245)))
        
        # Spatial coherence
        fg_h = r_fg[:, :-1] & r_fg[:, 1:]
        same_h = np.all(rrgb[:, :-1] == rrgb[:, 1:], axis=-1) & fg_h
        same_p = np.sum(same_h)
        tot_p = np.sum(fg_h)
        if tot_p > 0:
            real_coherence.append(100.0 * same_p / tot_p)
            
    print(f"Real Training LPC Sprites (N={len(real_colors)}):")
    print(f"  Mean Unique 8-Bit Colors : {np.mean(real_colors):.1f}")
    print(f"  Alpha Binary Purity      : {np.mean(real_alphas):.2f}%")
    print(f"  Spatial Color Coherence  : {np.mean(real_coherence):.2f}%")
    
    # -------------------------------------------------------------------------
    # 3. Model Weight Forward Pass & 100 Sample Synthesis
    # -------------------------------------------------------------------------
    weights_path = os.path.join(lpc_results_dir, "generator_weights_final.weights.h5")
    fixed_z_path = os.path.join(lpc_results_dir, "config", "fixed_latent_vectors.npy")
    fixed_z = np.load(fixed_z_path)
    
    def load_generator_weights_numpy(h5_path):
        w = {}
        with h5py.File(h5_path, 'r') as f:
            w['dense_w'] = np.array(f['layers/dense/vars/0'])
            w['bn0_gamma'] = np.array(f['layers/batch_normalization/vars/0'])
            w['bn0_beta'] = np.array(f['layers/batch_normalization/vars/1'])
            w['bn0_mean'] = np.array(f['layers/batch_normalization/vars/2'])
            w['bn0_var'] = np.array(f['layers/batch_normalization/vars/3'])
            
            w['conv1_w'] = np.array(f['layers/conv2d/vars/0'])
            w['bn1_gamma'] = np.array(f['layers/batch_normalization_1/vars/0'])
            w['bn1_beta'] = np.array(f['layers/batch_normalization_1/vars/1'])
            w['bn1_mean'] = np.array(f['layers/batch_normalization_1/vars/2'])
            w['bn1_var'] = np.array(f['layers/batch_normalization_1/vars/3'])
            
            w['conv2_w'] = np.array(f['layers/conv2d_1/vars/0'])
            w['bn2_gamma'] = np.array(f['layers/batch_normalization_2/vars/0'])
            w['bn2_beta'] = np.array(f['layers/batch_normalization_2/vars/1'])
            w['bn2_mean'] = np.array(f['layers/batch_normalization_2/vars/2'])
            w['bn2_var'] = np.array(f['layers/batch_normalization_2/vars/3'])
            
            w['conv3_w'] = np.array(f['layers/conv2d_2/vars/0'])
            w['bn3_gamma'] = np.array(f['layers/batch_normalization_3/vars/0'])
            w['bn3_beta'] = np.array(f['layers/batch_normalization_3/vars/1'])
            w['bn3_mean'] = np.array(f['layers/batch_normalization_3/vars/2'])
            w['bn3_var'] = np.array(f['layers/batch_normalization_3/vars/3'])
            
            w['conv4_w'] = np.array(f['layers/conv2d_3/vars/0'])
            w['conv4_b'] = np.array(f['layers/conv2d_3/vars/1'])
        return w

    def conv2d_same(x, w, b=None):
        B, H, W, C_in = x.shape
        kH, kW, _, C_out = w.shape
        pad_h, pad_w = kH // 2, kW // 2
        x_pad = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        cols = []
        for dy in range(kH):
            for dx in range(kW):
                cols.append(x_pad[:, dy:dy+H, dx:dx+W, :])
        col_mat = np.concatenate(cols, axis=-1)
        w_mat = w.reshape(-1, C_out)
        out = col_mat @ w_mat
        if b is not None:
            out += b
        return out

    def forward_generator(z, w, eps=1e-5):
        x = z @ w['dense_w']
        x = (x - w['bn0_mean']) / np.sqrt(w['bn0_var'] + eps) * w['bn0_gamma'] + w['bn0_beta']
        x = np.maximum(x, 0.2 * x)
        x = x.reshape(-1, 4, 4, 512)
        
        # Block 1
        x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        x = conv2d_same(x, w['conv1_w'])
        x = (x - w['bn1_mean']) / np.sqrt(w['bn1_var'] + eps) * w['bn1_gamma'] + w['bn1_beta']
        x = np.maximum(x, 0.2 * x)
        
        # Block 2
        x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        x = conv2d_same(x, w['conv2_w'])
        x = (x - w['bn2_mean']) / np.sqrt(w['bn2_var'] + eps) * w['bn2_gamma'] + w['bn2_beta']
        x = np.maximum(x, 0.2 * x)
        
        # Block 3
        x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        x = conv2d_same(x, w['conv3_w'])
        x = (x - w['bn3_mean']) / np.sqrt(w['bn3_var'] + eps) * w['bn3_gamma'] + w['bn3_beta']
        x = np.maximum(x, 0.2 * x)
        
        # Block 4
        x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        x = conv2d_same(x, w['conv4_w'], w['conv4_b'])
        x = np.tanh(x)
        return x

    w = load_generator_weights_numpy(weights_path)
    print("\nSynthesizing 100 generated sprites from fixed latents...")
    raw_gen = forward_generator(fixed_z, w)
    
    # Evaluate channels
    rgb_gen = raw_gen[:, :, :, :3]
    alpha_gen = raw_gen[:, :, :, 3]
    
    print("\n--- Output Channels Distribution across 100 Synthesized Sprites ---")
    print(f"Alpha min: {alpha_gen.min():.4f}, max: {alpha_gen.max():.4f}, mean: {alpha_gen.mean():.4f}, std: {alpha_gen.std():.4f}")
    
    # Discrete 8-bit metrics
    norm = np.clip((raw_gen + 1.0) / 2.0, 0.0, 1.0)
    gen_u8 = np.round(norm * 255.0).astype(np.uint8)
    
    gen_colors = []
    gen_alphas = []
    gen_coherence = []
    gen_fg_pixels = []
    
    for i in range(100):
        im = gen_u8[i]
        a = im[:, :, 3]
        rgb = im[:, :, :3]
        
        fg = a > 20
        fg_px = rgb[fg]
        gen_fg_pixels.append(len(fg_px))
        
        if len(fg_px) > 10:
            packed = (fg_px[:, 0].astype(np.uint32) << 16) | (fg_px[:, 1].astype(np.uint32) << 8) | fg_px[:, 2].astype(np.uint32)
            gen_colors.append(len(np.unique(packed)))
            
            fg_h = fg[:, :-1] & fg[:, 1:]
            same_h = np.all(rgb[:, :-1] == rgb[:, 1:], axis=-1) & fg_h
            same_p = np.sum(same_h)
            tot_p = np.sum(fg_h)
            if tot_p > 0:
                gen_coherence.append(100.0 * same_p / tot_p)
                
        gen_alphas.append(100.0 * np.mean((a < 10) | (a > 245)))
        
    print(f"\nGenerated Sprites Metrics (Epoch 50, N=100):")
    print(f"  Mean Foreground Pixels (alpha > 20) : {np.mean(gen_fg_pixels):.1f} / 4096 ({100.0 * np.mean(gen_fg_pixels) / 4096:.2f}%)")
    print(f"  Sprites with zero foreground content: {sum(1 for c in gen_fg_pixels if c < 10)} / 100")
    if len(gen_colors) > 0:
        print(f"  Mean Unique 8-Bit Colors (when fg>10): {np.mean(gen_colors):.1f}")
        print(f"  Mean Spatial Color Coherence         : {np.mean(gen_coherence):.2f}%")
    print(f"  Alpha Binary Purity                  : {np.mean(gen_alphas):.2f}%")
    
    # -------------------------------------------------------------------------
    # 4. Comparative Scorecard Across 4 Iterations
    # -------------------------------------------------------------------------
    last_d = df.iloc[-1]['d_loss']
    last_fm = df.iloc[-1]['fm_loss']
    gen_col_str = f"{np.mean(gen_colors):.1f}" if gen_colors else "N/A (Empty)"
    fg_px_str = f"{np.mean(gen_fg_pixels):.1f} px"
    
    print("\n" + "=" * 90)
    print("4-EXPERIMENT COMPARATIVE BENCHMARK SCORECARD")
    print("=" * 90)
    print(f"{'Experiment':<30} | {'D_Loss (Final)':<14} | {'FM_Loss (Final)':<15} | {'Unique Colors':<14} | {'Foreground Px'}")
    print("-" * 90)
    print(f"{'1. Alucard 169K (50 Ep)':<30} | {'0.0085':<14} | {'0.4403':<15} | {'1,415.7':<14} | {'~1,180 px'}")
    print(f"{'2. Alucard Front 46K (50 Ep)':<30} | {'0.0358':<14} | {'0.4052':<15} | {'1,428.3':<14} | {'~1,180 px'}")
    print(f"{'3. Alucard Pilot 1 (10 Ep)':<30} | {'0.0219':<14} | {'0.3003':<15} | {'1,664.7':<14} | {'~1,180 px'}")
    print(f"{'4. LPC Front-Facing (50 Ep)':<30} | {last_d:<14.4f} | {last_fm:<15.4f} | {gen_col_str:<14} | {fg_px_str}")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # 5. Create Comparison Contact Sheet
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 8, figsize=(16, 9), facecolor='#0f172a')
    
    # Row 0: Real LPC Training Sprites (Ground Truth)
    for c in range(8):
        ax = axes[0, c]
        if c < len(extracted_idles):
            im = Image.open(extracted_idles[c]).convert("RGBA")
            ax.imshow(im)
            ax.set_title(f"Real LPC #{c}", color="#38bdf8", fontsize=8)
        ax.axis('off')
        
    # Row 1: Alucard Baseline 50-Epoch (Speckled)
    base_showcase = r"C:\Users\manis\Downloads\dcgan_frontfacing_46k_50ep_results\samples\final_master_showcase_100_characters.png"
    if os.path.exists(base_showcase):
        bim = Image.open(base_showcase)
        bw = bim.width // 10
        bh = bim.height // 10
        for c in range(8):
            ax = axes[1, c]
            box = (c * bw, 0, (c + 1) * bw, bh)
            ax.imshow(bim.crop(box))
            ax.set_title(f"Alucard 50Ep", color="#f87171", fontsize=8)
            ax.axis('off')
            
    # Row 2: Alucard Pilot 1 (10 Ep Centroid FM)
    pilot1_showcase = r"C:\Users\manis\Downloads\dcgan_frontfacing_centroid_pilot_results\samples\final_master_showcase_100_characters.png"
    if os.path.exists(pilot1_showcase):
        pim = Image.open(pilot1_showcase)
        pw = pim.width // 10
        ph = pim.height // 10
        for c in range(8):
            ax = axes[2, c]
            box = (c * pw, 0, (c + 1) * pw, ph)
            ax.imshow(pim.crop(box))
            ax.set_title(f"Pilot 1 10Ep", color="#fbbf24", fontsize=8)
            ax.axis('off')
            
    # Row 3: LPC DCGAN Generated (Epoch 50)
    for c in range(8):
        ax = axes[3, c]
        ax.imshow(gen_u8[c])
        ax.set_title(f"LPC Ep 50", color="#a3e635", fontsize=8)
        ax.axis('off')
        
    plt.suptitle("4-Tier Empirical Comparison:\nRow 0: Real LPC Sprites | Row 1: Alucard 50Ep Baseline | Row 2: Alucard Pilot 1 | Row 3: LPC DCGAN (50 Ep)", 
                 color="#f8fafc", fontsize=12, y=0.98, fontweight="bold")
    plt.tight_layout()
    out_comparison = r"phase_dcgan_improvement\lpc_investigation\lpc_vs_alucard_4tier_comparison.png"
    plt.savefig(out_comparison, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"\nSaved 4-tier visual comparison contact sheet to: {out_comparison}")

if __name__ == "__main__":
    main()
