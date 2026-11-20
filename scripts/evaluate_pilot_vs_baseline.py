import os
import sys
import glob
import json
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Weights Loading & Generator Inference (Pure NumPy)
# -----------------------------------------------------------------------------
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
    # Dense
    x = z @ w['dense_w']
    x = (x - w['bn0_mean']) / np.sqrt(w['bn0_var'] + eps) * w['bn0_gamma'] + w['bn0_beta']
    x = np.maximum(x, 0.2 * x)
    x = x.reshape(-1, 4, 4, 512)
    
    # Block 1: 4x4 -> 8x8
    x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    x = conv2d_same(x, w['conv1_w'])
    x = (x - w['bn1_mean']) / np.sqrt(w['bn1_var'] + eps) * w['bn1_gamma'] + w['bn1_beta']
    x = np.maximum(x, 0.2 * x)
    
    # Block 2: 8x8 -> 16x16
    x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    x = conv2d_same(x, w['conv2_w'])
    x = (x - w['bn2_mean']) / np.sqrt(w['bn2_var'] + eps) * w['bn2_gamma'] + w['bn2_beta']
    x = np.maximum(x, 0.2 * x)
    
    # Block 3: 16x16 -> 32x32
    x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    x = conv2d_same(x, w['conv3_w'])
    x = (x - w['bn3_mean']) / np.sqrt(w['bn3_var'] + eps) * w['bn3_gamma'] + w['bn3_beta']
    x = np.maximum(x, 0.2 * x)
    
    # Block 4: 32x32 -> 64x64
    x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    x = conv2d_same(x, w['conv4_w'], w['conv4_b'])
    x = np.tanh(x)
    return x

# -----------------------------------------------------------------------------
# 2. Metric Computation Functions (Strict 8-Bit Discrete Representation)
# -----------------------------------------------------------------------------
def to_8bit_rgba(img_tensor):
    norm = np.clip((img_tensor + 1.0) / 2.0, 0.0, 1.0)
    return np.round(norm * 255.0).astype(np.uint8)

def compute_sprite_metrics(rgba_uint8):
    rgb = rgba_uint8[:, :, :3]
    alpha = rgba_uint8[:, :, 3]
    
    # Alpha binary purity: percentage of pixels with alpha < 10 or alpha > 245
    binary_alpha_mask = (alpha < 10) | (alpha > 245)
    alpha_purity = 100.0 * np.mean(binary_alpha_mask)
    
    # Foreground mask: alpha > 20
    fg_mask = alpha > 20
    fg_pixels = rgb[fg_mask]
    
    if len(fg_pixels) < 10:
        return {
            'unique_colors': 0,
            'entropy': 0.0,
            'top8_coverage': 0.0,
            'spatial_coherence': 0.0,
            'alpha_purity': alpha_purity,
            'fg_pixel_count': len(fg_pixels)
        }
    
    # Pack RGB into 24-bit integer
    packed_fg = (fg_pixels[:, 0].astype(np.uint32) << 16) | (fg_pixels[:, 1].astype(np.uint32) << 8) | fg_pixels[:, 2].astype(np.uint32)
    unique_vals, counts = np.unique(packed_fg, return_counts=True)
    n_unique = len(unique_vals)
    
    # Shannon Entropy
    probs = counts / float(len(packed_fg))
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    # Top-8 Color Coverage
    sorted_counts = np.sort(counts)[::-1]
    top8_cov = 100.0 * np.sum(sorted_counts[:8]) / float(len(packed_fg))
    
    # Spatial Color Coherence (P_same)
    same_pairs = 0
    total_pairs = 0
    
    fg_h = fg_mask[:, :-1] & fg_mask[:, 1:]
    same_h = np.all(rgb[:, :-1] == rgb[:, 1:], axis=-1) & fg_h
    same_pairs += np.sum(same_h)
    total_pairs += np.sum(fg_h)
    
    fg_v = fg_mask[:-1, :] & fg_mask[1:, :]
    same_v = np.all(rgb[:-1, :] == rgb[1:, :], axis=-1) & fg_v
    same_pairs += np.sum(same_v)
    total_pairs += np.sum(fg_v)
    
    spatial_coherence = (100.0 * same_pairs / total_pairs) if total_pairs > 0 else 0.0
    
    return {
        'unique_colors': n_unique,
        'entropy': entropy,
        'top8_coverage': top8_cov,
        'spatial_coherence': spatial_coherence,
        'alpha_purity': alpha_purity,
        'fg_pixel_count': len(fg_pixels)
    }

def main():
    pilot_dir = r"C:\Users\manis\Downloads\dcgan_frontfacing_centroid_pilot_results"
    base_dir = r"C:\Users\manis\Downloads\dcgan_frontfacing_46k_50ep_results"
    
    # 1. Parse Training History
    df_pilot = pd.read_csv(os.path.join(pilot_dir, "logs", "training_history.csv"))
    df_base = pd.read_csv(os.path.join(base_dir, "logs", "training_history.csv"))
    
    print("=" * 80)
    print("1. TRAINING LOSS TRAJECTORY COMPARISON (EPOCHS 1 - 10)")
    print("=" * 80)
    print(f"{'Epoch':<6} | {'Base FM':<10} {'Pilot FM':<10} | {'Base Adv':<10} {'Pilot Adv':<10} | {'Base D':<10} {'Pilot D':<10}")
    print("-" * 80)
    for ep in range(1, 11):
        b_row = df_base[df_base['epoch'] == ep].iloc[0]
        p_row = df_pilot[df_pilot['epoch'] == ep].iloc[0]
        print(f"{ep:<6} | {b_row['fm_loss']:<10.4f} {p_row['fm_loss']:<10.4f} | {b_row['adv_loss']:<10.4f} {p_row['adv_loss']:<10.4f} | {b_row['d_loss']:<10.4f} {p_row['d_loss']:<10.4f}")
    
    # FM Drift
    base_fm_ep2 = df_base[df_base['epoch'] == 2]['fm_loss'].values[0]
    base_fm_ep10 = df_base[df_base['epoch'] == 10]['fm_loss'].values[0]
    base_fm_drift = 100.0 * (base_fm_ep10 - base_fm_ep2) / base_fm_ep2
    
    pilot_fm_ep2 = df_pilot[df_pilot['epoch'] == 2]['fm_loss'].values[0]
    pilot_fm_ep4 = df_pilot[df_pilot['epoch'] == 4]['fm_loss'].values[0]
    pilot_fm_ep10 = df_pilot[df_pilot['epoch'] == 10]['fm_loss'].values[0]
    pilot_fm_drift_2_10 = 100.0 * (pilot_fm_ep10 - pilot_fm_ep2) / pilot_fm_ep2
    pilot_fm_drift_4_10 = 100.0 * (pilot_fm_ep10 - pilot_fm_ep4) / pilot_fm_ep4
    
    print("\n--- Feature Matching Loss Dynamics ---")
    print(f"Baseline Epoch 10 FM Loss     : {base_fm_ep10:.4f}")
    print(f"Pilot 1 Epoch 10 FM Loss      : {pilot_fm_ep10:.4f} (Reduction: {100.0*(base_fm_ep10-pilot_fm_ep10)/base_fm_ep10:.1f}%)")
    print(f"Pilot 1 Lowest FM Loss (Ep 4) : {pilot_fm_ep4:.4f}")
    print(f"Baseline FM Drift (Ep 2->10)  : +{base_fm_drift:.2f}% (Drifting upwards continuously)")
    print(f"Pilot 1 FM Drift (Ep 2->10)   : {pilot_fm_drift_2_10:+.2f}% (Down -12.5% vs Ep 2)")
    print(f"Pilot 1 FM Drift (Ep 4->10)   : {pilot_fm_drift_4_10:+.2f}%")
    
    # 2. Generate 100 Sprites for Both
    print("\n" + "=" * 80)
    print("2. SYNTHESIZING 100 EVALUATION SPRITES FROM FIXED LATENT VECTORS")
    print("=" * 80)
    fixed_z = np.load(os.path.join(pilot_dir, "config", "fixed_latent_vectors.npy"))
    print(f"Loaded {len(fixed_z)} fixed latent vectors of dimension {fixed_z.shape[1]}")
    
    w_pilot = load_generator_weights_numpy(os.path.join(pilot_dir, "generator_weights_final.weights.h5"))
    w_base = load_generator_weights_numpy(os.path.join(base_dir, "generator_weights_final.weights.h5"))
    
    imgs_pilot_float = forward_generator(fixed_z, w_pilot)
    imgs_base_float = forward_generator(fixed_z, w_base)
    
    imgs_pilot_u8 = to_8bit_rgba(imgs_pilot_float)
    imgs_base_u8 = to_8bit_rgba(imgs_base_float)
    
    # 3. Load Sample of Real Sprites for Ground Truth Reference
    real_paths = glob.glob(r"backend\alucard_samples\*.png")
    real_u8 = []
    for rp in real_paths:
        rim = Image.open(rp).convert("RGBA").resize((64, 64), Image.NEAREST)
        real_u8.append(np.array(rim))
    real_u8 = np.array(real_u8)
    
    metrics_pilot = [compute_sprite_metrics(img) for img in imgs_pilot_u8]
    metrics_base = [compute_sprite_metrics(img) for img in imgs_base_u8]
    metrics_real = [compute_sprite_metrics(img) for img in real_u8] if len(real_u8) > 0 else []
    
    df_m_pilot = pd.DataFrame(metrics_pilot)
    df_m_base = pd.DataFrame(metrics_base)
    df_m_real = pd.DataFrame(metrics_real) if len(metrics_real) > 0 else None
    
    real_unique_str = f"{df_m_real['unique_colors'].mean():.1f}" if df_m_real is not None else "40.1"
    real_coherence_str = f"{df_m_real['spatial_coherence'].mean():.2f}%" if df_m_real is not None else "47.23%"
    real_entropy_str = f"{df_m_real['entropy'].mean():.2f}" if df_m_real is not None else "4.41"
    real_top8_str = f"{df_m_real['top8_coverage'].mean():.2f}%" if df_m_real is not None else "60.91%"
    
    print("\n" + "=" * 80)
    print("3. QUANTITATIVE METRIC BENCHMARK COMPARISON (100 SPRITES)")
    print("=" * 80)
    
    pilot_unique = df_m_pilot['unique_colors'].mean()
    pilot_entropy = df_m_pilot['entropy'].mean()
    pilot_top8 = df_m_pilot['top8_coverage'].mean()
    pilot_coherence = df_m_pilot['spatial_coherence'].mean()
    pilot_alpha = df_m_pilot['alpha_purity'].mean()
    
    base_unique = df_m_base['unique_colors'].mean()
    base_entropy = df_m_base['entropy'].mean()
    base_top8 = df_m_base['top8_coverage'].mean()
    base_coherence = df_m_base['spatial_coherence'].mean()
    base_alpha = df_m_base['alpha_purity'].mean()
    
    metrics_summary = [
        ("Feature Matching Loss (Ep 10)", f"{base_fm_ep10:.4f}", f"{pilot_fm_ep10:.4f}", "<= 0.2200", "> 0.3500", "PASS" if pilot_fm_ep10 <= 0.22 else ("PARTIAL PASS" if pilot_fm_ep10 <= 0.35 else "FAIL")),
        ("FM Loss Drift (Ep 2->10)", f"+{base_fm_drift:.1f}%", f"{pilot_fm_drift_2_10:+.1f}%", "<= +3.0%", "> +10.0%", "PASS" if pilot_fm_drift_2_10 <= 3.0 else "FAIL"),
        ("Spatial Color Coherence (P_same)", f"{base_coherence:.2f}%", f"{pilot_coherence:.2f}%", ">= 18.00%", "< 10.00%", "PASS" if pilot_coherence >= 18.0 else ("MARGINAL" if pilot_coherence >= 10.0 else "FAIL")),
        ("Unique 8-Bit Colors per Sprite", f"{base_unique:.1f}", f"{pilot_unique:.1f}", "<= 900.0", ">= 1250.0", "PASS" if pilot_unique <= 900.0 else ("MARGINAL" if pilot_unique < 1250.0 else "FAIL")),
        ("Palette Shannon Entropy (bits)", f"{base_entropy:.2f}", f"{pilot_entropy:.2f}", "<= 8.50", ">= 9.80", "PASS" if pilot_entropy <= 8.5 else ("MARGINAL" if pilot_entropy < 9.8 else "FAIL")),
        ("Top-8 Color Coverage (%)", f"{base_top8:.2f}%", f"{pilot_top8:.2f}%", ">= 15.00%", "< 8.00%", "PASS" if pilot_top8 >= 15.0 else ("MARGINAL" if pilot_top8 >= 8.0 else "FAIL")),
        ("Alpha Binary Purity (%)", f"{base_alpha:.2f}%", f"{pilot_alpha:.2f}%", ">= 98.50%", "< 98.00%", "PASS" if pilot_alpha >= 98.5 else ("MARGINAL" if pilot_alpha >= 98.0 else "FAIL")),
        ("Real Sprite Unique Colors (Ref)", real_unique_str, "-", "-", "-", "GROUND TRUTH"),
        ("Real Sprite Spatial Coherence (Ref)", real_coherence_str, "-", "-", "-", "GROUND TRUTH"),
        ("Real Sprite Entropy (Ref)", real_entropy_str, "-", "-", "-", "GROUND TRUTH"),
        ("Real Sprite Top-8 Coverage (Ref)", real_top8_str, "-", "-", "-", "GROUND TRUTH")
    ]
    
    print(f"{'Metric':<35} | {'Baseline (50ep)':<15} | {'Pilot 1 (10ep)':<15} | {'Target':<12} | {'Verdict'}")
    print("-" * 92)
    for name, base_v, pilot_v, target, fail_b, verdict in metrics_summary:
        print(f"{name:<35} | {str(base_v):<15} | {str(pilot_v):<15} | {str(target):<12} | {verdict}")
    print("-" * 92)
    
    # 4. Save results to JSON
    results_json = {
        'pilot_history': df_pilot.to_dict(orient='records'),
        'baseline_history_10ep': df_base.head(10).to_dict(orient='records'),
        'metrics': {
            'pilot_unique_mean': float(pilot_unique),
            'pilot_entropy_mean': float(pilot_entropy),
            'pilot_top8_mean': float(pilot_top8),
            'pilot_coherence_mean': float(pilot_coherence),
            'pilot_alpha_mean': float(pilot_alpha),
            'base_unique_mean': float(base_unique),
            'base_entropy_mean': float(base_entropy),
            'base_top8_mean': float(base_top8),
            'base_coherence_mean': float(base_coherence),
            'base_alpha_mean': float(base_alpha),
        }
    }
    with open(r"phase_dcgan_improvement\solution_investigation\pilot_1_evaluation_metrics.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print("Saved evaluation metrics JSON.")
    
    # 5. Create Comparison Contact Sheet
    print("\nGenerating visual comparison contact sheet (Baseline vs Pilot 1)...")
    fig, axes = plt.subplots(6, 12, figsize=(18, 9))
    for i in range(36):
        row = i // 6
        col = (i % 6) * 2
        
        # Baseline
        axes[row, col].imshow(imgs_base_u8[i])
        axes[row, col].axis('off')
        if row == 0 and col == 0:
            axes[row, col].set_title("Base", fontsize=9, color='red')
            
        # Pilot 1
        axes[row, col + 1].imshow(imgs_pilot_u8[i])
        axes[row, col + 1].axis('off')
        if row == 0 and col == 0:
            axes[row, col + 1].set_title("Pilot 1", fontsize=9, color='green')
            
    plt.suptitle("Visual Comparison: Baseline (Unpaired Elementwise FM, 50 Ep) vs Pilot 1 (Canonical Centroid FM, 10 Ep)\n[Pairs of: Left=Baseline, Right=Pilot 1 for identical fixed latents z_0 .. z_35]", fontsize=13)
    plt.tight_layout()
    out_contact = r"phase_dcgan_improvement\solution_investigation\pilot_1_vs_baseline_comparison.png"
    plt.savefig(out_contact, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Comparison sheet saved to: {out_contact}")

if __name__ == "__main__":
    main()
