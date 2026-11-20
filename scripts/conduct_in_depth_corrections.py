import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datasets import load_dataset
from collections import Counter
import math

RESULTS_DIR = r"C:\Users\manis\Downloads\dcgan_frontfacing_46k_50ep_results"
OUTPUT_DIR = "phase_dcgan_improvement/solution_investigation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("RUNNING IN-DEPTH RIGOROUS FORENSIC VALIDATION & CORRECTIONS SCRIPT")
print("="*80)

IMG_HEIGHT, IMG_WIDTH, CHANNELS = 64, 64, 4
LATENT_DIM = 128

# 1. Load Architecture & Checkpoint Weights
def build_generator(latent_dim=128):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,), name="latent_input"),
        layers.Dense(4 * 4 * 512, use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((4, 4, 512)),
        layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
        layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
        layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
        layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
        layers.Conv2D(4, kernel_size=3, padding="same", activation="tanh", kernel_initializer=init)
    ])
    return model

def build_discriminator():
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    img_in = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=True, kernel_initializer=init)(img_in)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    feat_out = x
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    logit_out = layers.Dense(1)(x)
    return keras.Model(inputs=img_in, outputs=[logit_out, feat_out])

gen = build_generator()
gen.load_weights(os.path.join(RESULTS_DIR, "generator_weights_final.weights.h5"))

disc = build_discriminator()
disc.load_weights(os.path.join(RESULTS_DIR, "discriminator_weights_final.weights.h5"))

# Load dataset manifest & 100 fixed latents
manifest_df = pd.read_csv("phase_dcgan_improvement/dataset_research/final_alucard_frontfacing_manifest.csv")
train_df = manifest_df[manifest_df['grouped_split'] == 'train'].reset_index(drop=True)
sample_indices = train_df['archive_index'].iloc[:100].tolist()

raw_ds = load_dataset("evilsocket/alucard-sprites", split="train")

real_uint8 = []
real_norm = []
for idx in sample_indices:
    img = raw_ds[idx]['image'].convert('RGBA').resize((64, 64), Image.Resampling.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    real_uint8.append(arr)
    # Normalized
    arr_f32 = arr.astype(np.float32)
    transparent_mask = arr_f32[:, :, 3] < 10.0
    arr_f32[transparent_mask, :3] = 0.0
    arr_f32[transparent_mask, 3] = 0.0
    real_norm.append((arr_f32 / 127.5) - 1.0)

real_uint8 = np.array(real_uint8, dtype=np.uint8)
real_norm = np.array(real_norm, dtype=np.float32)

fixed_z = np.load(os.path.join(RESULTS_DIR, "config", "fixed_latent_vectors.npy"))
gen_norm = gen(fixed_z, training=False).numpy()
gen_uint8 = np.clip((gen_norm + 1.0) * 127.5, 0, 255).astype(np.uint8)

print("Dataset and generated samples loaded successfully.")

# ==============================================================================
# SECTION 1: DERIVATION & MEASUREMENT OF EXACT LSGAN GRADIENTS
# ==============================================================================
print("\n--- SECTION 1: Exact Gradient Measurement ---")
z_batch = tf.convert_to_tensor(fixed_z[:32])
real_batch = tf.convert_to_tensor(real_norm[:32])

with tf.GradientTape(persistent=True) as tape:
    fake_b = gen(z_batch, training=True)
    tape.watch(fake_b)
    fake_logits, fake_feats = disc(fake_b, training=False)
    _, real_feats = disc(real_batch, training=False)
    
    # Adversarial Loss
    g_adv_loss = 0.5 * tf.reduce_mean(tf.square(fake_logits - 1.0))
    # Feature Matching Loss (Notebook implementation: unpaired elementwise)
    fm_loss_unpaired = tf.reduce_mean(tf.abs(real_feats - fake_feats))
    # Corrected Feature Matching Loss (True centroid distance)
    fm_loss_centroid = tf.reduce_mean(tf.abs(tf.reduce_mean(real_feats, axis=0) - tf.reduce_mean(fake_feats, axis=0)))
    
    g_total_loss = g_adv_loss + 0.20 * fm_loss_unpaired

# 1. Scalar derivative dL / d(D(fake))
scalar_deriv = fake_logits - 1.0
print(f"Mean Fake Logit D(G(z)):                {float(tf.reduce_mean(fake_logits)):.4f}")
print(f"Scalar Derivative dL_adv / d(logit):    {float(tf.reduce_mean(scalar_deriv)):.4f}")

# 2. Discriminator input Jacobian ||dD/dx||
d_input_grad = tape.gradient(fake_logits, fake_b)
d_input_grad_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(d_input_grad), axis=[1, 2, 3])))
print(f"Discriminator Input Jacobian ||dD/dx||: {float(d_input_grad_norm):.4f}")

# 3. Generator Parameter Gradients
g_adv_grads = tape.gradient(g_adv_loss, gen.trainable_variables)
g_fm_grads = tape.gradient(fm_loss_unpaired, gen.trainable_variables)
g_tot_grads = tape.gradient(g_total_loss, gen.trainable_variables)

g_adv_grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in g_adv_grads if g is not None]))
g_fm_grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in g_fm_grads if g is not None]))
g_tot_grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in g_tot_grads if g is not None]))

print(f"Generator Adv Gradient Global Norm:     {float(g_adv_grad_norm):.4f}")
print(f"Generator FM Gradient Global Norm:      {float(g_fm_grad_norm):.4f}")
print(f"Total Generator Gradient Global Norm:   {float(g_tot_grad_norm):.4f}")
print(f"Ratio of (0.20 * FM_Grad) / Adv_Grad:   {float(0.20 * g_fm_grad_norm / g_adv_grad_norm):.4f}")

# ==============================================================================
# SECTION 2: RIGOROUS 8-BIT INTEGER COLOUR & PALETTE ANALYSIS
# ==============================================================================
print("\n--- SECTION 2: 8-Bit Integer Colour & Palette Analysis ---")

def analyze_sprite_palette(arr_uint8):
    alpha = arr_uint8[:, :, 3]
    fg_mask = alpha > 20
    if np.sum(fg_mask) == 0:
        return {
            "unique_8bit": 0, "entropy": 0.0, "top4_pct": 100.0, "top8_pct": 100.0,
            "quantized_16": 0, "quantized_32": 0, "spatial_coherence": 1.0
        }
    
    fg_pixels = arr_uint8[fg_mask, :3]
    total_fg_pixels = len(fg_pixels)
    
    # Exact unique 8-bit colors
    unique_colors, counts = np.unique(fg_pixels, axis=0, return_counts=True)
    n_unique = len(unique_colors)
    
    # Palette entropy
    probs = counts / total_fg_pixels
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    # Dominant color frequencies (Top 4 and Top 8 coverage)
    sorted_counts = np.sort(counts)[::-1]
    top4_pct = (np.sum(sorted_counts[:4]) / total_fg_pixels) * 100.0
    top8_pct = (np.sum(sorted_counts[:8]) / total_fg_pixels) * 100.0
    
    # 5-bit quantized palette (32 bins per channel: 32x32x32 = 32,768 space)
    fg_quantized_32 = (fg_pixels // 8) * 8
    n_quant_32 = len(np.unique(fg_quantized_32, axis=0))
    
    # 4-bit quantized palette (16 bins per channel: 16x16x16 = 4,096 space)
    fg_quantized_16 = (fg_pixels // 16) * 16
    n_quant_16 = len(np.unique(fg_quantized_16, axis=0))
    
    # Spatial color coherence: Probability that horizontal or vertical neighbor in foreground has same color (<= 10 L1 distance)
    rgb = arr_uint8[:, :, :3].astype(np.float32)
    # Horizontal pairs where both are foreground
    m_h = fg_mask[:, :-1] & fg_mask[:, 1:]
    diff_h = np.sum(np.abs(rgb[:, :-1] - rgb[:, 1:]), axis=2)
    same_h = (diff_h <= 10.0) & m_h
    
    # Vertical pairs
    m_v = fg_mask[:-1, :] & fg_mask[1:, :]
    diff_v = np.sum(np.abs(rgb[:-1, :] - rgb[1:, :]), axis=2)
    same_v = (diff_v <= 10.0) & m_v
    
    total_neighbor_pairs = np.sum(m_h) + np.sum(m_v)
    same_neighbor_pairs = np.sum(same_h) + np.sum(same_v)
    spatial_coherence = (same_neighbor_pairs / max(1, total_neighbor_pairs)) * 100.0
    
    return {
        "unique_8bit": n_unique,
        "entropy": float(entropy),
        "top4_pct": float(top4_pct),
        "top8_pct": float(top8_pct),
        "quantized_16": n_quant_16,
        "quantized_32": n_quant_32,
        "spatial_coherence": float(spatial_coherence)
    }

real_palette_stats = [analyze_sprite_palette(im) for im in real_uint8]
gen_palette_stats = [analyze_sprite_palette(im) for im in gen_uint8]

def get_mean_std(stat_list, key):
    vals = [s[key] for s in stat_list]
    return np.mean(vals), np.std(vals), np.median(vals)

print("8-Bit Rigorous Palette Metrics (Mean ± Std [Median]):")
for k in ["unique_8bit", "entropy", "top4_pct", "top8_pct", "quantized_32", "quantized_16", "spatial_coherence"]:
    r_m, r_s, r_med = get_mean_std(real_palette_stats, k)
    g_m, g_s, g_med = get_mean_std(gen_palette_stats, k)
    print(f"  {k:<18} | Real: {r_m:6.2f} ± {r_s:5.2f} [{r_med:6.2f}] | Gen: {g_m:6.2f} ± {g_s:5.2f} [{g_med:6.2f}]")

# ==============================================================================
# SECTION 3: RGBA REPRESENTATION DIAGNOSTIC BATCH TEST
# ==============================================================================
print("\n--- SECTION 3: Diagnostic Batch RGBA Representation Testing ---")
# Test 4 representations on 32 Real and 32 Generated:
# A. Existing RGBA
# B. Canonicalized Transparent RGB (RGB = -1.0 where alpha < -0.8)
# C. Premultiplied Alpha: RGB_pre = RGB * ((alpha + 1)/2)
# D. Hard Binary Alpha: alpha in {-1.0, +1.0}

def test_representations(tensor_norm):
    results = {}
    
    # 1. Existing
    l1, _ = disc(tensor_norm, training=False)
    results["existing"] = {
        "mean_logit": float(tf.reduce_mean(l1)),
        "std_logit": float(tf.math.reduce_std(l1)),
        "mean_input_var": float(tf.reduce_mean(tf.math.reduce_variance(tensor_norm, axis=[1, 2, 3])))
    }
    
    # 2. Canonicalized (Transparent RGB = -1.0)
    canon = tensor_norm.copy()
    for i in range(len(canon)):
        m = canon[i, :, :, 3] < -0.8
        canon[i, m, :3] = -1.0
    l2, _ = disc(canon, training=False)
    results["canonicalized"] = {
        "mean_logit": float(tf.reduce_mean(l2)),
        "std_logit": float(tf.math.reduce_std(l2)),
        "mean_input_var": float(tf.reduce_mean(tf.math.reduce_variance(canon, axis=[1, 2, 3])))
    }
    
    # 3. Premultiplied Alpha
    premul = tensor_norm.copy()
    alpha_01 = (premul[:, :, :, 3:4] + 1.0) / 2.0
    rgb_01 = (premul[:, :, :, :3] + 1.0) / 2.0
    rgb_pre_01 = rgb_01 * alpha_01
    premul[:, :, :, :3] = rgb_pre_01 * 2.0 - 1.0
    l3, _ = disc(premul, training=False)
    results["premultiplied"] = {
        "mean_logit": float(tf.reduce_mean(l3)),
        "std_logit": float(tf.math.reduce_std(l3)),
        "mean_input_var": float(tf.reduce_mean(tf.math.reduce_variance(premul, axis=[1, 2, 3])))
    }
    
    # 4. Binary Alpha
    bin_alpha = tensor_norm.copy()
    bin_alpha[:, :, :, 3] = np.where(bin_alpha[:, :, :, 3] > 0.0, 1.0, -1.0)
    l4, _ = disc(bin_alpha, training=False)
    results["binary_alpha"] = {
        "mean_logit": float(tf.reduce_mean(l4)),
        "std_logit": float(tf.math.reduce_std(l4)),
        "mean_input_var": float(tf.reduce_mean(tf.math.reduce_variance(bin_alpha, axis=[1, 2, 3])))
    }
    
    return results

real_diag = test_representations(real_norm[:32])
gen_diag = test_representations(gen_norm[:32])

print(f"{'Representation':<16} | {'Real Mean Logit':<16} | {'Gen Mean Logit':<16} | {'Real Inp Var':<12} | {'Gen Inp Var':<12}")
for rep in ["existing", "canonicalized", "premultiplied", "binary_alpha"]:
    print(f"{rep:<16} | {real_diag[rep]['mean_logit']:<16.4f} | {gen_diag[rep]['mean_logit']:<16.4f} | {real_diag[rep]['mean_input_var']:<12.4f} | {gen_diag[rep]['mean_input_var']:<12.4f}")

# ==============================================================================
# SECTION 4: SPATIAL TOTAL VARIATION: INTERIOR VS BOUNDARIES
# ==============================================================================
print("\n--- SECTION 4: Spatial TV Analysis: Interior vs Boundary ---")

def compute_decomposed_tv(arr_uint8):
    alpha = arr_uint8[:, :, 3]
    fg_mask = alpha > 20
    
    if np.sum(fg_mask) < 20:
        return 0.0, 0.0, 0.0, 0.0
    
    # Pure numpy binary morphological erosion for 64x64
    # Interior mask: pixel and all 4 neighbors are foreground
    pad_m = np.pad(fg_mask, 1, mode='constant', constant_values=False)
    interior_mask = pad_m[1:-1, 1:-1] & pad_m[:-2, 1:-1] & pad_m[2:, 1:-1] & pad_m[1:-1, :-2] & pad_m[1:-1, 2:]
    
    # Boundary mask: foreground pixels that are NOT interior (1-2 pixel border)
    boundary_mask = fg_mask & (~interior_mask)
    
    rgb = arr_uint8[:, :, :3].astype(np.float32)
    diff_h = np.abs(rgb[1:, :, :] - rgb[:-1, :, :])
    diff_w = np.abs(rgb[:, 1:, :] - rgb[:, :-1, :])
    
    # 1. Total Foreground TV
    m_fg_h = fg_mask[1:, :] & fg_mask[:-1, :]
    m_fg_w = fg_mask[:, 1:] & fg_mask[:, :-1]
    tv_fg = (np.sum(diff_h[m_fg_h]) + np.sum(diff_w[m_fg_w])) / max(1.0, np.sum(m_fg_h) + np.sum(m_fg_w))
    
    # 2. Interior TV (Within solid armor/skin/cloth)
    m_int_h = interior_mask[1:, :] & interior_mask[:-1, :]
    m_int_w = interior_mask[:, 1:] & interior_mask[:, :-1]
    tv_int = (np.sum(diff_h[m_int_h]) + np.sum(diff_w[m_int_w])) / max(1.0, np.sum(m_int_h) + np.sum(m_int_w))
    
    # 3. Boundary TV (Outlines and edge transitions)
    m_bnd_h = boundary_mask[1:, :] | boundary_mask[:-1, :]
    m_bnd_w = boundary_mask[:, 1:] | boundary_mask[:, :-1]
    tv_bnd = (np.sum(diff_h[m_bnd_h & m_fg_h]) + np.sum(diff_w[m_bnd_w & m_fg_w])) / max(1.0, np.sum(m_bnd_h & m_fg_h) + np.sum(m_bnd_w & m_fg_w))
    
    # 4. Charbonnier / Edge-Preserving TV: sqrt(diff^2 + eps^2)
    eps = 5.0
    charb_h = np.sqrt(diff_h**2 + eps**2) - eps
    charb_w = np.sqrt(diff_w**2 + eps**2) - eps
    charb_fg = (np.sum(charb_h[m_fg_h]) + np.sum(charb_w[m_fg_w])) / max(1.0, np.sum(m_fg_h) + np.sum(m_fg_w))
    
    return float(tv_fg), float(tv_int), float(tv_bnd), float(charb_fg)

real_tvs = [compute_decomposed_tv(im) for im in real_uint8]
gen_tvs = [compute_decomposed_tv(im) for im in gen_uint8]

print(f"{'Region':<25} | {'Real Sprites (Mean)':<20} | {'Gen Sprites (Mean)':<20} | {'Ratio (Gen/Real)':<16}")
regions = ["Total Foreground TV", "Interior Flat TV", "Boundary Outline TV", "Charbonnier Soft TV"]
for idx, reg in enumerate(regions):
    r_val = np.mean([t[idx] for t in real_tvs])
    g_val = np.mean([t[idx] for t in gen_tvs])
    print(f"{reg:<25} | {r_val:<20.2f} | {g_val:<20.2f} | {g_val/r_val:<16.2f}")

# ==============================================================================
# SECTION 5: FEATURE MATCHING MATHEMATICAL AUDIT
# ==============================================================================
print("\n--- SECTION 5: Feature Matching Mathematical Audit ---")
print(f"Unpaired Mini-Batch FM Loss:            {float(fm_loss_unpaired):.4f}")
print(f"True Centroid-Matching FM Loss:         {float(fm_loss_centroid):.4f}")
print(f"Divergence Factor (Unpaired / Centroid): {float(fm_loss_unpaired / fm_loss_centroid):.2f}x")

# Save all quantitative data into a comprehensive JSON file
all_results = {
    "gradient_diagnostics": {
        "mean_fake_logit": float(tf.reduce_mean(fake_logits)),
        "scalar_dL_dlogit": float(tf.reduce_mean(scalar_deriv)),
        "d_input_jacobian_norm": float(d_input_grad_norm),
        "g_adv_grad_norm": float(g_adv_grad_norm),
        "g_fm_grad_norm": float(g_fm_grad_norm),
        "g_total_grad_norm": float(g_tot_grad_norm),
        "fm_to_adv_grad_ratio": float(0.20 * g_fm_grad_norm / g_adv_grad_norm)
    },
    "palette_diagnostics_8bit": {
        "real": {k: {"mean": float(np.mean([s[k] for s in real_palette_stats])), "std": float(np.std([s[k] for s in real_palette_stats])), "median": float(np.median([s[k] for s in real_palette_stats]))} for k in real_palette_stats[0].keys()},
        "gen": {k: {"mean": float(np.mean([s[k] for s in gen_palette_stats])), "std": float(np.std([s[k] for s in gen_palette_stats])), "median": float(np.median([s[k] for s in gen_palette_stats]))} for k in gen_palette_stats[0].keys()}
    },
    "rgba_representation_diagnostics": {
        "real": real_diag,
        "gen": gen_diag
    },
    "tv_decomposed_diagnostics": {
        "total_foreground": {"real": float(np.mean([t[0] for t in real_tvs])), "gen": float(np.mean([t[0] for t in gen_tvs]))},
        "interior_flat": {"real": float(np.mean([t[1] for t in real_tvs])), "gen": float(np.mean([t[1] for t in gen_tvs]))},
        "boundary_outline": {"real": float(np.mean([t[2] for t in real_tvs])), "gen": float(np.mean([t[2] for t in gen_tvs]))},
        "charbonnier_soft": {"real": float(np.mean([t[3] for t in real_tvs])), "gen": float(np.mean([t[3] for t in gen_tvs]))}
    },
    "feature_matching_diagnostics": {
        "unpaired_fm_loss": float(fm_loss_unpaired),
        "centroid_fm_loss": float(fm_loss_centroid),
        "ratio": float(fm_loss_unpaired / fm_loss_centroid)
    }
}

with open(os.path.join(OUTPUT_DIR, "rigorous_corrections_audit.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print("\nSuccessfully saved rigorous_corrections_audit.json.")
print("="*80)
