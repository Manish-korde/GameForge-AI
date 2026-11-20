import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

# Setup output paths
RESULTS_DIR = r"C:\Users\manis\Downloads\dcgan_frontfacing_46k_50ep_results"
OUTPUT_DIR = "phase_dcgan_improvement/solution_investigation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. Loading Real Dataset Manifest...")
manifest_path = "phase_dcgan_improvement/dataset_research/final_alucard_frontfacing_manifest.csv"
manifest_df = pd.read_csv(manifest_path)
train_df = manifest_df[manifest_df['grouped_split'] == 'train'].reset_index(drop=True)
print(f"Total training manifest sprites: {len(train_df):,}")

# Select first 200 deterministic training sprites
sample_indices = train_df['archive_index'].iloc[:200].tolist()

print("2. Ingesting 200 Real Sprites from evilsocket/alucard-sprites...")
raw_ds = load_dataset("evilsocket/alucard-sprites", split="train")

IMG_WIDTH, IMG_HEIGHT, CHANNELS = 64, 64, 4

real_raw_rgba = []         # Unmodified raw [0, 255]
real_preprocessed = []     # Normalized [-1, 1] with alpha mask cleanup
real_raw_transparent_rgb = []

for idx in sample_indices:
    item = raw_ds[idx]
    img = item['image'].convert('RGBA')
    img_64 = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.NEAREST)
    arr_uint8 = np.array(img_64, dtype=np.uint8)
    real_raw_rgba.append(arr_uint8)
    
    # Check raw transparent pixels
    mask_zero_alpha = arr_uint8[:, :, 3] < 10
    if np.any(mask_zero_alpha):
        trans_rgb = arr_uint8[mask_zero_alpha, :3]
        real_raw_transparent_rgb.extend(trans_rgb.tolist())
        
    # Preprocessed as in notebook Cell 9
    arr_f32 = arr_uint8.astype(np.float32)
    transparent_mask = arr_f32[:, :, 3] < 10.0
    arr_f32[transparent_mask, :3] = 0.0
    arr_f32[transparent_mask, 3] = 0.0
    norm_arr = (arr_f32 / 127.5) - 1.0
    real_preprocessed.append(norm_arr)

real_raw_rgba = np.array(real_raw_rgba, dtype=np.uint8)              # (200, 64, 64, 4)
real_preprocessed = np.array(real_preprocessed, dtype=np.float32)    # (200, 64, 64, 4)
print(f"Loaded {len(real_raw_rgba)} real sprites.")

print("3. Synthesizing 100 Generated Sprites using final model weights...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

fixed_z = np.load(os.path.join(RESULTS_DIR, "config", "fixed_latent_vectors.npy"))
gen_preds = gen(fixed_z, training=False).numpy()  # (100, 64, 64, 4) in [-1, 1]
gen_uint8 = np.clip((gen_preds + 1.0) * 127.5, 0, 255).astype(np.uint8)

print("4. Computing Quantitative Color, Alpha, and Speckling Metrics...")

# A. Unique colors per sprite (on RGB where alpha > 20)
def count_unique_colors(img_rgba_uint8):
    alpha = img_rgba_uint8[:, :, 3]
    visible_pixels = img_rgba_uint8[alpha > 20, :3]
    if len(visible_pixels) == 0:
        return 0
    unique = np.unique(visible_pixels, axis=0)
    return len(unique)

real_unique_colors = [count_unique_colors(im) for im in real_raw_rgba]
gen_unique_colors = [count_unique_colors(im) for im in gen_uint8]

# B. Alpha distributions
# Binary vs intermediate
def alpha_stats(alpha_channel):
    # binary: < 10 or > 245
    binary_pct = np.mean((alpha_channel < 10) | (alpha_channel > 245)) * 100.0
    semi_trans_pct = np.mean((alpha_channel >= 10) & (alpha_channel <= 245)) * 100.0
    return binary_pct, semi_trans_pct

real_alpha_stats = [alpha_stats(im[:, :, 3]) for im in real_raw_rgba]
gen_alpha_stats = [alpha_stats(im[:, :, 3]) for im in gen_uint8]

# C. Color Speckling / High-Frequency Noise (Laplacian variance on RGB)
def color_speckle_metric(img_rgba_uint8):
    # Convert visible character to grayscale in pure numpy
    rgb = img_rgba_uint8[:, :, :3].astype(np.float32)
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    mask = (img_rgba_uint8[:, :, 3] > 20).astype(np.uint8)
    if np.sum(mask) == 0:
        return 0.0
    # Discrete 2D Laplacian using 3x3 kernel
    # [ 0,  1,  0]
    # [ 1, -4,  1]
    # [ 0,  1,  0]
    padded = np.pad(gray, 1, mode='edge')
    lap = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * padded[1:-1, 1:-1])
    val = np.var(lap[mask > 0])
    return float(val)

# Also measure total variation inside character
def tv_norm(img_rgba_uint8):
    alpha = (img_rgba_uint8[:, :, 3] > 20).astype(np.float32)
    rgb = img_rgba_uint8[:, :, :3].astype(np.float32)
    diff_h = np.abs(rgb[1:, :, :] - rgb[:-1, :, :]) * alpha[1:, :, np.newaxis] * alpha[:-1, :, np.newaxis]
    diff_w = np.abs(rgb[:, 1:, :] - rgb[:, :-1, :]) * alpha[:, 1:, np.newaxis] * alpha[:, :-1, np.newaxis]
    norm_val = np.sum(diff_h) + np.sum(diff_w)
    char_area = max(1.0, np.sum(alpha))
    return float(norm_val / char_area)

real_speckle = [color_speckle_metric(im) for im in real_raw_rgba]
gen_speckle = [color_speckle_metric(im) for im in gen_uint8]

real_tv = [tv_norm(im) for im in real_raw_rgba]
gen_tv = [tv_norm(im) for im in gen_uint8]

# D. Background Pixel RGB Corruption
# In real raw images, do transparent pixels have arbitrary RGB values?
real_trans_rgb_arr = np.array(real_raw_transparent_rgb)
print(f"Raw real transparent pixel RGB sample size: {len(real_trans_rgb_arr)}")
if len(real_trans_rgb_arr) > 0:
    print(f"Raw real transparent RGB mean: {np.mean(real_trans_rgb_arr, axis=0)}, std: {np.std(real_trans_rgb_arr, axis=0)}")
    print(f"Percentage of transparent pixels with non-zero RGB in RAW Alucard: {np.mean(np.any(real_trans_rgb_arr > 0, axis=1))*100:.2f}%")

# In generated images, what are the RGB values where alpha < 20?
gen_trans_rgb = []
for im in gen_uint8:
    m = im[:, :, 3] < 20
    if np.any(m):
        gen_trans_rgb.extend(im[m, :3].tolist())
gen_trans_rgb_arr = np.array(gen_trans_rgb)
print(f"Generated transparent RGB mean: {np.mean(gen_trans_rgb_arr, axis=0)}, std: {np.std(gen_trans_rgb_arr, axis=0)}")

# E. Discriminator Sensitivity Test: Does D exploit transparent RGB?
print("5. Testing Discriminator Sensitivity to Transparent RGB vs Alpha...")
# Test 1: Real preprocessed images -> D scores
real_sample_tensor = real_preprocessed[:50]
real_logits, _ = disc(real_sample_tensor, training=False)
mean_real_logit = float(tf.reduce_mean(real_logits))

# Test 2: Fake generated images -> D scores
fake_sample_tensor = gen_preds[:50]
fake_logits, _ = disc(fake_sample_tensor, training=False)
mean_fake_logit = float(tf.reduce_mean(fake_logits))

# Test 3: Real images with NOISY transparent RGB (corrupting the background)
real_corrupted_bg = real_sample_tensor.copy()
for i in range(len(real_corrupted_bg)):
    bg_mask = real_corrupted_bg[i, :, :, 3] < -0.8
    # inject random RGB in [-1, 1] into transparent region
    noise_rgb = np.random.uniform(-1.0, 1.0, size=(np.sum(bg_mask), 3))
    real_corrupted_bg[i, bg_mask, :3] = noise_rgb

real_corrupted_logits, _ = disc(real_corrupted_bg, training=False)
mean_corrupted_logit = float(tf.reduce_mean(real_corrupted_logits))

# Test 4: Fake images with ZEROED-OUT transparent RGB
fake_cleaned_bg = fake_sample_tensor.copy()
for i in range(len(fake_cleaned_bg)):
    bg_mask = fake_cleaned_bg[i, :, :, 3] < -0.8
    fake_cleaned_bg[i, bg_mask, :3] = -1.0 # zero in tanh space
    fake_cleaned_bg[i, bg_mask, 3] = -1.0

fake_cleaned_logits, _ = disc(fake_cleaned_bg, training=False)
mean_cleaned_fake_logit = float(tf.reduce_mean(fake_cleaned_logits))

print(f"Discriminator Diagnostic Scores (Target Real = +0.90, Target Fake = 0.00):")
print(f"  Real Images Score (Pristine):            {mean_real_logit:.4f}")
print(f"  Real Images with Noisy Background RGB:   {mean_corrupted_logit:.4f}")
print(f"  Fake Images Score (Raw Output):          {mean_fake_logit:.4f}")
print(f"  Fake Images with Cleaned Background RGB: {mean_cleaned_fake_logit:.4f}")

# Save quantitative summary json
metrics_summary = {
    "real_unique_colors": {
        "mean": float(np.mean(real_unique_colors)),
        "median": float(np.median(real_unique_colors)),
        "min": int(np.min(real_unique_colors)),
        "max": int(np.max(real_unique_colors)),
        "p90": float(np.percentile(real_unique_colors, 90))
    },
    "gen_unique_colors": {
        "mean": float(np.mean(gen_unique_colors)),
        "median": float(np.median(gen_unique_colors)),
        "min": int(np.min(gen_unique_colors)),
        "max": int(np.max(gen_unique_colors)),
        "p90": float(np.percentile(gen_unique_colors, 90))
    },
    "real_alpha_stats": {
        "mean_binary_pct": float(np.mean([s[0] for s in real_alpha_stats])),
        "mean_semi_trans_pct": float(np.mean([s[1] for s in real_alpha_stats]))
    },
    "gen_alpha_stats": {
        "mean_binary_pct": float(np.mean([s[0] for s in gen_alpha_stats])),
        "mean_semi_trans_pct": float(np.mean([s[1] for s in gen_alpha_stats]))
    },
    "speckle_laplacian_var": {
        "real_mean": float(np.mean(real_speckle)),
        "gen_mean": float(np.mean(gen_speckle))
    },
    "total_variation_per_pixel": {
        "real_mean": float(np.mean(real_tv)),
        "gen_mean": float(np.mean(gen_tv))
    },
    "discriminator_experiment": {
        "mean_real_logit": mean_real_logit,
        "mean_corrupted_real_logit": mean_corrupted_logit,
        "mean_fake_logit": mean_fake_logit,
        "mean_cleaned_fake_logit": mean_cleaned_fake_logit
    }
}

with open(os.path.join(OUTPUT_DIR, "forensic_metrics_summary.json"), "w") as f:
    json.dump(metrics_summary, f, indent=2)
print("Saved forensic_metrics_summary.json.")

print("6. Generating Visual Evidence Comparison Figures...")

# Figure 1: Real vs Generated Image Contact Sheet (10x10 comparison: 5 rows real, 5 rows fake)
fig, axes = plt.subplots(10, 10, figsize=(16, 16))
for r in range(5):
    for c in range(10):
        idx = r * 10 + c
        ax = axes[r, c]
        # Real sprite composite on subtle checkered background
        im = real_raw_rgba[idx]
        ax.imshow(im)
        ax.axis('off')
        if r == 0 and c == 0:
            ax.set_title("Real Sprites (Rows 1-5)", fontsize=10, loc='left', color='blue', weight='bold')

for r in range(5, 10):
    for c in range(10):
        idx = (r - 5) * 10 + c
        ax = axes[r, c]
        im = gen_uint8[idx]
        ax.imshow(im)
        ax.axis('off')
        if r == 5 and c == 0:
            ax.set_title("Generated Sprites (Rows 6-10)", fontsize=10, loc='left', color='red', weight='bold')

plt.suptitle("GameForge AI Forensic Comparison: Real Training Sprites vs DCGAN 46K Generated Samples", fontsize=15, y=0.92)
plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "real_vs_generated_contact_sheet.png")
plt.savefig(fig1_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path}")

# Figure 2: RGB Channel Decomposition Comparison (4 Real vs 4 Generated)
fig, axes = plt.subplots(8, 4, figsize=(12, 18))
cols = ["RGB Composite", "Red Channel", "Green Channel", "Blue Channel"]
for c, col_name in enumerate(cols):
    axes[0, c].set_title(col_name, fontsize=12, pad=10)

for i in range(4):
    row = i
    im = real_raw_rgba[i]
    axes[row, 0].imshow(im)
    axes[row, 0].set_ylabel(f"Real {i+1}", fontsize=10, color='blue', weight='bold')
    axes[row, 1].imshow(im[:, :, 0], cmap='Reds_r')
    axes[row, 2].imshow(im[:, :, 1], cmap='Greens_r')
    axes[row, 3].imshow(im[:, :, 2], cmap='Blues_r')
    for c in range(4):
        axes[row, c].set_xticks([])
        axes[row, c].set_yticks([])

for i in range(4):
    row = 4 + i
    im = gen_uint8[i]
    axes[row, 0].imshow(im)
    axes[row, 0].set_ylabel(f"Generated {i+1}", fontsize=10, color='red', weight='bold')
    axes[row, 1].imshow(im[:, :, 0], cmap='Reds_r')
    axes[row, 2].imshow(im[:, :, 1], cmap='Greens_r')
    axes[row, 3].imshow(im[:, :, 2], cmap='Blues_r')
    for c in range(4):
        axes[row, c].set_xticks([])
        axes[row, c].set_yticks([])

plt.suptitle("RGB Channel Spatial Decomposition: Coherent Real Ramps vs Noisy High-Frequency Speckle", fontsize=14, y=0.92)
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "rgb_channel_comparison.png")
plt.savefig(fig2_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

# Figure 3: Alpha Channel Decomposition & Edge Profile Comparison
fig, axes = plt.subplots(4, 4, figsize=(14, 12))
for i in range(4):
    # Real
    im_r = real_raw_rgba[i]
    axes[i, 0].imshow(im_r)
    axes[i, 0].set_title(f"Real #{i+1} RGBA")
    axes[i, 0].axis('off')
    
    alpha_r = im_r[:, :, 3]
    im_alpha_r = axes[i, 1].imshow(alpha_r, cmap='gray', vmin=0, vmax=255)
    axes[i, 1].set_title(f"Real #{i+1} Alpha (Binary)")
    axes[i, 1].axis('off')
    
    # Fake
    im_f = gen_uint8[i]
    axes[i, 2].imshow(im_f)
    axes[i, 2].set_title(f"Generated #{i+1} RGBA")
    axes[i, 2].axis('off')
    
    alpha_f = im_f[:, :, 3]
    axes[i, 3].imshow(alpha_f, cmap='gray', vmin=0, vmax=255)
    axes[i, 3].set_title(f"Generated #{i+1} Alpha (Halo)")
    axes[i, 3].axis('off')

plt.suptitle("Alpha Transparency Comparison: Real Discrete Silhouettes vs Generated Smudged Boundaries", fontsize=14, y=0.92)
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "alpha_channel_comparison.png")
plt.savefig(fig3_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3_path}")

# Figure 4: Colour Distribution Comparison (Histograms & Unique Color Counts)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Unique color count distribution
ax1.hist(real_unique_colors, bins=25, alpha=0.7, color='blue', label=f'Real Sprites (μ={np.mean(real_unique_colors):.1f})', density=True)
ax1.hist(gen_unique_colors, bins=25, alpha=0.7, color='red', label=f'Generated Sprites (μ={np.mean(gen_unique_colors):.1f})', density=True)
ax1.set_title("Unique Colors Per Sprite\n(Pixel-Art Palette vs Continuous Spectrum)", fontsize=11)
ax1.set_xlabel("Unique RGB Colors in Visible Silhouette")
ax1.set_ylabel("Density")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Alpha Channel Histogram
real_all_alphas = real_raw_rgba[:, :, :, 3].flatten()
gen_all_alphas = gen_uint8[:, :, :, 3].flatten()
r_pct = metrics_summary["real_alpha_stats"]["mean_binary_pct"]
g_pct = metrics_summary["gen_alpha_stats"]["mean_binary_pct"]
ax2.hist(real_all_alphas, bins=50, alpha=0.7, color='blue', label=f'Real Alpha ({r_pct:.1f}% binary)', density=True, log=True)
ax2.hist(gen_all_alphas, bins=50, alpha=0.7, color='red', label=f'Generated Alpha ({g_pct:.1f}% binary)', density=True, log=True)
ax2.set_title("Alpha Channel Distribution (Log Scale)\nBinary Crispness vs Continuous Halo", fontsize=11)
ax2.set_xlabel("Alpha Channel Value [0-255]")
ax2.set_ylabel("Log Density")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Color Speckle / High-Frequency Gradient Variance
ax3.hist(real_speckle, bins=25, alpha=0.7, color='blue', label=f'Real Laplacian Var (μ={np.mean(real_speckle):.1f})', density=True)
ax3.hist(gen_speckle, bins=25, alpha=0.7, color='red', label=f'Generated Laplacian Var (μ={np.mean(gen_speckle):.1f})', density=True)
ax3.set_title("Color Speckling Index (Laplacian Spatial Variance)\nSmooth Flat Regions vs High-Freq Noise", fontsize=11)
ax3.set_xlabel("Variance of Laplacian on Foreground RGB")
ax3.set_ylabel("Density")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, "colour_distribution_comparison.png")
plt.savefig(fig4_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {fig4_path}")

# Figure 5: Representative Pixel-Art Palette Swatch Analysis (Top 3 Real vs Top 3 Fake)
from collections import Counter

def extract_palette(im_rgba_uint8, k=16):
    alpha = im_rgba_uint8[:, :, 3]
    vis = im_rgba_uint8[alpha > 20, :3]
    if len(vis) == 0:
        return [(0,0,0)] * k
    tuples = [tuple(p) for p in vis]
    counts = Counter(tuples)
    most_common = [c[0] for c in counts.most_common(k)]
    while len(most_common) < k:
        most_common.append((0,0,0))
    return most_common

fig, axes = plt.subplots(6, 2, figsize=(10, 12), gridspec_kw={'width_ratios': [1, 4]})

# 3 Real
for i in range(3):
    r_ax_img = axes[i, 0]
    r_ax_pal = axes[i, 1]
    im = real_raw_rgba[i]
    r_ax_img.imshow(im)
    r_ax_img.set_title(f"Real #{i+1}", fontsize=10)
    r_ax_img.axis('off')
    
    pal = extract_palette(im, k=16)
    pal_img = np.zeros((1, 16, 3), dtype=np.uint8)
    for j, col in enumerate(pal):
        pal_img[0, j] = col
    r_ax_pal.imshow(pal_img, aspect='auto')
    r_ax_pal.set_title(f"Dominant 16 Colors (Cohesive Clustered Palette)", fontsize=10)
    r_ax_pal.set_xticks([])
    r_ax_pal.set_yticks([])

# 3 Fake
for i in range(3):
    f_ax_img = axes[3+i, 0]
    f_ax_pal = axes[3+i, 1]
    im = gen_uint8[i]
    f_ax_img.imshow(im)
    f_ax_img.set_title(f"Generated #{i+1}", fontsize=10)
    f_ax_img.axis('off')
    
    pal = extract_palette(im, k=16)
    pal_img = np.zeros((1, 16, 3), dtype=np.uint8)
    for j, col in enumerate(pal):
        pal_img[0, j] = col
    f_ax_pal.imshow(pal_img, aspect='auto')
    f_ax_pal.set_title(f"Dominant 16 Colors (Muted Muddy Continuous Drift)", fontsize=10)
    f_ax_pal.set_xticks([])
    f_ax_pal.set_yticks([])

plt.suptitle("Palette Extraction: Real Discrete Pixel Ramps vs Generated Disorganized Swatches", fontsize=13, y=0.92)
plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, "pixel_art_palette_analysis.png")
plt.savefig(fig5_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {fig5_path}")

print("Forensic analysis and visual evidence generation 100% COMPLETE.")
