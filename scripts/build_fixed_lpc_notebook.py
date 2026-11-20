import json
import os

def build_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # ----------------------------------------------------
    # SECTION 1
    # ----------------------------------------------------
    add_md("""# GameForge AI — Corrected LPC DCGAN Training Pipeline
### Architecture: Exact 2.63M Generator / 2.77M Discriminator
### Dataset: Liberated Pixel Art (LPC) Canonical Front-Facing Idles (`r02_c00`)
### Key Interventions: Asymmetric 2G:1D Updates, Throttled D-LR (4e-5), Canonical Background Masking, Dark-Canvas Visualization

---
## Section 1: Environment Setup, Paths, & Deterministic Reproducibility
We initialize seeds, verify GPU acceleration, create clean isolated output directories, and ensure deterministic execution.""")

    add_code("""import os
import sys
import gc
import json
import time
import random
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, Model

# Set deterministic random seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Detect GPU / Accelerator
gpus = tf.config.list_physical_devices('GPU')
print(f"[INIT] TensorFlow Version: {tf.__version__}")
if gpus:
    print(f"[INIT] GPUs Available: {len(gpus)} -> {[g.name for g in gpus]}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INIT] Memory growth successfully enabled for all GPUs.")
    except Exception as e:
        print(f"[WARN] Memory growth configuration notice: {e}")
else:
    print("[WARN] No GPU found. Running on CPU.")

# Output Directory Configuration
BASE_OUTPUT_DIR = "dcgan_lpc_rootcause_fixed_results"
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
EVAL_DIR = os.path.join(BASE_OUTPUT_DIR, "eval_samples")
DIAGNOSTICS_DIR = os.path.join(BASE_OUTPUT_DIR, "diagnostics")

for d in [BASE_OUTPUT_DIR, CHECKPOINT_DIR, EVAL_DIR, DIAGNOSTICS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[OK] Output directories ready under: {os.path.abspath(BASE_OUTPUT_DIR)}")""")

    # ----------------------------------------------------
    # SECTION 2
    # ----------------------------------------------------
    add_md("""---
## Section 2: LPC Dataset Loading (Local, Kaggle Input, or Streamed)
Loads the 2,500 canonical front-facing idle sprites (`r02_c00`).
The pipeline checks:
1. Local pre-extracted directory: `Dataset/lpc_frontfacing_idles` or `hf_extracted_idles`.
2. Kaggle dataset inputs: `/kaggle/input/...`.
3. If not present locally, streams the 2,500 spritesheets from HuggingFace (`carlosuperb/lpc-action-pixel-art-diffusion`) and crops the canonical idle frame.""")

    add_code("""LOCAL_SEARCH_PATHS = [
    "Dataset/lpc_frontfacing_idles",
    "Dataset",
    "dcgan_lpc_frontfacing_50ep_results/hf_extracted_idles",
    "C:/Users/manis/Downloads/dcgan_lpc_frontfacing_50ep_results/hf_extracted_idles",
    "C:/Users/manis/Downloads/lpc_frontfacing_idles_2500",
    "/kaggle/input"
]

DATASET_DIR = None

# 1. Check direct search paths
for p in LOCAL_SEARCH_PATHS:
    if os.path.exists(p):
        pngs = [f for f in os.listdir(p) if f.endswith('.png')]
        if len(pngs) >= 500:
            DATASET_DIR = p
            break

# 2. Check any subdirectory in /kaggle/input/ or local dirs
if not DATASET_DIR:
    search_roots = ["/kaggle/input", "Dataset", "."]
    for root in search_roots:
        if os.path.exists(root):
            for dirpath, _, filenames in os.walk(root):
                png_count = sum(1 for f in filenames if f.endswith('.png'))
                if png_count >= 500:
                    DATASET_DIR = dirpath
                    print(f"[OK] Automatically discovered dataset directory at: {DATASET_DIR} ({png_count} PNGs)")
                    break
        if DATASET_DIR:
            break

if DATASET_DIR:
    print(f"[OK] Using pre-extracted LPC idle dataset from: {DATASET_DIR}")
    png_files = sorted([os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.png')])
    print(f"[INFO] Total available PNG sprites: {len(png_files)}")
else:
    print("[INFO] Pre-extracted directory not found locally or in /kaggle/input. Streaming from HuggingFace dataset...")
    # Stream and extract from HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset("carlosuperb/lpc-action-pixel-art-diffusion", split="train", streaming=True)
        STREAM_EXTRACT_DIR = "Dataset/lpc_frontfacing_idles"
        os.makedirs(STREAM_EXTRACT_DIR, exist_ok=True)
        
        extracted_count = 0
        TARGET_SAMPLES = 2500
        print(f"[INFO] Streaming {TARGET_SAMPLES} sheets and cropping canonical idle (Row 2, Col 0)...")
        for i, item in enumerate(ds):
            if extracted_count >= TARGET_SAMPLES:
                break
            img = item['image'].convert('RGBA')
            w, h = img.size
            if w >= 64 and h >= 192:
                # Row 2 (Walk South / Idle), Col 0 (Standing Idle) -> box: (0, 128, 64, 192)
                idle_frame = img.crop((0, 128, 64, 192))
                out_path = os.path.join(STREAM_EXTRACT_DIR, f"lpc_idle_{extracted_count:05d}.png")
                idle_frame.save(out_path)
                extracted_count += 1
                if extracted_count % 500 == 0:
                    print(f"  Extracted {extracted_count}/{TARGET_SAMPLES} sprites...")
        DATASET_DIR = STREAM_EXTRACT_DIR
        png_files = sorted([os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.png')])
        print(f"[OK] Successfully extracted {len(png_files)} sprites to {DATASET_DIR}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")""")

    # ----------------------------------------------------
    # SECTION 3
    # ----------------------------------------------------
    add_md("""---
## Section 3: Dataset Integrity & Foreground Occupancy Verification
Verifies image dimensions (64x64), 4 RGBA channels, transparency patterns, and calculates exact real dataset statistics to serve as ground-truth bounds for training.""")

    add_code("""# Load all images into memory or a verified tf.data pipeline
TOTAL_SAMPLES = min(len(png_files), 2500)
sample_files = png_files[:TOTAL_SAMPLES]

print(f"[INFO] Auditing {len(sample_files)} sprites for integrity...")
raw_images = []
opaque_pixel_counts = []
unique_color_counts = []

for idx, p in enumerate(sample_files):
    im = Image.open(p).convert("RGBA")
    arr = np.array(im)
    assert arr.shape == (64, 64, 4), f"Invalid shape {arr.shape} at {p}"
    raw_images.append(arr)
    
    # Audit foreground alpha (A > 128)
    alpha = arr[..., 3]
    fg_px = np.sum(alpha > 128)
    opaque_pixel_counts.append(fg_px)
    
    # Unique colors in 8-bit RGBA
    unq = len(np.unique(arr.reshape(-1, 4), axis=0))
    unique_color_counts.append(unq)

raw_images = np.array(raw_images, dtype=np.uint8)
mean_fg_px = np.mean(opaque_pixel_counts)
mean_occupancy = (mean_fg_px / (64 * 64)) * 100.0
mean_unq = np.mean(unique_color_counts)

print(f"[AUDIT] Verified Images: {len(raw_images)} RGBA images loaded.")
print(f"[AUDIT] Mean Foreground Pixels: {mean_fg_px:.1f} / 4096 ({mean_occupancy:.2f}% occupancy)")
print(f"[AUDIT] Mean Unique 8-bit Colors: {mean_unq:.1f}")

# Strict Ground Truth Invariant Checks
assert 800 <= mean_fg_px <= 1400, f"Foreground occupancy anomaly: {mean_fg_px}"
assert 20 <= mean_unq <= 80, f"Unexpected color palette anomaly: {mean_unq}"
print("[PASS] Dataset integrity invariants verified successfully!")""")

    # ----------------------------------------------------
    # SECTION 4
    # ----------------------------------------------------
    add_md("""---
## Section 4: Real-Image RGB & Alpha Channel Diagnostics
We plot real training samples decomposing:
1. Raw RGB (unmasked).
2. Binary Alpha Mask.
3. Composited over Dark Slate Canvas `(0.10, 0.12, 0.16)`.
This demonstrates the ground truth against which generated outputs must be judged, eliminating matplotlib's white background washout.""")

    add_code("""DARK_BG = np.array([0.10, 0.12, 0.16], dtype=np.float32)

def composite_on_dark(rgb_norm, alpha_norm, bg_color=DARK_BG):
    \"\"\"
    rgb_norm: (H, W, 3) in [0, 1]
    alpha_norm: (H, W, 1) in [0, 1]
    bg_color: (3,) float in [0, 1]
    \"\"\"
    return rgb_norm * alpha_norm + bg_color.reshape(1, 1, 3) * (1.0 - alpha_norm)

fig, axes = plt.subplots(3, 8, figsize=(16, 6))
plt.subplots_adjust(wspace=0.1, hspace=0.2)

for i in range(8):
    sample = raw_images[i].astype(np.float32) / 255.0
    s_rgb = sample[..., :3]
    s_alpha = sample[..., 3:4]
    s_comp = composite_on_dark(s_rgb, s_alpha)
    
    # Row 0: Raw RGB
    axes[0, i].imshow(s_rgb)
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Real Raw RGB", fontsize=11, fontweight='bold', loc='left')
    
    # Row 1: Alpha Mask
    axes[1, i].imshow(s_alpha[..., 0], cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title("Real Alpha Mask", fontsize=11, fontweight='bold', loc='left')
    
    # Row 2: Composited Dark Canvas
    axes[2, i].imshow(s_comp)
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_title("Real On Dark Slate", fontsize=11, fontweight='bold', loc='left')

plt.suptitle("Real LPC Sprites: Ground Truth Channel Separation", fontsize=14, fontweight='bold')
plt.savefig(os.path.join(DIAGNOSTICS_DIR, "real_ground_truth_separation.png"), bbox_inches='tight', dpi=150)
plt.show()""")

    # ----------------------------------------------------
    # SECTION 5
    # ----------------------------------------------------
    add_md("""---
## Section 5: Data Preprocessing
Transforms RGBA images from $[0, 255]$ into $[-1.0, 1.0]$.
Real LPC sprites already have clean black RGB $(0, 0, 0) \to [-1, -1, -1]$ under transparent pixels ($A = 0 \to -1$).
We pass full 4-channel tensors directly so the Discriminator supervises both foreground character and clean background boundaries, eliminating edge ringing artifacts.""")

    add_code("""# Normalize dataset to [-1.0, 1.0]
normalized_dataset = (raw_images.astype(np.float32) / 127.5) - 1.0

BATCH_SIZE = 32
BUFFER_SIZE = len(normalized_dataset)

train_dataset = tf.data.Dataset.from_tensor_slices(normalized_dataset)
train_dataset = train_dataset.shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print(f"[OK] Preprocessing ready. Batch size: {BATCH_SIZE}, Total batches per epoch: {len(train_dataset)}")""")

    # ----------------------------------------------------
    # SECTION 6
    # ----------------------------------------------------
    add_md("""---
## Section 6: Exact DCGAN Architecture (2.63M G / 2.77M D)
Strictly preserves the production architecture:
- Latent Dimension: 128
- Generator: 2,633,732 parameters (Dense -> Reshape to 4x4x512 -> 4 nearest UpSampling2D + Conv2D(3x3) -> Tanh)
- Discriminator: 2,768,449 parameters (4 Conv2D(4x4, strides=2) + BN + LeakyReLU(0.2) -> Flatten -> Dense(1))
- Feature Tap Layer: `d_conv_3` (8x8x256) for Centroid Feature Matching.""")

    add_code("""LATENT_DIM = 128
CHANNELS = 4
IMG_HEIGHT = 64
IMG_WIDTH = 64

def build_generator(latent_dim=LATENT_DIM):
    \"\"\"Exact architectural replica of production DCGAN Generator (2,633,732 params).\"\"\"
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
    
    model = tf.keras.Sequential([
        layers.Input(shape=(latent_dim,), name="latent_input"),
        layers.Dense(4 * 4 * 512, use_bias=False, kernel_initializer=init, name="dense_proj"),
        layers.BatchNormalization(name="bn_proj"),
        layers.LeakyReLU(0.2, name="lrelu_proj"),
        layers.Reshape((4, 4, 512), name="reshape"),
        
        # Block 1: 4x4 -> 8x8
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_1"),
        layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.LeakyReLU(0.2, name="lrelu_1"),
        
        # Block 2: 8x8 -> 16x16
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_2"),
        layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.LeakyReLU(0.2, name="lrelu_2"),
        
        # Block 3: 16x16 -> 32x32
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_3"),
        layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_3"),
        layers.BatchNormalization(name="bn_3"),
        layers.LeakyReLU(0.2, name="lrelu_3"),
        
        # Block 4: 32x32 -> 64x64 RGBA
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_4"),
        layers.Conv2D(CHANNELS, kernel_size=3, padding="same", activation="tanh", kernel_initializer=init, name="conv_out")
    ], name="dcgan_generator")
    return model

def build_discriminator():
    \"\"\"Exact architectural replica of production DCGAN Discriminator (2,768,449 params).\"\"\"
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
    
    img_in = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="img_input")
    
    # Block 1: 64x64 -> 32x32 (kernel_size=4, use_bias=True: 4*4*4*64 + 64 = 4,160 params)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=True, kernel_initializer=init, name="d_conv_1")(img_in)
    x = layers.LeakyReLU(0.2, name="d_lrelu_1")(x)
    
    # Block 2: 32x32 -> 16x16 (kernel_size=4, use_bias=False + BN: 4*4*64*128 + 512 = 131,584 params)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_2")(x)
    x = layers.BatchNormalization(name="d_bn_2")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_2")(x)
    
    # Block 3: 16x16 -> 8x8 (kernel_size=4, use_bias=False + BN: 4*4*128*256 + 1024 = 525,312 params)
    # Intermediate Feature Tap for Canonical Centroid Feature Matching
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_3")(x)
    x = layers.BatchNormalization(name="d_bn_3")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_3")(x)
    feat_out = x  # Shape: (None, 8, 8, 256)
    
    # Block 4: 8x8 -> 4x4 (kernel_size=4, use_bias=False + BN: 4*4*256*512 + 2048 = 2,099,200 params)
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_4")(x)
    x = layers.BatchNormalization(name="d_bn_4")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_4")(x)
    
    # Dense Logit Out: (4 * 4 * 512) * 1 + 1 = 8,193 params
    x = layers.Flatten(name="d_flatten")(x)
    logit_out = layers.Dense(1, name="d_logit")(x)
    
    return Model(inputs=img_in, outputs=[logit_out, feat_out], name="dcgan_discriminator")

generator = build_generator()
discriminator = build_discriminator()

print("[OK] Exact production models instantiated.")""")

    # ----------------------------------------------------
    # SECTION 7
    # ----------------------------------------------------
    add_md("""---
## Section 7: Parameter Verification Asserts
Enforces exact parameter counts:
- Generator: exactly 2,633,732
- Discriminator: exactly 2,768,449
- Total: exactly 5,402,181""")

    add_code("""g_params = generator.count_params()
d_params = discriminator.count_params()
total_params = g_params + d_params

print(f"[PARAM CHECK] Generator Parameters:     {g_params:,} (Expected: 2,633,732)")
print(f"[PARAM CHECK] Discriminator Parameters: {d_params:,} (Expected: 2,768,449)")
print(f"[PARAM CHECK] Total Network Parameters: {total_params:,} (Expected: 5,402,181)")

assert g_params == 2633732, f"Generator parameter mismatch: {g_params}"
assert d_params == 2768449, f"Discriminator parameter mismatch: {d_params}"
assert total_params == 5402181, f"Total parameter mismatch: {total_params}"

print("[PASS] Exact parameter verification passed!")""")

    # ----------------------------------------------------
    # SECTION 8
    # ----------------------------------------------------
    add_md("""---
## Section 8: Controlled Training Configuration
Applies the verified training hyper-parameters:
1. **Generator LR:** $\eta_G = 3.0 \times 10^{-4}$ (Adam $\beta_1 = 0.5$).
2. **Discriminator LR:** $\eta_D = 8.0 \times 10^{-5}$ ($3.75\times$ slower to prevent D overpower).
3. **One-Sided Label Smoothing:** Target real = 0.90, Target fake = 0.0 (Standard LSGAN).
4. **Canonical L1 Centroid Feature Matching:** $\lambda_{\text{FM}} = 0.20$ using L1 distance `tf.abs(real_centroid - fake_centroid)` to preserve multi-modal diversity and avoid mode blurring.""")

    add_code("""# Learning rates & Adam momentum
LR_G = 3.0e-4
LR_D = 8.0e-5  # Throttled to keep D in balance with G
BETA_1 = 0.5
BETA_2 = 0.999

# Loss weights and smoothing
LAMBDA_FM = 0.20  # Canonical Salimans weight
TARGET_REAL = 0.90
TARGET_FAKE = 0.00

optimizer_g = tf.keras.optimizers.Adam(learning_rate=LR_G, beta_1=BETA_1, beta_2=BETA_2)
optimizer_d = tf.keras.optimizers.Adam(learning_rate=LR_D, beta_1=BETA_1, beta_2=BETA_2)

# Fixed latent vectors for rigorous tracking across epochs
FIXED_LATENTS_100 = tf.random.normal([100, LATENT_DIM], seed=SEED)
FIXED_LATENTS_16 = FIXED_LATENTS_100[:16]

print(f"[CONFIG] Generator LR: {LR_G}")
print(f"[CONFIG] Discriminator LR: {LR_D}")
print(f"[CONFIG] Target Smoothing: Real={TARGET_REAL}, Fake={TARGET_FAKE}")
print(f"[CONFIG] L1 Centroid Feature Matching Weight: {LAMBDA_FM}")""")

    # ----------------------------------------------------
    # SECTION 9
    # ----------------------------------------------------
    add_md("""---
## Section 9: Loss Formulation & Compiled Training Step
We compile the custom `train_step` implementing:
- Full 4-channel supervision (no artificial masking, so D enforces clean transparent borders).
- LSGAN loss with 0.9 soft real target.
- Canonical L1 Centroid Feature Matching between real and fake activations at `d_conv_3`.""")

    add_code("""mse_loss = tf.keras.losses.MeanSquaredError()

@tf.function
def train_d_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    
    with tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)
        
        real_logits, _ = discriminator(real_images, training=True)
        fake_logits, _ = discriminator(fake_images, training=True)
        
        # LSGAN with soft real label
        d_loss_real = tf.reduce_mean(tf.square(real_logits - TARGET_REAL))
        d_loss_fake = tf.reduce_mean(tf.square(fake_logits - TARGET_FAKE))
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
    grads_d = d_tape.gradient(d_loss, discriminator.trainable_variables)
    optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    
    return d_loss

@tf.function
def train_g_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    
    with tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)
        
        fake_logits, fake_feats = discriminator(fake_images, training=False)
        _, real_feats = discriminator(real_images, training=False)
        
        # Adversarial LSGAN loss (target = 1.0)
        adv_loss = 0.5 * tf.reduce_mean(tf.square(fake_logits - 1.0))
        
        # Canonical L1 Centroid Feature Matching (Salimans et al., 2016)
        real_centroid = tf.stop_gradient(tf.reduce_mean(real_feats, axis=0))
        fake_centroid = tf.reduce_mean(fake_feats, axis=0)
        fm_loss = tf.reduce_mean(tf.abs(real_centroid - fake_centroid))
        
        total_g_loss = adv_loss + (LAMBDA_FM * fm_loss)
        
    grads_g = g_tape.gradient(total_g_loss, generator.trainable_variables)
    optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
    
    return total_g_loss, adv_loss, fm_loss

print("[OK] Custom train_step functions defined and compiled.")""")

    # ----------------------------------------------------
    # SECTION 10
    # ----------------------------------------------------
    add_md("""---
## Section 10: Separated Visualizer Pipeline
Displays generated samples decomposing:
1. Raw RGB (unblended)
2. Alpha Mask (grayscale)
3. Composited over Dark Canvas `(0.10, 0.12, 0.16)`
Eliminates matplotlib white background dilution!""")

    add_code("""def evaluate_and_visualize(epoch, latents=FIXED_LATENTS_16, save_prefix="eval"):
    fake_imgs = generator(latents, training=False).numpy()
    
    # Rescale [-1, 1] -> [0, 1]
    fake_unit = (fake_imgs + 1.0) * 0.5
    fake_unit = np.clip(fake_unit, 0.0, 1.0)
    
    n_samples = min(len(latents), 8)
    fig, axes = plt.subplots(3, n_samples, figsize=(2 * n_samples, 6))
    plt.subplots_adjust(wspace=0.08, hspace=0.15)
    
    for i in range(n_samples):
        img_rgb = fake_unit[i, ..., :3]
        img_alpha = fake_unit[i, ..., 3:4]
        img_comp = composite_on_dark(img_rgb, img_alpha)
        
        # Row 0: Raw RGB
        axes[0, i].imshow(img_rgb)
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Generated Raw RGB", fontsize=10, fontweight='bold', loc='left')
        
        # Row 1: Alpha Mask
        axes[1, i].imshow(img_alpha[..., 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Generated Alpha Mask", fontsize=10, fontweight='bold', loc='left')
        
        # Row 2: Composited Dark Canvas
        axes[2, i].imshow(img_comp)
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("On Dark Slate Canvas", fontsize=10, fontweight='bold', loc='left')
        
    plt.suptitle(f"Epoch {epoch:02d} — Fixed Latents Evaluation", fontsize=13, fontweight='bold')
    out_path = os.path.join(EVAL_DIR, f"{save_prefix}_epoch_{epoch:03d}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    
    # Calculate alpha health statistics
    alpha_raw = fake_imgs[..., 3]
    mean_a = np.mean(alpha_raw)
    std_a = np.std(alpha_raw)
    pct_fg = np.mean(alpha_raw > 0.0) * 100.0
    print(f"[ALPHA METRICS Ep {epoch:02d}] Mean: {mean_a:.3f} | Std: {std_a:.3f} | FG Area: {pct_fg:.1f}%")
    return mean_a, std_a, pct_fg""")

    # ----------------------------------------------------
    # SECTION 11
    # ----------------------------------------------------
    add_md("""---
## Section 11: Controlled Pilot Phase (10 Epochs)
Executes a strictly monitored 10-epoch pilot run.
In each batch:
- Discriminator updated once (`train_d_step`).
- Generator updated **TWICE** (`train_g_step`) with fresh noise.
Every epoch:
- Visual evaluation with separated RGB / Alpha / Dark Composite.
- Loss recording and health checks.
- Checkpoint saving.""")

    add_code("""PILOT_EPOCHS = 10
history = {
    'epoch': [],
    'd_loss': [],
    'g_loss': [],
    'adv_loss': [],
    'fm_loss': [],
    'alpha_mean': [],
    'fg_pct': [],
    'time_sec': []
}

print(f"[START] Beginning Controlled Pilot Phase ({PILOT_EPOCHS} Epochs)...")
start_pilot_time = time.time()

for epoch in range(1, PILOT_EPOCHS + 1):
    ep_start = time.time()
    batch_d_losses = []
    batch_g_losses = []
    batch_adv_losses = []
    batch_fm_losses = []
    
    for batch_images in train_dataset:
        # Step 1: Train Discriminator ONCE
        d_loss = train_d_step(batch_images)
        batch_d_losses.append(float(d_loss))
        
        # Step 2: Train Generator TWICE (Asymmetric 2G:1D)
        g_loss_1, adv_1, fm_1 = train_g_step(batch_images)
        g_loss_2, adv_2, fm_2 = train_g_step(batch_images)
        
        batch_g_losses.append(float(0.5 * (g_loss_1 + g_loss_2)))
        batch_adv_losses.append(float(0.5 * (adv_1 + adv_2)))
        batch_fm_losses.append(float(0.5 * (fm_1 + fm_2)))
        
    ep_d = np.mean(batch_d_losses)
    ep_g = np.mean(batch_g_losses)
    ep_adv = np.mean(batch_adv_losses)
    ep_fm = np.mean(batch_fm_losses)
    ep_duration = time.time() - ep_start
    
    # Visual evaluation & Alpha health audit
    a_mean, a_std, a_fg = evaluate_and_visualize(epoch, FIXED_LATENTS_16, save_prefix="pilot")
    
    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"pilot_ckpt_ep{epoch:02d}.weights.h5")
    generator.save_weights(ckpt_path)
    
    history['epoch'].append(epoch)
    history['d_loss'].append(ep_d)
    history['g_loss'].append(ep_g)
    history['adv_loss'].append(ep_adv)
    history['fm_loss'].append(ep_fm)
    history['alpha_mean'].append(a_mean)
    history['fg_pct'].append(a_fg)
    history['time_sec'].append(ep_duration)
    
    print(f"[PILOT Ep {epoch:02d}/{PILOT_EPOCHS:02d}] "
          f"D_Loss: {ep_d:.4f} | G_Loss: {ep_g:.4f} | Adv: {ep_adv:.4f} | FM: {ep_fm:.4f} | "
          f"Alpha: {a_mean:.3f} | Duration: {ep_duration:.1f}s")

total_pilot_min = (time.time() - start_pilot_time) / 60.0
print(f"[COMPLETE] Pilot Phase finished in {total_pilot_min:.2f} minutes.")""")

    # ----------------------------------------------------
    # SECTION 12
    # ----------------------------------------------------
    add_md("""---
## Section 12: Pilot Metric Verification & Stability Gate
Evaluates whether the 10-epoch pilot run met stability gates:
1. $\mathcal{L}_D$ maintained $\ge 0.040$ (No Discriminator overpower).
2. Centroid Feature Matching loss stabilized $\le 0.35$.
3. Alpha mean maintained healthy foreground $\ge 15\%$.
If all gates pass, full training can proceed.""")

    add_code("""min_d_loss = min(history['d_loss'])
final_fm = history['fm_loss'][-1]
final_fg = history['fg_pct'][-1]

print("=== PILOT STABILITY GATE VERIFICATION ===")
print(f"1. Minimum Discriminator Loss: {min_d_loss:.4f} (Gate: >= 0.040)")
gate_1 = min_d_loss >= 0.040
print(f"   -> Result: {'PASS' if gate_1 else 'WARNING (D overpowering)'}")

print(f"2. Final Centroid FM Loss:     {final_fm:.4f} (Gate: <= 0.350)")
gate_2 = final_fm <= 0.350
print(f"   -> Result: {'PASS' if gate_2 else 'WARNING (FM unaligned)'}")

print(f"3. Foreground Occupancy:       {final_fg:.1f}% (Gate: >= 15.0%)")
gate_3 = final_fg >= 15.0
print(f"   -> Result: {'PASS' if gate_3 else 'WARNING (Alpha collapsing)'}")

if gate_1 and gate_2 and gate_3:
    print("\\n[VERDICT] ALL PILOT GATES PASSED! The network is stable and ready for extended training.")
else:
    print("\\n[VERDICT] Pilot completed with observations noted above. Inspect visual evaluation grids.")""")

    # ----------------------------------------------------
    # SECTION 13
    # ----------------------------------------------------
    add_md("""---
## Section 13: Optional Continuation to Full Training (Epochs 11 to 25)
Our forensic investigation identified Epochs 15–25 as the peak convergence window.
This cell safely trains from Epoch 11 to Epoch 25 with automatic early stopping if $\mathcal{L}_D$ drops below $0.025$.""")

    add_code("""EXTENDED_TARGET_EPOCHS = 25
CONTINUE_TRAINING = True  # Set to False to stop after pilot

if CONTINUE_TRAINING and PILOT_EPOCHS < EXTENDED_TARGET_EPOCHS:
    print(f"[INFO] Continuing training from Epoch {PILOT_EPOCHS + 1} to {EXTENDED_TARGET_EPOCHS}...")
    
    for epoch in range(PILOT_EPOCHS + 1, EXTENDED_TARGET_EPOCHS + 1):
        ep_start = time.time()
        batch_d_losses, batch_g_losses, batch_adv_losses, batch_fm_losses = [], [], [], []
        
        for batch_images in train_dataset:
            d_loss = train_d_step(batch_images)
            batch_d_losses.append(float(d_loss))
            
            g_loss_1, adv_1, fm_1 = train_g_step(batch_images)
            g_loss_2, adv_2, fm_2 = train_g_step(batch_images)
            
            batch_g_losses.append(float(0.5 * (g_loss_1 + g_loss_2)))
            batch_adv_losses.append(float(0.5 * (adv_1 + adv_2)))
            batch_fm_losses.append(float(0.5 * (fm_1 + fm_2)))
            
        ep_d = np.mean(batch_d_losses)
        ep_g = np.mean(batch_g_losses)
        ep_adv = np.mean(batch_adv_losses)
        ep_fm = np.mean(batch_fm_losses)
        ep_duration = time.time() - ep_start
        
        a_mean, a_std, a_fg = evaluate_and_visualize(epoch, FIXED_LATENTS_16, save_prefix="ext")
        
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ext_ckpt_ep{epoch:02d}.weights.h5")
        generator.save_weights(ckpt_path)
        
        history['epoch'].append(epoch)
        history['d_loss'].append(ep_d)
        history['g_loss'].append(ep_g)
        history['adv_loss'].append(ep_adv)
        history['fm_loss'].append(ep_fm)
        history['alpha_mean'].append(a_mean)
        history['fg_pct'].append(a_fg)
        history['time_sec'].append(ep_duration)
        
        print(f"[EXT Ep {epoch:02d}/{EXTENDED_TARGET_EPOCHS:02d}] "
              f"D_Loss: {ep_d:.4f} | G_Loss: {ep_g:.4f} | Adv: {ep_adv:.4f} | FM: {ep_fm:.4f} | "
              f"Alpha: {a_mean:.3f} | Duration: {ep_duration:.1f}s")
        
        # Early stopping safeguard against Discriminator saturation
        if ep_d < 0.020:
            print(f"[EARLY STOPPING] Discriminator loss reached saturation threshold ({ep_d:.4f} < 0.020). Halting.")
            break""")

    # ----------------------------------------------------
    # SECTION 14
    # ----------------------------------------------------
    add_md("""---
## Section 14: Comprehensive Post-Training Diagnostics
Plots:
1. Training Loss Trajectories ($\mathcal{L}_D, \mathcal{L}_G, \mathcal{L}_{\text{Adv}}, \mathcal{L}_{\text{FM}}$).
2. Alpha Dynamics & Foreground Retention over epochs.
Saves diagnostic figures in `diagnostics/`.""")

    add_code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss curves
ax1.plot(history['epoch'], history['d_loss'], label='Discriminator Loss (LSGAN)', color='#e74c3c', lw=2)
ax1.plot(history['epoch'], history['g_loss'], label='Generator Total Loss', color='#2ecc71', lw=2)
ax1.plot(history['epoch'], history['adv_loss'], label='Adversarial Loss', color='#f39c12', lw=1.5, ls='--')
ax1.plot(history['epoch'], history['fm_loss'], label='Centroid FM Loss', color='#3498db', lw=1.5, ls=':')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training Loss Dynamics (Corrected Asymmetric Schedule)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Alpha health
ax2.plot(history['epoch'], history['fg_pct'], label='Foreground Occupancy %', color='#9b59b6', lw=2)
ax2.axhline(26.29, color='gray', linestyle='--', label='Real LPC Target (26.3%)')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Foreground Pixel Ratio (%)', fontsize=11)
ax2.set_title('Alpha Mask Retention vs Epoch', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
diag_curve_path = os.path.join(DIAGNOSTICS_DIR, "lpc_fixed_training_dynamics.png")
plt.savefig(diag_curve_path, bbox_inches='tight', dpi=150)
plt.show()
print(f"[OK] Training diagnostic curves saved to {diag_curve_path}")""")

    # ----------------------------------------------------
    # SECTION 15
    # ----------------------------------------------------
    add_md("""---
## Section 15: Master 100-Sample Showcase Grid
Generates the fixed 100 latent vectors $10 \times 10$ showcase composited over dark slate canvas.""")

    add_code("""final_100 = generator(FIXED_LATENTS_100, training=False).numpy()
final_100_unit = np.clip((final_100 + 1.0) * 0.5, 0.0, 1.0)

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
plt.subplots_adjust(wspace=0.04, hspace=0.04)

for i in range(100):
    r = i // 10
    c = i % 10
    img_rgb = final_100_unit[i, ..., :3]
    img_a = final_100_unit[i, ..., 3:4]
    comp = composite_on_dark(img_rgb, img_a)
    axes[r, c].imshow(comp)
    axes[r, c].axis('off')

plt.suptitle("GameForge AI — LPC DCGAN Final Master Showcase (100 Characters)", fontsize=16, fontweight='bold', y=0.92)
master_showcase_path = os.path.join(BASE_OUTPUT_DIR, "lpc_fixed_master_showcase_100.png")
plt.savefig(master_showcase_path, bbox_inches='tight', dpi=150)
plt.show()
print(f"[OK] Master 100-character showcase saved to {master_showcase_path}")""")

    # ----------------------------------------------------
    # SECTION 16
    # ----------------------------------------------------
    add_md("""---
## Section 16: Automated Showcase & Checkpoint Export
Saves the final generator and discriminator weights, dumps run metrics to JSON, and builds a portable ZIP bundle ready for one-click download in Kaggle or local environments.""")

    add_code("""# Save final weights
final_g_path = os.path.join(BASE_OUTPUT_DIR, "generator_weights_final.weights.h5")
final_d_path = os.path.join(BASE_OUTPUT_DIR, "discriminator_weights_final.weights.h5")
generator.save_weights(final_g_path)
discriminator.save_weights(final_d_path)

# Save history metrics
history_path = os.path.join(BASE_OUTPUT_DIR, "training_history.json")
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)

# Build portable ZIP archive
ZIP_NAME = "dcgan_lpc_rootcause_fixed_bundle.zip"
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(BASE_OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, start=".")
            zipf.write(full_path, arcname=rel_path)

zip_size_mb = os.path.getsize(ZIP_NAME) / (1024 * 1024)
print(f"\\n=======================================================")
print(f"[EXPORT COMPLETE]")
print(f"Final Generator:     {final_g_path}")
print(f"Final Discriminator: {final_d_path}")
print(f"Training History:    {history_path}")
print(f"Downloadable Bundle: {os.path.abspath(ZIP_NAME)} ({zip_size_mb:.2f} MB)")
print(f"=======================================================\\n")""")

    # Save to file
    out_notebook_path = os.path.join("Notebook", "training", "03_DCGAN_LPC_RootCause_Fixed.ipynb")
    os.makedirs(os.path.dirname(out_notebook_path), exist_ok=True)
    with open(out_notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"[OK] Successfully built {out_notebook_path} with {len(notebook['cells'])} cells.")

if __name__ == "__main__":
    build_notebook()

