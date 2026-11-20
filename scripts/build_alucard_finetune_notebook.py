import json
import os

def build_alucard_notebook():
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
    add_md("""# GameForge AI: Alucard Front-Facing DCGAN Fine-Tuning Pipeline
### Resumed from Epoch 50 Checkpoint -> Fine-Tuned with Throttled D & L1 Feature Matching
### Goal: Eliminate color speckles, sharpen pixel-art edges, and maintain dynamic character silhouettes

---
## Section 1: Environment Setup, Multi-GPU Detection, & Output Isolation""")

    add_code("""import os
import sys
import gc
import json
import time
import random
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

# Seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Detect GPUs and set memory growth for ALL devices
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

# Output directory configuration
BASE_OUTPUT_DIR = "dcgan_alucard_finetune_results"
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
EVAL_DIR = os.path.join(BASE_OUTPUT_DIR, "eval_samples")
DIAGNOSTICS_DIR = os.path.join(BASE_OUTPUT_DIR, "diagnostics")

for d in [BASE_OUTPUT_DIR, CHECKPOINT_DIR, EVAL_DIR, DIAGNOSTICS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[OK] Output directory ready at: {os.path.abspath(BASE_OUTPUT_DIR)}")""")

    # ----------------------------------------------------
    # SECTION 2
    # ----------------------------------------------------
    add_md("""---
## Section 2: Locate Weights, Manifest, & Latents
Automatically discovers:
1. `generator_weights_final.weights.h5`
2. `discriminator_weights_final.weights.h5`
3. `fixed_latent_vectors.npy`
4. `final_alucard_frontfacing_manifest.csv`
Searches `/kaggle/input/`, Downloads, and workspace folders.""")

    add_code("""SEARCH_PATHS = [
    ".",
    "Dataset",
    "/kaggle/input",
    "/kaggle/input/alucard-46k-finetune-package",
    "/kaggle/input/dcgan-frontfacing-46k-50ep-results",
    "C:/Users/manis/Downloads/alucard_46k_finetune_package",
    "C:/Users/manis/Downloads/dcgan_frontfacing_46k_50ep_results",
    "phase_dcgan_improvement/dataset_research"
]

def find_file(filename):
    for root in SEARCH_PATHS:
        if os.path.exists(root):
            for dirpath, _, files in os.walk(root):
                if filename in files:
                    return os.path.join(dirpath, filename)
    return None

G_WEIGHTS_PATH = find_file("generator_weights_final.weights.h5")
D_WEIGHTS_PATH = find_file("discriminator_weights_final.weights.h5")
LATENTS_PATH = find_file("fixed_latent_vectors.npy")
MANIFEST_PATH = find_file("final_alucard_frontfacing_manifest.csv")

print("=== DISCOVERED ASSETS ===")
print(f"Generator Weights:     {G_WEIGHTS_PATH}")
print(f"Discriminator Weights: {D_WEIGHTS_PATH}")
print(f"Fixed Latents:         {LATENTS_PATH}")
print(f"Dataset Manifest:      {MANIFEST_PATH}")

assert G_WEIGHTS_PATH and os.path.exists(G_WEIGHTS_PATH), "Could not find generator weights!"
assert D_WEIGHTS_PATH and os.path.exists(D_WEIGHTS_PATH), "Could not find discriminator weights!"
assert MANIFEST_PATH and os.path.exists(MANIFEST_PATH), "Could not find Alucard front-facing manifest!"
print("[PASS] All required assets discovered successfully!")""")

    # ----------------------------------------------------
    # SECTION 3
    # ----------------------------------------------------
    add_md("""---
## Section 3: Exact Production DCGAN Architecture & Checkpoint Loading
Instantiates the bit-exact architecture:
- Generator: 2,633,732 parameters
- Discriminator: 2,768,449 parameters
Loads the 50-epoch weights into both networks.""")

    add_code("""LATENT_DIM = 128
CHANNELS = 4
IMG_HEIGHT = 64
IMG_WIDTH = 64

def build_generator(latent_dim=LATENT_DIM):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
    model = Sequential([
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
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
    img_in = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="img_input")
    
    # Block 1: 64x64 -> 32x32
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=True, kernel_initializer=init, name="d_conv_1")(img_in)
    x = layers.LeakyReLU(0.2, name="d_lrelu_1")(x)
    
    # Block 2: 32x32 -> 16x16
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_2")(x)
    x = layers.BatchNormalization(name="d_bn_2")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_2")(x)
    
    # Block 3: 16x16 -> 8x8 (Feature Tap for Centroid Feature Matching)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_3")(x)
    x = layers.BatchNormalization(name="d_bn_3")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_3")(x)
    feat_out = x
    
    # Block 4: 8x8 -> 4x4
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_4")(x)
    x = layers.BatchNormalization(name="d_bn_4")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_4")(x)
    
    # Dense Logit Out
    x = layers.Flatten(name="d_flatten")(x)
    logit_out = layers.Dense(1, name="d_logit")(x)
    
    return Model(inputs=img_in, outputs=[logit_out, feat_out], name="dcgan_discriminator")

generator = build_generator()
discriminator = build_discriminator()

assert generator.count_params() == 2633732, f"G param mismatch: {generator.count_params()}"
assert discriminator.count_params() == 2768449, f"D param mismatch: {discriminator.count_params()}"
print(f"[OK] Exact architecture verified (G: {generator.count_params():,}, D: {discriminator.count_params():,}).")

# Load Epoch 50 weights
print("[INFO] Restoring Epoch 50 checkpoint weights...")
generator.load_weights(G_WEIGHTS_PATH)
discriminator.load_weights(D_WEIGHTS_PATH)
print("[PASS] Checkpoint weights successfully loaded!")""")

    # ----------------------------------------------------
    # SECTION 4
    # ----------------------------------------------------
    add_md("""---
## Section 4: Baseline Verification (Epoch 50 Pre-Fine-Tuning Showcase)
Loads the fixed latents and visualizes the exact loaded checkpoint before fine-tuning starts.""")

    add_code("""if LATENTS_PATH and os.path.exists(LATENTS_PATH):
    print(f"[INFO] Loading fixed 100 latents from {LATENTS_PATH}")
    fixed_latents_100 = np.load(LATENTS_PATH)
    FIXED_LATENTS_100 = tf.convert_to_tensor(fixed_latents_100, dtype=tf.float32)
else:
    print("[INFO] Generating deterministic fixed latents...")
    FIXED_LATENTS_100 = tf.random.normal([100, LATENT_DIM], seed=SEED)

FIXED_LATENTS_16 = FIXED_LATENTS_100[:16]

DARK_BG = np.array([0.10, 0.12, 0.16], dtype=np.float32)

def composite_on_dark(rgb_norm, alpha_norm, bg_color=DARK_BG):
    return rgb_norm * alpha_norm + bg_color.reshape(1, 1, 3) * (1.0 - alpha_norm)

# Generate baseline preview
baseline_fake = generator(FIXED_LATENTS_16[:8], training=False).numpy()
baseline_unit = np.clip((baseline_fake + 1.0) * 0.5, 0.0, 1.0)

fig, axes = plt.subplots(3, 8, figsize=(16, 6))
plt.subplots_adjust(wspace=0.08, hspace=0.18)

for i in range(8):
    rgb = baseline_unit[i, ..., :3]
    alpha = baseline_unit[i, ..., 3:4]
    comp = composite_on_dark(rgb, alpha)
    
    axes[0, i].imshow(rgb)
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Loaded Raw RGB", fontsize=10, fontweight='bold', loc='left')
    
    axes[1, i].imshow(alpha[..., 0], cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title("Loaded Alpha", fontsize=10, fontweight='bold', loc='left')
    
    axes[2, i].imshow(comp)
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_title("Loaded on Slate", fontsize=10, fontweight='bold', loc='left')

plt.suptitle("ALUCARD EPOCH 50 BASELINE CHECKPOINT (PRE-FINE-TUNING)", fontsize=13, fontweight='bold')
plt.savefig(os.path.join(DIAGNOSTICS_DIR, "alucard_epoch50_baseline_preview.png"), bbox_inches='tight', dpi=150)
plt.show()""")

    # ----------------------------------------------------
    # SECTION 5
    # ----------------------------------------------------
    add_md("""---
## Section 5: Dataset Ingestion (42,212 Audited Front-Facing Alucard Sprites)
Loads `evilsocket/alucard-sprites` from HuggingFace and filters using `final_alucard_frontfacing_manifest.csv`.""")

    add_code("""print(f"[INFO] Reading manifest: {MANIFEST_PATH}")
manifest_df = pd.read_csv(MANIFEST_PATH)
print(f"[INFO] Manifest total rows: {len(manifest_df)}")

# Train split filter
if 'split' in manifest_df.columns:
    train_df = manifest_df[manifest_df['split'] == 'train']
else:
    train_df = manifest_df

train_indices = set(train_df['archive_index'].tolist())
print(f"[INFO] Target training sprites: {len(train_indices):,}")

# Load HuggingFace dataset
try:
    from datasets import load_dataset
    print("[INFO] Streaming evilsocket/alucard-sprites dataset from HuggingFace Hub...")
    alucard_ds = load_dataset("evilsocket/alucard-sprites", split="train")
    print(f"[INFO] Total dataset size: {len(alucard_ds):,} sprites.")
    
    images_list = []
    print("[INFO] Filtering front-facing training sprites...")
    for idx in train_df['archive_index'].values:
        item = alucard_ds[int(idx)]
        img = item['image'].convert('RGBA')
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.NEAREST)
        arr = np.array(img, dtype=np.uint8)
        images_list.append(arr)
        if len(images_list) % 5000 == 0:
            print(f"  Loaded {len(images_list):,} / {len(train_df):,}...")
            
    images_array = np.array(images_list, dtype=np.uint8)
    print(f"[OK] Successfully loaded {len(images_array):,} images into memory.")
except Exception as e:
    print(f"[WARN] HuggingFace loading note: {e}")
    # Fallback to local image search if already downloaded
    LOCAL_IMG_DIR = "Dataset/alucard_frontfacing_images"
    if os.path.exists(LOCAL_IMG_DIR):
        pngs = sorted([os.path.join(LOCAL_IMG_DIR, f) for f in os.listdir(LOCAL_IMG_DIR) if f.endswith('.png')])
        images_array = np.array([np.array(Image.open(p).convert('RGBA')) for p in pngs], dtype=np.uint8)
        print(f"[OK] Loaded {len(images_array)} images from local directory {LOCAL_IMG_DIR}")
    else:
        raise RuntimeError("Please ensure internet access is on to stream from HuggingFace, or supply local sprites.")

# Normalize to [-1.0, 1.0]
normalized_dataset = (images_array.astype(np.float32) / 127.5) - 1.0

BATCH_SIZE = 32
BUFFER_SIZE = min(len(normalized_dataset), 10000)

train_dataset = tf.data.Dataset.from_tensor_slices(normalized_dataset)
train_dataset = train_dataset.shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print(f"[OK] Training pipeline ready: {len(normalized_dataset):,} samples, {len(train_dataset)} batches per epoch.")""")

    # ----------------------------------------------------
    # SECTION 6
    # ----------------------------------------------------
    add_md("""---
## Section 6: Fine-Tuning Optimization Setup
Applies the calibrated fine-tuning schedule:
- **Generator LR:** $\eta_G = 1.5 \times 10^{-4}$ (Adam $\beta_1 = 0.5$).
- **Discriminator LR:** $\eta_D = 3.0 \times 10^{-5}$ ($5\times$ slower than G to prevent overpower!).
- **Canonical L1 Feature Matching:** $\lambda_{\text{FM}} = 0.20$ to guide G toward real feature centroids without blurring.
- **Asymmetric 2G : 1D Updates:** G trains twice per D step.""")

    add_code("""LR_G = 1.5e-4       # Fine-tuning rate for Generator
LR_D = 3.0e-5       # Heavily throttled for Discriminator
BETA_1 = 0.5
BETA_2 = 0.999

LAMBDA_FM = 0.20    # Canonical Salimans L1 Feature Matching
TARGET_REAL = 0.90
TARGET_FAKE = 0.00

optimizer_g = tf.keras.optimizers.Adam(learning_rate=LR_G, beta_1=BETA_1, beta_2=BETA_2)
optimizer_d = tf.keras.optimizers.Adam(learning_rate=LR_D, beta_1=BETA_1, beta_2=BETA_2)

print(f"[CONFIG] Generator LR:     {LR_G}")
print(f"[CONFIG] Discriminator LR: {LR_D} (Throttled)")
print(f"[CONFIG] Feature Matching: lambda={LAMBDA_FM} (L1 Centroid)")
print(f"[CONFIG] Update Schedule:  2 G steps per 1 D step")""")

    # ----------------------------------------------------
    # SECTION 7
    # ----------------------------------------------------
    add_md("""---
## Section 7: Training Steps with L1 Feature Matching""")

    add_code("""mse_loss = tf.keras.losses.MeanSquaredError()

@tf.function
def train_d_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    
    with tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)
        
        real_logits, _ = discriminator(real_images, training=True)
        fake_logits, _ = discriminator(fake_images, training=True)
        
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
        
        # Adversarial LSGAN loss
        adv_loss = 0.5 * tf.reduce_mean(tf.square(fake_logits - 1.0))
        
        # Canonical L1 Centroid Feature Matching
        real_centroid = tf.stop_gradient(tf.reduce_mean(real_feats, axis=0))
        fake_centroid = tf.reduce_mean(fake_feats, axis=0)
        fm_loss = tf.reduce_mean(tf.abs(real_centroid - fake_centroid))
        
        total_g_loss = adv_loss + (LAMBDA_FM * fm_loss)
        
    grads_g = g_tape.gradient(total_g_loss, generator.trainable_variables)
    optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
    return total_g_loss, adv_loss, fm_loss

print("[OK] Custom fine-tuning train_step compiled.")""")

    # ----------------------------------------------------
    # SECTION 8
    # ----------------------------------------------------
    add_md("""---
## Section 8: Visualizer Function""")

    add_code("""def evaluate_and_visualize(epoch, latents=FIXED_LATENTS_16, save_prefix="finetune"):
    fake_imgs = generator(latents, training=False).numpy()
    fake_unit = np.clip((fake_imgs + 1.0) * 0.5, 0.0, 1.0)
    
    n_samples = min(len(latents), 8)
    fig, axes = plt.subplots(3, n_samples, figsize=(2 * n_samples, 6))
    plt.subplots_adjust(wspace=0.08, hspace=0.15)
    
    for i in range(n_samples):
        img_rgb = fake_unit[i, ..., :3]
        img_alpha = fake_unit[i, ..., 3:4]
        img_comp = composite_on_dark(img_rgb, img_alpha)
        
        axes[0, i].imshow(img_rgb)
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Fine-Tuned Raw RGB", fontsize=10, fontweight='bold', loc='left')
        
        axes[1, i].imshow(img_alpha[..., 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Fine-Tuned Alpha", fontsize=10, fontweight='bold', loc='left')
        
        axes[2, i].imshow(img_comp)
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("On Dark Slate Canvas", fontsize=10, fontweight='bold', loc='left')
        
    plt.suptitle(f"Epoch {epoch:02d} — Fixed Latents Evaluation", fontsize=13, fontweight='bold')
    out_path = os.path.join(EVAL_DIR, f"{save_prefix}_epoch_{epoch:03d}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    
    alpha_raw = fake_imgs[..., 3]
    mean_a = np.mean(alpha_raw)
    pct_fg = np.mean(alpha_raw > 0.0) * 100.0
    return mean_a, pct_fg""")

    # ----------------------------------------------------
    # SECTION 9
    # ----------------------------------------------------
    add_md("""---
## Section 9: Fine-Tuning Execution (25 Epochs)
Runs fine-tuning with per-epoch visual evaluations and checkpointing.""")

    add_code("""FINETUNE_EPOCHS = 25

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

print(f"[START] Running Alucard Fine-Tuning for {FINETUNE_EPOCHS} Epochs...")
start_time = time.time()

for epoch in range(51, 51 + FINETUNE_EPOCHS):
    ep_start = time.time()
    batch_d, batch_g, batch_adv, batch_fm = [], [], [], []
    
    for batch_images in train_dataset:
        d_loss = train_d_step(batch_images)
        batch_d.append(float(d_loss))
        
        g_loss_1, adv_1, fm_1 = train_g_step(batch_images)
        g_loss_2, adv_2, fm_2 = train_g_step(batch_images)
        
        batch_g.append(float(0.5 * (g_loss_1 + g_loss_2)))
        batch_adv.append(float(0.5 * (adv_1 + adv_2)))
        batch_fm.append(float(0.5 * (fm_1 + fm_2)))
        
    ep_d = np.mean(batch_d)
    ep_g = np.mean(batch_g)
    ep_adv = np.mean(batch_adv)
    ep_fm = np.mean(batch_fm)
    ep_duration = time.time() - ep_start
    
    a_mean, a_fg = evaluate_and_visualize(epoch, FIXED_LATENTS_16)
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"alucard_finetune_ep{epoch:03d}.weights.h5")
    generator.save_weights(ckpt_path)
    
    history['epoch'].append(epoch)
    history['d_loss'].append(ep_d)
    history['g_loss'].append(ep_g)
    history['adv_loss'].append(ep_adv)
    history['fm_loss'].append(ep_fm)
    history['alpha_mean'].append(a_mean)
    history['fg_pct'].append(a_fg)
    history['time_sec'].append(ep_duration)
    
    print(f"[FT Ep {epoch:03d}/{50 + FINETUNE_EPOCHS:03d}] "
          f"D_Loss: {ep_d:.4f} | G_Loss: {ep_g:.4f} | Adv: {ep_adv:.4f} | FM: {ep_fm:.4f} | "
          f"FG: {a_fg:.1f}% | Duration: {ep_duration:.1f}s")
    
    # Early stopping if D approaches dangerous saturation
    if ep_d < 0.012:
        print(f"[SAFEGUARD] Discriminator loss reached saturation zone ({ep_d:.4f} < 0.012). Halting to preserve optimal checkpoint.")
        break

total_min = (time.time() - start_time) / 60.0
print(f"[COMPLETE] Fine-tuning finished in {total_min:.2f} minutes.")""")

    # ----------------------------------------------------
    # SECTION 10
    # ----------------------------------------------------
    add_md("""---
## Section 10: Master 100-Sample Showcase & ZIP Export""")

    add_code("""# Generate 100-sample grid
final_100 = generator(FIXED_LATENTS_100, training=False).numpy()
final_100_unit = np.clip((final_100 + 1.0) * 0.5, 0.0, 1.0)

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
plt.subplots_adjust(wspace=0.04, hspace=0.04)

for i in range(100):
    r, c = i // 10, i % 10
    rgb = final_100_unit[i, ..., :3]
    alpha = final_100_unit[i, ..., 3:4]
    comp = composite_on_dark(rgb, alpha)
    axes[r, c].imshow(comp)
    axes[r, c].axis('off')

plt.suptitle("GameForge AI — Alucard DCGAN Fine-Tuned Master Showcase (100 Sprites)", fontsize=16, fontweight='bold', y=0.92)
showcase_path = os.path.join(BASE_OUTPUT_DIR, "alucard_finetuned_master_showcase_100.png")
plt.savefig(showcase_path, bbox_inches='tight', dpi=150)
plt.show()

# Save final weights
final_g_path = os.path.join(BASE_OUTPUT_DIR, "generator_weights_finetuned.weights.h5")
final_d_path = os.path.join(BASE_OUTPUT_DIR, "discriminator_weights_finetuned.weights.h5")
generator.save_weights(final_g_path)
discriminator.save_weights(final_d_path)

# Build ZIP bundle
ZIP_NAME = "dcgan_alucard_finetune_results_bundle.zip"
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(BASE_OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, start=".")
            zipf.write(full_path, arcname=rel_path)

size_mb = os.path.getsize(ZIP_NAME) / (1024 * 1024)
print(f"\\n[EXPORT COMPLETE] Created {ZIP_NAME} ({size_mb:.2f} MB)")""")

    # Save to file
    out_notebook_path = os.path.join("Notebook", "training", "02_DCGAN_Alucard_FineTune_Resumed.ipynb")
    os.makedirs(os.path.dirname(out_notebook_path), exist_ok=True)
    with open(out_notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"[OK] Successfully built {out_notebook_path} with {len(notebook['cells'])} cells.")

if __name__ == "__main__":
    build_alucard_notebook()

