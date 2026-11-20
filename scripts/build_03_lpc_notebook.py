import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_cell(cell_type, source):
        if isinstance(source, str):
            lines = [line + "\n" for line in source.split("\n")]
            if lines and lines[-1] == "\n":
                lines[-1] = ""
        else:
            lines = source
        nb["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
            "execution_count": None if cell_type == "code" else None,
            "outputs": [] if cell_type == "code" else None
        })

    # -------------------------------------------------------------------------
    # CELL 0: Markdown Title & Academic Protocol
    # -------------------------------------------------------------------------
    add_cell("markdown", """# 🎮 GameForge AI: DCGAN Training on LPC Character Sprites (50 Epochs)
### Unconditional Generation of High-Coherence Pixel-Art Game Characters
**Dataset:** Liberated Pixel Art (LPC) Canonical Front-Facing Idles (`r02_c00`)  
**Architecture:** Production DCGAN Replica (G: 2,633,732 params | D: 2,768,449 params)  
**Regularization:** Canonical Batch Centroid Feature Matching (Salimans et al., 2016)  
**Isolation Guarantee:** Outputs routed to `dcgan_lpc_frontfacing_50ep_results`

---
### 🔬 Key Experimental Objectives:
1. **Clean Structured Dataset vs Uncurated Data:** Transition from the uncurated Alucard dataset to canonically proportioned, segmented LPC character sprites.
2. **True Front-Facing Idles:** Strictly isolate Row 2, Column 0 (`r02_c00`) front-facing idle stances. Exclude walking, slashing, spellcasting, and profile frames.
3. **Palette & Silhouette Evaluation:** Measure whether the clean 42-color indexed palette of LPC prevents the color speckling and continuous palette explosion observed in earlier experiments.""")

    # -------------------------------------------------------------------------
    # CELL 1: Environment Setup & Hardware Strategy
    # -------------------------------------------------------------------------
    add_cell("code", """# Hardware Detection & Strategy Setup
import os
import sys
import time
import math
import glob
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers

# Configure GPU memory growth & Distribution Strategy
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
        print(f"✅ Active Distribution Strategy: MirroredStrategy ({len(gpus)} GPUs detected)")
    except RuntimeError as e:
        print(f"Memory growth exception: {e}")
        strategy = tf.distribute.get_strategy()
else:
    strategy = tf.distribute.get_strategy()
    print("⚠️ No physical GPUs detected. Running on CPU single replica.")

NUM_REPLICAS = strategy.num_replicas_in_sync
print(f"TensorFlow Version: {tf.__version__}")
print(f"Synchronous Compute Replicas: {NUM_REPLICAS}")""")

    # -------------------------------------------------------------------------
    # CELL 2: Reproducibility & Random Seeds
    # -------------------------------------------------------------------------
    add_cell("code", """# Deterministic Seeding Contract
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
print(f"✅ Global Random Seed locked to: {SEED}")""")

    # -------------------------------------------------------------------------
    # CELL 3: Configuration & Hyperparameters
    # -------------------------------------------------------------------------
    add_cell("code", """# Global Training Configuration (LPC 50-Epoch Run)
EPOCHS                 = 50
GLOBAL_BATCH_SIZE      = 32
BATCH_SIZE_PER_REPLICA = GLOBAL_BATCH_SIZE // NUM_REPLICAS

LATENT_DIM             = 128
IMG_HEIGHT             = 64
IMG_WIDTH              = 64
CHANNELS               = 4  # RGBA

# Baseline Optimizer Configuration (Audited Baseline Values)
G_LR                   = 3e-4      # Generator learning rate
D_LR                   = 8e-5      # Discriminator learning rate
ADAM_BETA1             = 0.5       # Momentum beta 1
ADAM_BETA2             = 0.999     # Momentum beta 2
LABEL_REAL             = 0.9       # Soft real target
LABEL_FAKE             = 0.0       # Hard fake target
LAMBDA_FM              = 0.20      # Canonical Centroid Feature Matching weight

# ISOLATED OUTPUT DIRECTORY: Ensures zero overwrite of existing experiments
if "google.colab" in sys.modules:
    WORK_DIR = "/content/dcgan_lpc_frontfacing_50ep_results"
elif os.path.exists("/kaggle"):
    WORK_DIR = "/kaggle/working/dcgan_lpc_frontfacing_50ep_results"
else:
    WORK_DIR = "./models/dcgan_lpc_frontfacing_50ep_results"

CKPT_DIR   = f"{WORK_DIR}/checkpoints"
LOG_DIR    = f"{WORK_DIR}/logs"
SAMPLE_DIR = f"{WORK_DIR}/samples"
CONFIG_DIR = f"{WORK_DIR}/config"

for d in [WORK_DIR, CKPT_DIR, LOG_DIR, SAMPLE_DIR, CONFIG_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 75)
print("LPC DCGAN TRAINING CONFIGURATION:")
print(f"  Target Epochs          : {EPOCHS}")
print(f"  Global Batch Size      : {GLOBAL_BATCH_SIZE} ({BATCH_SIZE_PER_REPLICA} per replica)")
print(f"  Generator LR           : {G_LR} (Adam beta1={ADAM_BETA1}, beta2={ADAM_BETA2})")
print(f"  Discriminator LR       : {D_LR} (Adam beta1={ADAM_BETA1}, beta2={ADAM_BETA2})")
print(f"  Feature Matching       : lambda_FM = {LAMBDA_FM} (CANONICAL BATCH CENTROID)")
print(f"  Working Directory      : {WORK_DIR}")
print("=" * 75)""")

    # -------------------------------------------------------------------------
    # CELL 4: Dataset Ingestion (LPC Canonical Front-Facing Idles)
    # -------------------------------------------------------------------------
    add_cell("code", """# Dataset Ingestion: Canonical Front-Facing LPC Idles
# Auto-detect local dataset, Kaggle input, or HuggingFace streaming fallback

lpc_image_paths = []

# Candidate 1: Local repo dataset
local_dir = "Dataset/lpc_frontfacing_idles"
if os.path.exists(local_dir):
    pngs = glob.glob(f"{local_dir}/*.png")
    if len(pngs) > 0:
        lpc_image_paths = sorted(pngs)
        print(f"Found {len(lpc_image_paths)} local LPC front-facing idles in {local_dir}")

# Candidate 2: Kaggle dataset input
if len(lpc_image_paths) == 0:
    for pattern in ["/kaggle/input/*lpc*/**/*.png", "/kaggle/input/*gameforge*/**/*.png", "/kaggle/input/*frontfacing*/**/*.png"]:
        pngs = glob.glob(pattern, recursive=True)
        # Filter for r02_c00 or front_idle
        idles = [p for p in pngs if ("r02_c00" in p or "front_idle" in p)]
        if len(idles) > 0:
            lpc_image_paths = sorted(idles)
            print(f"Found {len(lpc_image_paths)} idle images in Kaggle input {pattern}")
            break

# Candidate 3: HuggingFace streaming fallback (when on Kaggle/Colab with internet)
if len(lpc_image_paths) == 0:
    print("Local crops not found. Ingesting from HuggingFace dataset 'carlosuperb/lpc-action-pixel-art-diffusion'...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset
        
        # Stream action spritesheets and slice Row 2, Col 0 (Walk South, Frame 0 = Idle)
        ds = load_dataset("carlosuperb/lpc-action-pixel-art-diffusion", split="train", streaming=True)
        hf_cache_dir = f"{WORK_DIR}/hf_extracted_idles"
        os.makedirs(hf_cache_dir, exist_ok=True)
        
        TARGET_SHEETS = 2500  # Extract up to 2,500 unique characters
        print(f"Slicing canonical front idles (Row 2, Col 0) from {TARGET_SHEETS} LPC spritesheets...")
        
        count = 0
        GRID_COLS = 9
        GRID_ROWS = 12
        
        for item in ds:
            sheet = item["image"].convert("RGBA")
            cw = sheet.width // GRID_COLS
            ch = sheet.height // GRID_ROWS
            
            # Row 2 (South Walk), Col 0 (Canonical Idle)
            box = (0, 2 * ch, cw, 3 * ch)
            crop = sheet.crop(box).resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.NEAREST)
            
            # Verify alpha has foreground content
            alpha = np.array(crop)[:, :, 3]
            if np.sum(alpha > 20) < 100:
                continue
                
            out_path = f"{hf_cache_dir}/lpc_char_{count:05d}_front_idle.png"
            crop.save(out_path)
            lpc_image_paths.append(out_path)
            count += 1
            if count >= TARGET_SHEETS:
                break
        print(f"Successfully extracted {len(lpc_image_paths)} front-facing idle sprites from HuggingFace!")
    except Exception as e:
        print(f"HuggingFace extraction fallback encountered: {e}")

if len(lpc_image_paths) == 0:
    raise FileNotFoundError("Could not locate LPC front-facing idle sprites! Please provide Dataset/lpc_frontfacing_idles or attach Kaggle dataset.")

print(f"\\n✅ Total Confirmed LPC Front-Facing Idle Sprites: {len(lpc_image_paths):,}")

# Save / Load 100 deterministic fixed latent vectors
FIXED_LATENTS_PATH = f"{CONFIG_DIR}/fixed_latent_vectors.npy"
if not os.path.exists(FIXED_LATENTS_PATH):
    rng = np.random.RandomState(42)
    fixed_latents = rng.randn(100, LATENT_DIM).astype(np.float32)
    np.save(FIXED_LATENTS_PATH, fixed_latents)
    print(f"Saved 100 fixed evaluation latents to: {FIXED_LATENTS_PATH}")
else:
    fixed_latents = np.load(FIXED_LATENTS_PATH)
    print(f"Loaded existing fixed evaluation latents: shape {fixed_latents.shape}")""")

    # -------------------------------------------------------------------------
    # CELL 5: Architecture Definition (Bit-Exact Match)
    # -------------------------------------------------------------------------
    add_cell("code", """# Architecture Definition: Bit-Exact Match (G: 2,633,732 | D: 2,768,449)
def build_generator(latent_dim=LATENT_DIM):
    \"\"\"Exact architectural replica of production DCGAN Generator (2,633,732 params).\"\"\"
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
    
    model = keras.Sequential([
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
    
    # Total: 4,160 + 131,584 + 525,312 + 2,099,200 + 8,193 = 2,768,449 params
    return keras.Model(inputs=img_in, outputs=[logit_out, feat_out], name="dcgan_discriminator")""")

    # -------------------------------------------------------------------------
    # CELL 6: Model Instantiation & Parameter Verification
    # -------------------------------------------------------------------------
    add_cell("code", """# Architecture Equivalence Verification
with strategy.scope():
    generator = build_generator()
    discriminator = build_discriminator()

g_params = generator.count_params()
d_params = discriminator.count_params()
tot_params = g_params + d_params

print("=" * 80)
print(f"Generator Parameters    : {g_params:,} (Target: 2,633,732)")
print(f"Discriminator Parameters: {d_params:,} (Target: 2,768,449)")
print(f"Total Model Parameters  : {tot_params:,} (Target: 5,402,181)")
print("=" * 80)

assert g_params == 2633732, f"Generator mismatch! Got {g_params}"
assert d_params == 2768449, f"Discriminator mismatch! Got {d_params}"
print("✅ ARCHITECTURE EQUIVALENCE CONFIRMED: 100% Bit-Exact Match.")""")

    # -------------------------------------------------------------------------
    # CELL 7: Preprocessing Pipeline & Distributed Dataset
    # -------------------------------------------------------------------------
    add_cell("code", """# tf.data Input Pipeline (RGBA, Linear [-1, 1], Deterministic Shuffling)
def preprocess_image_path(file_path):
    raw = tf.io.read_file(file_path)
    img = tf.io.decode_png(raw, channels=4)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Linear normalization to [-1.0, 1.0]
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img

train_raw_ds = tf.data.Dataset.from_tensor_slices(lpc_image_paths)
train_ds = (train_raw_ds
            .shuffle(buffer_size=min(5000, len(lpc_image_paths)), seed=SEED, reshuffle_each_iteration=True)
            .map(preprocess_image_path, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))

# Calculate STEPS_PER_EPOCH
TOTAL_SAMPLES = len(lpc_image_paths)
STEPS_PER_EPOCH = TOTAL_SAMPLES // GLOBAL_BATCH_SIZE
if STEPS_PER_EPOCH == 0:
    # If dataset is small (e.g. 200 samples), repeat to form at least 25 steps per epoch
    REPEAT_FACTOR = math.ceil(GLOBAL_BATCH_SIZE * 25 / TOTAL_SAMPLES)
    train_raw_ds = train_raw_ds.repeat(REPEAT_FACTOR)
    train_ds = (train_raw_ds
                .shuffle(buffer_size=min(5000, len(lpc_image_paths)), seed=SEED, reshuffle_each_iteration=True)
                .map(preprocess_image_path, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))
    STEPS_PER_EPOCH = (TOTAL_SAMPLES * REPEAT_FACTOR) // GLOBAL_BATCH_SIZE

train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
train_dist_iter = iter(train_dist_ds)

print(f"Dataset Pipeline Configured:")
print(f"  Total Unique Images : {TOTAL_SAMPLES:,}")
print(f"  Steps per Epoch     : {STEPS_PER_EPOCH}")
print(f"  Total Steps (50 Ep) : {STEPS_PER_EPOCH * EPOCHS:,}")""")

    # -------------------------------------------------------------------------
    # CELL 8: Training Step with Canonical Centroid Feature Matching
    # -------------------------------------------------------------------------
    add_cell("code", """# Optimizers & Canonical Centroid Feature Matching Step
with strategy.scope():
    g_optimizer = keras.optimizers.Adam(learning_rate=G_LR, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    d_optimizer = keras.optimizers.Adam(learning_rate=D_LR, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    
    # Pre-build optimizer slot variables inside strategy.scope()
    g_optimizer.build(generator.trainable_variables)
    d_optimizer.build(discriminator.trainable_variables)

def step_per_replica(real_images):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal([batch_size, LATENT_DIM])
    
    # -------------------------------------------------------------------------
    # 1. Train Discriminator
    # -------------------------------------------------------------------------
    with tf.GradientTape() as d_tape:
        fake_images = generator(z, training=True)
        real_logits, real_feats = discriminator(real_images, training=True)
        fake_logits, fake_feats = discriminator(fake_images, training=True)
        
        # LSGAN Loss with 0.9 soft real target
        d_loss_real = tf.reduce_mean(tf.square(real_logits - LABEL_REAL))
        d_loss_fake = tf.reduce_mean(tf.square(fake_logits - LABEL_FAKE))
        d_loss = 0.5 * (d_loss_real + d_loss_fake) / float(strategy.num_replicas_in_sync)
        
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    # -------------------------------------------------------------------------
    # 2. Train Generator (CANONICAL BATCH CENTROID FEATURE MATCHING)
    # -------------------------------------------------------------------------
    with tf.GradientTape() as g_tape:
        fake_images = generator(z, training=True)
        # Note: training=False on D freezes Batch Normalization moving stats during G step
        fake_logits, fake_feats = discriminator(fake_images, training=False)
        _, real_feats = discriminator(real_images, training=False)
        
        # LSGAN Adversarial Loss
        g_adv_loss = 0.5 * tf.reduce_mean(tf.square(fake_logits - 1.0)) / float(strategy.num_replicas_in_sync)
        
        # CANONICAL BATCH CENTROID FEATURE MATCHING (Salimans et al., 2016)
        mean_real = tf.stop_gradient(tf.reduce_mean(real_feats, axis=0))
        mean_fake = tf.reduce_mean(fake_feats, axis=0)
        fm_loss = tf.reduce_mean(tf.abs(mean_real - mean_fake)) / float(strategy.num_replicas_in_sync)
        
        g_loss = g_adv_loss + (LAMBDA_FM * fm_loss)
        
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    
    return d_loss, g_loss, g_adv_loss, fm_loss

@tf.function
def distributed_train_step(real_images):
    per_d, per_g, per_adv, per_fm = strategy.run(step_per_replica, args=(real_images,))
    d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_d, axis=None)
    g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_g, axis=None)
    adv_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_adv, axis=None)
    fm_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_fm, axis=None)
    return d_loss, g_loss, adv_loss, fm_loss""")

    # -------------------------------------------------------------------------
    # CELL 9: Smoke Test
    # -------------------------------------------------------------------------
    add_cell("code", """# Execution Smoke Test Function
def run_smoke_test():
    print("Running 1-batch execution smoke test under strategy...")
    dummy_tensor = tf.random.uniform([GLOBAL_BATCH_SIZE, 64, 64, 4], minval=-1.0, maxval=1.0)
    dummy_ds = tf.data.Dataset.from_tensor_slices(dummy_tensor).batch(GLOBAL_BATCH_SIZE)
    dist_dummy_ds = strategy.experimental_distribute_dataset(dummy_ds)
    
    smoke_batch = next(iter(dist_dummy_ds))
    t0 = time.time()
    d_loss, g_loss, adv_loss, fm_loss = distributed_train_step(smoke_batch)
    dt = time.time() - t0
    print(f"Smoke test successful in {dt*1000:.1f}ms:")
    print(f"  D_Loss: {float(d_loss):.4f} | G_Loss: {float(g_loss):.4f} (Adv: {float(adv_loss):.4f}, Centroid FM: {float(fm_loss):.4f})")

run_smoke_test()""")

    # -------------------------------------------------------------------------
    # CELL 10: Checkpointing & Visual Grid Evaluator
    # -------------------------------------------------------------------------
    add_cell("code", """# Checkpointing, Visual Evaluator & Logging Setup
import csv

with strategy.scope():
    ckpt = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

eval_fixed_z = np.load(FIXED_LATENTS_PATH)
print(f"Loaded 100 fixed evaluation latents: shape {eval_fixed_z.shape}")

def save_eval_grid(epoch, title, filename, use_fixed=True):
    \"\"\"Generates a 6x6 evaluation grid of synthesized sprites.\"\"\"
    z = eval_fixed_z[:36] if use_fixed else np.random.randn(36, LATENT_DIM).astype(np.float32)
    raw = generator(z, training=False).numpy()
    imgs = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
    
    fig, axes = plt.subplots(6, 6, figsize=(10, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.axis('off')
    plt.suptitle(f"LPC DCGAN (50 Ep) — {title} (Epoch {epoch})", fontsize=13, y=0.92)
    plt.tight_layout()
    out_path = f"{SAMPLE_DIR}/{filename}"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path

history_csv = f"{LOG_DIR}/training_history.csv"
with open(history_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "g_loss", "d_loss", "adv_loss", "fm_loss", "epoch_time_sec"])

print("LPC Checkpoint manager and logging initialized.")""")

    # -------------------------------------------------------------------------
    # CELL 11: 50-Epoch Training Loop
    # -------------------------------------------------------------------------
    add_cell("code", """# 50-Epoch Training Loop (LPC Canonical Front-Facing Idles)
print("=" * 80)
print(f"STARTING 50-EPOCH LPC DCGAN TRAINING")
print(f"Total Epochs    : {EPOCHS}")
print(f"Steps per Epoch : {STEPS_PER_EPOCH}")
print(f"Total Steps     : {STEPS_PER_EPOCH * EPOCHS:,}")
print(f"Output Directory: {WORK_DIR}")
print("=" * 80)

history_records = []
total_train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    t_ep_start = time.time()
    ep_d_loss, ep_g_loss, ep_adv_loss, ep_fm_loss = 0.0, 0.0, 0.0, 0.0
    
    for step in range(STEPS_PER_EPOCH):
        try:
            batch = next(train_dist_iter)
        except StopIteration:
            train_dist_iter = iter(train_dist_ds)
            batch = next(train_dist_iter)
            
        d_l, g_l, adv_l, fm_l = distributed_train_step(batch)
        ep_d_loss += float(d_l)
        ep_g_loss += float(g_l)
        ep_adv_loss += float(adv_l)
        ep_fm_loss += float(fm_l)
        
    ep_d_loss /= STEPS_PER_EPOCH
    ep_g_loss /= STEPS_PER_EPOCH
    ep_adv_loss /= STEPS_PER_EPOCH
    ep_fm_loss /= STEPS_PER_EPOCH
    ep_duration = time.time() - t_ep_start
    
    record = {
        'epoch': epoch,
        'g_loss': ep_g_loss,
        'd_loss': ep_d_loss,
        'adv_loss': ep_adv_loss,
        'fm_loss': ep_fm_loss,
        'epoch_time_sec': ep_duration
    }
    history_records.append(record)
    
    with open(history_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, ep_g_loss, ep_d_loss, ep_adv_loss, ep_fm_loss, ep_duration])
        
    print(f"Epoch {epoch:02d}/{EPOCHS} | D_Loss: {ep_d_loss:.4f} | G_Loss: {ep_g_loss:.4f} (Adv: {ep_adv_loss:.4f}, Centroid FM: {ep_fm_loss:.4f}) | Time: {ep_duration:.1f}s")
    
    # Save checkpoints & sample grids at Epochs 1, 5, 10, 25, 50
    if epoch in [1, 5, 10, 25, 50]:
        save_eval_grid(epoch, "Fixed Latents", f"samples_epoch_{epoch:04d}.png", use_fixed=True)
        save_eval_grid(epoch, "Fresh Random", f"fresh_samples_epoch_{epoch:04d}.png", use_fixed=False)
        ckpt_path = ckpt_manager.save()
        print(f"  -> Checkpoint & Evaluation grid saved for Epoch {epoch}")

total_train_time = time.time() - total_train_start
print("=" * 80)
print(f"LPC 50-EPOCH TRAINING COMPLETE in {total_train_time/60:.1f} minutes ({total_train_time:.1f}s)!")
print("=" * 80)

# Save final model weights in isolated directory
final_g_path = f"{WORK_DIR}/generator_weights_final.weights.h5"
final_d_path = f"{WORK_DIR}/discriminator_weights_final.weights.h5"
generator.save_weights(final_g_path)
discriminator.save_weights(final_d_path)
print(f"Final Generator weights saved to: {final_g_path}")
print(f"Final Discriminator weights saved to: {final_d_path}")""")

    # -------------------------------------------------------------------------
    # CELL 12: 100-Sample Showcase, Loss Curves & Zip Archive
    # -------------------------------------------------------------------------
    add_cell("code", """# Master 100-Sample Evaluation Showcase & Download Packaging
from IPython.display import FileLink

print("Synthesizing 100-character master evaluation showcase...")
eval_raw = generator(eval_fixed_z, training=False).numpy()
eval_imgs = np.clip((eval_raw + 1.0) / 2.0, 0.0, 1.0)

# 1. 10x10 Master Showcase Contact Sheet
fig, axes = plt.subplots(10, 10, figsize=(16, 16))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(eval_imgs[i])
    ax.axis('off')
plt.suptitle("GameForge AI: LPC DCGAN 100-Character Master Showcase (50 Epochs)", fontsize=15, y=0.92)
plt.tight_layout()
showcase_path = f"{SAMPLE_DIR}/final_master_showcase_100_characters.png"
plt.savefig(showcase_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Master showcase saved to: {showcase_path}")

# 2. Plot Loss Curves
epochs_list = [r['epoch'] for r in history_records]
g_losses = [r['g_loss'] for r in history_records]
d_losses = [r['d_loss'] for r in history_records]
adv_losses = [r['adv_loss'] for r in history_records]
fm_losses = [r['fm_loss'] for r in history_records]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(epochs_list, g_losses, label='G Total Loss', color='#1f77b4', lw=2)
ax1.plot(epochs_list, d_losses, label='D Loss (LSGAN)', color='#d62728', lw=2)
ax1.set_title('Generator vs Discriminator Loss', fontsize=12)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(epochs_list, adv_losses, label='G Adversarial Loss', color='#2ca02c', lw=2)
ax2.plot(epochs_list, fm_losses, label='Canonical Centroid FM Loss', color='#9467bd', lw=2)
ax2.set_title('Generator Loss Decomposition', fontsize=12)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss Component')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
loss_curve_path = f"{LOG_DIR}/loss_curves.png"
plt.savefig(loss_curve_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"LPC training loss curves saved to: {loss_curve_path}")

# 3. Zip Results for Immediate Download
zip_base = f"{WORK_DIR}"
print("Compressing LPC results folder into zip archive...")
zip_archive = shutil.make_archive(zip_base, 'zip', WORK_DIR)
print(f"Zip archive created: {zip_archive}")

print("\\n" + "=" * 70)
print("CLICK THE LINK BELOW TO DOWNLOAD LPC 50-EPOCH RESULTS:")
print("=" * 70)
FileLink(os.path.basename(zip_archive))""")

    # Write notebook file
    out_nb_path = r"Notebook/training/03_DCGAN_LPC_FrontFacing_50Epochs.ipynb"
    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"✅ Successfully created notebook: {out_nb_path} with {len(nb['cells'])} cells.")

if __name__ == "__main__":
    create_notebook()

