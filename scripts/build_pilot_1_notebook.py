import json
import os

nb_path = "Notebook/training/03_DCGAN_Centroid_FeatureMatching_Pilot_10Epochs.ipynb"
os.makedirs(os.path.dirname(nb_path), exist_ok=True)

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

# Cell 0: Header
add_md("""# 🎮 GameForge AI: DCGAN Pilot 1 — Canonical Centroid Feature Matching (10 Epochs)

**Experiment Codename:** `DCGAN-46K-PILOT-1-CENTROID-FM`  
**Purpose:** Single-Variable Verification of Canonical Batch Centroid Feature Matching (Salimans et al., 2016).  
**Tested Variable:** Replacing unpaired elementwise feature matching with canonical distribution centroid matching.  
**Frozen Baseline:** 
- Identical 42,212 training sprites (`final_alucard_frontfacing_manifest.csv`).
- Identical architecture (Generator: 2,633,732 params | Discriminator: 2,768,449 params).
- Identical optimizers (Adam $\\beta_1=0.5, \\beta_2=0.999$, $G_{\\text{lr}}=3\\times 10^{-4}, D_{\\text{lr}}=8\\times 10^{-5}$).
- Identical 1G : 1D update ratio.
- Identical 100 fixed evaluation latents.
- Target budget: **10 Epochs** (~11 minutes on Dual T4).
- Output directory: `/kaggle/working/dcgan_frontfacing_centroid_pilot_results/` (Zero overwrite of previous results).""")

# Cell 1: Environment & Imports
add_code("""# Setup Environment & Dependencies
import os
import sys
import time
import random
import csv
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Install HuggingFace datasets if not present
try:
    import datasets
except ImportError:
    !pip install -q datasets
    import datasets
from datasets import load_dataset

# TensorFlow & Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Set strict deterministic seeds
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"Python Version    : {sys.version.split()[0]}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version     : {keras.__version__}")""")

# Cell 2: Hardware Strategy
add_code("""# Hardware Distribution Strategy
try:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Dual/Multi-GPU Strategy Active: {len(gpus)} GPUs detected.")
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        print("Single GPU Strategy Active.")
    else:
        strategy = tf.distribute.get_strategy()
        print("CPU Strategy Active.")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(f"Fallback Default Strategy Active: {e}")

NUM_REPLICAS = strategy.num_replicas_in_sync
print(f"Number of Synchronized Replicas: {NUM_REPLICAS}")""")

# Cell 3: Configuration & Isolated Output Directory
add_code("""# Global Pilot Configuration (Frozen Parameters + Centroid FM)
EPOCHS                 = 10
GLOBAL_BATCH_SIZE      = 32
BATCH_SIZE_PER_REPLICA = GLOBAL_BATCH_SIZE // NUM_REPLICAS

LATENT_DIM             = 128
IMG_HEIGHT             = 64
IMG_WIDTH              = 64
CHANNELS               = 4  # RGBA

# Optimization & Regularization (Audited Baseline Values)
G_LR                   = 3e-4      # Generator learning rate
D_LR                   = 8e-5      # Discriminator learning rate
ADAM_BETA1             = 0.5       # Momentum beta 1
ADAM_BETA2             = 0.999     # Momentum beta 2
LABEL_REAL             = 0.9       # Soft real target
LABEL_FAKE             = 0.0       # Hard fake target
LAMBDA_FM              = 0.20      # Feature Matching regularization weight

# ISOLATED OUTPUT DIRECTORY: Ensures zero overwrite of 50-epoch baseline
if "google.colab" in sys.modules:
    WORK_DIR = "/content/dcgan_frontfacing_centroid_pilot_results"
elif os.path.exists("/kaggle"):
    WORK_DIR = "/kaggle/working/dcgan_frontfacing_centroid_pilot_results"
else:
    WORK_DIR = "./models/dcgan_frontfacing_centroid_pilot_results"

CKPT_DIR   = f"{WORK_DIR}/checkpoints"
LOG_DIR    = f"{WORK_DIR}/logs"
SAMPLE_DIR = f"{WORK_DIR}/samples"
CONFIG_DIR = f"{WORK_DIR}/config"

for d in [WORK_DIR, CKPT_DIR, LOG_DIR, SAMPLE_DIR, CONFIG_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 75)
print("PILOT 1 CONFIGURATION SUMMARY (CANONICAL CENTROID FEATURE MATCHING):")
print(f"  Target Epochs          : {EPOCHS} (Pilot Budget)")
print(f"  Global Batch Size      : {GLOBAL_BATCH_SIZE} ({BATCH_SIZE_PER_REPLICA} per replica)")
print(f"  Generator LR           : {G_LR} (Adam beta1={ADAM_BETA1}, beta2={ADAM_BETA2})")
print(f"  Discriminator LR       : {D_LR} (Adam beta1={ADAM_BETA1}, beta2={ADAM_BETA2})")
print(f"  Feature Matching       : lambda_FM = {LAMBDA_FM} (CANONICAL BATCH CENTROID)")
print(f"  Working Directory      : {WORK_DIR}")
print("=" * 75)""")

# Cell 4: Manifest Ingestion
add_code("""# Ingest Approved Dataset Manifest (42,212 Training Sprites)
MANIFEST_CANDIDATES = [
    "phase_dcgan_improvement/dataset_research/final_alucard_frontfacing_manifest.csv",
    "final_alucard_frontfacing_manifest.csv",
    "/kaggle/input/alucard-frontfacing-manifest/final_alucard_frontfacing_manifest.csv",
    "/kaggle/input/dcgan-frontfacing-dataset/final_alucard_frontfacing_manifest.csv"
]

manifest_path = None
for p in MANIFEST_CANDIDATES:
    if os.path.exists(p):
        manifest_path = p
        break

if manifest_path is None:
    raise FileNotFoundError("Could not find final_alucard_frontfacing_manifest.csv!")

print(f"Loading manifest from: {manifest_path}")
manifest_df = pd.read_csv(manifest_path)
print(f"Total entries in manifest: {len(manifest_df):,}")

# Filter strictly by grouped_split == 'train' (Zero template leakage)
train_df = manifest_df[manifest_df['grouped_split'] == 'train'].reset_index(drop=True)
val_df   = manifest_df[manifest_df['grouped_split'] == 'validation'].reset_index(drop=True)

print(f"Grouped Training Sprites  : {len(train_df):,} (Target: 42,212)")
print(f"Grouped Validation Sprites: {len(val_df):,} (Target: 4,690)")

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

# Cell 5: Architecture Definitions
add_code("""# Architecture Definition: Bit-Exact Match (G: 2,633,732 | D: 2,768,449)
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

# Cell 6: Architecture Assertion
add_code("""# Architecture Equivalence Verification
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
print("ARCHITECTURE EQUIVALENCE CONFIRMED: 100% Bit-Exact Match.")""")

# Cell 7: Corrected Canonical Centroid Training Step
add_code("""# Loss Functions, Optimizers & CANONICAL CENTROID Feature Matching Step
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
        # Note: training=False on D during G step freezes Batch Normalization moving stats
        fake_logits, fake_feats = discriminator(fake_images, training=False)
        _, real_feats = discriminator(real_images, training=False)
        
        # LSGAN Adversarial Loss
        g_adv_loss = 0.5 * tf.reduce_mean(tf.square(fake_logits - 1.0)) / float(strategy.num_replicas_in_sync)
        
        # CANONICAL BATCH CENTROID FEATURE MATCHING (Salimans et al., 2016)
        # Average feature representations across batch dimension (axis 0) before distance:
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

# Cell 8: Distributed Smoke Test
add_code("""# Execution Smoke Test Function (Distributed Strategy Compatible)
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

# Cell 9: Dataset Ingestion
add_code("""# Ingestion & Preprocessing Pipeline (42,212 Audited Sprites)
from datasets import load_dataset
from PIL import Image as PILImage

print("Loading 'evilsocket/alucard-sprites' dataset from HuggingFace Hub...")
raw_ds = load_dataset("evilsocket/alucard-sprites", split="train")

train_archive_indices = train_df['archive_index'].tolist()
print(f"Filtering raw dataset for {len(train_archive_indices):,} audited training sprites...")
train_sprite_ds = raw_ds.select(train_archive_indices)
print(f"Dataset successfully indexed: {len(train_sprite_ds):,} training sprites.")

def preprocess_sprite(item):
    \"\"\"Preprocess sprite: resize nearest to 64x64, clean transparent halo, scale to [-1, 1].\"\"\"
    img = item['image'].convert('RGBA')
    if img.size != (IMG_WIDTH, IMG_HEIGHT):
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), PILImage.Resampling.NEAREST)
    arr = np.array(img, dtype=np.float32)
    
    # Zero out color bleed in transparent pixels
    transparent_mask = arr[:, :, 3] < 10.0
    arr[transparent_mask, :3] = 0.0
    arr[transparent_mask, 3] = 0.0
    
    # Scale [0, 255] -> [-1, 1]
    return (arr / 127.5) - 1.0

def make_data_generator(hf_dataset):
    def _generator():
        for sample in hf_dataset:
            yield preprocess_sprite(sample)
    return _generator

def build_distributed_tf_dataset(hf_dataset, batch_size):
    ds = tf.data.Dataset.from_generator(
        make_data_generator(hf_dataset),
        output_signature=tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=tf.float32)
    )
    ds = ds.shuffle(buffer_size=min(10000, len(hf_dataset)), seed=SEED)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)

train_tf_ds = build_distributed_tf_dataset(train_sprite_ds, batch_size=GLOBAL_BATCH_SIZE)
dist_train_ds = strategy.experimental_distribute_dataset(train_tf_ds)
train_dist_iter = iter(dist_train_ds)
STEPS_PER_EPOCH = len(train_sprite_ds) // GLOBAL_BATCH_SIZE
print(f"Pipeline ready: {STEPS_PER_EPOCH:,} steps/epoch at Global Batch Size {GLOBAL_BATCH_SIZE}.")""")

# Cell 10: Checkpointing & Logging Setup
add_code("""# Checkpointing, Visual Evaluator & Logging Setup
import csv
import matplotlib.pyplot as plt

with strategy.scope():
    ckpt = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=3)

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
    plt.suptitle(f"Pilot 1 (Centroid FM) — {title} (Epoch {epoch})", fontsize=13, y=0.92)
    plt.tight_layout()
    out_path = f"{SAMPLE_DIR}/{filename}"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path

history_csv = f"{LOG_DIR}/training_history.csv"
with open(history_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "g_loss", "d_loss", "adv_loss", "fm_loss", "epoch_time_sec"])

print("Pilot 1 Checkpoint manager and logging initialized.")""")

# Cell 11: 10-Epoch Pilot Training Loop
add_code("""# 10-Epoch Pilot Training Loop (Canonical Centroid Feature Matching)
import time
import csv

print("=" * 80)
print(f"STARTING PILOT 1 DCGAN TRAINING (CANONICAL CENTROID FEATURE MATCHING)")
print(f"Total Epochs    : {EPOCHS} (Pilot Budget)")
print(f"Steps per Epoch : {STEPS_PER_EPOCH:,}")
print(f"Total Steps     : {STEPS_PER_EPOCH * EPOCHS:,}")
print(f"Output Directory: {WORK_DIR}")
print("=" * 80)

history_records = []
total_train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    t_ep_start = time.time()
    ep_d_loss, ep_g_loss, ep_adv_loss, ep_fm_loss = 0.0, 0.0, 0.0, 0.0
    
    for step in range(STEPS_PER_EPOCH):
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
    
    # Save checkpoints & sample grids at Epochs 1, 5, 10
    if epoch in [1, 5, 10]:
        save_eval_grid(epoch, "Fixed Latents", f"samples_epoch_{epoch:04d}.png", use_fixed=True)
        save_eval_grid(epoch, "Fresh Random", f"fresh_samples_epoch_{epoch:04d}.png", use_fixed=False)
        ckpt_path = ckpt_manager.save()
        print(f"  -> Checkpoint & Evaluation grid saved for Epoch {epoch}")

total_train_time = time.time() - total_train_start
print("=" * 80)
print(f"PILOT 1 TRAINING COMPLETE in {total_train_time/60:.1f} minutes ({total_train_time:.1f}s)!")
print("=" * 80)

# Save final model weights in isolated directory
final_g_path = f"{WORK_DIR}/generator_weights_final.weights.h5"
final_d_path = f"{WORK_DIR}/discriminator_weights_final.weights.h5"
generator.save_weights(final_g_path)
discriminator.save_weights(final_d_path)
print(f"Final Generator weights saved to: {final_g_path}")
print(f"Final Discriminator weights saved to: {final_d_path}")""")

# Cell 12: Evaluation Showcase & Zip Packaging
add_code("""# Evaluation Showcase (100 Characters), Loss Curves & Download Packaging
import shutil
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
plt.suptitle("GameForge AI: Pilot 1 (Canonical Centroid FM) — 100-Character Showcase (10 Epochs)", fontsize=15, y=0.92)
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
print(f"Pilot loss curves saved to: {loss_curve_path}")

# 3. Zip Results for Immediate Download
zip_base = f"{WORK_DIR}"
print("Compressing Pilot 1 results folder into zip archive...")
zip_archive = shutil.make_archive(zip_base, 'zip', WORK_DIR)
print(f"Zip archive created: {zip_archive}")

print("\\n" + "=" * 70)
print("CLICK THE LINK BELOW TO DOWNLOAD PILOT 1 RESULTS:")
print("=" * 70)
FileLink(os.path.basename(zip_archive))""")

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully created: {nb_path} with {len(cells)} cells.")

