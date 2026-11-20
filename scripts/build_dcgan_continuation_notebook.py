"""
scripts/build_dcgan_continuation_notebook.py
Builds the Notebook/training/02_DCGAN_Discrete_Continuation_Epochs51_to_100.ipynb notebook.
Keeps 100% pure DCGAN architecture so it can resume directly from the 50-epoch weights:
- generator_weights_final.weights.h5
- discriminator_weights_final.weights.h5

Targeted DCGAN-Specific Improvements:
1. Resumes from Epoch 50 weights (or trains fresh if weights not uploaded).
2. Reduced Learning Rate with Cosine Decay: Refines sub-pixel boundaries without thrashing.
3. TTUR Rebalancing: G is updated with a stabilizing ratio to pull out of late-epoch mode collapse.
4. Strengthened DiffAugment: Prevents D from memorizing front-facing poses.
5. G-EMA (Exponential Moving Average): Shadow weights for silky-smooth sprite rendering.
6. Temperature Annealing: Hardens straight-through Gumbel palette assignment.
7. Best-Gate Checkpointing: Saves the visual peak rather than an overcooked final epoch.
8. Largest Connected Component (LCC) post-processor for floater removal at export.
"""
import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_NB_PATH = os.path.join(PROJECT_ROOT, "Notebook", "training", "02_DCGAN_Discrete_Continuation_Epochs51_to_100.ipynb")

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
    "nbformat_minor": 5
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# ==============================================================================
# CELL 0: TITLE & MANIFESTO
# ==============================================================================
add_md(r"""# 🎮 02 — Discrete DCGAN Continuation: Epochs 51 to 100 (Refinement Edition)

**Project:** Prompt-to-Game-Asset Generator  
**Model:** 100% Pure Discrete DCGAN (Exact Weight-Compatible Architecture)  
**Starting Point:** Resumes from Epoch 50 weights (`generator_weights_final.weights.h5`)  
**Target:** 50 Additional Epochs (Epoch 51 $\to$ 100) of Fine-Grained Convergence

---

### 🚀 Targeted DCGAN Improvements (Preserving 100% Weight Compatibility)

1. **Seamless Weight Resumption:**
   - Loads your existing Epoch 50 weights (`generator_weights_final.weights.h5` and `discriminator_weights_final.weights.h5`) directly without changing layer shapes.
2. **Fine-Tuning Cosine Learning Rate Schedule:**
   - Drops LR from initial $3\times 10^{-4}$ down to $1\times 10^{-4} \to 2\times 10^{-5}$ via Cosine Decay. Prevents the generator from thrashing and allows delicate eye/limb articulation to converge.
3. **TTUR Balance Restoration:**
   - In late Epochs (35–50), Discriminator was slightly overpowered ($D=0.88, G=1.12$), triggering partial mode collapse toward the pale archer. We tune the relative learning rates to give G more capacity to escape local minima.
4. **Generator Exponential Moving Average (G-EMA):**
   - Tracks shadow weights (decay 0.999) to smooth out high-frequency gradient jitter and produce clean, continuous contours.
5. **Gumbel Softmax Temperature Annealing ($\tau: 0.9 \to 0.4$):**
   - Forces the categorical color distribution into sharp, high-contrast retro palette choices.
6. **Best-Gate Checkpointing:**
   - Saves `generator_best_gate.weights.h5` based on multi-metric visual quality ($S = \text{Diversity} \times \text{Coherence} \times \frac{\text{Colors}}{18}$).
7. **Connected Component Floater Pruning:**
   - Automatically cleans up detached stray pixels on export.
""")

# ==============================================================================
# CELL 1: ENVIRONMENT & GPU DISCOVERY
# ==============================================================================
add_code(r"""import os
import sys
import json
import time
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import label
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow Version: {tf.__version__}")
print(f"Eager Execution: {tf.executing_eagerly()}")

# Distributed Multi-GPU setup
gpus = tf.config.list_physical_devices('GPU')
print(f"Physical GPUs Available: {len(gpus)}")
for g in gpus:
    print(f"  - {g}")

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"MirroredStrategy active across {strategy.num_replicas_in_sync} GPUs.")
elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy()
    print("Single GPU active.")
else:
    strategy = tf.distribute.get_strategy()
    print("Running on CPU.")

assert hasattr(layers, "SpectralNormalization"), "SpectralNormalization is strictly required."
""")

# ==============================================================================
# CELL 2: HYPERPARAMETERS & CONFIGURATION
# ==============================================================================
add_code(r"""# Canvas & Architecture Constants (Identical to 50-Epoch Run for 100% Weight Compatibility)
IMG_HEIGHT             = 64
IMG_WIDTH              = 64
CHANNELS               = 4         # RGBA
PALETTE_SIZE           = 32        # Empirical global 32-color palette
LATENT_DIM             = 128

G_CHANNELS             = [512, 256, 128, 64]
D_CHANNELS             = [64, 128, 256, 512]
G_INIT_SPATIAL         = 4

# Continuation Epoch Schedule
START_EPOCH            = 51
ADDITIONAL_EPOCHS      = 50
END_EPOCH              = START_EPOCH + ADDITIONAL_EPOCHS - 1  # Epoch 100

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE      = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Fine-Tuning Learning Rates (Reduced for gentle refinement)
G_LR_INITIAL           = 1.5e-4
D_LR_INITIAL           = 0.8e-4
ADAM_BETA1             = 0.0
ADAM_BETA2             = 0.9

# Annealing and EMA
TAU_START              = 0.9
TAU_END                = 0.4
EMA_DECAY              = 0.999

# Directories
OUTPUT_DIR             = "dcgan_continuation_51_to_100_results"
SAMPLES_DIR            = os.path.join(OUTPUT_DIR, "samples")
CHECKPOINTS_DIR        = os.path.join(OUTPUT_DIR, "checkpoints")
CONFIG_DIR             = os.path.join(OUTPUT_DIR, "config")
LOGS_DIR               = os.path.join(OUTPUT_DIR, "logs")
ASSETS_DIR             = os.path.join(OUTPUT_DIR, "game_ready_assets")

for d in [SAMPLES_DIR, CHECKPOINTS_DIR, CONFIG_DIR, LOGS_DIR, ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 2026
tf.random.set_seed(SEED)
np.random.seed(SEED)
print(f"Continuation Run: Epochs {START_EPOCH} -> {END_EPOCH} | Global Batch Size: {GLOBAL_BATCH_SIZE}")
""")

# ==============================================================================
# CELL 3: DATASET DISCOVERY & PRE-EXTRACTED FRONT-FACING SPRITES
# ==============================================================================
add_code(r"""# Section 3: Universal Dataset Loader (HuggingFace Hub + Auto-Extract Front-Facing Sprites)
import os, json, re, time
import numpy as np
from PIL import Image

HF_DATASET_NAME = "evilsocket/alucard-sprites"
INDICES_FILE_NAME = "alucard_frontfacing_42k_indices.json"

# Search for existing indices file
curated_indices = None
INDICES_CANDIDATES = [
    f"{CONFIG_DIR}/{INDICES_FILE_NAME}",
    f"{INDICES_FILE_NAME}",
    f"/kaggle/working/{INDICES_FILE_NAME}",
    f"/kaggle/input/{INDICES_FILE_NAME}",
    f"/kaggle/input/alucard-frontfacing/{INDICES_FILE_NAME}",
    f"/kaggle/input/alucard-sprites/{INDICES_FILE_NAME}",
    f"/kaggle/input/dcgan-frontfacing-results/config/{INDICES_FILE_NAME}",
    f"Notebook/training/{INDICES_FILE_NAME}",
    f"Notebook/{INDICES_FILE_NAME}"
]

for p in INDICES_CANDIDATES:
    if os.path.isfile(p):
        try:
            with open(p, "r") as f:
                d = json.load(f)
                curated_indices = d if isinstance(d, list) else d.get("indices", [])
            print(f"Loaded {len(curated_indices):,} indices from {p}")
            break
        except Exception:
            pass

# Check for local directory; fallback to HuggingFace Hub
LOCAL_CANDIDATES = [
    "/kaggle/input/alucard-sprites",
    "/kaggle/input/evilsocket-alucard-sprites",
    "../input/alucard-sprites",
    "./alucard-sprites",
    "dataset/alucard-sprites"
]
DATASET_ROOT = None
for p in LOCAL_CANDIDATES:
    if os.path.isdir(p):
        DATASET_ROOT = p
        break

if DATASET_ROOT is not None:
    print(f"Loading sprites from local directory: {DATASET_ROOT}")
    all_files = sorted([
        os.path.join(DATASET_ROOT, f) for f in os.listdir(DATASET_ROOT)
        if f.lower().endswith(('.png', '.webp'))
    ])
    if curated_indices is None:
        curated_indices = list(range(len(all_files)))
    curated_items = [all_files[i] for i in curated_indices if i < len(all_files)]
    is_local_files = True
else:
    print(f"Loading '{HF_DATASET_NAME}' from HuggingFace Hub...")
    from datasets import load_dataset
    raw_ds = load_dataset(HF_DATASET_NAME, split="train")
    
    if curated_indices is None:
        print("Auto-extracting strictly front-facing sprites from dataset captions...")
        RE_FRONT = re.compile(r"\b(front view|facing front|front facing|canonical south)\b", re.IGNORECASE)
        RE_BACK  = re.compile(r"\b(back view|back|rear|behind|facing away|facing north|north|walk_north)\b", re.IGNORECASE)
        RE_SIDE  = re.compile(r"\b(side view|profile|facing east|facing west|walk_east|walk_west|east|west)\b", re.IGNORECASE)
        
        t_filter = time.time()
        curated_indices = [
            i for i, text in enumerate(raw_ds["text"])
            if RE_FRONT.search(text) and not RE_BACK.search(text) and not RE_SIDE.search(text)
        ]
        print(f"Auto-extracted {len(curated_indices):,} front-facing sprites in {time.time()-t_filter:.1f}s!")
        
        with open(os.path.join(CONFIG_DIR, INDICES_FILE_NAME), "w") as f:
            json.dump(curated_indices, f)
            
    curated_items = raw_ds.select(curated_indices)
    is_local_files = False

N_SPRITES = len(curated_items)
print(f"Total Curated Front-Facing Sprites: {N_SPRITES:,}")

# Look for existing palette from 50-epoch run or recompute
PALETTE_CANDIDATES = [
    "/kaggle/input/dcgan-frontfacing-results/config/global_palette_32.npy",
    "config/global_palette_32.npy",
    f"{CONFIG_DIR}/global_palette_32.npy"
]
existing_palette = None
for pp in PALETTE_CANDIDATES:
    if os.path.isfile(pp):
        existing_palette = np.load(pp)
        print(f"Found existing 32-color palette at: {pp}")
        break

# Preload sprites with nearest downsampling & 1-bit alpha
print("Preloading sprites with Nearest-Neighbor resize & binary alpha into RAM...")
t0 = time.time()
raw_images = np.zeros((N_SPRITES, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)

for i in range(N_SPRITES):
    if is_local_files:
        with Image.open(curated_items[i]) as im:
            im = im.convert("RGBA").resize((64, 64), Image.NEAREST)
            arr = np.array(im)
    else:
        im = curated_items[i]["image"].convert("RGBA").resize((64, 64), Image.NEAREST)
        arr = np.array(im)
        
    fg = arr[..., 3] > 127
    arr_clean = np.zeros_like(arr)
    arr_clean[fg, :3] = arr[fg, :3]
    arr_clean[fg, 3] = 255
    raw_images[i] = arr_clean
    
    if (i + 1) % 10000 == 0 or (i + 1) == N_SPRITES:
        print(f"  Loaded {i+1:,}/{N_SPRITES:,} sprites ({time.time() - t0:.1f}s)")

if existing_palette is not None:
    global_palette_u8 = existing_palette
else:
    print("Computing 32-Color Global Palette...")
    sample_indices = np.random.choice(N_SPRITES, size=min(3000, N_SPRITES), replace=False)
    sample_pixels = []
    for idx in sample_indices:
        spr = raw_images[idx]
        fg = spr[..., 3] > 0
        if np.any(fg):
            sample_pixels.append(spr[..., :3][fg])
    sample_pixels = np.concatenate(sample_pixels, axis=0)
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=PALETTE_SIZE, random_state=SEED, batch_size=4096, n_init=3)
    kmeans.fit(sample_pixels)
    global_palette_u8 = np.clip(np.round(kmeans.cluster_centers_), 0, 255).astype(np.uint8)

np.save(os.path.join(CONFIG_DIR, "global_palette_32.npy"), global_palette_u8)
print(f"Global 32-Color Palette active and saved to {CONFIG_DIR}.")
""")

# ==============================================================================
# CELL 4: MEMORY-SAFE UINT8 INDEX DATASET & LOOKUP TABLE
# ==============================================================================
add_code(r"""# Quantize into uint8 index array (~172 MB in RAM)
BG_INDEX = PALETTE_SIZE
idx_dataset = np.full((N_SPRITES, IMG_HEIGHT, IMG_WIDTH), BG_INDEX, dtype=np.uint8)
pal_32_f32 = global_palette_u8.astype(np.float32)

t0 = time.time()
print(f"Quantizing {N_SPRITES:,} sprites into uint8 index matrix...")
for i in range(N_SPRITES):
    spr = raw_images[i]
    fg = spr[..., 3] > 0
    if np.any(fg):
        px = spr[..., :3][fg].astype(np.float32)
        dists = np.sum((px[:, None, :] - pal_32_f32[None, :, :]) ** 2, axis=-1)
        nearest_idx = np.argmin(dists, axis=-1).astype(np.uint8)
        idx_dataset[i][fg] = nearest_idx
    if (i + 1) % 10000 == 0 or (i + 1) == N_SPRITES:
        print(f"  Quantized {i+1:,}/{N_SPRITES:,} sprites ({(time.time()-t0):.1f}s)")

import gc
del raw_images
gc.collect()

print(f"Total uint8 Index Matrix Size: {idx_dataset.nbytes / (1024*1024):.1f} MB (No 2GB protobuf limit risk).")

# RGBA Lookup Table in [-1.0, 1.0]
lut = np.zeros((PALETTE_SIZE + 1, 4), dtype=np.float32)
lut[:PALETTE_SIZE, :3] = (global_palette_u8 / 127.5) - 1.0
lut[:PALETTE_SIZE, 3] = 1.0
lut[BG_INDEX] = -1.0
lut_tf = tf.constant(lut, dtype=tf.float32)

def expand_indices_to_rgba(idx_batch):
    return tf.gather(lut_tf, tf.cast(idx_batch, tf.int32))

tf_dataset = (
    tf.data.Dataset.from_tensor_slices(idx_dataset)
    .shuffle(10000, seed=SEED)
    .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    .map(expand_indices_to_rgba, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

dist_dataset = strategy.experimental_distribute_dataset(tf_dataset)
print("Distributed tf.data pipeline created.")
""")

# ==============================================================================
# CELL 5: EXACT DCGAN ARCHITECTURE (WEIGHT-COMPATIBLE)
# ==============================================================================
add_code(r"""# Exact 100% Architecture Match to Attempt 4 / 50-Epoch Run
def build_generator(latent_dim=LATENT_DIM):
    z_in = layers.Input(shape=(latent_dim,), name="latent_vector")
    x = layers.Dense(G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0], use_bias=False)(z_in)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((G_INIT_SPATIAL, G_INIT_SPATIAL, G_CHANNELS[0]))(x)
    
    # 4x4 -> 8x8
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    x = layers.Conv2D(G_CHANNELS[1], (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # 8x8 -> 16x16
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    x = layers.Conv2D(G_CHANNELS[2], (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # 16x16 -> 32x32
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    x = layers.Conv2D(G_CHANNELS[3], (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # 32x32 -> 64x64
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Full-Resolution 64x64 Refinement Layer
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Discrete K+1 Logits Head (32 colors + 1 alpha logit)
    logits = layers.Conv2D(PALETTE_SIZE + 1, (3, 3), padding="same", name="palette_logits")(x)
    return keras.Model(z_in, logits, name="Discrete_DCGAN_Generator")

def build_discriminator():
    img_in = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="sprite_input")
    
    def sn_conv(filters, strides=2):
        conv = layers.Conv2D(filters, (4, 4), strides=strides, padding="same")
        return layers.SpectralNormalization(conv)
        
    x = sn_conv(D_CHANNELS[0])(img_in)
    x = layers.LeakyReLU(0.2)(x)
    
    x = sn_conv(D_CHANNELS[1])(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = sn_conv(D_CHANNELS[2])(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = sn_conv(D_CHANNELS[3])(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Flatten()(x)
    out = layers.SpectralNormalization(layers.Dense(1))(x)
    return keras.Model(img_in, out, name="Discrete_DCGAN_Discriminator")

palette_01_tf = tf.constant(global_palette_u8 / 255.0, dtype=tf.float32)

def logits_to_image(logits, tau=1.0, hard=True):
    pl = logits[..., :PALETTE_SIZE]
    al = logits[..., PALETTE_SIZE:]
    ys = tf.nn.softmax(pl / tau, axis=-1)
    if hard:
        y_hard = tf.one_hot(tf.argmax(ys, axis=-1), depth=PALETTE_SIZE, dtype=tf.float32)
        y = tf.stop_gradient(y_hard - ys) + ys
    else:
        y = ys
    rgb = tf.tensordot(y, palette_01_tf, axes=[[-1], [0]])
    a_s = tf.sigmoid(al)
    if hard:
        a_hard = tf.cast(a_s > 0.5, dtype=tf.float32)
        a = tf.stop_gradient(a_hard - a_s) + a_s
    else:
        a = a_s
    img_01 = tf.concat([rgb * a, a], axis=-1)
    return img_01 * 2.0 - 1.0

print("Exact DCGAN Generator & Discriminator architectures compiled successfully.")
""")

# ==============================================================================
# CELL 6: WEIGHT RESUMPTION, EMA & DISTRIBUTED TRAINING STEP
# ==============================================================================
add_code(r"""# Enhanced DiffAugment Translation (±15% to break front-facing mode collapse)
diff_aug_layer = layers.RandomTranslation(
    height_factor=(-0.15, 0.15),
    width_factor=(-0.15, 0.15),
    fill_mode="constant",
    fill_value=-1.0
)

# Search for Epoch 50 weights
WEIGHT_CANDIDATES_G = [
    "/kaggle/input/dcgan-frontfacing-results/game_ready_assets/generator_weights_final.weights.h5",
    "/kaggle/input/dcgan-frontfacing-results/checkpoints/ckpt-25.data-00000-of-00001",
    "game_ready_assets/generator_weights_final.weights.h5",
    f"{ASSETS_DIR}/generator_weights_final.weights.h5",
    "/kaggle/working/generator_weights_final.weights.h5"
]

WEIGHT_CANDIDATES_D = [
    "/kaggle/input/dcgan-frontfacing-results/game_ready_assets/discriminator_weights_final.weights.h5",
    "game_ready_assets/discriminator_weights_final.weights.h5",
    f"{ASSETS_DIR}/discriminator_weights_final.weights.h5",
    "/kaggle/working/discriminator_weights_final.weights.h5"
]

with strategy.scope():
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Cosine Decay Learning Rate for fine refinement
    g_lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=G_LR_INITIAL,
        decay_steps=ADDITIONAL_EPOCHS * (N_SPRITES // GLOBAL_BATCH_SIZE),
        alpha=0.1
    )
    d_lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=D_LR_INITIAL,
        decay_steps=ADDITIONAL_EPOCHS * (N_SPRITES // GLOBAL_BATCH_SIZE),
        alpha=0.1
    )
    
    g_optimizer = keras.optimizers.Adam(learning_rate=g_lr_schedule, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    d_optimizer = keras.optimizers.Adam(learning_rate=d_lr_schedule, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    
    # Generator Exponential Moving Average (G-EMA)
    ema_generator = build_generator()
    
    g_optimizer.build(generator.trainable_variables)
    d_optimizer.build(discriminator.trainable_variables)

# Load Epoch 50 weights if found
resumed_g = False
for g_path in WEIGHT_CANDIDATES_G:
    if os.path.isfile(g_path):
        try:
            generator.load_weights(g_path)
            ema_generator.load_weights(g_path)
            print(f"SUCCESS: Resumed Generator from Epoch 50 weights: {g_path}")
            resumed_g = True
            break
        except Exception as e:
            print(f"Failed loading G from {g_path}: {e}")

resumed_d = False
for d_path in WEIGHT_CANDIDATES_D:
    if os.path.isfile(d_path):
        try:
            discriminator.load_weights(d_path)
            print(f"SUCCESS: Resumed Discriminator from Epoch 50 weights: {d_path}")
            resumed_d = True
            break
        except Exception as e:
            print(f"Failed loading D from {d_path}: {e}")

if not resumed_g:
    print("NOTE: Epoch 50 weights not found. Training DCGAN fresh with Cosine Decay & G-EMA.")
    ema_generator.set_weights(generator.get_weights())

def update_ema():
    for ew, gw in zip(ema_generator.weights, generator.weights):
        ew.assign(EMA_DECAY * ew + (1.0 - EMA_DECAY) * gw)

def train_step_per_replica(real_imgs, tau_val):
    local_batch = tf.shape(real_imgs)[0]
    z = tf.random.normal((local_batch, LATENT_DIM))
    
    # 1. Update Discriminator
    with tf.GradientTape() as d_tape:
        fake_logits = generator(z, training=True)
        fake_imgs = logits_to_image(fake_logits, tau=tau_val, hard=True)
        
        real_aug = diff_aug_layer(real_imgs, training=True)
        fake_aug = diff_aug_layer(fake_imgs, training=True)
        
        d_real = discriminator(real_aug, training=True)
        d_fake = discriminator(fake_aug, training=True)
        d_loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real)) + tf.reduce_mean(tf.nn.relu(1.0 + d_fake))
        
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    # 2. Update Generator
    with tf.GradientTape() as g_tape:
        fake_logits_g = generator(z, training=True)
        fake_imgs_g = logits_to_image(fake_logits_g, tau=tau_val, hard=True)
        fake_aug_g = diff_aug_layer(fake_imgs_g, training=True)
        d_fake_g = discriminator(fake_aug_g, training=True)
        g_loss = -tf.reduce_mean(d_fake_g)
        
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    
    return d_loss, g_loss

@tf.function
def distributed_train_step(dist_real_images, tau_val):
    per_d, per_g = strategy.run(train_step_per_replica, args=(dist_real_images, tau_val))
    d_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_d, axis=None)
    g_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_g, axis=None)
    return d_loss, g_loss

print("Running 1-step warmup across GPU replicas...")
dummy_ds = tf.data.Dataset.from_tensors(
    tf.zeros((GLOBAL_BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=tf.float32)
)
dist_dummy = strategy.experimental_distribute_dataset(dummy_ds)
distributed_train_step(next(iter(dist_dummy)), tf.constant(1.0, dtype=tf.float32))
print("Warmup: SUCCESS! Replicas and optimizer states ready.")
""")

# ==============================================================================
# CELL 7: CONTINUATION TRAINING LOOP (EPOCHS 51 TO 100)
# ==============================================================================
add_code(r"""# Load fixed latent vectors from 50-epoch run if available for direct comparison
FIXED_Z_CANDIDATES = [
    "/kaggle/input/dcgan-frontfacing-results/config/fixed_latent_vectors.npy",
    "config/fixed_latent_vectors.npy",
    f"{CONFIG_DIR}/fixed_latent_vectors.npy"
]
fixed_eval_z = None
for fzp in FIXED_Z_CANDIDATES:
    if os.path.isfile(fzp):
        fixed_eval_z = tf.constant(np.load(fzp))
        print(f"Loaded original fixed latent vectors from: {fzp}")
        break

if fixed_eval_z is None:
    fixed_eval_z = tf.random.normal([36, LATENT_DIM], seed=42)
    np.save(os.path.join(CONFIG_DIR, "fixed_latent_vectors.npy"), fixed_eval_z.numpy())

steps_per_epoch = N_SPRITES // GLOBAL_BATCH_SIZE
history = []
best_gate_score = -1.0
t_start = time.time()

print(f"Starting DCGAN Continuation Training (Epochs {START_EPOCH} to {END_EPOCH})...")
print(f"Steps per Epoch: {steps_per_epoch} | Batch Size: {GLOBAL_BATCH_SIZE}")

for epoch in range(START_EPOCH, END_EPOCH + 1):
    epoch_t0 = time.time()
    
    # Anneal Gumbel Temperature from TAU_START (0.9) to TAU_END (0.4)
    progress = float(epoch - START_EPOCH) / float(max(1, ADDITIONAL_EPOCHS - 1))
    tau_current = TAU_START - (TAU_START - TAU_END) * progress
    
    d_loss_accum = 0.0
    g_loss_accum = 0.0
    
    for step, real_batch in enumerate(dist_dataset):
        d_l, g_l = distributed_train_step(real_batch, tf.constant(tau_current, dtype=tf.float32))
        d_loss_accum += float(d_l)
        g_loss_accum += float(g_l)
        update_ema()
        
    d_loss_avg = d_loss_accum / steps_per_epoch
    g_loss_avg = g_loss_accum / steps_per_epoch
    
    # Visual Evaluation using EMA Generator
    eval_logits = ema_generator(fixed_eval_z, training=False)
    eval_imgs_tf = logits_to_image(eval_logits, tau=tau_current, hard=True)
    eval_imgs = tf.cast((eval_imgs_tf + 1.0) * 127.5, tf.uint8).numpy()
    
    # Save visual sample grid every 2 epochs
    if epoch == START_EPOCH or epoch % 2 == 0 or epoch == END_EPOCH:
        grid = Image.new("RGBA", (6 * 64, 6 * 64), (255, 255, 255, 255))
        for idx in range(36):
            r, c = idx // 6, idx % 6
            p_img = Image.fromarray(eval_imgs[idx])
            grid.paste(p_img, (c * 64, r * 64), mask=p_img.split()[3])
        grid.save(os.path.join(SAMPLES_DIR, f"samples_epoch_{epoch:04d}.png"))
    
    # Visual Gate Metrics
    colors_list = []
    fg_area_list = []
    for img in eval_imgs:
        fg = img[..., 3] > 127
        fg_area_list.append(np.mean(fg))
        if np.any(fg):
            colors_list.append(len(np.unique(img[..., :3][fg], axis=0)))
            
    mean_colors = float(np.mean(colors_list)) if colors_list else 0.0
    mean_fg = float(np.mean(fg_area_list))
    
    flat = eval_imgs.astype(np.float32).reshape((36, -1)) / 255.0
    dists = [np.mean(np.abs(flat[i] - flat[j])) for i in range(36) for j in range(i+1, 36)]
    mean_div = float(np.mean(dists))
    mean_coherence = float(np.clip(1.0 - abs(mean_fg - 0.35) * 2.0, 0.0, 1.0))
    gate_score = mean_div * mean_coherence * min(1.0, mean_colors / 18.0)
    
    epoch_dur = time.time() - epoch_t0
    print(f"Epoch {epoch:03d}/{END_EPOCH} ({tau_current:.2f}tau) | D: {d_loss_avg:.4f} | G: {g_loss_avg:.4f} | "
          f"Colors: {mean_colors:.1f} | Div: {mean_div:.3f} | GateScore: {gate_score:.4f} ({epoch_dur:.1f}s)")
          
    history.append({
        "epoch": epoch, "d_loss": d_loss_avg, "g_loss": g_loss_avg,
        "colors": mean_colors, "diversity": mean_div, "gate_score": gate_score
    })
    
    # Save Best Checkpoint by Visual Gate Score
    if gate_score > best_gate_score:
        best_gate_score = gate_score
        ema_generator.save_weights(os.path.join(CHECKPOINTS_DIR, "generator_best_gate.weights.h5"))
        print(f"  --> Saved new best checkpoint (Gate Score: {best_gate_score:.4f})")

# Save training history and final weights
with open(os.path.join(LOGS_DIR, "continuation_history.csv"), "w") as f:
    f.write("epoch,d_loss,g_loss,colors,diversity,gate_score\n")
    for h in history:
        f.write(f"{h['epoch']},{h['d_loss']:.4f},{h['g_loss']:.4f},{h['colors']:.2f},{h['diversity']:.4f},{h['gate_score']:.4f}\n")

generator.save_weights(os.path.join(ASSETS_DIR, "generator_continuation_final.weights.h5"))
ema_generator.save_weights(os.path.join(ASSETS_DIR, "generator_ema_continuation_final.weights.h5"))
print(f"Continuation complete in {(time.time() - t_start)/60:.1f} minutes!")
""")

# ==============================================================================
# CELL 8: LARGEST CONNECTED COMPONENT (LCC) POST-PROCESSING & EXPORT
# ==============================================================================
add_code(r"""def clean_floaters(img_rgba):
    alpha = img_rgba[..., 3] > 127
    labeled, num_features = label(alpha)
    if num_features <= 1:
        return img_rgba
    sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
    largest = np.argmax(sizes) + 1
    clean_mask = (labeled == largest)
    cleaned = img_rgba.copy()
    cleaned[~clean_mask] = 0
    return cleaned

# Export 36 production game-ready assets from the best gate checkpoint
print("Exporting production assets using best gate checkpoint + LCC filter...")
ema_generator.load_weights(os.path.join(CHECKPOINTS_DIR, "generator_best_gate.weights.h5"))

prod_z = tf.random.normal([36, LATENT_DIM], seed=888)
raw_logits = ema_generator(prod_z, training=False)
raw_imgs = tf.cast((logits_to_image(raw_logits, tau=TAU_END, hard=True) + 1.0) * 127.5, tf.uint8).numpy()

fig, axes = plt.subplots(6, 6, figsize=(12, 12))
for i in range(36):
    r, c = i // 6, i % 6
    clean = clean_floaters(raw_imgs[i])
    axes[r, c].imshow(clean)
    axes[r, c].axis("off")
    Image.fromarray(clean).save(os.path.join(ASSETS_DIR, f"sprite_dcgan_epoch100_{i+1:03d}.png"))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "production_sprites_grid.png"), dpi=150)
plt.show()
print(f"Exported 36 refined discrete sprites to {ASSETS_DIR}.")
""")

# ==============================================================================
# CELL 9: ONE-CLICK ZIP DOWNLOAD
# ==============================================================================
add_code(r"""import shutil
from IPython.display import FileLink, display

zip_base = "dcgan_continuation_51_to_100_results"
zip_output_path = f"{zip_base}.zip"

print(f"Compressing {OUTPUT_DIR} into {zip_output_path}...")
shutil.make_archive(zip_base, 'zip', OUTPUT_DIR)
print(f"Archive created: {zip_output_path} ({os.path.getsize(zip_output_path) / (1024*1024):.2f} MB)")

display(FileLink(zip_output_path))
print("Download ready.")
""")

with open(TARGET_NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated {TARGET_NB_PATH} ({os.path.getsize(TARGET_NB_PATH):,} bytes)")

