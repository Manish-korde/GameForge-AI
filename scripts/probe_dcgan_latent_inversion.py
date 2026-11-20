import os
import sys
import numpy as np
from PIL import Image

# Ensure environment is configured
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow Version: {tf.__version__}")

# Project Root Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "phase_dcgan_improvement", "latent_inversion_probe")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Locate Generator Weights
WEIGHTS_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "models", "dcgan_all_characters_50ep_results", "generator_weights_final.weights.h5"),
    r"C:\Users\manis\Downloads\dcgan_frontfacing_46k_50ep_results\generator_weights_final.weights.h5"
]

WEIGHTS_PATH = None
for p in WEIGHTS_CANDIDATES:
    if os.path.exists(p):
        WEIGHTS_PATH = p
        break

if not WEIGHTS_PATH:
    print(f"ERROR: Could not find generator weights in candidates: {WEIGHTS_CANDIDATES}")
    sys.exit(1)

print(f"Loading Generator weights from: {WEIGHTS_PATH}")

# 2. Build 128-D DCGAN Generator
LATENT_DIM = 128
G_CHANNELS = [512, 256, 128, 64]
G_INIT_SPATIAL = 4

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
    
    # 32x32 -> 64x64 RGBA
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    img_out = layers.Conv2D(4, (3, 3), padding="same", activation="tanh", name="rgba_output")(x)
    return keras.Model(z_in, img_out, name="DCGAN_Generator_128D")

generator = build_generator(LATENT_DIM)
generator.load_weights(WEIGHTS_PATH)
generator.trainable = False  # STRICTLY FREEZE GENERATOR
print("DCGAN Generator initialized and strictly frozen.")

# 3. Load 5 Real Reference Sprites
SAMPLE_IDS = [0, 1, 4, 6, 19]
real_images = []
for sid in SAMPLE_IDS:
    img_path = os.path.join(BACKEND_DIR, "alucard_samples", f"alucard_{sid}.png")
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found.")
        continue
    img = Image.open(img_path).convert("RGBA").resize((64, 64), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # Scale to [-1, 1]
    real_images.append(arr)

real_tensor = tf.convert_to_tensor(np.array(real_images), dtype=tf.float32)
N = len(real_images)
print(f"Loaded {N} real test sprites for Latent Inversion Probe.")

# 4. Run Latent Inversion Probe (Gradient Descent on z)
# Multi-start initialization for global exploration
print("\n--- Starting Latent Inversion Gradient Descent (500 Steps) ---")

# Best z tracking
best_z = tf.Variable(tf.random.normal([N, LATENT_DIM], stddev=0.8), trainable=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.08)

# Denoising and palette snap helper
def snap_to_palette(img_arr, n_colors=16):
    from sklearn.cluster import KMeans
    h, w, c = img_arr.shape
    fg_mask = img_arr[:, :, 3] > 0.1
    if not np.any(fg_mask):
        return (img_arr * 255).astype(np.uint8)
    
    rgb = img_arr[:, :, :3]
    fg_pixels = rgb[fg_mask]
    
    n_c = min(n_colors, len(fg_pixels))
    kmeans = KMeans(n_clusters=n_c, n_init=3, random_state=42).fit(fg_pixels)
    snapped_rgb = kmeans.cluster_centers_[kmeans.labels_]
    
    out_img = np.zeros_like(img_arr)
    out_img[fg_mask, :3] = snapped_rgb
    out_img[fg_mask, 3] = 1.0  # Clean 1-bit alpha
    return np.clip(out_img * 255.0, 0, 255).astype(np.uint8)

loss_history = []
for step in range(500):
    with tf.GradientTape() as tape:
        fake = generator(best_z, training=False)
        
        # 1. Pixel L1 loss
        l1_loss = tf.reduce_mean(tf.abs(fake - real_tensor))
        
        # 2. SSIM structural loss (RGB channels)
        ssim_val = tf.reduce_mean(tf.image.ssim(fake[:, :, :, :3] * 0.5 + 0.5, 
                                                real_tensor[:, :, :, :3] * 0.5 + 0.5, 
                                                max_val=1.0))
        ssim_loss = 1.0 - ssim_val
        
        # 3. Latent prior regularization (keep z on Gaussian sphere)
        reg_loss = 0.01 * tf.reduce_mean(tf.square(best_z))
        
        # 4. Total Variation loss (suppress salt-and-pepper noise)
        tv_loss = 0.02 * tf.reduce_mean(tf.image.total_variation(fake[:, :, :, :3]))
        
        total_loss = l1_loss + 0.5 * ssim_loss + reg_loss + tv_loss

    grads = tape.gradient(total_loss, [best_z])
    optimizer.apply_gradients(zip(grads, [best_z]))
    
    if (step + 1) % 100 == 0 or step == 0:
        print(f"Step {step+1:3d}/500 | Total Loss: {total_loss.numpy():.4f} | L1: {l1_loss.numpy():.4f} | SSIM: {ssim_val.numpy():.4f}")

# 5. Extract Final Inverted Reconstructions
final_fakes = generator(best_z, training=False).numpy()

# 6. Build High-Resolution Comparison Contact Sheet
# Rows: [Real | Inverted G(z*) | Inverted G(z*) Palette-Snapped | Diff Error Map]
# Scale up each 64x64 sprite by 4x (256x256) with Nearest-Neighbor interpolation for pixel clarity

SCALE = 4
SPRITE_SIZE = 64 * SCALE
GRID_WIDTH = N * SPRITE_SIZE
GRID_HEIGHT = 4 * SPRITE_SIZE

contact_sheet = Image.new("RGBA", (GRID_WIDTH, GRID_HEIGHT), (20, 24, 33, 255))

for i in range(N):
    real_np = ((real_tensor[i].numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    fake_np = ((final_fakes[i] * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    
    # Snapped version
    fake_float = (final_fakes[i] * 0.5 + 0.5).clip(0, 1)
    snapped_np = snap_to_palette(fake_float, n_colors=16)
    
    # Diff map
    diff_np = np.abs(real_np.astype(np.int32) - fake_np.astype(np.int32)).clip(0, 255).astype(np.uint8)
    diff_np[:, :, 3] = 255
    
    img_real = Image.fromarray(real_np, mode="RGBA").resize((SPRITE_SIZE, SPRITE_SIZE), Image.NEAREST)
    img_fake = Image.fromarray(fake_np, mode="RGBA").resize((SPRITE_SIZE, SPRITE_SIZE), Image.NEAREST)
    img_snap = Image.fromarray(snapped_np, mode="RGBA").resize((SPRITE_SIZE, SPRITE_SIZE), Image.NEAREST)
    img_diff = Image.fromarray(diff_np, mode="RGBA").resize((SPRITE_SIZE, SPRITE_SIZE), Image.NEAREST)
    
    contact_sheet.paste(img_real, (i * SPRITE_SIZE, 0 * SPRITE_SIZE))
    contact_sheet.paste(img_fake, (i * SPRITE_SIZE, 1 * SPRITE_SIZE))
    contact_sheet.paste(img_snap, (i * SPRITE_SIZE, 2 * SPRITE_SIZE))
    contact_sheet.paste(img_diff, (i * SPRITE_SIZE, 3 * SPRITE_SIZE))

output_img_path = os.path.join(OUTPUT_DIR, "dcgan_latent_inversion_contact_sheet.png")
contact_sheet.save(output_img_path)
print(f"\nSaved Latent Inversion Probe Contact Sheet to: {output_img_path}")

# 7. Compute Detailed Color & Silhouette Statistics
print("\n=== LATENT INVERSION CAPACITY EVALUATION ===")
for i in range(N):
    real_colors = len(np.unique(real_tensor[i].numpy().reshape(-1, 4), axis=0))
    raw_inv_colors = len(np.unique(final_fakes[i].reshape(-1, 4), axis=0))
    psnr = tf.image.psnr(final_fakes[i:i+1], real_tensor[i:i+1], max_val=2.0).numpy()[0]
    ssim = tf.image.ssim(final_fakes[i:i+1, :, :, :3]*0.5+0.5, real_tensor[i:i+1, :, :, :3]*0.5+0.5, max_val=1.0).numpy()[0]
    print(f"Sprite #{i+1} (Sample {SAMPLE_IDS[i]}): PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | Real Unique Colors: {real_colors} | Inverted Unique Colors: {raw_inv_colors}")

print("\n--- Latent Inversion Probe Completed Successfully ---")

