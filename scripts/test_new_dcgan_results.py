import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Ensure output directories exist
os.makedirs("phase_dcgan_improvement", exist_ok=True)
artifact_dir = r"C:\Users\manis\.gemini\antigravity\brain\0e75ca3f-2ae4-4b95-8ef8-c8e5292ad012"

RESULTS_DIR = r"C:\Users\manis\Downloads\dcgan_frontfacing_results"
PALETTE_PATH = os.path.join(RESULTS_DIR, "config", "global_palette_32.npy")
FIXED_Z_PATH = os.path.join(RESULTS_DIR, "config", "fixed_latent_vectors.npy")
G_WEIGHTS_PATH = os.path.join(RESULTS_DIR, "game_ready_assets", "generator_weights_final.weights.h5")

# Load palette
global_palette_u8 = np.load(PALETTE_PATH)  # (32, 3) uint8
palette_tf = tf.constant(global_palette_u8 / 255.0, dtype=tf.float32)
PALETTE_SIZE = len(global_palette_u8)
LATENT_DIM = 128
G_CHANNELS = [512, 256, 128, 64]
G_INIT_SPATIAL = 4

print(f"Loaded Palette: {global_palette_u8.shape}, range [{global_palette_u8.min()}, {global_palette_u8.max()}]")

def logits_to_image(logits, palette_tf, tau=1.0, hard=True, noise=False):
    """
    Converts (B, 64, 64, K+1) logits into discrete RGBA images in [0, 255] uint8.
    """
    pl = logits[..., :PALETTE_SIZE]
    al = logits[..., PALETTE_SIZE:]
    
    if noise:
        u = tf.random.uniform(tf.shape(pl), minval=1e-6, maxval=1.0)
        gumbel_noise = -tf.math.log(-tf.math.log(u))
        pl = pl + gumbel_noise
        
    ys = tf.nn.softmax(pl / tau, axis=-1)
    if hard:
        y_hard = tf.one_hot(tf.argmax(ys, axis=-1), depth=PALETTE_SIZE, dtype=tf.float32)
        y = tf.stop_gradient(y_hard - ys) + ys
    else:
        y = ys
        
    rgb = tf.tensordot(y, palette_tf, axes=[[-1], [0]]) # (B, 64, 64, 3) in [0, 1]
    a_s = tf.sigmoid(al)
    if hard:
        a_hard = tf.cast(a_s > 0.5, dtype=tf.float32)
        a = tf.stop_gradient(a_hard - a_s) + a_s
    else:
        a = a_s
        
    rgba = tf.concat([rgb * a, a], axis=-1) # (B, 64, 64, 4) in [0, 1]
    return tf.cast(tf.clip_by_value(rgba * 255.0 + 0.5, 0.0, 255.0), tf.uint8).numpy()

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
    
    # Discrete K+1 Logits Head (32 color logits + 1 alpha logit)
    logits = layers.Conv2D(PALETTE_SIZE + 1, (3, 3), padding="same", name="palette_logits")(x)
    return keras.Model(z_in, logits, name="Discrete_DCGAN_Generator")

generator = build_generator()
generator.load_weights(G_WEIGHTS_PATH)
print("Loaded generator weights successfully!")

def make_contact_sheet(images, grid_size=(6, 6), pad=4):
    n_rows, n_cols = grid_size
    img_h, img_w, _ = images[0].shape
    sheet_w = n_cols * img_w + (n_cols + 1) * pad
    sheet_h = n_rows * img_h + (n_rows + 1) * pad
    
    # White background for contact sheet
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (255, 255, 255, 255))
    
    for idx, img_arr in enumerate(images[:n_rows * n_cols]):
        r = idx // n_cols
        c = idx % n_cols
        x = pad + c * (img_w + pad)
        y = pad + r * (img_h + pad)
        pil_img = Image.fromarray(img_arr, mode="RGBA")
        sheet.paste(pil_img, (x, y), mask=pil_img.split()[3])
    return sheet

# 1. Generate with Fixed Latents (Reproduce Epoch 50)
fixed_z = np.load(FIXED_Z_PATH)
logits_fixed = generator(fixed_z, training=False)
imgs_fixed = logits_to_image(logits_fixed, palette_tf, hard=True, noise=False)
sheet_fixed = make_contact_sheet(imgs_fixed)
sheet_fixed.save("phase_dcgan_improvement/fixed_latents_epoch50_eval.png")
sheet_fixed.save(os.path.join(artifact_dir, "discrete_fixed_eval.png"))

# 2. Generate with 36 Fresh Random Seeds
np.random.seed(42)
fresh_z = np.random.normal(0.0, 1.0, (36, LATENT_DIM)).astype(np.float32)
logits_fresh = generator(fresh_z, training=False)
imgs_fresh = logits_to_image(logits_fresh, palette_tf, hard=True, noise=False)
sheet_fresh = make_contact_sheet(imgs_fresh)
sheet_fresh.save("phase_dcgan_improvement/fresh_random_seeds_eval.png")
sheet_fresh.save(os.path.join(artifact_dir, "discrete_fresh_eval.png"))

# 3. Save 4 individual zoomed-in crops (4x scale) for deep forensic inspection
for i in [0, 4, 13, 20]:
    crop = Image.fromarray(imgs_fresh[i], mode="RGBA")
    crop_zoomed = crop.resize((256, 256), Image.NEAREST)
    crop_zoomed.save(f"phase_dcgan_improvement/zoomed_sprite_{i}.png")
    crop_zoomed.save(os.path.join(artifact_dir, f"discrete_zoomed_{i}.png"))

# 4. Forensic Measurements
def audit_sprites(imgs):
    unique_colors_list = []
    alpha_vals_list = []
    fg_area_list = []
    
    for img in imgs:
        fg_mask = img[..., 3] > 127
        fg_area = np.mean(fg_mask)
        fg_area_list.append(fg_area)
        
        # Unique RGB colors in foreground
        if np.any(fg_mask):
            fg_rgb = img[..., :3][fg_mask]
            unique_colors = len(np.unique(fg_rgb, axis=0))
            unique_colors_list.append(unique_colors)
            
        alpha_vals = np.unique(img[..., 3])
        alpha_vals_list.extend(alpha_vals)
        
    alpha_uniques = np.unique(alpha_vals_list)
    avg_colors = np.mean(unique_colors_list)
    avg_fg = np.mean(fg_area_list)
    return avg_colors, alpha_uniques, avg_fg

avg_c, alphas, avg_fg = audit_sprites(imgs_fresh)
print(f"--- Forensic Visual Stats ---")
print(f"Average Unique Colors per Sprite: {avg_c:.1f} (Ground truth is ~18)")
print(f"Alpha Values Present in Output: {alphas.tolist()} (Must be only [0, 255])")
print(f"Average Foreground Area: {avg_fg * 100:.1f}%")
print("Visual evaluation complete and artifacts exported!")

