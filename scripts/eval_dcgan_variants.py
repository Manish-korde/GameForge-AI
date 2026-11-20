import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label

# Paths
WORKSPACE = r"c:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator"
MODEL_DIR = os.path.join(WORKSPACE, "models", "dcgan_discrete_50ep_results")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "game_ready_assets", "generator_weights_final.weights.h5")
PALETTE_PATH = os.path.join(MODEL_DIR, "config", "global_palette_32.npy")
FIXED_Z_PATH = os.path.join(MODEL_DIR, "config", "fixed_latent_vectors.npy")
OUTPUT_DIR = os.path.join(WORKSPACE, "phase_dcgan_improvement")
ARTIFACT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\0e75ca3f-2ae4-4b95-8ef8-c8e5292ad012"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load Palette
global_palette_u8 = np.load(PALETTE_PATH)  # (32, 3)
palette_tf = tf.constant(global_palette_u8 / 255.0, dtype=tf.float32)
PALETTE_SIZE = 32
LATENT_DIM = 128
G_CHANNELS = [512, 256, 128, 64]
G_INIT_SPATIAL = 4

print("Loaded Palette:", global_palette_u8.shape)

def build_discrete_generator():
    z_in = layers.Input(shape=(LATENT_DIM,), name="latent_vector")
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
    
    # Refinement
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Output Head: 32 color logits + 1 alpha logit = 33 channels
    logits = layers.Conv2D(PALETTE_SIZE + 1, (3, 3), padding="same", name="palette_logits")(x)
    return keras.Model(z_in, logits, name="Discrete_DCGAN_Generator")

def logits_to_image(logits, palette_tf, tau=1.0, hard=True, noise=False):
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
        
    rgb = tf.tensordot(y, palette_tf, axes=[[-1], [0]]) # [0, 1]
    a_s = tf.sigmoid(al)
    if hard:
        a_hard = tf.cast(a_s > 0.5, dtype=tf.float32)
        a = tf.stop_gradient(a_hard - a_s) + a_s
    else:
        a = a_s
        
    rgba = tf.concat([rgb * a, a], axis=-1)
    return tf.cast(tf.clip_by_value(rgba * 255.0 + 0.5, 0.0, 255.0), tf.uint8).numpy()

def remove_stray_floaters_weapon_safe(rgba_img, min_island_size=6, max_distance_to_body=14):
    """
    Cleans isolated GAN specks/floaters while preserving thin weapon parts (spears, swords, staves, bows).
    """
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return rgba_img, 0
        
    labeled_array, num_features = label(alpha)
    if num_features <= 1:
        return rgba_img, 0
        
    sizes = np.bincount(labeled_array.ravel())
    largest_label = np.argmax(sizes[1:]) + 1
    
    body_mask = (labeled_array == largest_label)
    body_y, body_x = np.where(body_mask)
    
    keep_mask = body_mask.copy()
    
    for i in range(1, num_features + 1):
        if i == largest_label:
            continue
        sz = sizes[i]
        # Keep if component size >= min_island_size (weapon, shield, hat)
        if sz >= min_island_size:
            keep_mask |= (labeled_array == i)
        elif sz >= 3:
            # Check proximity to body: if close enough, it's part of weapon tip or hand
            comp_y, comp_x = np.where(labeled_array == i)
            dy = np.min(np.abs(comp_y[:, None] - body_y[None, :]))
            dx = np.min(np.abs(comp_x[:, None] - body_x[None, :]))
            if (dy + dx) <= max_distance_to_body:
                keep_mask |= (labeled_array == i)
                
    removed_pixels = int(np.sum(alpha) - np.sum(keep_mask))
    cleaned = rgba_img.copy()
    cleaned[~keep_mask] = 0
    return cleaned, removed_pixels

# Build & Load Weights
generator = build_discrete_generator()
generator.load_weights(WEIGHTS_PATH)
print("Generator weights loaded successfully from:", WEIGHTS_PATH)

# Load Fixed Latent Vectors (exact 36 vectors from Epoch 50)
fixed_z = np.load(FIXED_Z_PATH)
print("Loaded fixed latent vectors:", fixed_z.shape)

# VARIANT 1: Baseline Epoch 50 Raw (sigma = 1.0, noise=False, no filter)
logits_var1 = generator(fixed_z, training=False)
imgs_var1_raw = logits_to_image(logits_var1, palette_tf, hard=True, noise=False)

# VARIANT 2: Truncated sigma = 0.7 (scaled fixed_z * 0.7, clipped)
z_trunc_07 = np.clip(fixed_z * 0.7, -1.8, 1.8).astype(np.float32)
logits_var2 = generator(z_trunc_07, training=False)
imgs_var2_trunc07 = logits_to_image(logits_var2, palette_tf, hard=True, noise=False)

# VARIANT 3: Truncated sigma = 0.85 (scaled fixed_z * 0.85, clipped)
z_trunc_085 = np.clip(fixed_z * 0.85, -2.0, 2.0).astype(np.float32)
logits_var3 = generator(z_trunc_085, training=False)
imgs_var3_trunc085 = logits_to_image(logits_var3, palette_tf, hard=True, noise=False)

# VARIANT 4: Truncated sigma = 0.7 + Weapon-Safe LCC Floater Filter
imgs_var4_trunc07_lcc = []
var4_removed = []
for img in imgs_var2_trunc07:
    cleaned, rem = remove_stray_floaters_weapon_safe(img, min_island_size=6)
    imgs_var4_trunc07_lcc.append(cleaned)
    var4_removed.append(rem)
imgs_var4_trunc07_lcc = np.array(imgs_var4_trunc07_lcc)

# VARIANT 5: Truncated sigma = 0.85 + Weapon-Safe LCC Floater Filter
imgs_var5_trunc085_lcc = []
var5_removed = []
for img in imgs_var3_trunc085:
    cleaned, rem = remove_stray_floaters_weapon_safe(img, min_island_size=6)
    imgs_var5_trunc085_lcc.append(cleaned)
    var5_removed.append(rem)
imgs_var5_trunc085_lcc = np.array(imgs_var5_trunc085_lcc)

# VARIANT 6: Best-of-8 Rejection Sampling around each fixed seed
# For each of the 36 seeds, explore 8 jittered candidates: z_k = 0.80 * z_seed + 0.20 * N(0, 0.8)
# Enforce: 0.15 <= fg <= 0.55. Score = orphans * 2.0 + |fg - 0.32| * 100 + |colors - 18| * 0.8
imgs_var6_rejection = []
rng = np.random.RandomState(42)

for i in range(36):
    z_base = fixed_z[i]
    candidates_z = [z_base * 0.8]  # include the direct truncated anchor
    for k in range(7):
        perturb = rng.normal(0, 0.8, size=(LATENT_DIM,)).astype(np.float32)
        cand_z = 0.75 * (z_base * 0.8) + 0.25 * perturb
        cand_z = np.clip(cand_z, -1.8, 1.8)
        candidates_z.append(cand_z)
    candidates_z = np.array(candidates_z, dtype=np.float32)
    
    cand_logits = generator(candidates_z, training=False)
    cand_imgs = logits_to_image(cand_logits, palette_tf, hard=True, noise=False)
    
    best_idx = 0
    best_score = float("inf")
    best_cleaned = cand_imgs[0]
    
    for idx, c_img in enumerate(cand_imgs):
        fg_mask = c_img[..., 3] > 127
        fg_frac = np.mean(fg_mask)
        
        # Hard rejection gate: blank blobs (< 15%) or shrouded square boxes (> 55%)
        if fg_frac < 0.15 or fg_frac > 0.55:
            continue
            
        cleaned_cand, orphans = remove_stray_floaters_weapon_safe(c_img, min_island_size=6)
        
        # Color diversity
        fg_rgb = c_img[..., :3][fg_mask]
        n_colors = len(np.unique(fg_rgb, axis=0)) if np.any(fg_mask) else 0
        
        # Composite score: lower is better
        score = (orphans * 2.0) + (abs(fg_frac - 0.32) * 100.0) + (abs(n_colors - 18) * 0.8)
        
        if score < best_score:
            best_score = score
            best_idx = idx
            best_cleaned = cleaned_cand
            
    imgs_var6_rejection.append(best_cleaned)

imgs_var6_rejection = np.array(imgs_var6_rejection)

# Compute Metrics Function
def compute_stats(img_batch, name=""):
    colors = []
    fg_ratios = []
    for img in img_batch:
        fg = img[..., 3] > 127
        fg_ratios.append(np.mean(fg))
        if np.any(fg):
            colors.append(len(np.unique(img[..., :3][fg], axis=0)))
    avg_c = np.mean(colors)
    avg_fg = np.mean(fg_ratios) * 100
    print(f"{name:35s} | Colors: {avg_c:4.1f} | FG Area: {avg_fg:4.1f}%")
    return avg_c, avg_fg

print("\n--- STATISTICAL COMPARISON ACROSS VARIANTS (36 SAME SEEDS) ---")
compute_stats(imgs_var1_raw, "Variant 1: Raw Epoch 50 (sigma=1.0)")
compute_stats(imgs_var2_trunc07, "Variant 2: Truncated sigma=0.7")
compute_stats(imgs_var3_trunc085, "Variant 3: Truncated sigma=0.85")
compute_stats(imgs_var4_trunc07_lcc, "Variant 4: Trunc 0.7 + Weapon-Safe LCC")
compute_stats(imgs_var5_trunc085_lcc, "Variant 5: Trunc 0.85 + Weapon-Safe LCC")
compute_stats(imgs_var6_rejection, "Variant 6: Best-of-8 Rejection + LCC")

# Function to create grid image with labels
def create_grid(imgs, grid_size=(6, 6), pad=4, bg_color=(255, 255, 255, 255)):
    rows, cols = grid_size
    h, w, _ = imgs[0].shape
    total_w = cols * w + (cols + 1) * pad
    total_h = rows * h + (rows + 1) * pad
    canvas = Image.new("RGBA", (total_w, total_h), bg_color)
    
    for idx, img in enumerate(imgs[:rows * cols]):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        pil_im = Image.fromarray(img, mode="RGBA")
        canvas.paste(pil_im, (x, y), mask=pil_im.split()[3])
    return canvas

# Save 6 individual 6x6 variant grids
grid1 = create_grid(imgs_var1_raw)
grid2 = create_grid(imgs_var2_trunc07)
grid3 = create_grid(imgs_var3_trunc085)
grid4 = create_grid(imgs_var4_trunc07_lcc)
grid5 = create_grid(imgs_var5_trunc085_lcc)
grid6 = create_grid(imgs_var6_rejection)

grid1.save(os.path.join(OUTPUT_DIR, "var1_raw_epoch50_36grid.png"))
grid2.save(os.path.join(OUTPUT_DIR, "var2_trunc07_36grid.png"))
grid3.save(os.path.join(OUTPUT_DIR, "var3_trunc085_36grid.png"))
grid4.save(os.path.join(OUTPUT_DIR, "var4_trunc07_lcc_36grid.png"))
grid5.save(os.path.join(OUTPUT_DIR, "var5_trunc085_lcc_36grid.png"))
grid6.save(os.path.join(OUTPUT_DIR, "var6_rejection_lcc_36grid.png"))

# Create Master Multi-Variant Controlled Contact Sheet (Side-by-Side 6 Panels)
# Canvas: 3 columns x 2 rows of 6x6 grids, with prominent headers
panel_w, panel_h = grid1.size
margin_top = 80
margin_side = 40
spacing_x = 40
spacing_y = 60

canvas_w = margin_side * 2 + 3 * panel_w + 2 * spacing_x
canvas_h = margin_top + 2 * (panel_h + 40) + spacing_y

master_sheet = Image.new("RGB", (canvas_w, canvas_h), (245, 247, 250))
draw = ImageDraw.Draw(master_sheet)

panels = [
    ("1. Epoch 50 Raw (sigma=1.0, No Filter)", grid1, "Baseline from master board"),
    ("2. Truncated (sigma=0.70)", grid2, "Higher coherence, reduced tail specks"),
    ("3. Truncated (sigma=0.85)", grid3, "Balanced diversity & sharper silhouette"),
    ("4. Trunc 0.70 + Weapon-Safe LCC", grid4, "Floaters removed, weapons preserved"),
    ("5. Trunc 0.85 + Weapon-Safe LCC", grid5, "Optimal diversity + zero floaters"),
    ("6. Best-of-8 Rejection + LCC", grid6, "Production GUI Candidate (Quality Gated)")
]

for idx, (title, grid_im, subtitle) in enumerate(panels):
    col = idx % 3
    row = idx // 3
    px = margin_side + col * (panel_w + spacing_x)
    py = margin_top + row * (panel_h + 40 + spacing_y)
    
    # White card background with subtle outline
    draw.rectangle([px - 4, py - 36, px + panel_w + 4, py + panel_h + 4], fill=(255, 255, 255), outline=(210, 215, 225), width=2)
    # Header bar
    draw.rectangle([px - 4, py - 36, px + panel_w + 4, py - 4], fill=(235, 240, 248))
    draw.text((px + 8, py - 30), title, fill=(20, 30, 50))
    draw.text((px + 8, py - 14), subtitle, fill=(100, 110, 125))
    
    master_sheet.paste(grid_im, (px, py))

# Top Title Header
draw.rectangle([0, 0, canvas_w, 65], fill=(24, 32, 47))
draw.text((margin_side, 14), "DISCRETE DCGAN (EPOCH 50) - CONTROLLED 36-SEED VARIANT EVALUATION", fill=(255, 255, 255))
draw.text((margin_side, 38), "All panels use the EXACT same 36 latent seeds. Weapon-safe LCC preserves spears/swords (>=6px / proximity).", fill=(180, 195, 215))

master_sheet.save(os.path.join(OUTPUT_DIR, "controlled_variant_comparison_master.png"))
master_sheet.save(os.path.join(ARTIFACT_DIR, "controlled_variant_comparison_master.png"))

# Create 8x Nearest-Neighbor Zoomed Comparisons for 4 key sprites (Archers, Knights, Weapons)
# Sprites #0, #4 (Archer), #13, #20
zoom_sprites = [0, 4, 13, 20]
zoom_canvas_w = 40 + 4 * (64 * 8 + 30)
zoom_canvas_h = 100 + 4 * (64 * 8 + 60)
zoom_sheet = Image.new("RGB", (zoom_canvas_w, zoom_canvas_h), (250, 252, 255))
zoom_draw = ImageDraw.Draw(zoom_sheet)

zoom_draw.rectangle([0, 0, zoom_canvas_w, 60], fill=(24, 32, 47))
zoom_draw.text((40, 15), "ZOOMED FORENSIC COMPARISON (8x NEAREST-NEIGHBOR): WEAPON & SILHOUETTE FIDELITY", fill=(255, 255, 255))
zoom_draw.text((40, 38), "Inspecting Archer (#4), Knight (#0), Warrior (#13), Mage (#20) across Raw vs Trunc 0.85 vs LCC vs Rejection", fill=(180, 195, 215))

zoom_cols = [
    ("Raw (sigma=1.0)", imgs_var1_raw),
    ("Trunc 0.85 Raw", imgs_var3_trunc085),
    ("Trunc 0.85 + LCC", imgs_var5_trunc085_lcc),
    ("Best-of-8 Rejection", imgs_var6_rejection)
]

for r_idx, s_id in enumerate(zoom_sprites):
    y_pos = 90 + r_idx * (64 * 8 + 60)
    for c_idx, (col_name, batch) in enumerate(zoom_cols):
        x_pos = 40 + c_idx * (64 * 8 + 30)
        
        # Draw border & title
        zoom_draw.rectangle([x_pos - 4, y_pos - 24, x_pos + 512 + 4, y_pos + 512 + 4], fill=(255, 255, 255), outline=(210, 220, 230), width=2)
        zoom_draw.text((x_pos + 8, y_pos - 20), f"Seed #{s_id} | {col_name}", fill=(30, 40, 60))
        
        # 8x upscale with nearest neighbor
        sprite_im = Image.fromarray(batch[s_id], mode="RGBA")
        sprite_up = sprite_im.resize((512, 512), Image.NEAREST)
        
        # Composite on light checkerboard or white
        white_bg = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
        white_bg.paste(sprite_up, (0, 0), mask=sprite_up.split()[3])
        zoom_sheet.paste(white_bg.convert("RGB"), (x_pos, y_pos))

zoom_sheet.save(os.path.join(OUTPUT_DIR, "zoomed_8x_forensic_comparison.png"))
zoom_sheet.save(os.path.join(ARTIFACT_DIR, "zoomed_8x_forensic_comparison.png"))

print("Evaluation sheets successfully generated and exported!")

