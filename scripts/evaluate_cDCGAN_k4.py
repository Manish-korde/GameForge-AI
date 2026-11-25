import os
import sys
import json
import time
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Backend imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import dcgan_service
from scipy.ndimage import label, binary_dilation

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ARTIFACT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "4_cDCGAN_k4_v1"))
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))

LATENT_DIM = 128
NUM_CLASSES = 4
PALETTE_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 4
G_CHANNELS = [512, 256, 128, 64]
G_INIT_SPATIAL = 4

print("=" * 80)
print("EVALUATION & BENCHMARK SUITE: cDCGAN (k=4) vs UNCONDITIONED DCGAN")
print("=" * 80)

# Load global palette from existing models/3_DCGAN_42k_v1
palette_path = os.path.join(os.path.dirname(__file__), "..", "models", "3_DCGAN_42k_v1", "config", "global_palette_32.npy")
global_palette_u8 = np.load(palette_path)
global_palette_tf = tf.constant(global_palette_u8 / 255.0, dtype=tf.float32)

# Build Conditional Generator
def build_cgenerator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
    z_in = layers.Input(shape=(latent_dim,), name="latent_vector")
    y_in = layers.Input(shape=(), dtype="int32", name="class_label")
    dense_proj = layers.Dense(G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0], use_bias=False, name="dense")(z_in)
    class_emb = layers.Embedding(num_classes, G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0],
                                 embeddings_initializer="zeros", name="class_embedding")(y_in)
    x = layers.Add(name="add_class")([dense_proj, class_emb])
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu")(x)
    x = layers.Reshape((G_INIT_SPATIAL, G_INIT_SPATIAL, G_CHANNELS[0]), name="reshape")(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_sampling2d")(x)
    x = layers.Conv2D(G_CHANNELS[1], (3, 3), padding="same", use_bias=False, name="conv2d")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_1")(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_sampling2d_1")(x)
    x = layers.Conv2D(G_CHANNELS[2], (3, 3), padding="same", use_bias=False, name="conv2d_1")(x)
    x = layers.BatchNormalization(name="batch_normalization_2")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_2")(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_sampling2d_2")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2d_2")(x)
    x = layers.BatchNormalization(name="batch_normalization_3")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_3")(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_sampling2d_3")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2d_3")(x)
    x = layers.BatchNormalization(name="batch_normalization_4")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_4")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2d_4")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_5")(x)
    logits = layers.Conv2D(PALETTE_SIZE + 1, (3, 3), padding="same", name="palette_logits")(x)
    return keras.Model([z_in, y_in], logits, name="Conditional_Discrete_DCGAN_Generator")

cgen = build_cgenerator()
cgen_weights_path = os.path.join(MODEL_DIR, "export", "cgenerator_ema_weights_final.weights.h5")
print(f"Loading cDCGAN weights from: {cgen_weights_path}")
cgen.load_weights(cgen_weights_path)
print("cDCGAN Generator loaded successfully!")

# Unconditioned model
ungen = dcgan_service.load_dcgan_generator()

def c_logits_to_image(logits, palette_tf, tau=0.8, hard=True):
    pl = logits[..., :PALETTE_SIZE]
    al = logits[..., PALETTE_SIZE:]
    ys = tf.nn.softmax(pl / tau, axis=-1)
    if hard:
        y_hard = tf.one_hot(tf.argmax(ys, axis=-1), depth=PALETTE_SIZE, dtype=tf.float32)
        y = tf.stop_gradient(y_hard - ys) + ys
    else:
        y = ys
    rgb = tf.tensordot(y, palette_tf, axes=[[-1], [0]])
    a_s = tf.sigmoid(al)
    a_hard = tf.cast(a_s > 0.5, dtype=tf.float32)
    a = tf.stop_gradient(a_hard - a_s) + a_s
    img_01 = tf.concat([rgb * a, a], axis=-1)
    img_u8 = tf.cast(tf.clip_by_value(img_01 * 255.0, 0.0, 255.0), tf.uint8)
    return img_u8.numpy()

# Metric Helpers
def clean_and_measure_v2(rgba_img, touch_dilation=2):
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return rgba_img, 0, 0
    labeled_array, num_features = label(alpha)
    if num_features <= 1:
        return rgba_img, 0, 0
    sizes = np.bincount(labeled_array.ravel())
    largest_label = np.argmax(sizes[1:]) + 1
    body_mask = (labeled_array == largest_label)
    dilated_body = binary_dilation(body_mask, iterations=touch_dilation)
    keep_mask = body_mask.copy()
    floater_count = 0
    stray_pixels = 0
    for i in range(1, num_features + 1):
        if i == largest_label:
            continue
        comp_mask = (labeled_array == i)
        if np.any(comp_mask & dilated_body):
            keep_mask |= comp_mask
        else:
            floater_count += 1
            stray_pixels += int(np.sum(comp_mask))
    cleaned = rgba_img.copy()
    cleaned[~keep_mask] = 0
    return cleaned, floater_count, stray_pixels

def compute_spatial_coherence(imgs_u8):
    coherences = []
    for im in imgs_u8:
        fg = im[..., 3] > 127
        if not np.any(fg):
            coherences.append(0.0)
            continue
        same_h = (im[:, 1:, :3] == im[:, :-1, :3]).all(-1)
        valid_adj = fg[:, 1:] & fg[:, :-1]
        if np.any(valid_adj):
            coherences.append(float(same_h[valid_adj].mean()))
        else:
            coherences.append(0.0)
    return float(np.mean(coherences))

def compute_unique_colors(img_u8):
    fg = img_u8[..., 3] > 127
    if not np.any(fg):
        return 0
    return len(np.unique(img_u8[..., :3][fg], axis=0))

def compute_palette_histogram(img_u8):
    fg = img_u8[..., 3] > 127
    if not np.any(fg):
        return np.zeros(32, dtype=np.float32)
    rgb = img_u8[..., :3][fg]
    diff = rgb[:, None, :] - global_palette_u8[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    nearest = np.argmin(dist_sq, axis=-1)
    hist = np.bincount(nearest, minlength=32).astype(np.float32)
    if np.sum(hist) > 0:
        hist /= np.sum(hist)
    return hist

# ── Part 1: Conditioned Grid (16 Samples per Cluster = 64 Sprites Total) ─────
print("\nGenerating conditioned grid (16 samples per cluster, 64 sprites total)...")
group_titles = [
    "Palette group 0 (Knight / Steel)",
    "Palette group 1 (Orc / Green)",
    "Palette group 2 (Ranger / Leather)",
    "Palette group 3 (Mage / Crimson)"
]

rng_64 = np.random.RandomState(1337)
N_PER_CLUSTER = 16
z_64_base = rng_64.normal(0, 1.0, size=(N_PER_CLUSTER, LATENT_DIM)).astype(np.float32)
z_64_base = np.clip(z_64_base, -2.0, 2.0)

cluster_sprites = {}
for c in range(NUM_CLASSES):
    z_tf = tf.constant(z_64_base)
    y_tf = tf.constant(np.full((N_PER_CLUSTER,), c, dtype=np.int32))
    logits = cgen([z_tf, y_tf], training=False)
    imgs = c_logits_to_image(logits, global_palette_tf, tau=0.8, hard=True)
    cluster_sprites[c] = imgs

# 4 rows x 16 columns composite
SPR_SZ = 64
SCALE = 2
TILE_SZ = SPR_SZ * SCALE

grid_w = N_PER_CLUSTER * TILE_SZ + 20
grid_h = NUM_CLASSES * (TILE_SZ + 30) + 40
canvas_64 = Image.new("RGBA", (grid_w, grid_h), (15, 23, 42, 255))
draw = ImageDraw.Draw(canvas_64)

for c in range(NUM_CLASSES):
    row_y = 30 + c * (TILE_SZ + 30)
    draw.text((15, row_y - 20), group_titles[c], fill=(226, 232, 240, 255))
    for i in range(N_PER_CLUSTER):
        spr = Image.fromarray(cluster_sprites[c][i]).resize((TILE_SZ, TILE_SZ), Image.NEAREST)
        col_x = 10 + i * TILE_SZ
        canvas_64.paste(spr, (col_x, row_y), spr)

out_64_path = os.path.join(ARTIFACT_DIR, "cdcgan_k4_64_grid.png")
canvas_64.save(out_64_path)
print(f"Saved 64-sprite conditioned grid to: {out_64_path}")

# ── Part 2: Measure 1,000 Raw cDCGAN Samples (250 per cluster) ───────────────
print("\n" + "=" * 80)
print("BENCHMARKING cDCGAN RAW ON 1,000 SAMPLES (250 PER CLUSTER)")
print("=" * 80)

N_BENCH = 1000
rng_1000 = np.random.RandomState(42)
BATCH = 50

cdc_facial_splits = 0
cdc_head_torso_clashes = 0
cdc_floaters = []
cdc_strays = []
cdc_colors = []
cdc_coherences = []
cluster_hists = {c: [] for c in range(NUM_CLASSES)}

t0 = time.time()
for b in range(N_BENCH // BATCH):
    c_target = (b * BATCH // 250) % NUM_CLASSES
    z_b = rng_1000.normal(0, 1.0, size=(BATCH, LATENT_DIM)).astype(np.float32)
    z_b = np.clip(z_b, -2.0, 2.0)
    y_b = np.full((BATCH,), c_target, dtype=np.int32)
    
    l_b = cgen([tf.constant(z_b), tf.constant(y_b)], training=False)
    imgs_b = c_logits_to_image(l_b, global_palette_tf, tau=0.8, hard=True)
    
    for im in imgs_b:
        ok_f, _ = dcgan_service.check_facial_hue_consistency(im, max_divergence=45.0)
        if not ok_f:
            cdc_facial_splits += 1
            
        ok_ht, _ = dcgan_service.check_head_torso_hue_agreement(im, max_clash=100.0)
        if not ok_ht:
            cdc_head_torso_clashes += 1
            
        _, fl, st = clean_and_measure_v2(im)
        cdc_floaters.append(fl)
        cdc_strays.append(st)
        cdc_colors.append(compute_unique_colors(im))
        cluster_hists[c_target].append(compute_palette_histogram(im))
        
    cdc_coherences.append(compute_spatial_coherence(imgs_b))

print(f"cDCGAN 1,000 raw samples evaluated in {time.time() - t0:.2f}s!")

cdcgan_raw_metrics = {
    "facial_split_rate_pct": (cdc_facial_splits / N_BENCH) * 100.0,
    "head_torso_clash_pct": (cdc_head_torso_clashes / N_BENCH) * 100.0,
    "mean_floater_count": float(np.mean(cdc_floaters)),
    "mean_stray_pixels": float(np.mean(cdc_strays)),
    "mean_unique_colors": float(np.mean(cdc_colors)),
    "spatial_coherence_pct": float(np.mean(cdc_coherences)) * 100.0,
}

# ── Part 3: Measure 1,000 Raw Unconditioned Samples ──────────────────────────
print("\nBenchmarking Unconditioned DCGAN Raw on 1,000 samples...")
un_facial_splits = 0
un_head_torso_clashes = 0
un_floaters = []
un_strays = []
un_colors = []
un_coherences = []

for b in range(N_BENCH // BATCH):
    z_b = rng_1000.normal(0, 1.0, size=(BATCH, LATENT_DIM)).astype(np.float32)
    z_b = np.clip(z_b, -2.0, 2.0)
    l_b = ungen(z_b, training=False)
    imgs_b = dcgan_service.logits_to_image(l_b, dcgan_service._palette_tf, hard=True, noise=False)
    
    for im in imgs_b:
        ok_f, _ = dcgan_service.check_facial_hue_consistency(im, max_divergence=45.0)
        if not ok_f:
            un_facial_splits += 1
        ok_ht, _ = dcgan_service.check_head_torso_hue_agreement(im, max_clash=100.0)
        if not ok_ht:
            un_head_torso_clashes += 1
        _, fl, st = clean_and_measure_v2(im)
        un_floaters.append(fl)
        un_strays.append(st)
        un_colors.append(compute_unique_colors(im))
        
    un_coherences.append(compute_spatial_coherence(imgs_b))

unconditioned_raw_metrics = {
    "facial_split_rate_pct": (un_facial_splits / N_BENCH) * 100.0,
    "head_torso_clash_pct": (un_head_torso_clashes / N_BENCH) * 100.0,
    "mean_floater_count": float(np.mean(un_floaters)),
    "mean_stray_pixels": float(np.mean(un_strays)),
    "mean_unique_colors": float(np.mean(un_colors)),
    "spatial_coherence_pct": float(np.mean(un_coherences)) * 100.0,
}

# ── Part 4: Gate 5 - Cross-Cluster Palette Cosine Similarity ─────────────────
print("\n" + "=" * 80)
print("GATE 5: PALETTE COSINE SIMILARITY")
print("=" * 80)

def cosine_similarity(u, v):
    dot = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    return float(dot / max(norm, 1e-8))

cluster_centroids = {c: np.mean(cluster_hists[c], axis=0) for c in range(NUM_CLASSES)}

pairwise_cosines = {}
off_diagonal_cosines = []
for i in range(NUM_CLASSES):
    for j in range(i + 1, NUM_CLASSES):
        cos_sim = cosine_similarity(cluster_centroids[i], cluster_centroids[j])
        pairwise_cosines[f"cluster_{i}_vs_{j}"] = round(cos_sim, 4)
        off_diagonal_cosines.append(cos_sim)

mean_cross_cluster_cosine = float(np.mean(off_diagonal_cosines))
max_cross_cluster_cosine = float(np.max(off_diagonal_cosines))

print("Pairwise Cosine Similarity Between Generated Clusters:")
for pair, val in pairwise_cosines.items():
    print(f"  {pair:18s} : {val:.4f}")
print(f"  Mean Cross-Cluster Cosine : {mean_cross_cluster_cosine:.4f}")
print(f"  Max Cross-Cluster Cosine  : {max_cross_cluster_cosine:.4f} (Requirement: <= 0.35)")

# ── Part 5: Evaluate 5 Gates ──────────────────────────────────────────────────
gates = [
    ("Gate 1: Facial Split Rate", cdcgan_raw_metrics["facial_split_rate_pct"], "<= 3.5%", cdcgan_raw_metrics["facial_split_rate_pct"] <= 3.5),
    ("Gate 2: Stray Pixels", cdcgan_raw_metrics["mean_stray_pixels"], "<= 1.5", cdcgan_raw_metrics["mean_stray_pixels"] <= 1.5),
    ("Gate 3: Floater Count", cdcgan_raw_metrics["mean_floater_count"], "<= 0.8", cdcgan_raw_metrics["mean_floater_count"] <= 0.8),
    ("Gate 4: Colours per Sprite", cdcgan_raw_metrics["mean_unique_colors"], "<= 18.0", cdcgan_raw_metrics["mean_unique_colors"] <= 18.0),
    ("Gate 5: Cross-Cluster Cosine", mean_cross_cluster_cosine, "<= 0.35", mean_cross_cluster_cosine <= 0.35),
]

print("\n" + "=" * 80)
print("FIVE GATES VERIFICATION SUMMARY")
print("=" * 80)
for name, val, thresh, passed in gates:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name:30s} : Measured = {val:.3f} (Requirement: {thresh})")

all_gates_passed = all(g[3] for g in gates)
print(f"\nAll Five Gates Passed: {all_gates_passed}")

# ── Part 6: Side-by-Side Comparison Matrix (4 columns x 4x4 tiles) ───────────
print("\nGenerating 4-way Side-by-Side Comparison Matrix...")
cmp_rng = np.random.RandomState(2026)
cmp_z = cmp_rng.normal(0, 1.0, size=(16, LATENT_DIM)).astype(np.float32)
cmp_z = np.clip(cmp_z, -2.0, 2.0)

# Unconditioned Raw
un_raw_l = ungen(cmp_z, training=False)
un_raw_imgs = dcgan_service.logits_to_image(un_raw_l, dcgan_service._palette_tf, hard=True, noise=False)

# Unconditioned Filtered (Filter v3 applied to 16 candidates)
un_filt_imgs = []
for i in range(16):
    s = 1000 + i
    r_cand = np.random.RandomState(s).normal(0, 1.0, size=(64, LATENT_DIM)).astype(np.float32)
    r_cand = np.clip(r_cand, -2.0, 2.0)
    c_l = ungen(r_cand, training=False)
    c_imgs = dcgan_service.logits_to_image(c_l, dcgan_service._palette_tf, hard=True, noise=False)
    scored = []
    for im in c_imgs:
        cl, fl, st = clean_and_measure_v2(im)
        ok_f, _ = dcgan_service.check_facial_hue_consistency(cl)
        ok_ht, _ = dcgan_service.check_head_torso_hue_agreement(cl)
        if ok_f and ok_ht:
            scored.append((dcgan_service.score_candidate(cl, fl), cl))
    if scored:
        scored.sort(key=lambda x: x[0])
        un_filt_imgs.append(scored[0][1])
    else:
        un_filt_imgs.append(clean_and_measure_v2(c_imgs[0])[0])

# cDCGAN Raw (4 from each cluster)
cdc_raw_list = []
for i in range(16):
    c_idx = i % 4
    z_s = cmp_z[i:i+1]
    y_s = tf.constant([c_idx], dtype=tf.int32)
    l_s = cgen([tf.constant(z_s), y_s], training=False)
    im_s = c_logits_to_image(l_s, global_palette_tf, tau=0.8, hard=True)[0]
    cdc_raw_list.append(im_s)
cdc_raw_imgs = np.array(cdc_raw_list)

# cDCGAN Filtered
cdc_filt_imgs = []
for i in range(16):
    c_idx = i % 4
    s = 2000 + i
    r_cand = np.random.RandomState(s).normal(0, 1.0, size=(64, LATENT_DIM)).astype(np.float32)
    r_cand = np.clip(r_cand, -2.0, 2.0)
    y_cand = np.full((64,), c_idx, dtype=np.int32)
    c_l = cgen([tf.constant(r_cand), tf.constant(y_cand)], training=False)
    c_imgs = c_logits_to_image(c_l, global_palette_tf, tau=0.8, hard=True)
    scored = []
    for im in c_imgs:
        cl, fl, st = clean_and_measure_v2(im)
        ok_f, _ = dcgan_service.check_facial_hue_consistency(cl)
        ok_ht, _ = dcgan_service.check_head_torso_hue_agreement(cl)
        if ok_f and ok_ht:
            scored.append((dcgan_service.score_candidate(cl, fl), cl))
    if scored:
        scored.sort(key=lambda x: x[0])
        cdc_filt_imgs.append(scored[0][1])
    else:
        cdc_filt_imgs.append(clean_and_measure_v2(c_imgs[0])[0])

MATRIX_TILE = 64 * 2
col_w = 4 * MATRIX_TILE
mat_w = 4 * col_w + 3 * 30 + 30
mat_h = 4 * MATRIX_TILE + 70

canvas_mat = Image.new("RGBA", (mat_w, mat_h), (15, 23, 42, 255))
mat_draw = ImageDraw.Draw(canvas_mat)

headers = [
    "1. Unconditioned Raw (50 Ep)",
    "2. Unconditioned + Filter v3",
    "3. cDCGAN Raw (18 Ep WarmStart)",
    "4. cDCGAN + Filter v3"
]

img_groups = [un_raw_imgs, un_filt_imgs, cdc_raw_imgs, cdc_filt_imgs]

for set_idx, imgs in enumerate(img_groups):
    col_x_start = 15 + set_idx * (col_w + 30)
    mat_draw.text((col_x_start + 10, 15), headers[set_idx], fill=(241, 245, 249, 255))
    
    for idx, im in enumerate(imgs):
        r = idx // 4
        c = idx % 4
        x = col_x_start + c * MATRIX_TILE
        y = 50 + r * MATRIX_TILE
        tile = Image.fromarray(im).resize((MATRIX_TILE, MATRIX_TILE), Image.NEAREST)
        canvas_mat.paste(tile, (x, y), tile)

out_mat_path = os.path.join(ARTIFACT_DIR, "cdcgan_comparison_matrix.png")
canvas_mat.save(out_mat_path)
print(f"Saved 4-way comparison matrix to: {out_mat_path}")

report_data = {
    "all_five_gates_passed": bool(all_gates_passed),
    "gates": [
        {"name": g[0], "measured": float(g[1]), "requirement": g[2], "passed": bool(g[3])} for g in gates
    ],
    "unconditioned_raw_1000": unconditioned_raw_metrics,
    "cdcgan_raw_1000": cdcgan_raw_metrics,
    "pairwise_cluster_cosines": pairwise_cosines,
    "mean_cross_cluster_cosine": mean_cross_cluster_cosine,
    "max_cross_cluster_cosine": max_cross_cluster_cosine,
    "output_grid_artifact": out_64_path,
    "comparison_matrix_artifact": out_mat_path
}

out_json_path = os.path.join(ARTIFACT_DIR, "cdcgan_evaluation_report.json")
with open(out_json_path, "w") as f:
    json.dump(report_data, f, indent=2)

print(f"Saved JSON report to: {out_json_path}")
print("Evaluation Complete!")
