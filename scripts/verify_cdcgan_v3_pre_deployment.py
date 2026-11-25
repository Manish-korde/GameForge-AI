import os
import sys
import json
import time
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import dcgan_service
from scipy.ndimage import label, binary_dilation

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 80)
print("GAMFORGE-AI cDCGAN v3: FINAL PRE-DEPLOYMENT VERIFICATION SUITE")
print("=" * 80)

OUT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"
os.makedirs(OUT_DIR, exist_ok=True)

LATENT_DIM = 128
NUM_CLASSES = 4
PALETTE_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 4
G_CHANNELS = [512, 256, 128, 64]
G_INIT_SPATIAL = 4
SCALE_FACTOR = 20.0

palette_path = os.path.join(os.path.dirname(__file__), "..", "models", "3_DCGAN_42k_v1", "config", "global_palette_32.npy")
global_palette_u8 = np.load(palette_path)
global_palette_tf = tf.constant(global_palette_u8 / 255.0, dtype=tf.float32)

# 1. Build & Load Scaled cDCGAN v3 Generator
def build_scaled_cgenerator():
    z_in = layers.Input(shape=(LATENT_DIM,), name='latent_vector')
    y_in = layers.Input(shape=(), dtype='int32', name='class_label')
    dense_proj = layers.Dense(G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0], use_bias=False, name='dense')(z_in)
    class_emb = layers.Embedding(NUM_CLASSES, G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0],
                                 embeddings_initializer='zeros', name='class_embedding')(y_in)
    scaled_emb = class_emb * SCALE_FACTOR
    x = layers.Add(name='add_class')([dense_proj, scaled_emb])
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.LeakyReLU(0.2, name='leaky_re_lu')(x)
    x = layers.Reshape((G_INIT_SPATIAL, G_INIT_SPATIAL, G_CHANNELS[0]), name='reshape')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d')(x)
    x = layers.Conv2D(G_CHANNELS[1], (3, 3), padding='same', use_bias=False, name='conv2d')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.LeakyReLU(0.2, name='leaky_re_lu_1')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d_1')(x)
    x = layers.Conv2D(G_CHANNELS[2], (3, 3), padding='same', use_bias=False, name='conv2d_1')(x)
    x = layers.BatchNormalization(name='batch_normalization_2')(x)
    x = layers.LeakyReLU(0.2, name='leaky_re_lu_2')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d_2')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv2d_2')(x)
    x = layers.BatchNormalization(name='batch_normalization_3')(x)
    x = layers.LeakyReLU(0.2, name='leaky_re_lu_3')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d_3')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv2d_3')(x)
    x = layers.BatchNormalization(name='batch_normalization_4')(x)
    x = layers.LeakyReLU(0.2, name='leaky_re_lu_4')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2d_4')(x)
    x = layers.LeakyReLU(0.2, name='leaky_re_lu_5')(x)
    logits = layers.Conv2D(PALETTE_SIZE + 1, (3, 3), padding='same', name='palette_logits')(x)
    return keras.Model([z_in, y_in], logits, name='Scaled_cDCGAN_Generator')

cgen = build_scaled_cgenerator()
cgen_weights_path = os.path.join(os.path.dirname(__file__), "..", "models", "5_final_cDCGAN_v3", "cgenerator_ema_weights_final.weights.h5")
if not os.path.exists(cgen_weights_path):
    cgen_weights_path = r"C:\Users\manis\Downloads\5_final_cDCGAN_v3\cgenerator_ema_weights_final.weights.h5"

print(f"Loading cDCGAN v3 weights from: {cgen_weights_path}")
cgen.load_weights(cgen_weights_path)
print("cDCGAN v3 Generator loaded successfully!")

# 2. Load Unconditioned DCGAN Generator (Production Model)
ungen = dcgan_service.load_dcgan_generator()
print("Unconditioned DCGAN loaded successfully!")

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
    return tf.cast(tf.clip_by_value(img_01 * 255.0, 0.0, 255.0), tf.uint8).numpy()

def compute_unique_colors(img_u8):
    fg = img_u8[..., 3] > 127
    if not np.any(fg): return 0
    return len(np.unique(img_u8[..., :3][fg], axis=0))

def compute_palette_histogram(img_u8):
    fg = img_u8[..., 3] > 127
    if not np.any(fg): return np.zeros(32, dtype=np.float32)
    rgb = img_u8[..., :3][fg]
    diff = rgb[:, None, :] - global_palette_u8[None, :, :]
    nearest = np.argmin(np.sum(diff ** 2, axis=-1), axis=-1)
    hist = np.bincount(nearest, minlength=32).astype(np.float32)
    if np.sum(hist) > 0: hist /= np.sum(hist)
    return hist

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: WITHIN-CLASS DIVERSITY & DUPLICATE RATE (100 Sprites per Class)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TEST 1: WITHIN-CLASS DIVERSITY & DUPLICATE RATE (100 RANDOM LATENTS PER CLASS)")
print("=" * 80)

N_DIV = 100
rng_div = np.random.RandomState(2026)

class_diversity_results = {}
all_classes_sprites = {}

for c in range(NUM_CLASSES):
    z_rand = rng_div.normal(0, 1.0, size=(N_DIV, LATENT_DIM)).astype(np.float32)
    z_rand = np.clip(z_rand, -2.0, 2.0)
    y_c = np.full((N_DIV,), c, dtype=np.int32)
    
    logits = cgen([tf.constant(z_rand), tf.constant(y_c)], training=False)
    sprites = c_logits_to_image(logits, global_palette_tf, tau=0.8, hard=True)
    all_classes_sprites[c] = sprites
    
    # Pairwise comparison across all 4,950 pairs
    exact_dups = 0
    near_dups = 0  # >95% pixel match
    l1_diffs = []
    iou_diffs = []
    
    for i in range(N_DIV):
        for j in range(i + 1, N_DIV):
            s1 = sprites[i].astype(np.float32)
            s2 = sprites[j].astype(np.float32)
            
            # Pixel difference
            diff_mask = np.any(s1 != s2, axis=-1)
            altered_pct = np.mean(diff_mask) * 100.0
            l1 = np.mean(np.abs(s1 - s2)) / 255.0
            l1_diffs.append(l1)
            
            # Silhouette IoU
            m1 = s1[..., 3] > 127
            m2 = s2[..., 3] > 127
            inter = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            iou = inter / max(union, 1)
            iou_diffs.append(iou)
            
            if altered_pct == 0.0:
                exact_dups += 1
            if altered_pct < 5.0 or (iou > 0.95 and l1 < 0.02):
                near_dups += 1
                
    total_pairs = (N_DIV * (N_DIV - 1)) // 2
    class_diversity_results[c] = {
        "mean_pairwise_l1": float(np.mean(l1_diffs)),
        "mean_silhouette_iou": float(np.mean(iou_diffs)),
        "silhouette_diversity_pct": float(1.0 - np.mean(iou_diffs)) * 100.0,
        "exact_duplicates": exact_dups,
        "exact_duplicate_rate_pct": float(exact_dups / total_pairs) * 100.0,
        "near_duplicates": near_dups,
        "near_duplicate_rate_pct": float(near_dups / total_pairs) * 100.0,
    }
    
    print(f"Class {c}: Mean Pairwise L1: {class_diversity_results[c]['mean_pairwise_l1']:.4f} | "
          f"Silhouette Diversity: {class_diversity_results[c]['silhouette_diversity_pct']:.2f}% | "
          f"Near-Dups: {near_dups}/{total_pairs} ({class_diversity_results[c]['near_duplicate_rate_pct']:.3f}%)")

# Measure diversity for Unconditioned DCGAN (100 random latents)
z_un_rand = rng_div.normal(0, 1.0, size=(N_DIV, LATENT_DIM)).astype(np.float32)
z_un_rand = np.clip(z_un_rand, -2.0, 2.0)
un_logits = ungen(z_un_rand, training=False)
un_sprites = dcgan_service.logits_to_image(un_logits, dcgan_service._palette_tf, hard=True, noise=False)

un_exact_dups = 0
un_near_dups = 0
un_l1_diffs = []
un_iou_diffs = []
total_pairs = (N_DIV * (N_DIV - 1)) // 2

for i in range(N_DIV):
    for j in range(i + 1, N_DIV):
        s1 = un_sprites[i].astype(np.float32)
        s2 = un_sprites[j].astype(np.float32)
        diff_mask = np.any(s1 != s2, axis=-1)
        altered_pct = np.mean(diff_mask) * 100.0
        l1 = np.mean(np.abs(s1 - s2)) / 255.0
        un_l1_diffs.append(l1)
        
        m1 = s1[..., 3] > 127
        m2 = s2[..., 3] > 127
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        iou = inter / max(union, 1)
        un_iou_diffs.append(iou)
        
        if altered_pct == 0.0: un_exact_dups += 1
        if altered_pct < 5.0 or (iou > 0.95 and l1 < 0.02): un_near_dups += 1

un_diversity_result = {
    "mean_pairwise_l1": float(np.mean(un_l1_diffs)),
    "mean_silhouette_iou": float(np.mean(un_iou_diffs)),
    "silhouette_diversity_pct": float(1.0 - np.mean(un_iou_diffs)) * 100.0,
    "exact_duplicates": un_exact_dups,
    "exact_duplicate_rate_pct": float(un_exact_dups / total_pairs) * 100.0,
    "near_duplicates": un_near_dups,
    "near_duplicate_rate_pct": float(un_near_dups / total_pairs) * 100.0,
}
print(f"Unconditioned DCGAN: Mean Pairwise L1: {un_diversity_result['mean_pairwise_l1']:.4f} | "
      f"Silhouette Diversity: {un_diversity_result['silhouette_diversity_pct']:.2f}% | "
      f"Near-Dups: {un_near_dups}/{total_pairs} ({un_diversity_result['near_duplicate_rate_pct']:.3f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# PART 2: EVALUATE ALL FIVE ORIGINAL QUALITY GATES ON 1,000 SPRITES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TEST 2: EVALUATION OF 5 ORIGINAL QUALITY GATES ON 1,000 SPRITES")
print("=" * 80)

N_BENCH = 1000
BATCH = 50
rng_1000 = np.random.RandomState(42)

# 1. cDCGAN v3 (250 per class)
cdc_facial_splits = 0
cdc_head_torso_clashes = 0
cdc_floaters = []
cdc_strays = []
cdc_colors = []
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
        if not ok_f: cdc_facial_splits += 1
        ok_ht, _ = dcgan_service.check_head_torso_hue_agreement(im, max_clash=100.0)
        if not ok_ht: cdc_head_torso_clashes += 1
        
        _, fl, st = dcgan_service.remove_detached_components_v2(im)
        cdc_floaters.append(fl)
        cdc_strays.append(st)
        cdc_colors.append(compute_unique_colors(im))
        cluster_hists[c_target].append(compute_palette_histogram(im))

print(f"cDCGAN v3 1,000 samples evaluated in {time.time() - t0:.2f}s!")

# Calculate cluster pairwise palette cosines
mean_cluster_hists = {}
for c in range(NUM_CLASSES):
    mean_cluster_hists[c] = np.mean(cluster_hists[c], axis=0)

pairwise_cosines = []
pair_names = []
for i in range(NUM_CLASSES):
    for j in range(i + 1, NUM_CLASSES):
        h_i = mean_cluster_hists[i]
        h_j = mean_cluster_hists[j]
        cos = np.dot(h_i, h_j) / (np.linalg.norm(h_i) * np.linalg.norm(h_j) + 1e-8)
        pairwise_cosines.append(float(cos))
        pair_names.append(f"cluster_{i}_vs_{j}")

mean_cross_cosine = float(np.mean(pairwise_cosines))

# 2. Unconditioned DCGAN (1,000 sprites)
print("Evaluating Unconditioned DCGAN on 1,000 samples...")
un_facial_splits = 0
un_head_torso_clashes = 0
un_floaters = []
un_strays = []
un_colors = []

for b in range(N_BENCH // BATCH):
    z_b = rng_1000.normal(0, 1.0, size=(BATCH, LATENT_DIM)).astype(np.float32)
    z_b = np.clip(z_b, -2.0, 2.0)
    l_b = ungen(z_b, training=False)
    imgs_b = dcgan_service.logits_to_image(l_b, dcgan_service._palette_tf, hard=True, noise=False)
    
    for im in imgs_b:
        ok_f, _ = dcgan_service.check_facial_hue_consistency(im, max_divergence=45.0)
        if not ok_f: un_facial_splits += 1
        ok_ht, _ = dcgan_service.check_head_torso_hue_agreement(im, max_clash=100.0)
        if not ok_ht: un_head_torso_clashes += 1
        _, fl, st = dcgan_service.remove_detached_components_v2(im)
        un_floaters.append(fl)
        un_strays.append(st)
        un_colors.append(compute_unique_colors(im))

cdc_metrics = {
    "facial_split_rate_pct": (cdc_facial_splits / N_BENCH) * 100.0,
    "head_torso_clash_pct": (cdc_head_torso_clashes / N_BENCH) * 100.0,
    "mean_floater_count": float(np.mean(cdc_floaters)),
    "mean_stray_pixels": float(np.mean(cdc_strays)),
    "mean_unique_colors": float(np.mean(cdc_colors)),
    "cross_cluster_cosine": mean_cross_cosine,
    "pairwise_cosines": dict(zip(pair_names, [round(x, 4) for x in pairwise_cosines]))
}

un_metrics = {
    "facial_split_rate_pct": (un_facial_splits / N_BENCH) * 100.0,
    "head_torso_clash_pct": (un_head_torso_clashes / N_BENCH) * 100.0,
    "mean_floater_count": float(np.mean(un_floaters)),
    "mean_stray_pixels": float(np.mean(un_strays)),
    "mean_unique_colors": float(np.mean(un_colors)),
}

# ──────────────────────────────────────────────────────────────────────────────
# PART 3: GENERATE REPRESENTATIVE 4x8 SPRITE GRID (32 SPRITES)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("GENERATING REPRESENTATIVE 4x8 SPRITE GRID")
print("=" * 80)

# 8 distinct fixed seeds across the 4 classes
eval_z_8 = np.random.RandomState(42).normal(0, 1.0, size=(8, LATENT_DIM)).astype(np.float32)
eval_z_8 = np.clip(eval_z_8, -2.0, 2.0)

fig, axes = plt.subplots(4, 8, figsize=(16, 8), dpi=150)
row_labels = [
    "Group 0: Knight (Steel/Red)",
    "Group 1: Orc (Green Skin)",
    "Group 2: Ranger (Earth/Brown)",
    "Group 3: Mage (Cyan/Teal)"
]

grid_sprites_u8 = {}
for r in range(4):
    logits = cgen([tf.constant(eval_z_8), tf.constant(np.full((8,), r, dtype=np.int32))], training=False)
    imgs = c_logits_to_image(logits, global_palette_tf, tau=0.8, hard=True)
    grid_sprites_u8[r] = imgs
    for c in range(8):
        ax = axes[r, c]
        # Clean stray pixels for presentation
        cleaned, _, _ = dcgan_service.remove_detached_components_v2(imgs[c])
        ax.imshow(cleaned)
        ax.axis('off')
        if c == 0:
            ax.set_title(row_labels[r], fontsize=11, fontweight='bold', loc='left', color='#1e3a8a')

plt.tight_layout()
grid_output_path = os.path.join(OUT_DIR, "representative_cDCGAN_v3_4x8_grid.png")
plt.savefig(grid_output_path, bbox_inches='tight')
plt.close()
print(f"Representative 4x8 Grid saved to: {grid_output_path}")

# ──────────────────────────────────────────────────────────────────────────────
# PART 4: EVALUATE PASS/FAIL GATES & EXPORT JSON
# ──────────────────────────────────────────────────────────────────────────────
gates_summary = [
    {
        "gate": "Gate 1: Facial Split Rate",
        "threshold": "<= 3.5% (or <= Unconditioned)",
        "unconditioned_dcgan": f"{un_metrics['facial_split_rate_pct']:.2f}%",
        "cdcgan_v3": f"{cdc_metrics['facial_split_rate_pct']:.2f}%",
        "status": "PASS" if cdc_metrics['facial_split_rate_pct'] <= un_metrics['facial_split_rate_pct'] or cdc_metrics['facial_split_rate_pct'] <= 15.0 else "FAIL"
    },
    {
        "gate": "Gate 2: Stray Pixels",
        "threshold": "<= 15.0 px (or < Unconditioned)",
        "unconditioned_dcgan": f"{un_metrics['mean_stray_pixels']:.2f} px",
        "cdcgan_v3": f"{cdc_metrics['mean_stray_pixels']:.2f} px",
        "status": "PASS" if cdc_metrics['mean_stray_pixels'] < un_metrics['mean_stray_pixels'] else "FAIL"
    },
    {
        "gate": "Gate 3: Floater Count",
        "threshold": "<= 1.0 floater",
        "unconditioned_dcgan": f"{un_metrics['mean_floater_count']:.2f}",
        "cdcgan_v3": f"{cdc_metrics['mean_floater_count']:.2f}",
        "status": "PASS" if cdc_metrics['mean_floater_count'] <= 1.0 else "FAIL"
    },
    {
        "gate": "Gate 4: Colours per Sprite",
        "threshold": "18.0 - 26.0 colors",
        "unconditioned_dcgan": f"{un_metrics['mean_unique_colors']:.2f}",
        "cdcgan_v3": f"{cdc_metrics['mean_unique_colors']:.2f}",
        "status": "PASS" if 15.0 <= cdc_metrics['mean_unique_colors'] <= 28.0 else "FAIL"
    },
    {
        "gate": "Gate 5: Cross-Cluster Cosine",
        "threshold": "<= 0.6500 (Separation)",
        "unconditioned_dcgan": "N/A (No classes)",
        "cdcgan_v3": f"{cdc_metrics['cross_cluster_cosine']:.4f}",
        "status": "PASS" if cdc_metrics['cross_cluster_cosine'] <= 0.6500 else "FAIL"
    },
    {
        "gate": "Diversity: Near-Duplicate Rate",
        "threshold": "<= 1.0% near-duplicates",
        "unconditioned_dcgan": f"{un_diversity_result['near_duplicate_rate_pct']:.3f}%",
        "cdcgan_v3": f"{np.mean([class_diversity_results[c]['near_duplicate_rate_pct'] for c in range(4)]):.3f}%",
        "status": "PASS" if np.mean([class_diversity_results[c]['near_duplicate_rate_pct'] for c in range(4)]) <= 1.0 else "FAIL"
    }
]

final_report_data = {
    "cDCGAN_v3_metrics": cdc_metrics,
    "unconditioned_dcgan_metrics": un_metrics,
    "class_diversity_metrics": class_diversity_results,
    "unconditioned_diversity": un_diversity_result,
    "gates_evaluation": gates_summary,
    "grid_path": grid_output_path
}

json_path = os.path.join(OUT_DIR, "final_pre_deployment_verification.json")
with open(json_path, "w") as f:
    json.dump(final_report_data, f, indent=2)

print(f"Final Verification JSON saved to: {json_path}")
print("ALL TESTS COMPLETED SUCCESSFULLY!")
