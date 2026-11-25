import os
import sys
import json
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import dcgan_service
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 80)
print("PHASE 3 DIAGNOSTIC SUITE: TESTS A THROUGH F")
print("=" * 80)

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

palette_path = os.path.join(os.path.dirname(__file__), "..", "models", "3_DCGAN_42k_v1", "config", "global_palette_32.npy")
palette_u8 = np.load(palette_path)
palette_tf = tf.constant(palette_u8 / 255.0, dtype=tf.float32)

# Conditional Generator Architecture
def build_cgenerator():
    z_in = layers.Input(shape=(LATENT_DIM,), name="latent_vector")
    y_in = layers.Input(shape=(), dtype="int32", name="class_label")
    dense_proj = layers.Dense(G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0], use_bias=False, name="dense")(z_in)
    class_emb = layers.Embedding(NUM_CLASSES, G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0],
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

def build_projection_discriminator():
    img_in = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="sprite_input")
    y_in = layers.Input(shape=(), dtype="int32", name="class_label")
    def sn_conv(filters, strides=2):
        conv = layers.Conv2D(filters, (4, 4), strides=strides, padding="same")
        return layers.SpectralNormalization(conv)
    x = sn_conv(64)(img_in)
    x = layers.LeakyReLU(0.2)(x)
    x = sn_conv(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = sn_conv(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = sn_conv(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    feat = layers.Flatten(name="feat_flatten")(x)
    base_score = layers.Dense(1, name="scalar_score")(feat)
    y_embed = layers.Embedding(NUM_CLASSES, 4 * 4 * 512, embeddings_initializer="zeros", name="proj_embedding")(y_in)
    proj_score = layers.Dot(axes=1, name="dot_projection")([feat, y_embed])
    out = layers.Add(name="total_score")([base_score, proj_score])
    return keras.Model([img_in, y_in], out, name="Projection_Spectral_Discriminator")

def c_logits_to_image(logits, tau=0.8, hard=True):
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

# Load models
cgen = build_cgenerator()
cgen_ema = build_cgenerator()
cdisc = build_projection_discriminator()

cgen_weights = os.path.join(MODEL_DIR, "export", "cgenerator_weights_final.weights.h5")
cgen_ema_weights = os.path.join(MODEL_DIR, "export", "cgenerator_ema_weights_final.weights.h5")
cdisc_weights = os.path.join(MODEL_DIR, "export", "cdiscriminator_weights_final.weights.h5")

cgen.load_weights(cgen_weights)
cgen_ema.load_weights(cgen_ema_weights)
cdisc.load_weights(cdisc_weights)

print("Loaded cgen (raw), cgen_ema, and cdisc successfully.")

# ══════════════════════════════════════════════════════════════════════════════
# TEST B — CLASS EMBEDDING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TEST B: CLASS EMBEDDING ANALYSIS")
print("=" * 80)

# In Generator:
g_emb_layer = cgen.get_layer("class_embedding")
g_emb_weights = g_emb_layer.get_weights()[0] # (4, 8192)

g_ema_emb_layer = cgen_ema.get_layer("class_embedding")
g_ema_emb_weights = g_ema_emb_layer.get_weights()[0] # (4, 8192)

# In Discriminator:
d_emb_layer = cdisc.get_layer("proj_embedding")
d_emb_weights = d_emb_layer.get_weights()[0] # (4, 8192)

# Latent projection dense layer weights
dense_layer = cgen.get_layer("dense")
dense_w = dense_layer.get_weights()[0] # (128, 8192)

# Compute norms and stats
print(f"Generator Class Embedding Shape     : {g_emb_weights.shape}")
print(f"Discriminator Proj Embedding Shape  : {d_emb_weights.shape}")
print(f"Generator Dense Projection Shape    : {dense_w.shape}")

# Average L2 norm of latent projection:
# For z ~ N(0, 1) in R^128, expected ||z @ W||
z_sample = np.random.RandomState(42).normal(0, 1.0, size=(1000, LATENT_DIM)).astype(np.float32)
proj_z = np.dot(z_sample, dense_w)
mean_proj_norm = float(np.mean(np.linalg.norm(proj_z, axis=1)))
print(f"\nMean L2 Norm of Latent Projection (z @ W) : {mean_proj_norm:.4f}")

g_norms = np.linalg.norm(g_emb_weights, axis=1)
g_ema_norms = np.linalg.norm(g_ema_emb_weights, axis=1)
d_norms = np.linalg.norm(d_emb_weights, axis=1)

print("\nGenerator Class Embedding L2 Norms:")
for c in range(4):
    ratio = g_norms[c] / mean_proj_norm
    print(f"   Class {c} L2 Norm (Raw): {g_norms[c]:.6f} (EMA: {g_ema_norms[c]:.6f}) | Ratio to z-proj: {ratio * 100:.3f}%")

print("\nDiscriminator Projection Embedding L2 Norms:")
for c in range(4):
    print(f"   Class {c} L2 Norm: {d_norms[c]:.6f}")

def pairwise_matrix(weights):
    cos_mat = np.zeros((4, 4))
    l2_mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dot = np.dot(weights[i], weights[j])
            denom = np.linalg.norm(weights[i]) * np.linalg.norm(weights[j])
            cos_mat[i, j] = dot / max(denom, 1e-12)
            l2_mat[i, j] = np.linalg.norm(weights[i] - weights[j])
    return cos_mat, l2_mat

g_cos, g_l2 = pairwise_matrix(g_emb_weights)
print("\nGenerator Class Embedding Pairwise Cosine Similarity:")
print(np.round(g_cos, 4))
print("\nGenerator Class Embedding Pairwise L2 Distance:")
print(np.round(g_l2, 6))

d_cos, d_l2 = pairwise_matrix(d_emb_weights)
print("\nDiscriminator Proj Embedding Pairwise Cosine Similarity:")
print(np.round(d_cos, 4))

# ══════════════════════════════════════════════════════════════════════════════
# TEST A — FIXED-LATENT LABEL SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TEST A: FIXED-LATENT LABEL SENSITIVITY (N = 100 LATENTS)")
print("=" * 80)

N_TEST_A = 100
rng_a = np.random.RandomState(42)
test_z = rng_a.normal(0, 1.0, size=(N_TEST_A, LATENT_DIM)).astype(np.float32)
test_z = np.clip(test_z, -2.0, 2.0)

# Generate samples for all 4 classes for both Raw and EMA
raw_samples_by_class = {}
ema_samples_by_class = {}

for c in range(4):
    z_tf = tf.constant(test_z)
    y_tf = tf.constant(np.full((N_TEST_A,), c, dtype=np.int32))
    l_raw = cgen([z_tf, y_tf], training=False)
    l_ema = cgen_ema([z_tf, y_tf], training=False)
    raw_samples_by_class[c] = c_logits_to_image(l_raw)
    ema_samples_by_class[c] = c_logits_to_image(l_ema)

# Pairwise differences between classes on same z
def measure_class_sensitivity(samples_by_class):
    l1_diffs = []
    fg_rgb_diffs = []
    alpha_diffs = []
    pct_altered_pixels = []
    
    for i in range(4):
        for j in range(i + 1, 4):
            for s in range(N_TEST_A):
                im_i = samples_by_class[i][s].astype(np.float32)
                im_j = samples_by_class[j][s].astype(np.float32)
                
                # Full image diff (0 - 255)
                diff = np.abs(im_i - im_j)
                l1_diffs.append(np.mean(diff))
                
                # Altered pixels fraction: where RGBA is not identical
                pixel_diff = np.any(im_i != im_j, axis=-1)
                pct_altered_pixels.append(np.mean(pixel_diff) * 100.0)
                
                # Alpha difference
                alpha_diffs.append(np.mean(np.abs(im_i[..., 3] - im_j[..., 3])))
                
                # Foreground RGB diff
                fg_both = (im_i[..., 3] > 127) & (im_j[..., 3] > 127)
                if np.any(fg_both):
                    fg_diff = np.mean(np.abs(im_i[..., :3][fg_both] - im_j[..., :3][fg_both]))
                    fg_rgb_diffs.append(fg_diff)
                    
    return {
        "mean_l1_diff": float(np.mean(l1_diffs)),
        "mean_pct_altered_pixels": float(np.mean(pct_altered_pixels)),
        "mean_alpha_diff": float(np.mean(alpha_diffs)),
        "mean_fg_rgb_diff": float(np.mean(fg_rgb_diffs))
    }

sens_raw = measure_class_sensitivity(raw_samples_by_class)
sens_ema = measure_class_sensitivity(ema_samples_by_class)

print(f"Raw Generator  : Mean Altered Pixels: {sens_raw['mean_pct_altered_pixels']:.2f}% | Mean L1 Diff: {sens_raw['mean_l1_diff']:.3f} | FG RGB Diff: {sens_raw['mean_fg_rgb_diff']:.3f}")
print(f"EMA Generator  : Mean Altered Pixels: {sens_ema['mean_pct_altered_pixels']:.2f}% | Mean L1 Diff: {sens_ema['mean_l1_diff']:.3f} | FG RGB Diff: {sens_ema['mean_fg_rgb_diff']:.3f}")

# Generate a 4 x 8 visual comparison grid (8 latents, 4 classes)
SPR_SZ = 64
TILE_SZ = 128
grid_4x8 = Image.new("RGBA", (8 * TILE_SZ, 4 * TILE_SZ), (15, 23, 42, 255))
for c in range(4):
    for z_idx in range(8):
        spr = Image.fromarray(ema_samples_by_class[c][z_idx]).resize((TILE_SZ, TILE_SZ), Image.NEAREST)
        grid_4x8.paste(spr, (z_idx * TILE_SZ, c * TILE_SZ), spr)

out_4x8_path = os.path.join(ARTIFACT_DIR, "diagnostic_test_a_4x8_grid.png")
grid_4x8.save(out_4x8_path)
print(f"Saved Test A 4x8 comparison grid to: {out_4x8_path}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST C — DISCRIMINATOR LABEL AWARENESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TEST C: DISCRIMINATOR LABEL AWARENESS (ON REAL DATA)")
print("=" * 80)

sprites_42k = np.load("Dataset/sprites_42k.npy")
labels_42k = np.load("Dataset/labels.npy")

# Sample 500 real sprites
sub_indices = np.random.RandomState(42).choice(len(sprites_42k), size=500, replace=False)
sub_x = (sprites_42k[sub_indices].astype(np.float32) / 127.5) - 1.0
sub_y_true = labels_42k[sub_indices]

# Test scores with true labels
scores_true = cdisc([tf.constant(sub_x), tf.constant(sub_y_true)], training=False).numpy().flatten()

# Test scores with wrong labels: (true + 1) % 4, (true + 2) % 4, (true + 3) % 4
wrong_margins = []
for shift in [1, 2, 3]:
    sub_y_wrong = (sub_y_true + shift) % 4
    scores_wrong = cdisc([tf.constant(sub_x), tf.constant(sub_y_wrong)], training=False).numpy().flatten()
    diff = scores_true - scores_wrong
    wrong_margins.append(float(np.mean(diff)))
    print(f"   Mean Score Margin (True vs Shift +{shift}) : {np.mean(diff):+.6f} (True Score Mean: {np.mean(scores_true):.4f}, Wrong Mean: {np.mean(scores_wrong):.4f})")

print(f"Overall True-vs-Wrong Margin in Projection Discriminator: {np.mean(wrong_margins):+.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST D — GRADIENT FLOW
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TEST D: GRADIENT FLOW ON GENERATOR EMBEDDING VS PRETRAINED LAYERS")
print("=" * 80)

# Simulate 1 training step gradient tape
dummy_z = tf.random.normal((64, LATENT_DIM))
dummy_y = tf.random.uniform((64,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

with tf.GradientTape() as tape:
    fake_logits = cgen([dummy_z, dummy_y], training=True)
    fake_x = c_logits_to_image(fake_logits) # non-differentiable numpy call, use tensor ops:
    
with tf.GradientTape() as tape:
    l = cgen([dummy_z, dummy_y], training=True)
    # Gumbel Softmax differentiable approximation
    pl = l[..., :PALETTE_SIZE]
    ys = tf.nn.softmax(pl, axis=-1)
    rgb = tf.tensordot(ys, palette_tf, axes=[[-1], [0]])
    a = tf.sigmoid(l[..., PALETTE_SIZE:])
    synth = tf.concat([rgb * a, a], axis=-1) * 2.0 - 1.0
    d_score = cdisc([synth, dummy_y], training=True)
    g_loss = -tf.reduce_mean(d_score)

grads = tape.gradient(g_loss, cgen.trainable_variables)
var_names = [v.name for v in cgen.trainable_variables]

print("Gradient Norms per Generator Layer:")
g_emb_grad_norm = 0.0
for v, g in zip(cgen.trainable_variables, grads):
    if g is not None:
        gn = float(tf.norm(g))
        if "class_embedding" in v.name:
            g_emb_grad_norm = gn
            print(f"   --> [CLASS EMBEDDING] {v.name:40s} : Grad Norm = {gn:.8f}")
        elif "dense" in v.name or "conv2d_4" in v.name or "palette_logits" in v.name:
            print(f"       {v.name:40s} : Grad Norm = {gn:.8f}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST F — CHECKPOINT PROGRESSION (EPOCHS 3, 6, 9, 12, 15, 18)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TEST F: CHECKPOINT PROGRESSION ACROSS EPOCHS 3 TO 18")
print("=" * 80)

ckpt_progression = []
for ep in [3, 6, 9, 12, 15, 18]:
    ckpt_p = os.path.join(MODEL_DIR, "checkpoints", f"cgenerator_epoch_{ep:02d}.weights.h5")
    if not os.path.exists(ckpt_p):
        continue
    cgen.load_weights(ckpt_p)
    emb_w = cgen.get_layer("class_embedding").get_weights()[0]
    norms = np.linalg.norm(emb_w, axis=1)
    
    # Measure sensitivity on 16 latents
    z_sub = test_z[:16]
    samples = {}
    for c in range(4):
        logits = cgen([tf.constant(z_sub), tf.constant(np.full((16,), c, dtype=np.int32))], training=False)
        samples[c] = c_logits_to_image(logits)
        
    diffs = []
    for i in range(4):
        for j in range(i+1, 4):
            for s in range(16):
                diffs.append(np.mean(np.abs(samples[i][s].astype(float) - samples[j][s].astype(float))))
                
    ckpt_progression.append({
        "epoch": ep,
        "mean_emb_norm": float(np.mean(norms)),
        "max_emb_norm": float(np.max(norms)),
        "mean_pairwise_l1_diff": float(np.mean(diffs))
    })
    print(f"   Epoch {ep:02d} | Mean Class Emb Norm: {np.mean(norms):.6f} | Max Norm: {np.max(norms):.6f} | Mean Output L1 Diff: {np.mean(diffs):.4f}")

# Save full Phase 3 summary
phase3_data = {
    "test_a_sensitivity_raw": sens_raw,
    "test_a_sensitivity_ema": sens_ema,
    "test_b_generator_emb_norms": g_norms.tolist(),
    "test_b_generator_ema_norms": g_ema_norms.tolist(),
    "test_b_discriminator_proj_norms": d_norms.tolist(),
    "test_b_mean_z_proj_norm": mean_proj_norm,
    "test_b_ratio_pct": float(np.mean(g_norms) / mean_proj_norm * 100.0),
    "test_c_discriminator_margins": wrong_margins,
    "test_d_emb_grad_norm": g_emb_grad_norm,
    "test_f_progression": ckpt_progression
}

out_p3_json = os.path.join(ARTIFACT_DIR, "phase3_diagnostics_report.json")
with open(out_p3_json, "w") as f:
    json.dump(phase3_data, f, indent=2)

print(f"\nPhase 3 Diagnostics Complete! Saved to: {out_p3_json}")
