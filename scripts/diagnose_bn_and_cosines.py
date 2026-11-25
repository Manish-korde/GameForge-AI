import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 80)
print("INVESTIGATION 2 & 3: BN SUPPRESSION PROFILE & EXACT REAL PALETTE COSINES")
print("=" * 80)

# Load Empirical Palette
palette_path = 'models/3_DCGAN_42k_v1/config/global_palette_32.npy'
palette_u8 = np.load(palette_path)
palette_tf = tf.constant(palette_u8 / 255.0, dtype=tf.float32)

# Load cgen
def build_cgenerator():
    z_in = layers.Input(shape=(128,), name='latent_vector')
    y_in = layers.Input(shape=(), dtype='int32', name='class_label')
    dense_proj = layers.Dense(4 * 4 * 512, use_bias=False, name='dense')(z_in)
    class_emb = layers.Embedding(4, 4 * 4 * 512, embeddings_initializer='zeros', name='class_embedding')(y_in)
    x_add = layers.Add(name='add_class')([dense_proj, class_emb])
    x_bn = layers.BatchNormalization(name='batch_normalization')(x_add)
    x_act = layers.LeakyReLU(0.2, name='leaky_re_lu')(x_bn)
    x_resh = layers.Reshape((4, 4, 512), name='reshape')(x_act)
    
    # 4x4 -> 8x8
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d')(x_resh)
    x = layers.Conv2D(256, (3, 3), padding='same', use_bias=False, name='conv2d')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x_8x8 = layers.LeakyReLU(0.2, name='leaky_re_lu_1')(x)
    
    # 8x8 -> 16x16
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d_1')(x_8x8)
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False, name='conv2d_1')(x)
    x = layers.BatchNormalization(name='batch_normalization_2')(x)
    x_16x16 = layers.LeakyReLU(0.2, name='leaky_re_lu_2')(x)
    
    # 16x16 -> 32x32
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d_2')(x_16x16)
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv2d_2')(x)
    x = layers.BatchNormalization(name='batch_normalization_3')(x)
    x_32x32 = layers.LeakyReLU(0.2, name='leaky_re_lu_3')(x)
    
    # 32x32 -> 64x64
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d_3')(x_32x32)
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv2d_3')(x)
    x = layers.BatchNormalization(name='batch_normalization_4')(x)
    x_64x64_pre = layers.LeakyReLU(0.2, name='leaky_re_lu_4')(x)
    
    # Refinement
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2d_4')(x_64x64_pre)
    x_64x64_post = layers.LeakyReLU(0.2, name='leaky_re_lu_5')(x)
    
    # Logits
    logits = layers.Conv2D(33, (3, 3), padding='same', name='palette_logits')(x_64x64_post)
    
    return keras.Model(
        [z_in, y_in],
        {
            "dense": dense_proj,
            "class_emb": class_emb,
            "before_bn": x_add,
            "after_bn": x_bn,
            "after_lrelu": x_act,
            "after_reshape": x_resh,
            "block_8x8": x_8x8,
            "block_16x16": x_16x16,
            "block_32x32": x_32x32,
            "block_64x64_pre": x_64x64_pre,
            "block_64x64_post": x_64x64_post,
            "palette_logits": logits
        },
        name='cgen_inspect'
    )

model = build_cgenerator()
model.load_weights('models/4_cDCGAN_k4_v1/export/cgenerator_weights_final.weights.h5')
print("Loaded cgenerator weights into multi-output inspection model.")

# ──────────────────────────────────────────────────────────────────────────────
# ISSUE 2: MEASURE CLASS-DEPENDENT ACTIVATION DIFFERENCES BEFORE/AFTER BN
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ISSUE 2: LAYER-BY-LAYER CLASS ACTIVATION DIFFERENCE PROFILE (N=100)")
print("=" * 80)

N_TEST = 100
rng = np.random.RandomState(42)
z_test = rng.normal(0, 1.0, size=(N_TEST, 128)).astype(np.float32)
z_test = np.clip(z_test, -2.0, 2.0)

# Run model for all 4 classes
acts = {}
for c in range(4):
    y_test = np.full((N_TEST,), c, dtype=np.int32)
    acts[c] = model([tf.constant(z_test), tf.constant(y_test)], training=False)

# Convert all outputs to numpy
for c in range(4):
    for k in acts[c]:
        acts[c][k] = acts[c][k].numpy()

layer_names = [
    "dense",
    "class_emb",
    "before_bn",
    "after_bn",
    "after_lrelu",
    "after_reshape",
    "block_8x8",
    "block_16x16",
    "block_32x32",
    "block_64x64_pre",
    "block_64x64_post",
    "palette_logits"
]

print(f"{'Layer / Block':20s} | {'Mean Shape':16s} | {'Mean ||Act||':14s} | {'Mean ||Delta||':14s} | {'Rel Diff (%)':12s} | {'Max Rel (%)':12s}")
print("-" * 96)

bn_profile_results = {}
for l_name in layer_names:
    l2_acts = []
    l2_deltas = []
    rel_diffs = []
    
    for i in range(4):
        for j in range(i + 1, 4):
            act_i = acts[i][l_name]
            act_j = acts[j][l_name]
            
            # Flatten spatial dimensions per sample
            act_i_flat = act_i.reshape(N_TEST, -1)
            act_j_flat = act_j.reshape(N_TEST, -1)
            
            norm_i = np.linalg.norm(act_i_flat, axis=1) # (N_TEST,)
            delta = np.linalg.norm(act_i_flat - act_j_flat, axis=1) # (N_TEST,)
            rel = delta / np.maximum(norm_i, 1e-8) * 100.0
            
            l2_acts.extend(norm_i.tolist())
            l2_deltas.extend(delta.tolist())
            rel_diffs.extend(rel.tolist())
            
    m_act = float(np.mean(l2_acts))
    m_delta = float(np.mean(l2_deltas))
    m_rel = float(np.mean(rel_diffs))
    max_rel = float(np.max(rel_diffs))
    sh_str = str(list(acts[0][l_name].shape[1:]))
    
    print(f"{l_name:20s} | {sh_str:16s} | {m_act:14.4f} | {m_delta:14.4f} | {m_rel:11.3f}% | {max_rel:11.3f}%")
    bn_profile_results[l_name] = {
        "mean_act_norm": m_act,
        "mean_delta_norm": m_delta,
        "rel_diff_pct": m_rel,
        "max_rel_diff_pct": max_rel
    }

# ──────────────────────────────────────────────────────────────────────────────
# ISSUE 3: EXACT REAL CLUSTER COSINE VALUES (IDENTICAL FOREGROUND MASKING)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ISSUE 3: ALL SIX PAIRWISE COSINES (REAL DATA vs GENERATED DATA)")
print("=" * 80)

def compute_palette_histogram(img_u8):
    fg = img_u8[..., 3] > 127
    if not np.any(fg):
        return np.zeros(32, dtype=np.float32)
    rgb = img_u8[..., :3][fg]
    diff = rgb[:, None, :] - palette_u8[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    nearest = np.argmin(dist_sq, axis=-1)
    hist = np.bincount(nearest, minlength=32).astype(np.float32)
    if np.sum(hist) > 0:
        hist /= np.sum(hist)
    return hist

def cosine_similarity(u, v):
    dot = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    return float(dot / max(norm, 1e-8))

sprites_42k = np.load("Dataset/sprites_42k.npy")
labels_42k = np.load("Dataset/labels.npy")

real_hists_per_cluster = {c: [] for c in range(4)}
for c in range(4):
    c_indices = np.where(labels_42k == c)[0]
    for idx in c_indices:
        h = compute_palette_histogram(sprites_42k[idx])
        real_hists_per_cluster[c].append(h)

real_centroids = {c: np.mean(real_hists_per_cluster[c], axis=0) for c in range(4)}

# Compute real pairwise cosines
print("Real Ground-Truth Cluster Centroid Pairwise Cosines:")
real_pairs = {}
for i in range(4):
    for j in range(i + 1, 4):
        cos_sim = cosine_similarity(real_centroids[i], real_centroids[j])
        pair_key = f"Group_{i}_vs_Group_{j}"
        real_pairs[pair_key] = round(cos_sim, 4)
        print(f"  {pair_key:20s} : {cos_sim:.4f}")

mean_real_pair_cos = float(np.mean(list(real_pairs.values())))
print(f"  {'MEAN REAL PAIRWISE':20s} : {mean_real_pair_cos:.4f}")
print(f"  {'MIN REAL PAIRWISE':20s}  : {min(real_pairs.values()):.4f}")

# Now load generated evaluation report to compare
with open('C:/Users/manis/.gemini/antigravity/brain/674ad8cd-af16-4b87-91e3-66973cf97bc5/cdcgan_evaluation_report.json', 'r') as f:
    eval_rep = json.load(f)

gen_pairs = eval_rep["pairwise_cluster_cosines"]
print("\nGenerated Model (18-Epoch cDCGAN) Cluster Centroid Pairwise Cosines:")
for k, v in gen_pairs.items():
    print(f"  {k:20s} : {v:.4f}")
print(f"  {'MEAN GENERATED PAIR':20s} : {eval_rep['mean_cross_cluster_cosine']:.4f}")

# Save JSON artifact
out_diag = {
    "issue_2_bn_profile": bn_profile_results,
    "issue_3_real_pairwise_cosines": real_pairs,
    "issue_3_generated_pairwise_cosines": gen_pairs,
    "real_centroids": {str(c): real_centroids[c].tolist() for c in range(4)}
}

np.save("config/real_c4_centroids.npy", np.array([real_centroids[c] for c in range(4)]))
out_diag_path = "C:/Users/manis/.gemini/antigravity/brain/674ad8cd-af16-4b87-91e3-66973cf97bc5/bn_and_cosines_diagnostic.json"
with open(out_diag_path, "w") as f:
    json.dump(out_diag, f, indent=2)

print(f"\nSaved config/real_c4_centroids.npy and {out_diag_path}")
print("Investigation 2 & 3 Complete!")
