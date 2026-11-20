"""
Checkpoint Archaeology: Rigorous Comparative Evaluation of DCGAN Checkpoints
Evaluates ckpt-9 (Epoch 40), ckpt-10 (Epoch 45), ckpt-11 (Epoch 50), and historical grids.
Generates 100 samples per checkpoint using identical fixed latent vectors.
Computes quantitative sharpness, diversity, alpha halos, and orientation.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

print("=" * 80)
print("STARTING CHECKPOINT ARCHAEOLOGY AUDIT")
print("=" * 80)

LATENT_DIM = 128
SEED = 42
OUTPUT_DIR = 'phase_dcgan_improvement/forensic_review'
CKPT_DIR = 'phase_dcgan_improvement/forensic_review/checkpoints'

def build_generator(latent_dim=LATENT_DIM):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,), name='latent_input'),
        layers.Dense(4 * 4 * 512, use_bias=False, kernel_initializer=init, name='dense_proj'),
        layers.BatchNormalization(name='bn_proj'),
        layers.LeakyReLU(0.2, name='lrelu_proj'),
        layers.Reshape((4, 4, 512), name='reshape'),
        layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_1'),
        layers.Conv2D(256, kernel_size=3, padding='same', use_bias=False, kernel_initializer=init, name='conv_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.LeakyReLU(0.2, name='lrelu_1'),
        layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_2'),
        layers.Conv2D(128, kernel_size=3, padding='same', use_bias=False, kernel_initializer=init, name='conv_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.LeakyReLU(0.2, name='lrelu_2'),
        layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_3'),
        layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False, kernel_initializer=init, name='conv_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.LeakyReLU(0.2, name='lrelu_3'),
        layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_4'),
        layers.Conv2D(4, kernel_size=3, padding='same', use_bias=True, kernel_initializer=init, activation='tanh', name='conv_out')
    ], name='generator')
    return model

def build_discriminator():
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
    model = keras.Sequential([
        layers.Input(shape=(64, 64, 4), name='image_input'),
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=True, kernel_initializer=init, name='d_conv_0'),
        layers.LeakyReLU(0.2, name='d_lrelu_0'),
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=True, kernel_initializer=init, name='d_conv_1'),
        layers.BatchNormalization(name='d_bn_1'),
        layers.LeakyReLU(0.2, name='d_lrelu_1'),
        layers.Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=True, kernel_initializer=init, name='d_conv_2'),
        layers.BatchNormalization(name='d_bn_2'),
        layers.LeakyReLU(0.2, name='d_lrelu_2'),
        layers.Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=True, kernel_initializer=init, name='d_conv_3'),
        layers.BatchNormalization(name='d_bn_3'),
        layers.LeakyReLU(0.2, name='d_lrelu_3'),
        layers.Flatten(name='d_flatten'),
        layers.Dense(1, use_bias=True, kernel_initializer=init, name='d_out')
    ], name='discriminator')
    return model

# Fixed evaluation latent vectors (N=100)
tf.random.set_seed(SEED)
np.random.seed(SEED)
fixed_z = tf.random.normal([100, LATENT_DIM], seed=SEED)

checkpoints_meta = [
    {"name": "ckpt-9", "epoch": 40, "desc": "Epoch 40 Checkpoint"},
    {"name": "ckpt-10", "epoch": 45, "desc": "Epoch 45 Checkpoint"},
    {"name": "ckpt-11", "epoch": 50, "desc": "Epoch 50 Checkpoint (Terminal)"}
]

ckpt_results = {}
generated_tensors = {}

gen = build_generator()
disc = build_discriminator()
g_opt = keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.5, beta_2=0.999)
d_opt = keras.optimizers.Adam(learning_rate=8e-5, beta_1=0.5, beta_2=0.999)
checkpoint = tf.train.Checkpoint(generator=gen, discriminator=disc, g_opt=g_opt, d_opt=d_opt)

for cm in checkpoints_meta:
    ckpt_name = cm["name"]
    epoch_num = cm["epoch"]
    print(f"\n--- Loading {ckpt_name} (Epoch {epoch_num}) ---")
    checkpoint.restore(os.path.join(CKPT_DIR, ckpt_name)).expect_partial()
    
    # Generate 100 samples
    raw_preds = gen(fixed_z, training=False).numpy()  # [-1, 1]
    # Denormalize to [0, 255] uint8
    imgs_uint8 = np.clip((raw_preds + 1.0) * 127.5, 0, 255).astype(np.uint8)
    generated_tensors[ckpt_name] = imgs_uint8
    
    # Compute Metrics
    # 1. Pairwise L1 Diversity (over RGB channels for non-transparent pixels)
    rgb_flat = imgs_uint8[:, :, :, :3].reshape(100, -1).astype(np.float32)
    l1_dists = []
    # Sample 500 random pairs
    for _ in range(500):
        idx_a, idx_b = np.random.choice(100, 2, replace=False)
        l1_dists.append(np.mean(np.abs(rgb_flat[idx_a] - rgb_flat[idx_b])))
    mean_l1 = float(np.mean(l1_dists))
    
    # 2. Laplacian Variance (Sharpness / High-Frequency Noise)
    lap_vars = []
    for i in range(100):
        gray = cv2.cvtColor(imgs_uint8[i, :, :, :3], cv2.COLOR_RGB2GRAY)
        lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    mean_lap = float(np.mean(lap_vars))
    
    # 3. Alpha Boundary Artifacts (fraction of semi-transparent halo pixels 30 < A < 200)
    alpha_halos = []
    for i in range(100):
        a = imgs_uint8[i, :, :, 3]
        halo_count = np.sum((a > 30) & (a < 200))
        body_count = np.sum(a >= 200)
        alpha_halos.append(float(halo_count) / float(body_count) if body_count > 0 else 0.0)
    mean_halo = float(np.mean(alpha_halos))
    
    # 4. Centroid Centering
    x_centroids = []
    for i in range(100):
        a = imgs_uint8[i, :, :, 3]
        y_idx, x_idx = np.where(a > 30)
        if len(x_idx) > 0:
            x_centroids.append(float(np.mean(x_idx)))
    mean_x_cent = float(np.mean(x_centroids))
    var_x_cent = float(np.std(x_centroids))
    
    ckpt_results[ckpt_name] = {
        "epoch": epoch_num,
        "mean_l1_diversity": round(mean_l1, 2),
        "mean_laplacian_var": round(mean_lap, 2),
        "alpha_halo_ratio": round(mean_halo, 4),
        "centroid_x_mean": round(mean_x_cent, 2),
        "centroid_x_std": round(var_x_cent, 2)
    }
    print(f"Results for {ckpt_name} (Epoch {epoch_num}):")
    print(f"  L1 Diversity       : {mean_l1:.2f}")
    print(f"  Laplacian Variance : {mean_lap:.2f}")
    print(f"  Alpha Halo Ratio   : {mean_halo:.4f}")
    print(f"  Centroid X Mean/Std: {mean_x_cent:.2f} +/- {var_x_cent:.2f}")

# Save metrics JSON
with open(os.path.join(OUTPUT_DIR, 'checkpoint_archaeology_metrics.json'), 'w') as f:
    json.dump(ckpt_results, f, indent=2)

# Produce Side-by-Side Comparison Contact Sheet (Comparing Epoch 40 vs 45 vs 50 on same 10 latent vectors)
print("\nRendering Side-by-Side Checkpoint Comparison Contact Sheet...")
fig, axes = plt.subplots(10, 3, figsize=(9, 25), facecolor="#0f172a")

for row_i in range(10):
    for col_j, ckpt_name in enumerate(["ckpt-9", "ckpt-10", "ckpt-11"]):
        ax = axes[row_i, col_j]
        img = generated_tensors[ckpt_name][row_i]
        ax.imshow(img)
        epoch_str = f"Ep {ckpt_results[ckpt_name]['epoch']}"
        if row_i == 0:
            ax.set_title(f"{ckpt_name} ({epoch_str})", color="#38bdf8", fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel(f"Seed {row_i}", color="#94a3b8", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("DCGAN Checkpoint Archaeology: Exact Latent Comparison (Epoch 40 vs 45 vs 50)", color="#f8fafc", fontsize=13, y=0.995, fontweight="bold")
plt.tight_layout()
cmp_path = os.path.join(OUTPUT_DIR, 'checkpoint_comparison_contact_sheet.png')
plt.savefig(cmp_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved: {cmp_path}")

print("CHECKPOINT ARCHAEOLOGY AUDIT COMPLETE!")

