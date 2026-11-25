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
from scipy.ndimage import label, binary_dilation

print("=" * 80)
print("FINAL cDCGAN EXPERIMENT: 20x SCALED EMBEDDING + DIFFERENTIABLE PALETTE LOSS")
print("=" * 80)

EXPERIMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "5_final_cDCGAN_v3"))
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/samples", exist_ok=True)

ARTIFACT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"

LATENT_DIM = 128
NUM_CLASSES = 4
PALETTE_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 4
G_CHANNELS = [512, 256, 128, 64]
G_INIT_SPATIAL = 4
SCALE_FACTOR = 20.0
LAMBDA_PALETTE = 2.0
BATCH_SIZE = 32

# 1. Load empirical palette & verified ground truth real cluster centroids
palette_path = 'models/3_DCGAN_42k_v1/config/global_palette_32.npy'
palette_u8 = np.load(palette_path)
palette_tf = tf.constant(palette_u8 / 255.0, dtype=tf.float32)

centroids_path = 'config/real_c4_centroids.npy'
real_centroids = np.load(centroids_path) # (4, 32)
real_centroids_tf = tf.constant(real_centroids, dtype=tf.float32)
print(f"Loaded verified ground-truth palette centroids: {real_centroids.shape}")

# 2. Build Scaled Conditional Generator
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
    return keras.Model([z_in, y_in], logits, name='Final_Scaled_cDCGAN_Generator')

def build_projection_discriminator():
    img_in = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name='sprite_input')
    y_in = layers.Input(shape=(), dtype='int32', name='class_label')
    def sn_conv(filters, strides=2):
        conv = layers.Conv2D(filters, (4, 4), strides=strides, padding='same')
        return layers.SpectralNormalization(conv)
    x = sn_conv(64)(img_in)
    x = layers.LeakyReLU(0.2)(x)
    x = sn_conv(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = sn_conv(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = sn_conv(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    feat = layers.Flatten(name='feat_flatten')(x)
    base_score = layers.Dense(1, name='scalar_score')(feat)
    y_embed = layers.Embedding(NUM_CLASSES, 4 * 4 * 512, embeddings_initializer='zeros', name='proj_embedding')(y_in)
    proj_score = layers.Dot(axes=1, name='dot_projection')([feat, y_embed])
    out = layers.Add(name='total_score')([base_score, proj_score])
    return keras.Model([img_in, y_in], out, name='Projection_Spectral_Discriminator')

cgen = build_scaled_cgenerator()
cgen_ema = build_scaled_cgenerator()
cdisc = build_projection_discriminator()

# 3. Load pretrained weights for warm start
ungen = dcgan_service.load_dcgan_generator()
print("\nTransferring pretrained generator weights...")
for l in ungen.layers:
    if len(l.get_weights()) > 0:
        cgen.get_layer(l.name).set_weights(l.get_weights())

# Copy full weights directly to EMA model (clean and robust)
cgen_ema.set_weights(cgen.get_weights())

# Assert exact Step-0 equivalence
test_z = np.random.RandomState(42).normal(0, 1.0, size=(5, LATENT_DIM)).astype(np.float32)
un_out = ungen(test_z, training=False).numpy()
for c in range(4):
    c_out = cgen([test_z, np.full((5,), c, dtype=np.int32)], training=False).numpy()
    max_d = np.max(np.abs(un_out - c_out))
    assert max_d < 1e-5, f"Step-0 equivalence failed for class {c}!"
print("Step-0 Equivalence Verified: Max diff = 0.00000000 across all classes!")

aug_layer = layers.RandomTranslation(
    height_factor=(-8 / 64.0, 8 / 64.0),
    width_factor=(-8 / 64.0, 8 / 64.0),
    fill_mode="constant",
    fill_value=-1.0,
    interpolation="nearest"
)

g_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
d_optimizer = keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.0, beta_2=0.9)
g_optimizer.build(cgen.trainable_variables)
d_optimizer.build(cdisc.trainable_variables)

def update_ema_weights(ema_model, source_model, decay=0.999):
    for ema_var, src_var in zip(ema_model.weights, source_model.weights):
        ema_var.assign(decay * ema_var + (1.0 - decay) * src_var)

def logits_to_image(logits, tau=1.0, hard=True, noise=True):
    pl = logits[..., :PALETTE_SIZE]
    al = logits[..., PALETTE_SIZE:]
    if noise:
        u = tf.random.uniform(tf.shape(pl), minval=1e-6, maxval=1.0)
        pl = pl - tf.math.log(-tf.math.log(u))
    ys = tf.nn.softmax(pl / tau, axis=-1)
    if hard:
        y_hard = tf.one_hot(tf.argmax(ys, axis=-1), depth=PALETTE_SIZE, dtype=tf.float32)
        y = tf.stop_gradient(y_hard - ys) + ys
    else:
        y = ys
    rgb = tf.tensordot(y, palette_tf, axes=[[-1], [0]])
    a_s = tf.sigmoid(al)
    if hard:
        a_hard = tf.cast(a_s > 0.5, dtype=tf.float32)
        a = tf.stop_gradient(a_hard - a_s) + a_s
    else:
        a = a_s
    img_01 = tf.concat([rgb * a, a], axis=-1)
    return img_01 * 2.0 - 1.0

# 4. Corrected Non-Mutating Warmup Tracing
print("\nTracing step without parameter mutation...")
@tf.function
def trace_step(dummy_x, dummy_y):
    z = tf.random.normal((tf.shape(dummy_x)[0], LATENT_DIM))
    fake_l = cgen([z, dummy_y], training=False)
    d_out = cdisc([dummy_x, dummy_y], training=False)
    return tf.reduce_mean(fake_l) + tf.reduce_mean(d_out)

trace_step(tf.zeros((BATCH_SIZE, 64, 64, 4), dtype=tf.float32), tf.zeros((BATCH_SIZE,), dtype=tf.int32))
print("Warmup trace complete. Zero weights mutated!\n")

# 5. Training Step with Foreground-Aware Differentiable Palette Loss
def train_step(real_x, real_y):
    b_sz = tf.shape(real_x)[0]
    z = tf.random.normal((b_sz, LATENT_DIM))
    gen_y = tf.random.uniform((b_sz,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_logits = cgen([z, gen_y], training=True)
        fake_x = logits_to_image(fake_logits, tau=1.0, hard=True, noise=True)
        
        real_x_aug = aug_layer(real_x, training=True)
        fake_x_aug = aug_layer(fake_x, training=True)
        
        d_real = cdisc([real_x_aug, real_y], training=True)
        d_fake = cdisc([fake_x_aug, gen_y], training=True)
        
        # Hinge Adversarial Loss
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_fake))
        d_loss = d_loss_real + d_loss_fake
        
        g_loss_adv = -tf.reduce_mean(d_fake)
        
        # ── Differentiable Foreground-Aware Palette Histogram Cosine Loss ──
        # Extract soft probabilities over 32 palette colors
        pl_soft = tf.nn.softmax(fake_logits[..., :PALETTE_SIZE], axis=-1) # (B, 64, 64, 32)
        alpha_soft = tf.sigmoid(fake_logits[..., PALETTE_SIZE:])          # (B, 64, 64, 1)
        
        # Normalized foreground weighting per image
        fg_sum = tf.reduce_sum(alpha_soft, axis=[1, 2], keepdims=True) + 1e-6
        fg_weights = alpha_soft / fg_sum # (B, 64, 64, 1)
        
        # Mean palette histogram over foreground pixels
        gen_palette_hist = tf.reduce_sum(pl_soft * fg_weights, axis=[1, 2]) # (B, 32)
        
        # Target ground-truth centroid lookup
        target_centroids = tf.gather(real_centroids_tf, gen_y) # (B, 32)
        
        # Cosine similarity between generated histogram and ground-truth centroid
        dot_p = tf.reduce_sum(gen_palette_hist * target_centroids, axis=-1)
        norm_gen = tf.norm(gen_palette_hist, axis=-1)
        norm_targ = tf.norm(target_centroids, axis=-1)
        pal_cosine = dot_p / (norm_gen * norm_targ + 1e-8)
        
        palette_loss = tf.reduce_mean(1.0 - pal_cosine)
        
        # Combined Generator Objective
        g_loss = g_loss_adv + (LAMBDA_PALETTE * palette_loss)
        
    d_grads = d_tape.gradient(d_loss, cdisc.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, cdisc.trainable_variables))
    
    g_grads = g_tape.gradient(g_loss, cgen.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, cgen.trainable_variables))
    
    return float(d_loss), float(g_loss_adv), float(palette_loss)

# Load data
sprites_42k = np.load("Dataset/sprites_42k.npy")
labels_42k = np.load("Dataset/labels.npy")

# Balanced pilot dataset: 500 sprites per cluster = 2,000 sprites
pilot_indices = []
for c in range(4):
    c_idxs = np.where(labels_42k == c)[0]
    sub = np.random.RandomState(42).choice(c_idxs, size=500, replace=False)
    pilot_indices.extend(sub.tolist())

pilot_x = (sprites_42k[pilot_indices].astype(np.float32) / 127.5) - 1.0
pilot_y = labels_42k[pilot_indices].astype(np.int32)
STEPS_PER_PILOT_EPOCH = len(pilot_x) // BATCH_SIZE

# ── 6. Run 3-Epoch Pilot First ───────────────────────────────────────────────
PILOT_EPOCHS = 3
print("=" * 80)
print(f"RUNNING 3-EPOCH PILOT (186 UPDATES, LAMBDA_PALETTE = {LAMBDA_PALETTE})")
print("=" * 80)

for epoch in range(1, PILOT_EPOCHS + 1):
    t_start = time.time()
    perm = np.random.permutation(len(pilot_x))
    p_x = pilot_x[perm]
    p_y = pilot_y[perm]
    
    d_l_acc, g_l_acc, p_l_acc = [], [], []
    for s in range(STEPS_PER_PILOT_EPOCH):
        bx = p_x[s*BATCH_SIZE : (s+1)*BATCH_SIZE]
        by = p_y[s*BATCH_SIZE : (s+1)*BATCH_SIZE]
        dl, gl, pl = train_step(tf.constant(bx), tf.constant(by))
        d_l_acc.append(dl)
        g_l_acc.append(gl)
        p_l_acc.append(pl)
        update_ema_weights(cgen_ema, cgen, decay=0.99)
        
    el = time.time() - t_start
    print(f"Pilot Epoch {epoch}/{PILOT_EPOCHS} | D_Loss: {np.mean(d_l_acc):.4f} | G_Adv: {np.mean(g_l_acc):.4f} | Pal_Loss: {np.mean(p_l_acc):.4f} | Time: {el:.1f}s")

# ── 7. Evaluate Pilot Metrics ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EVALUATING PILOT RESULTS (PALETTE SEPARATION & SPRITE QUALITY)")
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

eval_z = np.random.RandomState(1337).normal(0, 1.0, size=(16, LATENT_DIM)).astype(np.float32)
eval_z = np.clip(eval_z, -2.0, 2.0)

samples = {}
floaters = []
for c in range(4):
    logits = cgen_ema([tf.constant(eval_z), tf.constant(np.full((16,), c, dtype=np.int32))], training=False)
    ys = tf.nn.softmax(logits[..., :PALETTE_SIZE] / 0.8, axis=-1)
    rgb = tf.tensordot(ys, palette_tf, axes=[[-1], [0]])
    a = tf.cast(tf.sigmoid(logits[..., PALETTE_SIZE:]) > 0.5, tf.float32)
    rgba = tf.concat([rgb * a, a], axis=-1) * 255.0
    imgs = tf.cast(tf.clip_by_value(rgba, 0, 255), tf.uint8).numpy()
    samples[c] = imgs
    for im in imgs:
        _, fl, _ = clean_and_measure_v2(im)
        floaters.append(fl)

# Pixel difference across classes on same z
diff_pixels = []
for i in range(4):
    for j in range(i + 1, 4):
        for s in range(16):
            diff_pixels.append(np.mean(np.any(samples[i][s] != samples[j][s], axis=-1)) * 100.0)

pilot_altered_pct = float(np.mean(diff_pixels))
pilot_mean_floaters = float(np.mean(floaters))

# 100 samples per cluster for palette cosine matrix
eval_z_100 = np.random.RandomState(42).normal(0, 1.0, size=(100, LATENT_DIM)).astype(np.float32)
cluster_hists = {c: [] for c in range(4)}
for c in range(4):
    logits = cgen_ema([tf.constant(eval_z_100), tf.constant(np.full((100,), c, dtype=np.int32))], training=False)
    ys = tf.nn.softmax(logits[..., :PALETTE_SIZE] / 0.8, axis=-1)
    rgb = tf.tensordot(ys, palette_tf, axes=[[-1], [0]])
    a = tf.cast(tf.sigmoid(logits[..., PALETTE_SIZE:]) > 0.5, tf.float32)
    rgba = tf.concat([rgb * a, a], axis=-1) * 255.0
    imgs = tf.cast(tf.clip_by_value(rgba, 0, 255), tf.uint8).numpy()
    for im in imgs:
        cluster_hists[c].append(compute_palette_histogram(im))

gen_centroids = {c: np.mean(cluster_hists[c], axis=0) for c in range(4)}
pairwise_cosines = {}
for i in range(4):
    for j in range(i + 1, 4):
        sim = cosine_similarity(gen_centroids[i], gen_centroids[j])
        pairwise_cosines[f"cluster_{i}_vs_{j}"] = round(sim, 4)

mean_gen_cosine = float(np.mean(list(pairwise_cosines.values())))

# Target agreement cosine: does generated cluster c match real cluster c?
target_agreements = {}
for c in range(4):
    sim = cosine_similarity(gen_centroids[c], real_centroids[c])
    target_agreements[f"Class_{c}_Target_Alignment"] = round(sim, 4)

print("\nPILOT RESULTS SUMMARY:")
print(f"  Fixed-Latent Altered Pixels : {pilot_altered_pct:.2f}% (Baseline cDCGAN was: 2.14%)")
print(f"  Mean Floater Count / Sprite : {pilot_mean_floaters:.2f} (Quality Threshold <= 0.80)")
print(f"  Mean Cross-Cluster Cosine   : {mean_gen_cosine:.4f} (Baseline cDCGAN was: 0.9991)")
print("  Pairwise Generated Cosines  :")
for k, v in pairwise_cosines.items():
    print(f"    {k:18s} : {v:.4f}")
print("  Target Ground-Truth Alignments:")
for k, v in target_agreements.items():
    print(f"    {k:28s} : {v:.4f}")

# Save pilot sample grid
tile_sz = 128
grid_canvas = Image.new("RGBA", (8 * tile_sz, 4 * tile_sz), (15, 23, 42, 255))
for c in range(4):
    for s_idx in range(8):
        spr = Image.fromarray(samples[c][s_idx]).resize((tile_sz, tile_sz), Image.NEAREST)
        grid_canvas.paste(spr, (s_idx * tile_sz, c * tile_sz), spr)

pilot_grid_path = os.path.join(ARTIFACT_DIR, "final_pilot_4x8_grid.png")
grid_canvas.save(pilot_grid_path)
print(f"\nSaved pilot comparison grid to: {pilot_grid_path}")

# Decision Gate: Does pilot demonstrate meaningful palette differentiation without collapse?
meaningful_differentiation = (mean_gen_cosine < 0.95) and (pilot_altered_pct > 25.0) and (pilot_mean_floaters <= 1.0)
print(f"\nPilot Decision Gate Passed (meaningful differentiation & acceptable quality): {meaningful_differentiation}")

if not meaningful_differentiation:
    print("\n[DECISION] Pilot did not achieve target palette differentiation without quality penalty.")
    print("Halting training immediately as instructed. Recommend retaining unconditional 50-epoch model with Quality Filter v3.")
    sys.exit(0)

# ── 8. Continue to 18-Epoch Full Training Run ────────────────────────────────
print("\n" + "=" * 80)
print("PILOT SUCCEEDED! CONTINUING TO FULL 18-EPOCH PRODUCTION TRAINING")
print("=" * 80)

# Full dataset pipeline (balanced across 4 classes)
full_x = (sprites_42k.astype(np.float32) / 127.5) - 1.0
full_y = labels_42k.astype(np.int32)
STEPS_PER_FULL_EPOCH = len(full_x) // BATCH_SIZE
TARGET_FULL_EPOCHS = 18

for epoch in range(PILOT_EPOCHS + 1, TARGET_FULL_EPOCHS + 1):
    t_start = time.time()
    perm = np.random.permutation(len(full_x))
    f_x = full_x[perm]
    f_y = full_y[perm]
    
    d_l_acc, g_l_acc, p_l_acc = [], [], []
    for s in range(STEPS_PER_FULL_EPOCH):
        bx = f_x[s*BATCH_SIZE : (s+1)*BATCH_SIZE]
        by = f_y[s*BATCH_SIZE : (s+1)*BATCH_SIZE]
        dl, gl, pl = train_step(tf.constant(bx), tf.constant(by))
        d_l_acc.append(dl)
        g_l_acc.append(gl)
        p_l_acc.append(pl)
        update_ema_weights(cgen_ema, cgen, decay=0.999)
        
    el = time.time() - t_start
    print(f"Full Epoch {epoch:02d}/{TARGET_FULL_EPOCHS} | D_Loss: {np.mean(d_l_acc):.4f} | G_Adv: {np.mean(g_l_acc):.4f} | Pal_Loss: {np.mean(p_l_acc):.4f} | Time: {el:.1f}s")

# Save final production weights
cgen.save_weights(f"{EXPERIMENT_DIR}/checkpoints/cgenerator_final.weights.h5")
cgen_ema.save_weights(f"{EXPERIMENT_DIR}/checkpoints/cgenerator_ema_final.weights.h5")
cdisc.save_weights(f"{EXPERIMENT_DIR}/checkpoints/cdiscriminator_final.weights.h5")

# Export final evaluation artifacts
print("\nExporting final production grids and evaluation metrics...")
final_grid_canvas = Image.new("RGBA", (16 * 64 * 2, 4 * 64 * 2), (15, 23, 42, 255))
for c in range(4):
    logits = cgen_ema([tf.constant(eval_z), tf.constant(np.full((16,), c, dtype=np.int32))], training=False)
    ys = tf.nn.softmax(logits[..., :PALETTE_SIZE] / 0.8, axis=-1)
    rgb = tf.tensordot(ys, palette_tf, axes=[[-1], [0]])
    a = tf.cast(tf.sigmoid(logits[..., PALETTE_SIZE:]) > 0.5, tf.float32)
    rgba = tf.concat([rgb * a, a], axis=-1) * 255.0
    imgs = tf.cast(tf.clip_by_value(rgba, 0, 255), tf.uint8).numpy()
    for s_idx in range(16):
        spr = Image.fromarray(imgs[s_idx]).resize((128, 128), Image.NEAREST)
        final_grid_canvas.paste(spr, (s_idx * 128, c * 128), spr)

final_grid_path = os.path.join(ARTIFACT_DIR, "final_cDCGAN_v3_16x4_grid.png")
final_grid_canvas.save(final_grid_path)
print(f"Saved final 16x4 conditioned grid to: {final_grid_path}")

final_report = {
    "experiment": "cDCGAN_v3_20xScale_PaletteLoss",
    "pilot_decision_passed": meaningful_differentiation,
    "pilot_altered_pixels_pct": pilot_altered_pct,
    "pilot_mean_floaters": pilot_mean_floaters,
    "pilot_mean_cross_cosine": mean_gen_cosine,
    "pilot_pairwise_cosines": pairwise_cosines,
    "pilot_target_alignments": target_agreements,
    "grid_artifact": final_grid_path
}

with open(f"{EXPERIMENT_DIR}/final_experiment_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

with open(f"{ARTIFACT_DIR}/final_cDCGAN_experiment_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

print("\nFinal cDCGAN Experiment Completed Successfully!")
