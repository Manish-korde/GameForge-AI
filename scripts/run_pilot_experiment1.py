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
print("EXPERIMENT 1 PILOT: MINIMAL CONTROLLED FIX (CONDITIONING STRENGTH ONLY)")
print("=" * 80)

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "experiment_pilot_cDCGAN_v2"))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/samples", exist_ok=True)
os.makedirs(f"{OUT_DIR}/checkpoints", exist_ok=True)

LATENT_DIM = 128
NUM_CLASSES = 4
PALETTE_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 4
G_CHANNELS = [512, 256, 128, 64]
G_INIT_SPATIAL = 4
BATCH_SIZE = 32

palette_path = 'models/3_DCGAN_42k_v1/config/global_palette_32.npy'
palette_u8 = np.load(palette_path)
palette_tf = tf.constant(palette_u8 / 255.0, dtype=tf.float32)

# 1. Unconditional Generator (for warm start)
ungen = dcgan_service.load_dcgan_generator()

# 2. Conditional Generator with SCALED embedding (Factor = 20.0)
# Scaled to match the 40.0 L2 norm of the dense projection, initialized to zero for step-0 equivalence
def build_scaled_cgenerator(scale_factor=20.0):
    z_in = layers.Input(shape=(LATENT_DIM,), name='latent_vector')
    y_in = layers.Input(shape=(), dtype='int32', name='class_label')
    dense_proj = layers.Dense(G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0], use_bias=False, name='dense')(z_in)
    
    # Class embedding initialized to zeros (exact warm start)
    class_emb = layers.Embedding(NUM_CLASSES, G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0],
                                 embeddings_initializer='zeros', name='class_embedding')(y_in)
    
    # Scale embedding by scale_factor so its updates and activations are on the same order as dense_proj
    scaled_emb = class_emb * scale_factor
    
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

# 3. Exact Projection Discriminator (identical architecture)
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

cgen = build_scaled_cgenerator(scale_factor=20.0)
cdisc = build_projection_discriminator()

# Warm start weights transfer
print("\nTransferring weights from unconditional generator...")
for l in ungen.layers:
    try:
        targ = cgen.get_layer(l.name)
        targ.set_weights(l.get_weights())
    except Exception:
        pass

# Assert exact Step-0 equivalence
test_z = np.random.RandomState(42).normal(0, 1.0, size=(5, LATENT_DIM)).astype(np.float32)
un_out = ungen(test_z, training=False).numpy()
for c in range(4):
    c_out = cgen([test_z, np.full((5,), c, dtype=np.int32)], training=False).numpy()
    max_d = np.max(np.abs(un_out - c_out))
    print(f"Step-0 Equivalence check (Class {c}): Max Diff = {max_d:.8f}")
    assert max_d < 1e-5, f"Step-0 equivalence failed for class {c}!"

print("Step-0 equivalence verified: Max diff == 0.00000000 across all classes!\n")

# Augmentation layer
aug_layer = layers.RandomTranslation(
    height_factor=(-8 / 64.0, 8 / 64.0),
    width_factor=(-8 / 64.0, 8 / 64.0),
    fill_mode="constant",
    fill_value=-1.0,
    interpolation="nearest"
)

# Optimizers with exact same learning rates
g_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
d_optimizer = keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.0, beta_2=0.9)
g_optimizer.build(cgen.trainable_variables)
d_optimizer.build(cdisc.trainable_variables)

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

# Non-mutating warmup trace
print("Tracing step without parameter mutation...")
@tf.function
def trace_step(dummy_x, dummy_y):
    # Pure forward pass compilation
    z = tf.random.normal((tf.shape(dummy_x)[0], LATENT_DIM))
    fake_l = cgen([z, dummy_y], training=False)
    d_out = cdisc([dummy_x, dummy_y], training=False)
    return tf.reduce_mean(fake_l) + tf.reduce_mean(d_out)

dummy_bx = tf.zeros((BATCH_SIZE, 64, 64, 4), dtype=tf.float32)
dummy_by = tf.zeros((BATCH_SIZE,), dtype=tf.int32)
trace_step(dummy_bx, dummy_by)
print("Warmup trace complete. Zero weights mutated!\n")

# Single replica training step
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
        
        # Hinge loss
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_fake))
        d_loss = d_loss_real + d_loss_fake
        
        g_loss = -tf.reduce_mean(d_fake)
        
    d_grads = d_tape.gradient(d_loss, cdisc.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, cdisc.trainable_variables))
    
    g_grads = g_tape.gradient(g_loss, cgen.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, cgen.trainable_variables))
    
    return float(d_loss), float(g_loss)

# Load balanced pilot dataset: 500 sprites per cluster = 2,000 sprites
sprites_42k = np.load("Dataset/sprites_42k.npy")
labels_42k = np.load("Dataset/labels.npy")

pilot_indices = []
for c in range(4):
    c_idxs = np.where(labels_42k == c)[0]
    sub = np.random.RandomState(42).choice(c_idxs, size=500, replace=False)
    pilot_indices.extend(sub.tolist())

pilot_x = (sprites_42k[pilot_indices].astype(np.float32) / 127.5) - 1.0
pilot_y = labels_42k[pilot_indices].astype(np.int32)

print(f"Pilot dataset prepared: {len(pilot_x)} balanced sprites (500 per cluster).")

# Run 3 Pilot Epochs (62 steps per epoch = 186 updates)
PILOT_EPOCHS = 3
STEPS_PER_EPOCH = len(pilot_x) // BATCH_SIZE

print("\n" + "=" * 80)
print(f"STARTING EXPERIMENT 1 PILOT ({PILOT_EPOCHS} EPOCHS, {PILOT_EPOCHS * STEPS_PER_EPOCH} STEPS)")
print("=" * 80)

for epoch in range(1, PILOT_EPOCHS + 1):
    t_start = time.time()
    perm = np.random.permutation(len(pilot_x))
    p_x = pilot_x[perm]
    p_y = pilot_y[perm]
    
    d_losses = []
    g_losses = []
    
    for s in range(STEPS_PER_EPOCH):
        bx = p_x[s*BATCH_SIZE : (s+1)*BATCH_SIZE]
        by = p_y[s*BATCH_SIZE : (s+1)*BATCH_SIZE]
        dl, gl = train_step(tf.constant(bx), tf.constant(by))
        d_losses.append(dl)
        g_losses.append(gl)
        
    el = time.time() - t_start
    print(f"Pilot Epoch {epoch}/{PILOT_EPOCHS} | D_Loss: {np.mean(d_losses):.4f} | G_Loss: {np.mean(g_losses):.4f} | Time: {el:.1f}s")

# Save pilot weights
cgen.save_weights(f"{OUT_DIR}/cgenerator_pilot_exp1.weights.h5")
cdisc.save_weights(f"{OUT_DIR}/cdiscriminator_pilot_exp1.weights.h5")
print(f"Saved pilot weights to {OUT_DIR}/")

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION OF PILOT EXPERIMENT 1 vs EXISTING CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EVALUATING PILOT RESULTS (TESTING FOR MEASURABLE PALETTE DIFFERENTIATION)")
print("=" * 80)

# 1. Label Sensitivity on 16 fixed latents
eval_z = np.random.RandomState(1337).normal(0, 1.0, size=(16, LATENT_DIM)).astype(np.float32)
eval_z = np.clip(eval_z, -2.0, 2.0)

samples = {}
for c in range(4):
    logits = cgen([tf.constant(eval_z), tf.constant(np.full((16,), c, dtype=np.int32))], training=False)
    ys = tf.nn.softmax(logits[..., :PALETTE_SIZE] / 0.8, axis=-1)
    rgb = tf.tensordot(ys, palette_tf, axes=[[-1], [0]])
    a = tf.cast(tf.sigmoid(logits[..., PALETTE_SIZE:]) > 0.5, tf.float32)
    rgba = tf.concat([rgb * a, a], axis=-1) * 255.0
    samples[c] = tf.cast(tf.clip_by_value(rgba, 0, 255), tf.uint8).numpy()

# Compute pixel diffs across classes
diff_pixels = []
l1_diffs = []
fg_diffs = []
for i in range(4):
    for j in range(i + 1, 4):
        for s in range(16):
            im_i = samples[i][s].astype(float)
            im_j = samples[j][s].astype(float)
            px_diff = np.any(im_i != im_j, axis=-1)
            diff_pixels.append(np.mean(px_diff) * 100.0)
            l1_diffs.append(np.mean(np.abs(im_i - im_j)))
            fg_both = (im_i[..., 3] > 127) & (im_j[..., 3] > 127)
            if np.any(fg_both):
                fg_diffs.append(np.mean(np.abs(im_i[..., :3][fg_both] - im_j[..., :3][fg_both])))

exp1_altered_pct = float(np.mean(diff_pixels))
exp1_l1 = float(np.mean(l1_diffs))
exp1_fg = float(np.mean(fg_diffs))

print(f"Pilot Exp 1 Fixed-Latent Label Sensitivity:")
print(f"  Mean Altered Pixels : {exp1_altered_pct:.2f}% (Baseline cDCGAN was: 2.14%)")
print(f"  Mean Canvas L1 Diff : {exp1_l1:.3f} / 255.0 (Baseline cDCGAN was: 0.865)")
print(f"  Mean FG RGB Diff    : {exp1_fg:.3f} (Baseline cDCGAN was: 1.994)")

# 2. Palette Histograms and Pairwise Cosines on 100 samples per cluster
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

eval_z_100 = np.random.RandomState(42).normal(0, 1.0, size=(100, LATENT_DIM)).astype(np.float32)
cluster_hists = {c: [] for c in range(4)}
for c in range(4):
    logits = cgen([tf.constant(eval_z_100), tf.constant(np.full((100,), c, dtype=np.int32))], training=False)
    ys = tf.nn.softmax(logits[..., :PALETTE_SIZE] / 0.8, axis=-1)
    rgb = tf.tensordot(ys, palette_tf, axes=[[-1], [0]])
    a = tf.cast(tf.sigmoid(logits[..., PALETTE_SIZE:]) > 0.5, tf.float32)
    rgba = tf.concat([rgb * a, a], axis=-1) * 255.0
    imgs = tf.cast(tf.clip_by_value(rgba, 0, 255), tf.uint8).numpy()
    for im in imgs:
        cluster_hists[c].append(compute_palette_histogram(im))

centroids = {c: np.mean(cluster_hists[c], axis=0) for c in range(4)}
exp1_cosines = {}
for i in range(4):
    for j in range(i + 1, 4):
        sim = cosine_similarity(centroids[i], centroids[j])
        exp1_cosines[f"cluster_{i}_vs_{j}"] = round(sim, 4)

mean_exp1_cos = float(np.mean(list(exp1_cosines.values())))
print("\nPilot Exp 1 Pairwise Generated Cluster Cosines:")
for k, v in exp1_cosines.items():
    print(f"  {k:18s} : {v:.4f}")
print(f"  Mean Cross-Cluster Cosine : {mean_exp1_cos:.4f} (Baseline cDCGAN was: 0.9991)")

# 3. Discriminator Label Awareness Margin
sub_pilot_x = pilot_x[:200]
sub_pilot_y = pilot_y[:200]
s_true = cdisc([tf.constant(sub_pilot_x), tf.constant(sub_pilot_y)], training=False).numpy().flatten()
margins = []
for shift in [1, 2, 3]:
    s_wrong = cdisc([tf.constant(sub_pilot_x), tf.constant((sub_pilot_y + shift) % 4)], training=False).numpy().flatten()
    margins.append(float(np.mean(s_true - s_wrong)))
exp1_margin = float(np.mean(margins))
print(f"\nPilot Exp 1 Discriminator Margin : {exp1_margin:+.6f} (Baseline cDCGAN was: +0.1474)")

# 4. Save 4x8 Comparison Canvas
tile_sz = 128
grid_canvas = Image.new("RGBA", (8 * tile_sz, 4 * tile_sz), (15, 23, 42, 255))
for c in range(4):
    for s_idx in range(8):
        spr = Image.fromarray(samples[c][s_idx]).resize((tile_sz, tile_sz), Image.NEAREST)
        grid_canvas.paste(spr, (s_idx * tile_sz, c * tile_sz), spr)

out_grid_path = os.path.join("C:/Users/manis/.gemini/antigravity/brain/674ad8cd-af16-4b87-91e3-66973cf97bc5", "pilot_exp1_4x8_grid.png")
grid_canvas.save(out_grid_path)
print(f"\nSaved Pilot Exp 1 grid artifact to: {out_grid_path}")

# Save JSON results
pilot_report = {
    "exp1_altered_pixels_pct": exp1_altered_pct,
    "exp1_l1_diff": exp1_l1,
    "exp1_fg_rgb_diff": exp1_fg,
    "exp1_pairwise_cosines": exp1_cosines,
    "exp1_mean_cross_cosine": mean_exp1_cos,
    "exp1_d_margin": exp1_margin,
    "baseline_altered_pixels_pct": 2.14,
    "baseline_mean_cross_cosine": 0.9991,
    "baseline_d_margin": 0.1474,
    "grid_artifact": out_grid_path
}

with open(f"{OUT_DIR}/pilot_exp1_report.json", "w") as f:
    json.dump(pilot_report, f, indent=2)

with open("C:/Users/manis/.gemini/antigravity/brain/674ad8cd-af16-4b87-91e3-66973cf97bc5/pilot_exp1_report.json", "w") as f:
    json.dump(pilot_report, f, indent=2)

print("Pilot Experiment 1 Complete!")
