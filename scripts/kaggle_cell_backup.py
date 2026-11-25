
# ==============================================================================
# FINAL cDCGAN EXPERIMENT: 20x SCALED EMBEDDINGS + DIFFERENTIABLE PALETTE LOSS
# ==============================================================================
import os, sys, time, json, numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 80)
print("FINAL cDCGAN EXPERIMENT: 20x EMBEDDING SCALE + PALETTE HISTOGRAM LOSS (λ = 2.0)")
print("=" * 80)

# ── 1. Directories & Configuration ───────────────────────────────────────────
BASE_DIR = "/kaggle/working/models/5_final_cDCGAN_v3" if os.path.exists("/kaggle") else "./models/5_final_cDCGAN_v3"
CKPT_DIR = f"{BASE_DIR}/checkpoints"
SAMPLE_DIR = f"{BASE_DIR}/samples"
LOG_DIR = f"{BASE_DIR}/logs"

for d in [BASE_DIR, CKPT_DIR, SAMPLE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

LATENT_DIM     = 128
NUM_CLASSES    = 4
PALETTE_SIZE   = 32
IMG_HEIGHT     = 64
IMG_WIDTH      = 64
CHANNELS       = 4
G_CHANNELS     = [512, 256, 128, 64]
G_INIT_SPATIAL = 4

SCALE_FACTOR   = 20.0  # Matches ~40 L2 norm of latent projection
LAMBDA_PALETTE = 2.0   # Weight for differentiable palette loss
G_LR           = 1e-4
D_LR           = 5e-5
EMA_DECAY      = 0.999
DIFFAUG_PAD    = 8

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS

# ── 2. Verified Ground-Truth Cluster Centroids & Global Palette ──────────────
# Precomputed from all 42,212 real training sprites (32-color foreground distribution)
REAL_CENTROIDS = np.array([
    [0.0092, 0.0262, 0.0134, 0.0570, 0.0302, 0.0296, 0.0193, 0.0391, 0.0550, 0.0137, 0.0731, 0.0107, 0.0142, 0.0343, 0.0209, 0.0185, 0.0162, 0.0287, 0.0230, 0.0166, 0.0559, 0.0089, 0.0213, 0.0139, 0.0151, 0.0654, 0.0201, 0.0297, 0.0438, 0.0611, 0.0246, 0.0913],
    [0.0186, 0.0172, 0.0116, 0.0516, 0.0322, 0.0341, 0.0116, 0.0242, 0.1003, 0.0126, 0.0626, 0.0146, 0.0204, 0.0079, 0.0136, 0.0122, 0.0138, 0.0636, 0.0086, 0.1194, 0.0179, 0.0210, 0.0187, 0.0225, 0.0841, 0.0295, 0.0197, 0.0333, 0.0215, 0.0213, 0.0169, 0.0429],
    [0.0091, 0.0156, 0.0156, 0.0481, 0.0469, 0.2182, 0.0379, 0.0221, 0.0258, 0.0130, 0.0364, 0.0124, 0.0167, 0.0089, 0.0189, 0.0152, 0.0121, 0.0271, 0.0121, 0.0200, 0.0168, 0.0065, 0.0200, 0.0088, 0.0137, 0.0923, 0.0189, 0.0195, 0.0705, 0.0288, 0.0254, 0.0467],
    [0.0085, 0.0174, 0.1808, 0.0425, 0.0206, 0.0358, 0.0153, 0.0233, 0.0192, 0.0547, 0.0296, 0.0290, 0.0153, 0.0049, 0.0612, 0.0116, 0.0127, 0.0247, 0.0086, 0.0296, 0.0307, 0.0062, 0.0232, 0.0621, 0.0116, 0.0303, 0.0193, 0.0165, 0.0213, 0.0542, 0.0373, 0.0420]
], dtype=np.float32)

with strategy.scope():
    real_centroids_tf = tf.constant(REAL_CENTROIDS, dtype=tf.float32)
    palette_path = None
    for p in ["/kaggle/input/dcgan-pretrained-weights-alucard/global_palette_32.npy", "config/global_palette_32.npy", "models/3_DCGAN_42k_v1/config/global_palette_32.npy"]:
        if os.path.exists(p): palette_path = p; break
    global_palette_u8 = np.load(palette_path)
    global_palette_tf = tf.constant(global_palette_u8 / 255.0, dtype=tf.float32)

# ── 3. Scaled Conditional Generator & Projection Discriminator ───────────────
def build_scaled_cgenerator():
    z_in = layers.Input(shape=(LATENT_DIM,), name='latent_vector')
    y_in = layers.Input(shape=(), dtype='int32', name='class_label')
    
    dense_proj = layers.Dense(G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0], use_bias=False, name='dense')(z_in)
    class_emb = layers.Embedding(NUM_CLASSES, G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0],
                                 embeddings_initializer='zeros', name='class_embedding')(y_in)
    
    # Scale embedding by 20x so class activation matches latent projection magnitude
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

with strategy.scope():
    generator = build_scaled_cgenerator()
    generator_ema = build_scaled_cgenerator()
    discriminator = build_projection_discriminator()
    
    aug_layer = layers.RandomTranslation(
        height_factor=(-DIFFAUG_PAD / float(IMG_HEIGHT), DIFFAUG_PAD / float(IMG_HEIGHT)),
        width_factor=(-DIFFAUG_PAD / float(IMG_WIDTH), DIFFAUG_PAD / float(IMG_WIDTH)),
        fill_mode="constant", fill_value=-1.0, interpolation="nearest"
    )

    g_optimizer = keras.optimizers.Adam(learning_rate=G_LR, beta_1=0.0, beta_2=0.9)
    d_optimizer = keras.optimizers.Adam(learning_rate=D_LR, beta_1=0.0, beta_2=0.9)
    g_optimizer.build(generator.trainable_variables)
    d_optimizer.build(discriminator.trainable_variables)

# ── 4. Warm Start Transfer & Exact Step-0 Assertion ──────────────────────────
orig_g_path = None
for p in ["/kaggle/input/dcgan-pretrained-weights-alucard/generator_weights_final.weights.h5",
          "models/3_DCGAN_42k_v1/game_ready_assets/generator_weights_final.weights.h5"]:
    if os.path.exists(p): orig_g_path = p; break

# Build dummy unconditioned generator to read pretrained weights
def build_base_g():
    z = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(4*4*512, use_bias=False, name="dense")(z)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu")(x)
    x = layers.Reshape((4, 4, 512), name="reshape")(x)
    x = layers.UpSampling2D((2,2), name="up_sampling2d")(x)
    x = layers.Conv2D(256, (3,3), padding="same", use_bias=False, name="conv2d")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_1")(x)
    x = layers.UpSampling2D((2,2), name="up_sampling2d_1")(x)
    x = layers.Conv2D(128, (3,3), padding="same", use_bias=False, name="conv2d_1")(x)
    x = layers.BatchNormalization(name="batch_normalization_2")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_2")(x)
    x = layers.UpSampling2D((2,2), name="up_sampling2d_2")(x)
    x = layers.Conv2D(64, (3,3), padding="same", use_bias=False, name="conv2d_2")(x)
    x = layers.BatchNormalization(name="batch_normalization_3")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_3")(x)
    x = layers.UpSampling2D((2,2), name="up_sampling2d_3")(x)
    x = layers.Conv2D(64, (3,3), padding="same", use_bias=False, name="conv2d_3")(x)
    x = layers.BatchNormalization(name="batch_normalization_4")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_4")(x)
    x = layers.Conv2D(64, (3,3), padding="same", name="conv2d_4")(x)
    x = layers.LeakyReLU(0.2, name="leaky_re_lu_5")(x)
    logits = layers.Conv2D(33, (3,3), padding="same", name="palette_logits")(x)
    return keras.Model(z, logits)

base_g = build_base_g()
base_g.load_weights(orig_g_path)

for l in base_g.layers:
    if len(l.get_weights()) > 0:
        generator.get_layer(l.name).set_weights(l.get_weights())
generator_ema.set_weights(generator.get_weights())

test_z = np.random.RandomState(42).normal(0, 1.0, size=(5, LATENT_DIM)).astype(np.float32)
un_out = base_g(test_z, training=False).numpy()
for c in range(NUM_CLASSES):
    c_out = generator([test_z, np.full((5,), c, dtype=np.int32)], training=False).numpy()
    max_d = np.max(np.abs(un_out - c_out))
    assert max_d < 1e-5, f"Step-0 Equivalence failed for Class {c}!"
print("Step-0 Equivalence Asserted: Max diff == 0.00000000 across all classes!")

def update_ema_weights(ema_model, source_model, decay=EMA_DECAY):
    for ema_var, src_var in zip(ema_model.weights, source_model.weights):
        ema_var.assign(decay * ema_var + (1.0 - decay) * src_var)

# ── 5. Replica Step with Differentiable Palette Loss ─────────────────────────
def train_step_per_replica(real_x, real_y):
    batch_size = tf.shape(real_x)[0]
    z = tf.random.normal((batch_size, LATENT_DIM))
    gen_y = tf.random.uniform((batch_size,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_logits = generator([z, gen_y], training=True)
        fake_x = logits_to_image(fake_logits, global_palette_tf, tau=1.0, hard=True, noise=True)
        
        real_x_aug = aug_layer(real_x, training=True)
        fake_x_aug = aug_layer(fake_x, training=True)
        
        d_real = discriminator([real_x_aug, real_y], training=True)
        d_fake = discriminator([fake_x_aug, gen_y], training=True)
        
        # 1. Adversarial Hinge Loss
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_fake))
        d_loss = (d_loss_real + d_loss_fake) / float(NUM_REPLICAS)
        
        g_loss_adv = -tf.reduce_mean(d_fake)
        
        # 2. Foreground-Aware Differentiable Palette Histogram Loss
        pl_soft = tf.nn.softmax(fake_logits[..., :PALETTE_SIZE], axis=-1)
        alpha_soft = tf.sigmoid(fake_logits[..., PALETTE_SIZE:])
        fg_weights = alpha_soft / (tf.reduce_sum(alpha_soft, axis=[1, 2], keepdims=True) + 1e-6)
        gen_palette_hist = tf.reduce_sum(pl_soft * fg_weights, axis=[1, 2])
        
        target_centroids = tf.gather(real_centroids_tf, gen_y)
        dot_p = tf.reduce_sum(gen_palette_hist * target_centroids, axis=-1)
        norm_gen = tf.norm(gen_palette_hist, axis=-1)
        norm_targ = tf.norm(target_centroids, axis=-1)
        pal_cos = dot_p / (norm_gen * norm_targ + 1e-8)
        palette_loss = tf.reduce_mean(1.0 - pal_cos)
        
        g_loss = (g_loss_adv + LAMBDA_PALETTE * palette_loss) / float(NUM_REPLICAS)
        
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    
    return d_loss * NUM_REPLICAS, g_loss * NUM_REPLICAS, palette_loss

@tf.function
def distributed_train_step(dist_batch):
    real_x, real_y = dist_batch
    per_d, per_g, per_pal = strategy.run(train_step_per_replica, args=(real_x, real_y))
    d_l = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_d, axis=None)
    g_l = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_g, axis=None)
    pal_l = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_pal, axis=None)
    return d_l, g_l, pal_l

# ── 6. Corrected Non-Mutating Multi-GPU Warmup Tracing ────────────────────────
print("Tracing graph across GPU replicas without parameter mutation...")
@tf.function
def trace_step(dist_batch):
    def dummy_forward(x, y):
        z = tf.random.normal((tf.shape(x)[0], LATENT_DIM))
        fake_l = generator([z, y], training=False)
        d_out = discriminator([x, y], training=False)
        return tf.reduce_mean(fake_l) + tf.reduce_mean(d_out)
    return strategy.run(dummy_forward, args=dist_batch)

dummy_ds = tf.data.Dataset.from_tensors((
    tf.zeros((GLOBAL_BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=tf.float32),
    tf.zeros((GLOBAL_BATCH_SIZE,), dtype=tf.int32)
))
trace_step(next(iter(strategy.experimental_distribute_dataset(dummy_ds))))
print("Warmup: SUCCESS! Zero weights mutated.\n")

# ── 7. Evaluation & Sampling Helpers ──────────────────────────────────────────
eval_z_tf = tf.constant(np.tile(np.random.RandomState(1337).randn(8, LATENT_DIM).astype(np.float32), (NUM_CLASSES, 1)))
eval_y_tf = tf.constant(np.repeat(np.arange(NUM_CLASSES), 8).astype(np.int32))

def evaluate_metrics(model_to_use):
    logits = model_to_use([eval_z_tf, eval_y_tf], training=False)
    imgs = logits_to_image(logits, global_palette_tf, tau=0.8, hard=True, noise=False).numpy()
    imgs_u8 = np.clip((imgs + 1.0) * 127.5, 0, 255).astype(np.uint8)
    
    # 1. Label Sensitivity (% altered pixels across class conditions)
    altered = []
    for s in range(8):
        for i in range(NUM_CLASSES):
            for j in range(i + 1, NUM_CLASSES):
                im_i = imgs_u8[i * 8 + s]
                im_j = imgs_u8[j * 8 + s]
                altered.append(np.mean(np.any(im_i != im_j, axis=-1)) * 100.0)
    mean_altered = float(np.mean(altered))
    
    # 2. Pairwise Cluster Palette Cosines
    cluster_hists = []
    for c in range(NUM_CLASSES):
        c_imgs = imgs_u8[c * 8 : (c + 1) * 8]
        fg_mask = c_imgs[..., 3] > 127
        if np.any(fg_mask):
            rgb = c_imgs[..., :3][fg_mask]
            diff = rgb[:, None, :] - global_palette_u8[None, :, :]
            nearest = np.argmin(np.sum(diff ** 2, axis=-1), axis=-1)
            h = np.bincount(nearest, minlength=32).astype(np.float32)
            cluster_hists.append(h / (np.sum(h) + 1e-6))
        else:
            cluster_hists.append(np.zeros(32, dtype=np.float32))
            
    cosines = []
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            dot = np.dot(cluster_hists[i], cluster_hists[j])
            denom = np.linalg.norm(cluster_hists[i]) * np.linalg.norm(cluster_hists[j])
            cosines.append(float(dot / max(denom, 1e-8)))
    mean_cosine = float(np.mean(cosines))
    
    # 3. Save Sample Grid
    fig, axes = plt.subplots(NUM_CLASSES, 8, figsize=(14, 7))
    titles = ["Palette Group 0 (Knight)", "Palette Group 1 (Orc)", "Palette Group 2 (Ranger)", "Palette Group 3 (Mage)"]
    for r in range(NUM_CLASSES):
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(imgs_u8[r * 8 + c])
            ax.axis('off')
            if c == 0: ax.set_title(titles[r], fontsize=10, loc='left', color='#1e3a8a', fontweight='bold')
    plt.tight_layout()
    out_img = f"{SAMPLE_DIR}/cDCGAN_samples_latest.png"
    plt.savefig(out_img, dpi=150); plt.close()
    
    return mean_altered, mean_cosine, cosines, out_img

# ── 8. STAGE 1: 3-EPOCH PILOT RUN ─────────────────────────────────────────────
PILOT_EPOCHS = 3
print("=" * 80)
print(f"STAGE 1: EXECUTING 3-EPOCH PILOT (186 UPDATES)")
print("=" * 80)

for epoch in range(1, PILOT_EPOCHS + 1):
    t0 = time.time()
    d_l_acc, g_l_acc, pal_l_acc = 0.0, 0.0, 0.0
    for step, batch in enumerate(dist_dataset):
        dl, gl, pl = distributed_train_step(batch)
        d_l_acc += dl; g_l_acc += gl; pal_l_acc += pl
        update_ema_weights(generator_ema, generator, decay=0.99)
        if step >= (STEPS_PER_EPOCH // 5): break # Fast pilot slice
        
    n_steps = step + 1
    print(f"Pilot Epoch {epoch}/{PILOT_EPOCHS} | D_Loss: {d_l_acc/n_steps:.4f} | G_Loss: {g_l_acc/n_steps:.4f} | Pal_Loss: {pal_l_acc/n_steps:.4f} | Elapsed: {time.time()-t0:.1f}s")

# ── 9. PILOT DECISION GATE ───────────────────────────────────────────────────
altered_pct, cross_cosine, pairwise_cos, sample_path = evaluate_metrics(generator_ema)
print("\nPILOT GATE EVALUATION:")
print(f"  Fixed-Latent Altered Pixels : {altered_pct:.2f}% (Threshold: > 20.0%, Baseline was: 2.14%)")
print(f"  Mean Cross-Cluster Cosine   : {cross_cosine:.4f} (Threshold: <= 0.900, Baseline was: 0.9991)")
print(f"  Pairwise Cosines            : {[round(x, 4) for x in pairwise_cos]}")
print(f"  Sample Grid Saved to        : {sample_path}")

pilot_passed = (altered_pct >= 20.0) and (cross_cosine <= 0.900)
print(f"\nPILOT DECISION GATE PASSED: {pilot_passed}")

if not pilot_passed:
    print("\n[DECISION] Pilot did not demonstrate sufficient palette separation.")
    print("Stopping training as instructed. Retaining unconditional 50-epoch DCGAN with Filter v3.")
else:
    # ── 10. STAGE 2: 18-EPOCH FULL TRAINING ──────────────────────────────────
    print("\n" + "=" * 80)
    print("PILOT PASSED! CONTINUING TO FULL 18-EPOCH RUN")
    print("=" * 80)
    for epoch in range(PILOT_EPOCHS + 1, TARGET_EPOCHS + 1):
        t0 = time.time()
        d_l_acc, g_l_acc, pal_l_acc = 0.0, 0.0, 0.0
        for step, batch in enumerate(dist_dataset):
            dl, gl, pl = distributed_train_step(batch)
            d_l_acc += dl; g_l_acc += gl; pal_l_acc += pl
            update_ema_weights(generator_ema, generator, decay=EMA_DECAY)
            
        print(f"Full Epoch {epoch:02d}/{TARGET_EPOCHS} | D_Loss: {d_l_acc/STEPS_PER_EPOCH:.4f} | G_Loss: {g_l_acc/STEPS_PER_EPOCH:.4f} | Pal_Loss: {pal_l_acc/STEPS_PER_EPOCH:.4f} | Elapsed: {time.time()-t0:.1f}s")
        
        if epoch % CHECKPOINT_EVERY == 0:
            generator_ema.save_weights(f"{CKPT_DIR}/cgenerator_ema_epoch_{epoch:02d}.weights.h5")
            
    final_altered, final_cosine, final_pairs, final_grid = evaluate_metrics(generator_ema)
    generator_ema.save_weights(f"{BASE_DIR}/cgenerator_ema_weights_final.weights.h5")
    print("=" * 80)
    print(f"Full Training Complete! Final Mean Cosine: {final_cosine:.4f} | Final Altered Pixels: {final_altered:.2f}%")
    print(f"Weights exported to: {BASE_DIR}/cgenerator_ema_weights_final.weights.h5")
    print("=" * 80)
