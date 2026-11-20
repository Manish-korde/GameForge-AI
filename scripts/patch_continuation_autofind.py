import json

nb_path = "Notebook/training/02_DCGAN_Discrete_Continuation_Epochs51_to_100.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell 6
for idx, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "WEIGHT_CANDIDATES_G =" in src:
        cell6_idx = idx
        break

cell6_code = '''# Enhanced DiffAugment Translation (±15% to break front-facing mode collapse)
diff_aug_layer = layers.RandomTranslation(
    height_factor=(-0.15, 0.15),
    width_factor=(-0.15, 0.15),
    fill_mode="constant",
    fill_value=-1.0
)

# Dynamic weight discovery (recursively searches /kaggle/input and current workspace)
WEIGHT_CANDIDATES_G = [
    "models/dcgan_discrete_50ep_results/game_ready_assets/generator_weights_final.weights.h5",
    "game_ready_assets/generator_weights_final.weights.h5",
    f"{ASSETS_DIR}/generator_weights_final.weights.h5",
    "/kaggle/working/generator_weights_final.weights.h5"
]

WEIGHT_CANDIDATES_D = [
    "models/dcgan_discrete_50ep_results/game_ready_assets/discriminator_weights_final.weights.h5",
    "game_ready_assets/discriminator_weights_final.weights.h5",
    f"{ASSETS_DIR}/discriminator_weights_final.weights.h5",
    "/kaggle/working/discriminator_weights_final.weights.h5"
]

# Auto-scan /kaggle/input for any uploaded weights
if os.path.isdir("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            f_lower = f.lower()
            if "generator" in f_lower and f.endswith((".h5", ".weights.h5")):
                full_p = os.path.join(root, f)
                if full_p not in WEIGHT_CANDIDATES_G:
                    WEIGHT_CANDIDATES_G.insert(0, full_p)
            if "discriminator" in f_lower and f.endswith((".h5", ".weights.h5")):
                full_p = os.path.join(root, f)
                if full_p not in WEIGHT_CANDIDATES_D:
                    WEIGHT_CANDIDATES_D.insert(0, full_p)

with strategy.scope():
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Cosine Decay Learning Rate for fine refinement
    g_lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=G_LR_INITIAL,
        decay_steps=ADDITIONAL_EPOCHS * (N_SPRITES // GLOBAL_BATCH_SIZE),
        alpha=0.1
    )
    d_lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=D_LR_INITIAL,
        decay_steps=ADDITIONAL_EPOCHS * (N_SPRITES // GLOBAL_BATCH_SIZE),
        alpha=0.1
    )
    
    g_optimizer = keras.optimizers.Adam(learning_rate=g_lr_schedule, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    d_optimizer = keras.optimizers.Adam(learning_rate=d_lr_schedule, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    
    # Generator Exponential Moving Average (G-EMA)
    ema_generator = build_generator()
    
    g_optimizer.build(generator.trainable_variables)
    d_optimizer.build(discriminator.trainable_variables)

# Load Epoch 50 weights if found
resumed_g = False
for g_path in WEIGHT_CANDIDATES_G:
    if os.path.isfile(g_path):
        try:
            generator.load_weights(g_path)
            ema_generator.load_weights(g_path)
            print(f"SUCCESS: Resumed Generator from Epoch 50 weights: {g_path}")
            resumed_g = True
            break
        except Exception as e:
            print(f"Failed loading G from {g_path}: {e}")

resumed_d = False
for d_path in WEIGHT_CANDIDATES_D:
    if os.path.isfile(d_path):
        try:
            discriminator.load_weights(d_path)
            print(f"SUCCESS: Resumed Discriminator from Epoch 50 weights: {d_path}")
            resumed_d = True
            break
        except Exception as e:
            print(f"Failed loading D from {d_path}: {e}")

if not resumed_g:
    print("NOTE: Epoch 50 weights not found in /kaggle/input. Training DCGAN fresh with Cosine Decay & G-EMA.")
    ema_generator.set_weights(generator.get_weights())

def update_ema():
    for ew, gw in zip(ema_generator.weights, generator.weights):
        ew.assign(EMA_DECAY * ew + (1.0 - EMA_DECAY) * gw)

def train_step_per_replica(real_imgs, tau_val):
    local_batch = tf.shape(real_imgs)[0]
    z = tf.random.normal((local_batch, LATENT_DIM))
    
    # 1. Update Discriminator
    with tf.GradientTape() as d_tape:
        fake_logits = generator(z, training=True)
        fake_imgs = logits_to_image(fake_logits, tau=tau_val, hard=True)
        
        real_aug = diff_aug_layer(real_imgs, training=True)
        fake_aug = diff_aug_layer(fake_imgs, training=True)
        
        d_real = discriminator(real_aug, training=True)
        d_fake = discriminator(fake_aug, training=True)
        d_loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real)) + tf.reduce_mean(tf.nn.relu(1.0 + d_fake))
        
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    # 2. Update Generator
    with tf.GradientTape() as g_tape:
        fake_logits_g = generator(z, training=True)
        fake_imgs_g = logits_to_image(fake_logits_g, tau=tau_val, hard=True)
        fake_aug_g = diff_aug_layer(fake_imgs_g, training=True)
        d_fake_g = discriminator(fake_aug_g, training=True)
        g_loss = -tf.reduce_mean(d_fake_g)
        
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    
    return d_loss, g_loss

@tf.function
def distributed_train_step(dist_real_images, tau_val):
    per_d, per_g = strategy.run(train_step_per_replica, args=(dist_real_images, tau_val))
    d_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_d, axis=None)
    g_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_g, axis=None)
    return d_loss, g_loss

print("Running 1-step warmup across GPU replicas...")
dummy_ds = tf.data.Dataset.from_tensors(
    tf.zeros((GLOBAL_BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=tf.float32)
)
dist_dummy = strategy.experimental_distribute_dataset(dummy_ds)
distributed_train_step(next(iter(dist_dummy)), tf.constant(1.0, dtype=tf.float32))
print("Warmup: SUCCESS! Replicas and optimizer states ready.")
'''

nb["cells"][cell6_idx]["source"] = [line + "\n" for line in cell6_code.split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated Cell 6 in continuation notebook with recursive auto-scanner!")

