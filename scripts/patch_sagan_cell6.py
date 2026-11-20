import json

nb_path = "Notebook/training/03_Discrete_SAGAN_Production.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell6_code = '''# Cell 6: Distributed Training Step (MirroredStrategy Compatible) & 1-Step Warmup
diff_aug_layer = layers.RandomTranslation(
    height_factor=(-0.125, 0.125),
    width_factor=(-0.125, 0.125),
    fill_mode="constant",
    fill_value=-1.0
)

with strategy.scope():
    generator = build_generator()
    discriminator = build_discriminator()
    g_optimizer = keras.optimizers.Adam(learning_rate=G_LR, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    d_optimizer = keras.optimizers.Adam(learning_rate=D_LR, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    
    # Exponential Moving Average Shadow Weights
    ema_generator = build_generator()
    ema_generator.set_weights(generator.get_weights())
    
    g_optimizer.build(generator.trainable_variables)
    d_optimizer.build(discriminator.trainable_variables)

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

print("Distributed SAGAN train step initialized.")
print("Running 1-step warmup to initialize distributed slot variables...")
dummy_ds = tf.data.Dataset.from_tensors(
    tf.zeros((GLOBAL_BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=tf.float32)
)
dist_dummy = strategy.experimental_distribute_dataset(dummy_ds)
distributed_train_step(next(iter(dist_dummy)), tf.constant(1.0, dtype=tf.float32))
print("Warmup: SUCCESS! All replica optimizer states initialized cleanly.")
'''

# Cell 6 is index 6
nb["cells"][6]["source"] = [line + "\n" for line in cell6_code.split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated Cell 6 in 03_Discrete_SAGAN_Production.ipynb!")

