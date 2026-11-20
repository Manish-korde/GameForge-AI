with open('scripts/build_discrete_palette_dcgan_notebook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

replacement_code = '''with strategy.scope():
    g_optimizer = keras.optimizers.Adam(learning_rate=G_LR, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    d_optimizer = keras.optimizers.Adam(learning_rate=D_LR, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    
    # Pre-build optimizer slot variables under strategy scope to prevent placeholder uninitialized graph crash
    g_optimizer.build(generator.trainable_variables)
    d_optimizer.build(discriminator.trainable_variables)

# ── Unified Replica Step (MirroredStrategy Compatible) ─────────────────────
def train_step_per_replica(real_imgs):
    local_batch = tf.shape(real_imgs)[0]
    z = tf.random.normal((local_batch, LATENT_DIM))
    
    # 1. Update Discriminator
    with tf.GradientTape() as d_tape:
        fake_logits_d = generator(z, training=True)
        fake_imgs_d = logits_to_image(fake_logits_d, global_palette_tf, tau=1.0, hard=True, noise=True)
        
        real_aug = diff_aug_translation(real_imgs)
        fake_aug_d = diff_aug_translation(fake_imgs_d)
        
        d_real = discriminator(real_aug, training=True)
        d_fake = discriminator(fake_aug_d, training=True)
        
        d_loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real)) + tf.reduce_mean(tf.nn.relu(1.0 + d_fake))
        
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    # 2. Update Generator
    with tf.GradientTape() as g_tape:
        fake_logits_g = generator(z, training=True)
        fake_imgs_g = logits_to_image(fake_logits_g, global_palette_tf, tau=1.0, hard=True, noise=True)
        fake_aug_g = diff_aug_translation(fake_imgs_g)
        
        d_fake_g = discriminator(fake_aug_g, training=True)
        g_loss = -tf.reduce_mean(d_fake_g)
        
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    
    return d_loss, g_loss

@tf.function
def distributed_train_step(dist_real_images):
    per_d, per_g = strategy.run(train_step_per_replica, args=(dist_real_images,))
    d_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_d, axis=None)
    g_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_g, axis=None)
    return d_loss, g_loss

print("Distributed Hinge Training Step compiled.")
print("Executing 1-step warmup across GPU replicas to initialize all optimizer states...")
dummy_ds = tf.data.Dataset.from_tensors(
    tf.zeros((GLOBAL_BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=tf.float32)
)
dist_dummy = strategy.experimental_distribute_dataset(dummy_ds)
distributed_train_step(next(iter(dist_dummy)))
print("Warmup: SUCCESS! All replica optimizer states initialized cleanly.")""")
'''

new_lines = lines[:529] + [replacement_code] + lines[580:]
with open('scripts/build_discrete_palette_dcgan_notebook.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Lines 530-580 replaced cleanly!')

