"""
Patch cells 7 and 8 in Notebook/training/02_DCGAN_FrontFacing_46K_50Epochs.ipynb
to support multi-GPU MirroredStrategy with pre-built optimizers and distributed smoke testing.
"""

import json

nb_path = 'Notebook/training/02_DCGAN_FrontFacing_46K_50Epochs.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 7: Loss Functions, Optimizers & Custom Training Step
cell_7_code = '''# Loss Functions, Optimizers & Distributed Training Step
with strategy.scope():
    g_optimizer = keras.optimizers.Adam(learning_rate=G_LR, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    d_optimizer = keras.optimizers.Adam(learning_rate=D_LR, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2)
    
    # Pre-build optimizer slot variables (momentums/velocities) INSIDE strategy.scope()
    # This prevents the MirroredStrategy "Need to be inside strategy.scope()" RuntimeError
    g_optimizer.build(generator.trainable_variables)
    d_optimizer.build(discriminator.trainable_variables)

def step_per_replica(real_images):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal([batch_size, LATENT_DIM])
    
    # 1. Train Discriminator
    with tf.GradientTape() as d_tape:
        fake_images = generator(z, training=True)
        real_logits, real_feats = discriminator(real_images, training=True)
        fake_logits, fake_feats = discriminator(fake_images, training=True)
        
        # LSGAN Loss with 0.9 soft real target
        d_loss_real = tf.reduce_mean(tf.square(real_logits - LABEL_REAL))
        d_loss_fake = tf.reduce_mean(tf.square(fake_logits - LABEL_FAKE))
        d_loss = 0.5 * (d_loss_real + d_loss_fake) / float(strategy.num_replicas_in_sync)
        
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    # 2. Train Generator
    with tf.GradientTape() as g_tape:
        fake_images = generator(z, training=True)
        fake_logits, fake_feats = discriminator(fake_images, training=True)
        _, real_feats = discriminator(real_images, training=False)
        
        # LSGAN Adversarial Loss
        g_adv_loss = 0.5 * tf.reduce_mean(tf.square(fake_logits - 1.0)) / float(strategy.num_replicas_in_sync)
        # Feature Matching Loss
        fm_loss = tf.reduce_mean(tf.abs(real_feats - fake_feats)) / float(strategy.num_replicas_in_sync)
        g_loss = g_adv_loss + (LAMBDA_FM * fm_loss)
        
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    
    return d_loss, g_loss, g_adv_loss, fm_loss

@tf.function
def distributed_train_step(real_images):
    per_d, per_g, per_adv, per_fm = strategy.run(step_per_replica, args=(real_images,))
    d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_d, axis=None)
    g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_g, axis=None)
    adv_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_adv, axis=None)
    fm_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_fm, axis=None)
    return d_loss, g_loss, adv_loss, fm_loss'''

# Cell 8: Execution Smoke Test Function
cell_8_code = '''# Execution Smoke Test Function (Distributed Strategy Compatible)
def run_smoke_test():
    print("Running 1-batch execution smoke test under strategy...")
    dummy_tensor = tf.random.uniform([GLOBAL_BATCH_SIZE, 64, 64, 4], minval=-1.0, maxval=1.0)
    dummy_ds = tf.data.Dataset.from_tensor_slices(dummy_tensor).batch(GLOBAL_BATCH_SIZE)
    dist_dummy_ds = strategy.experimental_distribute_dataset(dummy_ds)
    
    smoke_batch = next(iter(dist_dummy_ds))
    t0 = time.time()
    d_loss, g_loss, adv_loss, fm_loss = distributed_train_step(smoke_batch)
    dt = time.time() - t0
    print(f"Smoke test successful in {dt*1000:.1f}ms:")
    print(f"  D_Loss: {float(d_loss):.4f} | G_Loss: {float(g_loss):.4f} (Adv: {float(adv_loss):.4f}, FM: {float(fm_loss):.4f})")

run_smoke_test()'''

nb['cells'][7]['source'] = [line + '\n' for line in cell_7_code.split('\n')]
nb['cells'][8]['source'] = [line + '\n' for line in cell_8_code.split('\n')]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Patched {nb_path} cells 7 and 8 successfully!")

