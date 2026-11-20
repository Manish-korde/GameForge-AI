"""
Script to patch 02_DCGAN_FrontFacing_46K_50Epochs.ipynb with the exact DCGAN Discriminator
architecture (kernel_size=4, strides=2, padding='same', use_bias=True for layer 1).
"""

import json

nb_path = 'Notebook/training/02_DCGAN_FrontFacing_46K_50Epochs.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'def build_discriminator()' in src:
        print(f"Found build_discriminator in cell {idx}")
        # Replace the entire cell with the exact bit-exact architecture
        new_cell_code = '''# Architecture Definition: Bit-Exact Replica of backend/dcgan_service.py
def build_generator(latent_dim=LATENT_DIM):
    """Exact architectural replica of production DCGAN Generator (2,633,732 params)."""
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
    
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,), name="latent_input"),
        layers.Dense(4 * 4 * 512, use_bias=False, kernel_initializer=init, name="dense_proj"),
        layers.BatchNormalization(name="bn_proj"),
        layers.LeakyReLU(0.2, name="lrelu_proj"),
        layers.Reshape((4, 4, 512), name="reshape"),
        
        # Block 1: 4x4 -> 8x8
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_1"),
        layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.LeakyReLU(0.2, name="lrelu_1"),
        
        # Block 2: 8x8 -> 16x16
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_2"),
        layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.LeakyReLU(0.2, name="lrelu_2"),
        
        # Block 3: 16x16 -> 32x32
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_3"),
        layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_3"),
        layers.BatchNormalization(name="bn_3"),
        layers.LeakyReLU(0.2, name="lrelu_3"),
        
        # Block 4: 32x32 -> 64x64 RGBA
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_4"),
        layers.Conv2D(CHANNELS, kernel_size=3, padding="same", activation="tanh", kernel_initializer=init, name="conv_out")
    ], name="dcgan_generator")
    return model

def build_discriminator():
    """Exact architectural replica of production DCGAN Discriminator (2,768,449 params)."""
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
    
    img_in = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="img_input")
    
    # Block 1: 64x64 -> 32x32 (kernel_size=4, use_bias=True: 4*4*4*64 + 64 = 4,160 params)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=True, kernel_initializer=init, name="d_conv_1")(img_in)
    x = layers.LeakyReLU(0.2, name="d_lrelu_1")(x)
    
    # Block 2: 32x32 -> 16x16 (kernel_size=4, use_bias=False + BN: 4*4*64*128 + 512 = 131,584 params)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_2")(x)
    x = layers.BatchNormalization(name="d_bn_2")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_2")(x)
    
    # Block 3: 16x16 -> 8x8 (kernel_size=4, use_bias=False + BN: 4*4*128*256 + 1024 = 525,312 params)
    # Intermediate Feature Tap for Feature Matching Loss
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_3")(x)
    x = layers.BatchNormalization(name="d_bn_3")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_3")(x)
    feat_out = x  # Shape: (None, 8, 8, 256)
    
    # Block 4: 8x8 -> 4x4 (kernel_size=4, use_bias=False + BN: 4*4*256*512 + 2048 = 2,099,200 params)
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_4")(x)
    x = layers.BatchNormalization(name="d_bn_4")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_4")(x)
    
    # Dense Logit Out: (4 * 4 * 512) * 1 + 1 = 8,193 params
    x = layers.Flatten(name="d_flatten")(x)
    logit_out = layers.Dense(1, name="d_logit")(x)
    
    # Total Discriminator Parameters: 4,160 + 131,584 + 525,312 + 2,099,200 + 8,193 = 2,768,449 params
    return keras.Model(inputs=img_in, outputs=[logit_out, feat_out], name="dcgan_discriminator")'''
        
        cell['source'] = [line + '\n' for line in new_cell_code.split('\n')]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Patched {nb_path} successfully!")

