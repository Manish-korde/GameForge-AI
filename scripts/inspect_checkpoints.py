import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

LATENT_DIM = 128
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 4

def build_generator(latent_dim=LATENT_DIM):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=42)
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
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=42)
    img_in = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="img_input")
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=True, kernel_initializer=init, name="d_conv_1")(img_in)
    x = layers.LeakyReLU(0.2, name="d_lrelu_1")(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_2")(x)
    x = layers.BatchNormalization(name="d_bn_2")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_2")(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_3")(x)
    x = layers.BatchNormalization(name="d_bn_3")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_3")(x)
    feat_out = x
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=init, name="d_conv_4")(x)
    x = layers.BatchNormalization(name="d_bn_4")(x)
    x = layers.LeakyReLU(0.2, name="d_lrelu_4")(x)
    x = layers.Flatten(name="d_flatten")(x)
    logit_out = layers.Dense(1, name="d_logit")(x)
    return keras.Model(inputs=img_in, outputs=[logit_out, feat_out], name="dcgan_discriminator")

def main():
    lpc_results_dir = r"C:\Users\manis\Downloads\dcgan_lpc_frontfacing_50ep_results"
    ckpt_dir = os.path.join(lpc_results_dir, "checkpoints")
    fixed_z = np.load(os.path.join(lpc_results_dir, "config", "fixed_latent_vectors.npy"))
    
    ckpt_map = {
        "ckpt-1": 1,
        "ckpt-2": 5,
        "ckpt-3": 10,
        "ckpt-4": 25,
        "ckpt-5": 50
    }
    
    generator = build_generator()
    discriminator = build_discriminator()
    g_opt = keras.optimizers.Adam()
    d_opt = keras.optimizers.Adam()
    
    ckpt = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_opt,
        d_optimizer=d_opt
    )
    
    checkpoint_stats = []
    generated_by_ckpt = {}
    
    for ckpt_name, ep in ckpt_map.items():
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        status = ckpt.restore(ckpt_path).expect_partial()
        
        # Generate 100 fixed latent samples
        raw = generator(fixed_z, training=False).numpy()
        generated_by_ckpt[ep] = raw
        
        rgb = raw[:, :, :, :3]
        alpha = raw[:, :, :, 3]
        
        a_mean = float(alpha.mean())
        a_std = float(alpha.std())
        a_min = float(alpha.min())
        a_max = float(alpha.max())
        
        u8_alpha = np.round(np.clip((alpha + 1.0) / 2.0, 0.0, 1.0) * 255.0).astype(np.uint8)
        u8_rgb = np.round(np.clip((rgb + 1.0) / 2.0, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        pct_transparent = 100.0 * np.mean(u8_alpha < 10)
        pct_opaque = 100.0 * np.mean(u8_alpha > 245)
        pct_semi = 100.0 * np.mean((u8_alpha >= 10) & (u8_alpha <= 245))
        
        fg_mask = u8_alpha > 20
        fg_pixels_per_sprite = np.sum(fg_mask, axis=(1, 2))
        mean_fg_px = float(np.mean(fg_pixels_per_sprite))
        
        if np.sum(fg_mask) > 100:
            fg_rgb = u8_rgb[fg_mask]
            rgb_mean = fg_rgb.mean(axis=0).tolist()
            rgb_std = fg_rgb.std(axis=0).tolist()
        else:
            rgb_mean = [0, 0, 0]
            rgb_std = [0, 0, 0]
            
        record = {
            'checkpoint': ckpt_name,
            'epoch': ep,
            'alpha_mean_tanh': a_mean,
            'alpha_std_tanh': a_std,
            'alpha_max_tanh': a_max,
            'alpha_min_tanh': a_min,
            'pct_transparent': pct_transparent,
            'pct_opaque': pct_opaque,
            'pct_semi': pct_semi,
            'mean_fg_pixels': mean_fg_px,
            'fg_rgb_mean': rgb_mean,
            'fg_rgb_std': rgb_std
        }
        checkpoint_stats.append(record)
        
    out_json = r"phase_dcgan_improvement\lpc_investigation\lpc_checkpoint_forensic_stats.json"
    with open(out_json, "w") as f:
        json.dump(checkpoint_stats, f, indent=2)
    print(f"✅ Checkpoint stats saved to: {out_json}")
    
    # Save the generated arrays for fast plotting
    for ep, raw in generated_by_ckpt.items():
        np_path = rf"phase_dcgan_improvement\lpc_investigation\raw_gen_ep_{ep:02d}.npy"
        np.save(np_path, raw)
    print("✅ All raw checkpoint generated tensors saved.")

if __name__ == "__main__":
    main()
