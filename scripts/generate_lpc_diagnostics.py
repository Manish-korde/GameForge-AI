import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# We use TensorFlow in Python 3.11 to load checkpoints
import tensorflow as tf
from tensorflow import keras
from keras import layers

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
        
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_1"),
        layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.LeakyReLU(0.2, name="lrelu_1"),
        
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_2"),
        layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.LeakyReLU(0.2, name="lrelu_2"),
        
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_3"),
        layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_3"),
        layers.BatchNormalization(name="bn_3"),
        layers.LeakyReLU(0.2, name="lrelu_3"),
        
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

def composite_on_dark(rgba_norm, bg_color=(0.1, 0.12, 0.16)):
    # rgba_norm: (H, W, 4) in [0, 1]
    rgb = rgba_norm[:, :, :3]
    alpha = rgba_norm[:, :, 3:4]
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    comp = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(comp, 0.0, 1.0)

def main():
    lpc_results_dir = r"C:\Users\manis\Downloads\dcgan_lpc_frontfacing_50ep_results"
    ckpt_dir = os.path.join(lpc_results_dir, "checkpoints")
    fixed_z_path = os.path.join(lpc_results_dir, "config", "fixed_latent_vectors.npy")
    fixed_z = np.load(fixed_z_path)
    
    out_dir = r"phase_dcgan_improvement\lpc_investigation"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load checkpoints and generate samples
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
    ckpt = tf.train.Checkpoint(generator=generator, discriminator=discriminator, g_optimizer=g_opt, d_optimizer=d_opt)
    
    generated_raw = {}
    for cname, ep in ckpt_map.items():
        cpath = os.path.join(ckpt_dir, cname)
        ckpt.restore(cpath).expect_partial()
        raw = generator(fixed_z, training=False).numpy()
        generated_raw[ep] = raw
        print(f"Generated raw tensors for Epoch {ep} ({cname})")
        
    # Load Real LPC samples
    real_files = sorted(glob.glob(os.path.join(lpc_results_dir, "hf_extracted_idles", "*.png")))[:100]
    real_raw = []
    for rf in real_files:
        im = Image.open(rf).convert("RGBA")
        arr = (np.array(im, dtype=np.float32) / 127.5) - 1.0
        real_raw.append(arr)
    real_raw = np.array(real_raw)
    print(f"Loaded {len(real_raw)} real LPC sprites for ground truth reference")

    # -------------------------------------------------------------------------
    # FIGURE 1: lpc_checkpoint_comparison.png
    # -------------------------------------------------------------------------
    # 6 rows (Real, Ep 1, Ep 5, Ep 10, Ep 25, Ep 50) x 8 samples
    fig, axes = plt.subplots(6, 8, figsize=(16, 12), facecolor='#0f172a')
    row_titles = [
        ("Real LPC Ground Truth", real_raw),
        ("LPC DCGAN Epoch 01 (ckpt-1)", generated_raw[1]),
        ("LPC DCGAN Epoch 05 (ckpt-2)", generated_raw[5]),
        ("LPC DCGAN Epoch 10 (ckpt-3)", generated_raw[10]),
        ("LPC DCGAN Epoch 25 (ckpt-4)", generated_raw[25]),
        ("LPC DCGAN Epoch 50 (ckpt-5)", generated_raw[50])
    ]
    
    for r_idx, (title, tensor_batch) in enumerate(row_titles):
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            raw_sprite = tensor_batch[c_idx]
            norm_sprite = np.clip((raw_sprite + 1.0) / 2.0, 0.0, 1.0)
            comp = composite_on_dark(norm_sprite)
            ax.imshow(comp)
            if c_idx == 0:
                ax.set_ylabel(title, color="#f8fafc", fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#334155")
                
    plt.suptitle("LPC DCGAN Checkpoint Progression: Real vs Epochs 1, 5, 10, 25, 50\n(Composited on Neutral Dark Canvas to Reveal Foreground Alpha & Anatomy)", 
                 color="#f8fafc", fontsize=13, y=0.98, fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "lpc_checkpoint_comparison.png")
    plt.savefig(p1, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------------
    # FIGURE 2: lpc_rgb_alpha_diagnostics.png
    # -------------------------------------------------------------------------
    # Inspect 4 samples (Real, Ep 10, Ep 25, Ep 50) across 4 visual modes:
    # 1. Raw RGB (Ignoring Alpha)
    # 2. Alpha Heatmap
    # 3. Composite on Dark Background
    # 4. Composite on White Background (Matplotlib default that caused the faint blur!)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), facecolor='#0f172a')
    targets = [
        ("Real LPC Sprite #0", real_raw[0]),
        ("Epoch 10 Sprite #0", generated_raw[10][0]),
        ("Epoch 25 Sprite #0", generated_raw[25][0]),
        ("Epoch 50 Sprite #0", generated_raw[50][0])
    ]
    
    for row_idx, (t_name, raw_sp) in enumerate(targets):
        norm_sp = np.clip((raw_sp + 1.0) / 2.0, 0.0, 1.0)
        rgb_raw = norm_sp[:, :, :3]
        alpha = norm_sp[:, :, 3]
        comp_dark = composite_on_dark(norm_sp, bg_color=(0.08, 0.10, 0.14))
        comp_white = composite_on_dark(norm_sp, bg_color=(1.0, 1.0, 1.0))
        
        # Col 0: Raw RGB
        axes[row_idx, 0].imshow(rgb_raw)
        axes[row_idx, 0].set_ylabel(t_name, color="#f8fafc", fontsize=9, fontweight="bold")
        if row_idx == 0:
            axes[row_idx, 0].set_title("1. Raw RGB (No Alpha)", color="#38bdf8", fontsize=10)
            
        # Col 1: Alpha Grayscale
        axes[row_idx, 1].imshow(alpha, cmap='gray', vmin=0, vmax=1)
        if row_idx == 0:
            axes[row_idx, 1].set_title("2. Alpha Channel Mask", color="#34d399", fontsize=10)
            
        # Col 2: Composite Dark
        axes[row_idx, 2].imshow(comp_dark)
        if row_idx == 0:
            axes[row_idx, 2].set_title("3. Composite (Dark BG)", color="#fbbf24", fontsize=10)
            
        # Col 3: Composite White
        axes[row_idx, 3].imshow(comp_white)
        if row_idx == 0:
            axes[row_idx, 3].set_title("4. Composite (White BG)\n[Matplotlib Default]", color="#f87171", fontsize=10)
            
        for c in range(4):
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])
            for spine in axes[row_idx, c].spines.values():
                spine.set_color("#334155")
                
    plt.suptitle("RGB vs Alpha Channel Forensic Decomposition\n(Demonstrating Why Low Alpha + White Default Canvas Made Epoch 50 Sprites Disappear)", 
                 color="#f8fafc", fontsize=12, y=0.98, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "lpc_rgb_alpha_diagnostics.png")
    plt.savefig(p2, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------------
    # FIGURE 3: lpc_training_dynamics.png
    # -------------------------------------------------------------------------
    history_csv = os.path.join(lpc_results_dir, "logs", "training_history.csv")
    df_h = pd.read_csv(history_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f172a')
    epochs = df_h['epoch']
    
    # Left: G vs D Loss
    ax1.plot(epochs, df_h['g_loss'], label='G Loss (Total)', color='#38bdf8', lw=2)
    ax1.plot(epochs, df_h['d_loss'], label='D Loss (LSGAN)', color='#f87171', lw=2)
    ax1.axvline(x=42, color='#fbbf24', ls='--', lw=1.5, label='Epoch 42 Overpower Destabilization')
    ax1.set_title("Generator vs Discriminator Loss Trajectory", color="#f8fafc", fontsize=12)
    ax1.set_xlabel("Epoch", color="#94a3b8")
    ax1.set_ylabel("Loss", color="#94a3b8")
    ax1.set_facecolor("#1e293b")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, alpha=0.2, color="#475569")
    ax1.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
    
    # Right: Adv vs Centroid FM Loss
    ax2.plot(epochs, df_h['adv_loss'], label='G Adversarial Loss', color='#34d399', lw=2)
    ax2.plot(epochs, df_h['fm_loss'], label='Canonical Centroid FM Loss', color='#c084fc', lw=2)
    ax2.axhline(y=0.22, color='#e2e8f0', ls=':', lw=1.2, label='Target FM Threshold (<=0.22)')
    ax2.axvline(x=42, color='#fbbf24', ls='--', lw=1.5, label='Epoch 42 Destabilization Spike')
    ax2.set_title("Adversarial vs Feature Matching Decomposition", color="#f8fafc", fontsize=12)
    ax2.set_xlabel("Epoch", color="#94a3b8")
    ax2.set_ylabel("Loss Component", color="#94a3b8")
    ax2.set_facecolor("#1e293b")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, alpha=0.2, color="#475569")
    ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
    
    plt.suptitle("LPC DCGAN Training Dynamics (50 Epochs): Stable Plateau at Ep 15-40 followed by Epoch 42 Overpower", 
                 color="#f8fafc", fontsize=13, y=0.98, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(out_dir, "lpc_training_dynamics.png")
    plt.savefig(p3, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------------
    # FIGURE 4: lpc_real_vs_generated_diagnostics.png
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 8, figsize=(16, 7), facecolor='#0f172a')
    
    # Row 0: Real LPC Sprites (Ground Truth)
    for c in range(8):
        ax = axes[0, c]
        norm_r = np.clip((real_raw[c] + 1.0) / 2.0, 0.0, 1.0)
        ax.imshow(composite_on_dark(norm_r))
        ax.set_title(f"Real LPC #{c}", color="#38bdf8", fontsize=8)
        ax.axis('off')
        
    # Row 1: Epoch 25 Generated (Optimal Checkpoint)
    for c in range(8):
        ax = axes[1, c]
        norm_25 = np.clip((generated_raw[25][c] + 1.0) / 2.0, 0.0, 1.0)
        ax.imshow(composite_on_dark(norm_25))
        ax.set_title(f"Ep 25 Gen (ckpt-4)", color="#34d399", fontsize=8)
        ax.axis('off')
        
    # Row 2: Epoch 50 Generated (Late Overpower Collapse)
    for c in range(8):
        ax = axes[2, c]
        norm_50 = np.clip((generated_raw[50][c] + 1.0) / 2.0, 0.0, 1.0)
        ax.imshow(composite_on_dark(norm_50))
        ax.set_title(f"Ep 50 Gen (ckpt-5)", color="#f87171", fontsize=8)
        ax.axis('off')
        
    plt.suptitle("LPC Real vs Generated Ground Truth Audit:\nRow 0: Real LPC Sprites | Row 1: Epoch 25 Generated (Optimal Peak) | Row 2: Epoch 50 Generated (Alpha Shrinkage)", 
                 color="#f8fafc", fontsize=12, y=0.98, fontweight="bold")
    plt.tight_layout()
    p4 = os.path.join(out_dir, "lpc_real_vs_generated_diagnostics.png")
    plt.savefig(p4, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {p4}")

if __name__ == "__main__":
    main()

