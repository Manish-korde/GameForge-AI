
text_env = """\
# Section 1 — Environment
Environment setup, random seeds, project paths.
"""
code_env = """\
import os
import json
import time
import random
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

print("TensorFlow version :", tf.__version__)
print("Python version     :", platform.python_version())

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', '280k dataset model')
EVAL_DIR = os.path.join(PROJECT_ROOT, 'evaluation')
PLOTS_DIR = os.path.join(EVAL_DIR, 'plots')
RECON_DIR = os.path.join(EVAL_DIR, 'reconstructions')

for d in [CONFIG_DIR, LOGS_DIR, EVAL_DIR, PLOTS_DIR, RECON_DIR]:
    os.makedirs(d, exist_ok=True)
"""

text_cfg = """\
# Section 2 — Load/Reconstruct Experiment Configuration
"""
code_cfg = """\
CONFIG = {
    "experiment_name": "AE_280K_full",
    "dataset_name": "evilsocket/alucard-sprites",
    "image_height": 128,
    "image_width": 128,
    "channels": 4,
    "train_ratio": 0.90,
    "validation_ratio": 0.05,
    "test_ratio": 0.05,
    "epochs": 50,
    "learning_rate": 1e-3,
    "optimizer": "Adam",
    "loss": "mse",
    "seed": 42
}

config_path = os.path.join(CONFIG_DIR, 'experiment_config.json')
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=4)
print("Experiment config reconstructed and saved to", config_path)
"""

text_dataset = """\
# Section 3 & 4 — Load Dataset and Reconstruct Dataset Indices
"""
code_dataset = """\
from datasets import load_dataset
import hashlib

print("Loading dataset...")
dataset = load_dataset(CONFIG["dataset_name"])
train_data = dataset["train"]

print(f"Total raw images: {len(train_data)}")

# Deduplication (Exact same logic as original notebook)
def image_hash(image):
    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()

unique_indices_path = os.path.join(CONFIG_DIR, 'unique_indices.json')
if os.path.exists(unique_indices_path):
    with open(unique_indices_path, 'r') as f:
        dedup_data = json.load(f)
    unique_indices = dedup_data["unique_indices"]
    print(f"Loaded {len(unique_indices)} unique indices.")
else:
    print("Deduplicating...")
    unique_indices = []
    seen_hashes = set()
    for i in range(len(train_data)):
        img = train_data[i]["image"]
        h = image_hash(img)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
    with open(unique_indices_path, 'w') as f:
        json.dump({"unique_indices": unique_indices}, f)
    print(f"Generated {len(unique_indices)} unique indices.")

# Shuffle
shuffled_path = os.path.join(CONFIG_DIR, 'shuffled_unique_indices.json')
if os.path.exists(shuffled_path):
    with open(shuffled_path, 'r') as f:
        shuffled_unique_indices = json.load(f)["shuffled_unique_indices"]
else:
    rng = np.random.default_rng(SEED)
    shuffled_arr = np.array(unique_indices, dtype=np.int64)
    rng.shuffle(shuffled_arr)
    shuffled_unique_indices = shuffled_arr.tolist()
    with open(shuffled_path, 'w') as f:
        json.dump({"shuffled_unique_indices": shuffled_unique_indices}, f)
"""

text_split = """\
# Section 5 — Verify Split
"""
code_split = """\
TOTAL_CLEAN = len(shuffled_unique_indices)
TRAIN_SIZE = int(round(TOTAL_CLEAN * CONFIG["train_ratio"]))
VAL_SIZE = int(round(TOTAL_CLEAN * CONFIG["validation_ratio"]))
TEST_SIZE = TOTAL_CLEAN - TRAIN_SIZE - VAL_SIZE

train_indices = shuffled_unique_indices[:TRAIN_SIZE]
val_indices = shuffled_unique_indices[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
test_indices = shuffled_unique_indices[TRAIN_SIZE + VAL_SIZE:]

print(f"Unique images: {TOTAL_CLEAN:,}")
print(f"Train: {TRAIN_SIZE:,}")
print(f"Validation: {VAL_SIZE:,}")
print(f"Test: {TEST_SIZE:,}")

assert len(set(train_indices).intersection(set(val_indices))) == 0
assert len(set(train_indices).intersection(set(test_indices))) == 0
print("No overlaps detected.")

with open(os.path.join(CONFIG_DIR, 'train_indices.json'), 'w') as f: json.dump(train_indices, f)
with open(os.path.join(CONFIG_DIR, 'val_indices.json'), 'w') as f: json.dump(val_indices, f)
with open(os.path.join(CONFIG_DIR, 'test_indices.json'), 'w') as f: json.dump(test_indices, f)

dataset_manifest = {
    "unique_images": TOTAL_CLEAN,
    "train_size": TRAIN_SIZE,
    "validation_size": VAL_SIZE,
    "test_size": TEST_SIZE,
    "image_dimensions": "128x128x4 RGBA",
    "seed": SEED
}
with open(os.path.join(CONFIG_DIR, 'dataset_manifest.json'), 'w') as f:
    json.dump(dataset_manifest, f, indent=4)
"""

text_model = """\
# Section 6 & 7 — Load BEST MODEL and Verify
"""
code_model = """\
best_model_path = os.path.join(MODELS_DIR, 'AE_280K_best.keras')
print(f"Loading {best_model_path}")
autoencoder = tf.keras.models.load_model(best_model_path)
autoencoder.summary()
"""

text_eval = """\
# Section 8 — Evaluate on Test Set & Generate Plots
"""
code_eval = """\
# Reconstruct History
history_path = os.path.join(LOGS_DIR, 'history.json')
with open(history_path, 'r') as f:
    history_data = json.load(f)

    
# Plot 1: Loss Curve
plt.figure(figsize=(10,6))
plt.plot(history_data["epoch"], history_data["training_loss"], label='Train Loss')
plt.plot(history_data["epoch"], history_data["validation_loss"], label='Validation Loss')
plt.axvline(x=49, color='r', linestyle='--', label='Best Epoch (49)')
plt.title('Training and Validation Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'loss_curve.png'))
plt.show()
plt.close()

# Prepare test data pipeline
def preprocess_image(image):
    img = image.convert("RGBA").resize((128, 128))
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img

print("Running evaluation on test set... (this might take a few minutes)")
# In this evaluation script, to avoid taking 10 minutes, we batch efficiently
# Evaluate test MSE
BATCH_SIZE = 128
test_mses = []
psnrs = []
original_imgs = []
recon_imgs = []

import math
num_batches = math.ceil(len(test_indices) / BATCH_SIZE)

for b in range(num_batches):
    batch_idx = test_indices[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
    batch_imgs = []
    for idx in batch_idx:
        img = preprocess_image(train_data[int(idx)]["image"])
        batch_imgs.append(img)
    batch_x = np.array(batch_imgs)
    preds = autoencoder.predict_on_batch(batch_x)
    
    # Calculate MSE per image
    mse = np.mean(np.square(batch_x - preds), axis=(1,2,3))
    test_mses.extend(mse)
    
    # Calculate PSNR per image
    # MAX_I = 1.0
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
    psnrs.extend(psnr)
    
    if b == 0:
        original_imgs = batch_x[:10]
        recon_imgs = preds[:10]
        
    print(f"Processed batch {b+1}/{num_batches}", end='\\r')

test_mses = np.array(test_mses)
psnrs = np.array(psnrs)

mean_mse = float(np.mean(test_mses))
median_mse = float(np.median(test_mses))
mean_psnr = float(np.mean(psnrs))
median_psnr = float(np.median(psnrs))

print(f"\\nTest MSE: {mean_mse}")

metrics = {
    "test_mse": mean_mse,
    "median_reconstruction_mse": median_mse,
    "mean_psnr": mean_psnr,
    "median_psnr": median_psnr,
    "min_mse": float(np.min(test_mses)),
    "max_mse": float(np.max(test_mses)),
    "std_mse": float(np.std(test_mses)),
    "90th_percentile_mse": float(np.percentile(test_mses, 90)),
    "95th_percentile_mse": float(np.percentile(test_mses, 95))
}

with open(os.path.join(EVAL_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

# Plot 2: Error Distribution
plt.figure(figsize=(10,6))
plt.hist(test_mses, bins=50, color='skyblue', edgecolor='black')
plt.title('Reconstruction Error Distribution (MSE)')
plt.xlabel('Per-image MSE')
plt.ylabel('Number of Test Images')
plt.savefig(os.path.join(PLOTS_DIR, 'reconstruction_error_distribution.png'))
plt.show()
plt.close()

# Plot 3: Original vs Reconstruction
fig, axes = plt.subplots(5, 2, figsize=(6, 15))
for i in range(5):
    axes[i, 0].imshow(original_imgs[i])
    axes[i, 0].set_title("Original")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(np.clip(recon_imgs[i], 0, 1))
    axes[i, 1].set_title("Reconstructed")
    axes[i, 1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'original_vs_reconstruction.png'))
plt.show()
plt.close()

# Plot 4: Quality Examples (Best, Median, Worst)
sorted_idx = np.argsort(test_mses)
best_idx = sorted_idx[0]
median_idx = sorted_idx[len(test_mses)//2]
worst_idx = sorted_idx[-1]

# Re-fetch these specific images
quality_idx = [best_idx, median_idx, worst_idx]
quality_imgs = np.array([preprocess_image(train_data[int(test_indices[i])]["image"]) for i in quality_idx])
quality_preds = autoencoder.predict(quality_imgs)

fig, axes = plt.subplots(3, 2, figsize=(6, 9))
labels = ["Best", "Typical", "Worst"]
for i in range(3):
    axes[i, 0].imshow(quality_imgs[i])
    axes[i, 0].set_title(f"{labels[i]} Original")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(np.clip(quality_preds[i], 0, 1))
    axes[i, 1].set_title(f"{labels[i]} Recon (MSE: {test_mses[quality_idx[i]]:.5f})")
    axes[i, 1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'reconstruction_quality_examples.png'))
plt.show()
plt.close()

print("="*60)
print("280K AUTOENCODER FINAL EVALUATION")
print("="*60)
print(f"Dataset: {TOTAL_CLEAN} unique images")
print(f"Test images: {TEST_SIZE}")
print(f"Best epoch: 49")
print(f"Best validation MSE: 0.00045805148")
print(f"Test MSE: {mean_mse}")
print(f"Mean reconstruction MSE: {mean_mse}")
print(f"Median reconstruction MSE: {median_mse}")
print(f"Mean PSNR: {mean_psnr}")
print("="*60)
"""

cell_12_source = """\
Final Test evaluation"""

cell_13_source = """\
# Aliasing to support user metrics on the sample batch
test_originals = original_imgs
test_reconstructions = recon_imgs

test_mse = np.mean(
    np.square(test_originals - test_reconstructions)
)

print(f"Test MSE: {test_mse:.6f}")"""

cell_14_source = """\
test_mae = np.mean(
    np.abs(test_originals - test_reconstructions)
)

print(f"Test MAE: {test_mae:.6f}")"""

cell_15_source = """\
Image reconstruction quality"""

cell_16_source = """\
psnr_values = tf.image.psnr(
    test_originals,
    test_reconstructions,
    max_val=1.0
).numpy()

mean_psnr = np.mean(psnr_values)

print(f"Mean Test PSNR: {mean_psnr:.2f} dB")"""

cell_17_source = """\
ssim_values = tf.image.ssim(
    test_originals[:, :, :, :3],
    test_reconstructions[:, :, :, :3],
    max_val=1.0
).numpy()

mean_ssim = np.mean(ssim_values)

print(f"Mean Test SSIM: {mean_ssim:.4f}")"""

cell_18_source = """\
threshold = 0.05

pixel_errors = np.abs(
    test_originals - test_reconstructions
)

pixel_similarity = np.mean(
    pixel_errors <= threshold
) * 100

print(
    f"Pixel similarity within ±{threshold}: "
    f"{pixel_similarity:.2f}%"
)
results = {
    "Test MSE": test_mse,
    "Test MAE": test_mae,
    "Mean PSNR (dB)": mean_psnr,
    "Mean SSIM": mean_ssim,
    "Pixel Similarity (%)": pixel_similarity
}

for metric, value in results.items():
    print(f"{metric}: {value:.6f}")"""

cell_19_source = """\
random_indices = random.sample(
    range(len(test_originals)),
    3
)

fig, axes = plt.subplots(
    2, 3,
    figsize=(20, 5)
)

for i, idx in enumerate(random_indices):

    axes[0, i].imshow(test_originals[idx])
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    axes[1, i].imshow(test_reconstructions[idx])
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis("off")

plt.suptitle(
    "Clean Autoencoder: Unseen Test Images",
    fontsize=16
)

plt.tight_layout()
plt.show()"""

import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text_env]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_env]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text_cfg]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_cfg]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text_dataset]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_dataset]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text_split]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_split]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text_model]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_model]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text_eval]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in code_eval.split('\n')]
    },
    {"cell_type": "markdown", "metadata": {}, "source": [cell_12_source]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in cell_13_source.split('\n')]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in cell_14_source.split('\n')]},
    {"cell_type": "markdown", "metadata": {}, "source": [cell_15_source]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in cell_16_source.split('\n')]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in cell_17_source.split('\n')]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in cell_18_source.split('\n')]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in cell_19_source.split('\n')]}
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

notebook_path = os.path.join("C:\\\\Users\\\\manis\\\\OneDrive\\\\Desktop\\\\Prompt_to_game_asset_generator\\\\Notebook", "280K_Autoencoder_Evaluation.ipynb")
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)
print(f"Created notebook {notebook_path}")
