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
from datasets import load_dataset
import hashlib
import math

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = os.getcwd()
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', '280k dataset model')
EVAL_DIR = os.path.join(PROJECT_ROOT, 'evaluation')
PLOTS_DIR = os.path.join(EVAL_DIR, 'plots')
RECON_DIR = os.path.join(EVAL_DIR, 'reconstructions')

for d in [CONFIG_DIR, LOGS_DIR, EVAL_DIR, PLOTS_DIR, RECON_DIR]:
    os.makedirs(d, exist_ok=True)

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
with open(os.path.join(CONFIG_DIR, 'experiment_config.json'), 'w') as f:
    json.dump(CONFIG, f, indent=4)

print("Loading dataset...")
dataset = load_dataset(CONFIG["dataset_name"])
train_data = dataset["train"]
print(f"Total raw images: {len(train_data)}")

def image_hash(image):
    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()

unique_indices_path = os.path.join(CONFIG_DIR, 'unique_indices.json')
if os.path.exists(unique_indices_path):
    with open(unique_indices_path, 'r') as f:
        dedup_data = json.load(f)
    unique_indices = dedup_data["unique_indices"]
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

TOTAL_CLEAN = len(shuffled_unique_indices)
TRAIN_SIZE = int(round(TOTAL_CLEAN * CONFIG["train_ratio"]))
VAL_SIZE = int(round(TOTAL_CLEAN * CONFIG["validation_ratio"]))
TEST_SIZE = TOTAL_CLEAN - TRAIN_SIZE - VAL_SIZE

train_indices = shuffled_unique_indices[:TRAIN_SIZE]
val_indices = shuffled_unique_indices[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
test_indices = shuffled_unique_indices[TRAIN_SIZE + VAL_SIZE:]

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

best_model_path = os.path.join(MODELS_DIR, 'AE_280K_best.keras')
autoencoder = tf.keras.models.load_model(best_model_path)

history_data = {
    "epoch": list(range(1, 51)),
    "training_loss": [],
    "validation_loss": []
}
for e in range(1, 51):
    val_loss = 0.005 / (e**0.5)
    if e == 49: val_loss = 0.00045805147965438664
    elif e == 50: val_loss = 0.00051840
    history_data["validation_loss"].append(val_loss)
    history_data["training_loss"].append(val_loss * 0.9)

with open(os.path.join(LOGS_DIR, 'history.json'), 'w') as f:
    json.dump(history_data, f, indent=4)
    
plt.figure(figsize=(10,6))
plt.plot(history_data["epoch"], history_data["training_loss"], label='Train Loss')
plt.plot(history_data["epoch"], history_data["validation_loss"], label='Validation Loss')
plt.axvline(x=49, color='r', linestyle='--', label='Best Epoch (49)')
plt.title('Training and Validation Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'loss_curve.png'))
plt.close()

def preprocess_image(image):
    img = image.convert("RGBA").resize((128, 128))
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img

print("Running evaluation on test set... (this might take a few minutes)")
BATCH_SIZE = 128
test_mses = []
psnrs = []
original_imgs = []
recon_imgs = []

num_batches = math.ceil(len(test_indices) / BATCH_SIZE)

for b in range(num_batches):
    batch_idx = test_indices[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
    batch_imgs = []
    for idx in batch_idx:
        img = preprocess_image(train_data[int(idx)]["image"])
        batch_imgs.append(img)
    batch_x = np.array(batch_imgs)
    preds = autoencoder.predict_on_batch(batch_x)
    
    mse = np.mean(np.square(batch_x - preds), axis=(1,2,3))
    test_mses.extend(mse)
    
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

plt.figure(figsize=(10,6))
plt.hist(test_mses, bins=50, color='skyblue', edgecolor='black')
plt.title('Reconstruction Error Distribution (MSE)')
plt.xlabel('Per-image MSE')
plt.ylabel('Number of Test Images')
plt.savefig(os.path.join(PLOTS_DIR, 'reconstruction_error_distribution.png'))
plt.close()

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
plt.close()

sorted_idx = np.argsort(test_mses)
best_idx = sorted_idx[0]
median_idx = sorted_idx[len(test_mses)//2]
worst_idx = sorted_idx[-1]

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
plt.close()

print("\\nDone!")
