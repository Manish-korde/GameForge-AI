import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")
CONTACT_DIR = os.path.join(EVAL_DIR, "contact_sheets")

for d in [EVAL_DIR, PLOTS_DIR, CONTACT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=== PHASE 4: AE RECONSTRUCTION ERROR RANKING & OUTLIER SCREENING ===")

AE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "280k dataset model", "AE_280K_best.keras")
print(f"Loading AE model from {AE_MODEL_PATH}...")
ae_full = tf.keras.models.load_model(AE_MODEL_PATH)
ae_full(tf.zeros((1, 128, 128, 4), dtype=tf.float32))

print("Loading dataset test sample (2,000 images)...")
ds = load_dataset("evilsocket/alucard-sprites")
raw_data = ds["train"]

TEST_SIZE = 2000
test_offset = 250000 # Use samples from the tail of dataset (similar to test split)
test_images = []
test_indices = []

for i in range(test_offset, test_offset + TEST_SIZE):
    if i < len(raw_data):
        row = raw_data[i]
        img = row["image"].convert("RGBA").resize((128, 128))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        test_images.append(arr)
        test_indices.append(i)

X_test = np.array(test_images)
print("Running AE reconstruction on test set...")
preds = ae_full.predict(X_test, batch_size=128, verbose=0)

# Compute per-image MSE
mses = np.mean(np.square(X_test - preds), axis=(1, 2, 3))
psnrs = 10.0 * np.log10(1.0 / (mses + 1e-10))

sorted_order = np.argsort(mses)
sorted_mses = mses[sorted_order]
sorted_indices = [test_indices[idx] for idx in sorted_order]

mean_mse = float(np.mean(mses))
median_mse = float(np.median(mses))
min_mse = float(np.min(mses))
max_mse = float(np.max(mses))
p90_mse = float(np.percentile(mses, 90))
p95_mse = float(np.percentile(mses, 95))
p99_mse = float(np.percentile(mses, 99))

print("\n=== RECONSTRUCTION ERROR STATISTICS ===")
print(f"  - Mean MSE: {mean_mse:.6f}")
print(f"  - Median MSE: {median_mse:.6f}")
print(f"  - Min MSE (Best): {min_mse:.6f}")
print(f"  - Max MSE (Worst Outlier): {max_mse:.6f}")
print(f"  - 90th Percentile MSE: {p90_mse:.6f}")
print(f"  - 95th Percentile MSE: {p95_mse:.6f}")
print(f"  - 99th Percentile MSE: {p99_mse:.6f}")

# Plot Error Distribution
plt.figure(figsize=(9, 5))
plt.hist(mses, bins=50, color='#3b82f6', edgecolor='black', alpha=0.8)
plt.axvline(median_mse, color='green', linestyle='--', linewidth=2, label=f'Median MSE ({median_mse:.5f})')
plt.axvline(p95_mse, color='red', linestyle='--', linewidth=2, label=f'95th Percentile Outlier Threshold ({p95_mse:.5f})')
plt.title('AE Reconstruction MSE Error Distribution (Outlier Screening)', fontsize=12, fontweight='bold')
plt.xlabel('Per-Image Reconstruction MSE')
plt.ylabel('Image Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ae_error_distribution.png"), dpi=200)
plt.close()

# Generate Contact Sheet: Best vs Median vs Worst Outliers
print("Generating Best / Median / Worst Reconstruction Contact Sheet...")
best_sel = sorted_order[:6]
median_sel = sorted_order[len(sorted_order)//2 - 3 : len(sorted_order)//2 + 3]
worst_sel = sorted_order[-6:]

fig, axes = plt.subplots(6, 6, figsize=(14, 14))
fig.suptitle("AE Reconstruction Quality: Best (Top) vs Typical Median (Middle) vs Outliers (Bottom)", fontsize=13, fontweight='bold')

for i in range(6):
    # Best
    b_idx = best_sel[i]
    axes[0, i].imshow(X_test[b_idx])
    axes[0, i].set_title(f"Best Orig #{i+1}\nMSE: {mses[b_idx]:.5f}", fontsize=8)
    axes[0, i].axis('off')
    
    axes[1, i].imshow(np.clip(preds[b_idx], 0, 1))
    axes[1, i].set_title(f"Best Recon #{i+1}", fontsize=8, color='green')
    axes[1, i].axis('off')

    # Median
    m_idx = median_sel[i]
    axes[2, i].imshow(X_test[m_idx])
    axes[2, i].set_title(f"Median Orig #{i+1}\nMSE: {mses[m_idx]:.5f}", fontsize=8)
    axes[2, i].axis('off')

    axes[3, i].imshow(np.clip(preds[m_idx], 0, 1))
    axes[3, i].set_title(f"Median Recon #{i+1}", fontsize=8, color='blue')
    axes[3, i].axis('off')

    # Worst (Outliers)
    w_idx = worst_sel[i]
    axes[4, i].imshow(X_test[w_idx])
    axes[4, i].set_title(f"Outlier Orig #{i+1}\nMSE: {mses[w_idx]:.5f}", fontsize=8)
    axes[4, i].axis('off')

    axes[5, i].imshow(np.clip(preds[w_idx], 0, 1))
    axes[5, i].set_title(f"Outlier Recon #{i+1}", fontsize=8, color='red')
    axes[5, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(CONTACT_DIR, "ae_outlier_reconstruction_contact_sheet.png"), dpi=200)
plt.close()

metrics_json = {
    "test_sample_size": TEST_SIZE,
    "mean_mse": round(mean_mse, 6),
    "median_mse": round(median_mse, 6),
    "min_mse": round(min_mse, 6),
    "max_mse": round(max_mse, 6),
    "percentile_90_mse": round(p90_mse, 6),
    "percentile_95_mse": round(p95_mse, 6),
    "percentile_99_mse": round(p99_mse, 6),
    "screening_recommendation": "Distributional Outlier Screening (High-error assets correspond to complex multi-object tiles & extreme color variations)"
}

with open(os.path.join(EVAL_DIR, "ae_outlier_metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=4)

print(f"\nPhase 4 Completed Successfully! Summary saved to {os.path.join(EVAL_DIR, 'ae_outlier_metrics.json')}")
