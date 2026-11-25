import os
import sys
import json
import time
import numpy as np
from PIL import Image
from collections import Counter

print("=" * 80)
print("PHASE 2 DIAGNOSTIC: REAL DATASET & K-MEANS PALETTE CLUSTER VALIDATION")
print("=" * 80)

# Load dataset and labels
sprites_path = "Dataset/sprites_42k.npy"
labels_path = "Dataset/labels.npy"
palette_path = "models/3_DCGAN_42k_v1/config/global_palette_32.npy"

t0 = time.time()
sprites = np.load(sprites_path) # (42212, 64, 64, 4) uint8
labels = np.load(labels_path)   # (42212,) int32
palette = np.load(palette_path) # (32, 3) uint8

N = len(sprites)
print(f"Loaded {N:,} sprites, {len(labels):,} labels in {time.time() - t0:.2f}s")
print(f"Sprites shape: {sprites.shape}, dtype: {sprites.dtype}")
print(f"Labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
print(f"Palette shape: {palette.shape}, dtype: {palette.dtype}")

# 1. Cluster counts and percentages
counts = Counter(labels)
print("\n1. Cluster Distribution:")
for c in range(4):
    cnt = counts[c]
    pct = cnt / N * 100.0
    print(f"   Cluster {c}: {cnt:,} sprites ({pct:.2f}%)")

# 2. Compute normalized foreground-only palette histograms for all 42,212 real sprites
print("\n2. Computing foreground-only 32-color histograms for all 42,212 real sprites...")
# Transparent pixels (alpha <= 127) must NOT contribute!
hists = np.zeros((N, 32), dtype=np.float32)
unique_colors_per_sprite = []

BATCH = 5000
for start in range(0, N, BATCH):
    end = min(start + BATCH, N)
    batch_sprites = sprites[start:end]
    for i in range(end - start):
        idx = start + i
        im = batch_sprites[i]
        fg_mask = im[..., 3] > 127
        if not np.any(fg_mask):
            continue
        fg_rgb = im[..., :3][fg_mask] # (P, 3)
        # Unique colors
        u_colors = len(np.unique(fg_rgb, axis=0))
        unique_colors_per_sprite.append(u_colors)
        
        # Distance to 32 palette colors
        # Vectorized mapping to nearest palette index
        diff = fg_rgb[:, None, :] - palette[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        nearest = np.argmin(dist_sq, axis=-1)
        h = np.bincount(nearest, minlength=32).astype(np.float32)
        if np.sum(h) > 0:
            h /= np.sum(h)
        hists[idx] = h

print(f"Computed histograms in {time.time() - t0:.2f}s")
print(f"Real Sprite Unique Colors: Mean = {np.mean(unique_colors_per_sprite):.2f}, Median = {np.median(unique_colors_per_sprite):.1f}, Min = {min(unique_colors_per_sprite)}, Max = {max(unique_colors_per_sprite)}")

# 3. Cluster centroids in real data
centroids = np.zeros((4, 32), dtype=np.float32)
for c in range(4):
    c_mask = (labels == c)
    centroids[c] = np.mean(hists[c_mask], axis=0)

def cosine_similarity(u, v):
    dot = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    return float(dot / max(norm, 1e-8))

print("\n3. Pairwise Cosine Similarity Between REAL Cluster Centroids:")
pairwise_real = {}
for i in range(4):
    for j in range(i + 1, 4):
        sim = cosine_similarity(centroids[i], centroids[j])
        pairwise_real[f"Real_Cluster_{i}_vs_{j}"] = round(sim, 4)
        print(f"   Real Cluster {i} vs {j} Cosine: {sim:.4f}")

mean_real_cross_cosine = float(np.mean(list(pairwise_real.values())))
print(f"   --> Mean Pairwise Cosine Between Real Clusters: {mean_real_cross_cosine:.4f}")

# 4. Within-Cluster vs Between-Cluster Variation (L2 and Cosine)
print("\n4. Within-Cluster vs Between-Cluster Variance:")
for c in range(4):
    c_mask = (labels == c)
    c_hists = hists[c_mask]
    # Sample 1000 for speed
    sample_sub = c_hists[np.random.RandomState(42).choice(len(c_hists), size=min(1000, len(c_hists)), replace=False)]
    within_cosines = [cosine_similarity(v, centroids[c]) for v in sample_sub]
    print(f"   Cluster {c} Mean Cosine with its OWN centroid: {np.mean(within_cosines):.4f}")

# Cross-similarity: what is cosine of sample with OTHER centroids?
for c in range(4):
    c_mask = (labels == c)
    c_hists = hists[c_mask]
    sample_sub = c_hists[np.random.RandomState(42).choice(len(c_hists), size=min(1000, len(c_hists)), replace=False)]
    other_cosines = []
    for other_c in range(4):
        if other_c == c: continue
        other_cosines.extend([cosine_similarity(v, centroids[other_c]) for v in sample_sub])
    print(f"   Cluster {c} Mean Cosine with OTHER centroids: {np.mean(other_cosines):.4f}")

# 5. Check Sparsity / Active Colors per Cluster
print("\n5. Active Palette Colors per Real Cluster (colors with > 2% mass in centroid):")
for c in range(4):
    cent = centroids[c]
    active_cols = np.where(cent > 0.02)[0]
    top5_cols = np.argsort(cent)[::-1][:5]
    print(f"   Cluster {c}: {len(active_cols)} active colors (>2% mass). Top 5 palette indices: {top5_cols.tolist()} (Mass: {cent[top5_cols].tolist()})")

# 6. Save results to JSON
res = {
    "num_sprites": N,
    "cluster_counts": {str(c): int(counts[c]) for c in range(4)},
    "cluster_percentages": {str(c): round(float(counts[c]/N*100), 2) for c in range(4)},
    "mean_unique_colors_real": float(np.mean(unique_colors_per_sprite)),
    "pairwise_real_cluster_cosines": pairwise_real,
    "mean_real_cross_cluster_cosine": mean_real_cross_cosine,
    "centroids": centroids.tolist()
}

out_json = "C:/Users/manis/.gemini/antigravity/brain/674ad8cd-af16-4b87-91e3-66973cf97bc5/real_cluster_analysis.json"
with open(out_json, "w") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved analysis to {out_json}")
