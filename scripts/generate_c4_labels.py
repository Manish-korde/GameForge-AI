import os
import sys
import json
import time
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import app
import dcgan_service

ARTIFACT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))

print("=" * 80)
print("GENERATING K=4 PALETTE-CLUSTER LABELS FOR 42,212 FRONT-FACING SPRITES")
print("=" * 80)

indices_path = os.path.join(CONFIG_DIR, "alucard_frontfacing_42k_indices.json")
with open(indices_path, "r") as f:
    frontfacing_indices = json.load(f)

N_TOTAL = len(frontfacing_indices)
print(f"Total front-facing sprites: {N_TOTAL:,}")

raw_ds = app.get_raw_dataset()
dcgan_service.load_dcgan_generator()
palette_u8 = dcgan_service._global_palette_u8
print(f"Global palette shape: {palette_u8.shape}")

# Vectorized distance computation helper
print("Computing 32-palette histograms for all 42,212 sprites...")
t0 = time.time()

palette_histograms = np.zeros((N_TOTAL, 32), dtype=np.float32)
# We will also keep candidate representative sprites for visualization
# To save memory, we can store 50 representative sprites per cluster during/after clustering

BATCH_SIZE = 2000
for start_idx in range(0, N_TOTAL, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, N_TOTAL)
    batch_raw_ids = frontfacing_indices[start_idx:end_idx]
    
    for i, raw_id in enumerate(batch_raw_ids):
        global_idx = start_idx + i
        im = raw_ds[int(raw_id)]["image"].convert("RGBA")
        if im.size != (64, 64):
            im = im.resize((64, 64), Image.NEAREST)
        arr = np.array(im)
        alpha_mask = arr[..., 3] > 127
        if np.any(alpha_mask):
            rgb_pts = arr[..., :3][alpha_mask] # (P, 3)
            # Distance to 32 colors
            diff = rgb_pts[:, None, :] - palette_u8[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            nearest = np.argmin(dist_sq, axis=-1)
            hist = np.bincount(nearest, minlength=32) / len(nearest)
            palette_histograms[global_idx] = hist

    if (end_idx % 10000 == 0) or end_idx == N_TOTAL:
        print(f"  Processed {end_idx:,}/{N_TOTAL:,} sprites in {time.time() - t0:.1f}s")

print(f"Feature matrix complete in {time.time() - t0:.2f}s. Shape: {palette_histograms.shape}")

# Run K-Means with k=4
print("\nFitting KMeans (k=4, random_state=42)...")
t_km = time.time()
km = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = km.fit_predict(palette_histograms)
print(f"KMeans converged in {time.time() - t_km:.2f}s.")

counts = Counter(cluster_labels)
print("\nCluster Distribution on 42,212 Sprites:")
for c in range(4):
    cnt = counts[c]
    pct = cnt / N_TOTAL * 100.0
    print(f"  Cluster {c}: {cnt:,} sprites ({pct:.2f}%)")

# Save labels dictionary and list
out_labels_dict = {
    "num_clusters": 4,
    "num_sprites": N_TOTAL,
    "cluster_counts": {str(c): int(counts[c]) for c in range(4)},
    "cluster_percentages": {str(c): round(float(counts[c] / N_TOTAL * 100.0), 2) for c in range(4)},
    "raw_id_to_cluster": {str(frontfacing_indices[i]): int(cluster_labels[i]) for i in range(N_TOTAL)},
    "index_to_cluster": [int(x) for x in cluster_labels]
}

out_labels_path = os.path.join(CONFIG_DIR, "alucard_frontfacing_42k_c4_labels.json")
with open(out_labels_path, "w") as f:
    json.dump(out_labels_dict, f, indent=2)
print(f"\nSaved cluster labels to {out_labels_path} ({os.path.getsize(out_labels_path):,} bytes)")

# Artifact path
art_labels_path = os.path.join(ARTIFACT_DIR, "alucard_frontfacing_42k_c4_labels.json")
with open(art_labels_path, "w") as f:
    json.dump(out_labels_dict, f, indent=2)

# Extract 8 representative sprites per cluster closest to centroid
print("\nExtracting 8 closest representative sprites per cluster...")
SPR_SZ = 64
composite = Image.new("RGBA", (8 * SPR_SZ, 4 * SPR_SZ), (15, 23, 42, 255))

archetype_names = {
    0: "Cluster 0 (Knight / Gray / Steel)",
    1: "Cluster 1 (Orc / Green / Nature)",
    2: "Cluster 2 (Ranger / Brown / Leather)",
    3: "Cluster 3 (Mage / Crimson / Royal)"
}

for c in range(4):
    c_indices = np.where(cluster_labels == c)[0]
    c_vecs = palette_histograms[c_indices]
    center = km.cluster_centers_[c]
    dists = np.sum((c_vecs - center) ** 2, axis=1)
    closest_idxs = c_indices[np.argsort(dists)[:8]]
    
    for col_idx, idx_in_c in enumerate(closest_idxs):
        raw_id = frontfacing_indices[idx_in_c]
        im = raw_ds[int(raw_id)]["image"].convert("RGBA")
        if im.size != (64, 64):
            im = im.resize((64, 64), Image.NEAREST)
        composite.paste(im, (col_idx * SPR_SZ, c * SPR_SZ), im)

out_img_config = os.path.join(CONFIG_DIR, "real_c4_cluster_examples.png")
composite.save(out_img_config)
out_img_art = os.path.join(ARTIFACT_DIR, "real_c4_cluster_examples.png")
composite.save(out_img_art)
print(f"Saved 8-sprite per cluster composite to {out_img_config} and {out_img_art}")

print("\nDone!")
