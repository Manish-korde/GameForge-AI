import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
npy_path = os.path.join(PROJECT_ROOT, "models", "latent_gallery_index.npy")

if not os.path.exists(npy_path):
    print(f"Error: Latent index file not found at {npy_path}")
    exit(1)

gallery = np.load(npy_path)
print(f"Loaded gallery index shape: {gallery.shape}")

# Calculate pairwise distances in memory-efficient chunks
n_vectors = len(gallery)
sample_size = min(5000, n_vectors)
print(f"Computing pairwise L2 distances for sample of size {sample_size}...")

# Sample if needed, but since it's 5000, we can compute the entire matrix or sample 2000 for speed
# Let's compute the full upper triangle of pairwise distances
dists = []
for i in range(sample_size):
    diff = gallery[i+1:sample_size] - gallery[i]
    dists.extend(np.linalg.norm(diff, axis=1))

dists = np.array(dists)
mean_dist = np.mean(dists)
std_dist = np.std(dists)
min_dist = np.min(dists)
max_dist = np.max(dists)

# Check 5th and 10th percentiles of distance distribution to understand typical nearest-neighbor bounds
p5 = np.percentile(dists, 5)
p10 = np.percentile(dists, 10)
p50 = np.percentile(dists, 50)

print("\n=== LATENT SPACE PAIRWISE L2 DISTANCE METRICS ===")
print(f"Mean Pairwise L2 Distance:  {mean_dist:.4f}")
print(f"Std Pairwise L2 Distance:   {std_dist:.4f}")
print(f"Min Pairwise L2 Distance:   {min_dist:.4f}")
print(f"Max Pairwise L2 Distance:   {max_dist:.4f}")
print(f"5th Percentile Distance:    {p5:.4f}")
print(f"10th Percentile Distance:   {p10:.4f}")
print(f"Median (50th%) Distance:    {p50:.4f}")
print("==================================================")
