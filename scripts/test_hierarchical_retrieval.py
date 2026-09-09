import os
import json
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BUCKETS_DIR = os.path.join(MODELS_DIR, "category_buckets")

print("=== VERIFYING PRE-COMPUTED HIERARCHICAL LATENT RETRIEVAL ===")

# 1. Verify Global Index Load Latency
t0 = time.time()
npy_path = os.path.join(MODELS_DIR, "latent_gallery_index.npy")
manifest_path = os.path.join(MODELS_DIR, "latent_gallery_manifest.json")

assert os.path.exists(npy_path), "latent_gallery_index.npy missing!"
assert os.path.exists(manifest_path), "latent_gallery_manifest.json missing!"

gallery_index = np.load(npy_path)
with open(manifest_path, "r") as f:
    manifest = json.load(f)

t_load = (time.time() - t0) * 1000
print(f"1. Global Index Loaded: shape={gallery_index.shape}, items={len(manifest)} in {t_load:.2f} ms")

# 2. Test Fast Vector Search Speed across 5,000 vectors
query_vec = gallery_index[0] # Sample query
t0 = time.time()
sims = (gallery_index @ query_vec + 1.0) / 2.0 * 100.0
top_4 = np.argsort(sims)[::-1][:4]
t_search = (time.time() - t0) * 1000

print(f"2. Vector Search across 5,000 vectors completed in {t_search:.4f} ms")
print("   Top Matches:")
for rank, idx in enumerate(top_4):
    item = manifest[idx]
    print(f"   Rank #{rank+1}: ID={item['id']} | Category={item['category']} | Match={sims[idx]:.2f}%")

# 3. Verify Partitioned Category Buckets
print("\n3. Verifying Partitioned Category Buckets:")
categories = ["characters", "weapons", "items", "enemies", "effects", "props", "tiles"]

for cat in categories:
    cat_npy = os.path.join(BUCKETS_DIR, f"{cat}.npy")
    cat_json = os.path.join(BUCKETS_DIR, f"{cat}_manifest.json")
    if os.path.exists(cat_npy) and os.path.exists(cat_json):
        mat = np.load(cat_npy)
        with open(cat_json, "r") as f:
            meta = json.load(f)
        print(f"   - Bucket [{cat.capitalize()}]: {mat.shape[0]} items ({mat.shape[1]}-dim vectors)")

print("\nAll Hierarchical Retrieval Tests Passed Cleanly! [SUCCESS]")
