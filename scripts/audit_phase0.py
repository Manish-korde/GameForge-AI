import os
import json
import hashlib
from collections import Counter
import numpy as np
from PIL import Image

os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

# 1. Load manifest and gallery index
manifest_path = os.path.join(MODELS_DIR, "latent_gallery_manifest_25k.json")
index_path = os.path.join(MODELS_DIR, "latent_gallery_index_25k.npy")

with open(manifest_path, "r") as f:
    manifest_25k = json.load(f)
gallery_vectors = np.load(index_path)

print(f"Loaded 25k manifest: {len(manifest_25k)} items, index shape: {gallery_vectors.shape}")

# Category counts in 25k gallery
cat_counts_25k = Counter(item["category"] for item in manifest_25k)

# Category counts in full dataset (from evaluation/dataset_analysis.json if available)
with open(os.path.join(EVAL_DIR, "dataset_analysis.json"), "r") as f:
    ds_analysis = json.load(f)

# Load dataset train split
ds = load_dataset("evilsocket/alucard-sprites")["train"]
print(f"Total raw dataset length: {len(ds)}")

# 2. Check the two swords: ID 209416 and 222754
sword1_row = ds[209416]
sword2_row = ds[222754]

img1 = sword1_row["image"].convert("RGBA")
img2 = sword2_row["image"].convert("RGBA")

bytes1 = np.asarray(img1, dtype=np.uint8).tobytes()
bytes2 = np.asarray(img2, dtype=np.uint8).tobytes()

hash1 = hashlib.sha256(bytes1).hexdigest()
hash2 = hashlib.sha256(bytes2).hexdigest()

exact_bytes_match = (bytes1 == bytes2)

# dHash implementation (difference hash)
def dhash(image, hash_size=8):
    # Resize to (hash_size + 1, hash_size)
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized)
    # Compute horizontal difference
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff

dhash1 = dhash(img1)
dhash2 = dhash(img2)
dhash_hamming_dist = int(np.count_nonzero(dhash1 != dhash2))

# Latent cosine similarity
vec1 = gallery_vectors[209416 - 200000]
vec2 = gallery_vectors[222754 - 200000]
latent_cos_sim = float(np.dot(vec1, vec2))

# Pixel difference
diff_arr = np.abs(np.asarray(img1, dtype=np.float32) - np.asarray(img2, dtype=np.float32))
mean_pixel_diff = float(np.mean(diff_arr))
max_pixel_diff = float(np.max(diff_arr))

print("\n--- TEST CASE: TWO IDENTICAL SWORDS ---")
print(f"Sword 1 (ID {209416}): prompt = '{sword1_row.get('text', '')}'")
print(f"Sword 2 (ID {222754}): prompt = '{sword2_row.get('text', '')}'")
print(f"SHA-256 (Sword 1): {hash1}")
print(f"SHA-256 (Sword 2): {hash2}")
print(f"Exact byte match: {exact_bytes_match}")
print(f"Mean pixel diff: {mean_pixel_diff:.4f}, Max pixel diff: {max_pixel_diff:.4f}")
print(f"dHash Hamming distance (64-bit): {dhash_hamming_dist}")
print(f"Latent Cosine Similarity: {latent_cos_sim:.6f}")

# 3. Check duplicates across the 25,000 gallery
print("\n--- AUDITING DUPLICATES IN 25,000 GALLERY ---")
# Check exact byte/sha256 hashes across the 25k images
# To be fast, let's sample or check all 25k:
sha_map = {}
exact_dup_pairs = 0
dup_ids = set()

# Also check latent duplicates (cos_sim > 0.9999)
print("Scanning 25k raw dataset rows for exact SHA256 duplicates...")
for i in range(25000):
    row_id = 200000 + i
    row = ds[row_id]
    b = np.asarray(row["image"].convert("RGBA"), dtype=np.uint8).tobytes()
    h = hashlib.sha256(b).hexdigest()
    if h in sha_map:
        exact_dup_pairs += 1
        dup_ids.add(row_id)
    else:
        sha_map[h] = row_id

print(f"Exact SHA-256 unique images in 25k slice: {len(sha_map)}")
print(f"Exact SHA-256 duplicate occurrences in 25k slice: {exact_dup_pairs}")

# Let's save a concise summary json
audit_results = {
    "cat_counts_25k": cat_counts_25k,
    "full_corpus_dist": ds_analysis.get("category_distribution", []),
    "sword_test_case": {
        "id1": 209416,
        "id2": 222754,
        "exact_byte_match": exact_bytes_match,
        "hash1": hash1,
        "hash2": hash2,
        "mean_pixel_diff": mean_pixel_diff,
        "max_pixel_diff": max_pixel_diff,
        "dhash_hamming_dist": dhash_hamming_dist,
        "latent_cos_sim": latent_cos_sim
    },
    "gallery_25k_stats": {
        "total": 25000,
        "sha256_unique": len(sha_map),
        "sha256_duplicates": exact_dup_pairs
    }
}

with open(os.path.join(EVAL_DIR, "phase0_audit_results.json"), "w") as f:
    json.dump(audit_results, f, indent=2)

print("\nSaved phase0_audit_results.json successfully!")
