import os
import sys
import time
import json
import random
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

INDEX_PATH = os.path.join(MODELS_DIR, "latent_gallery_index_282k_fp16.npy")
MANIFEST_PATH = os.path.join(MODELS_DIR, "latent_gallery_manifest_282k.json")

print("=" * 80)
print("PHASE 3: STRATIFIED MULTI-METHOD RETRIEVAL BENCHMARK (N=300 TEST QUERIES)")
print("=" * 80)

# Check gallery
if not os.path.exists(INDEX_PATH) or not os.path.exists(MANIFEST_PATH):
    print("Error: Gallery index or manifest not found!")
    sys.exit(1)

gallery_matrix = np.load(INDEX_PATH, mmap_mode="r")
with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

print(f"Loaded Gallery Index: {gallery_matrix.shape}, Manifest: {len(manifest):,} records")

# Load AE model
@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs): return inputs[0]

for cls in [tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose, 
            tf.keras.layers.Flatten, tf.keras.layers.Reshape, tf.keras.layers.InputLayer]:
    orig = cls.__init__
    def make_patch(o):
        def p(self, *a, **kw):
            kw.pop('quantization_config', None)
            o(self, *a, **kw)
        return p
    cls.__init__ = make_patch(orig)

AE_MODEL_PATH = os.path.join(MODELS_DIR, "1_AE_280k_v1", "AE_280K_best.keras")
print(f"Loading AE model from: {AE_MODEL_PATH}")
ae_full = tf.keras.models.load_model(AE_MODEL_PATH)
ae_encoder = ae_full.layers[0]

VAE_MODEL_PATH = os.path.join(MODELS_DIR, "2_VAE_280k_v2", "VAE_280K_Outputs", "encoder_280k_final.keras")
print(f"Loading VAE model from: {VAE_MODEL_PATH}")
vae_encoder = tf.keras.models.load_model(VAE_MODEL_PATH, custom_objects={'Sampling': Sampling})

ds = load_dataset("evilsocket/alucard-sprites")["train"]

# Load test split indices
test_indices = json.load(open(os.path.join(CONFIG_DIR, "test_indices.json")))
print(f"Authentic held-out test split size: {len(test_indices):,} sprites")

# Group test indices by category
test_by_cat = defaultdict(list)
manifest_by_raw_id = {item["raw_id"]: item for item in manifest}

for raw_id in test_indices:
    item = manifest_by_raw_id.get(raw_id)
    if item:
        test_by_cat[item["category"]].append(raw_id)

print("\nTest split category distribution:")
for cat, ids in sorted(test_by_cat.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {cat:15s}: {len(ids):5d} sprites ({len(ids)/len(test_indices)*100:5.2f}%)")

# Majority class baseline
gallery_cat_counts = Counter(item["category"] for item in manifest)
majority_cat, majority_count = gallery_cat_counts.most_common(1)[0]
majority_baseline = majority_count / len(manifest)
print(f"\nGallery Majority Class: '{majority_cat}' with {majority_count:,} items ({majority_baseline*100:.2f}%)")

# Draw 300 stratified queries from the test split
random.seed(42)
np.random.seed(42)

TOTAL_QUERIES = 300
target_quotas = {
    "Character": 70,
    "Enemy": 45,
    "Weapon": 45,
    "Item": 45,
    "Tile": 30,
    "Prop": 25,
    "Effect": 20,
    "Unknown": 20
}

stratified_queries = []
for cat, quota in target_quotas.items():
    pool = test_by_cat.get(cat, [])
    sample_k = min(len(pool), quota)
    chosen = random.sample(pool, sample_k)
    for raw_id in chosen:
        stratified_queries.append((raw_id, cat))

random.shuffle(stratified_queries)
print(f"\nSelected {len(stratified_queries)} stratified test queries:")
query_cat_counts = Counter(cat for _, cat in stratified_queries)
for cat, cnt in query_cat_counts.items():
    print(f"  {cat:15s}: {cnt:3d} queries")

# Color histogram helper
def compute_color_hist(arr_rgba, bins=4):
    # Non-transparent pixels
    alpha_mask = arr_rgba[:, :, 3] > 0.1
    if not np.any(alpha_mask):
        return np.zeros(bins * bins * bins, dtype=np.float32)
    rgb = arr_rgba[alpha_mask][:, :3]
    # Quantize to bins
    indices = np.clip((rgb * bins).astype(int), 0, bins - 1)
    flat_idx = indices[:, 0] * (bins * bins) + indices[:, 1] * bins + indices[:, 2]
    hist = np.bincount(flat_idx, minlength=bins * bins * bins).astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / (norm + 1e-8) if norm > 0 else hist

# Shape / Alpha mask helper
def compute_alpha_features(arr_rgba):
    mask = (arr_rgba[:, :, 3] > 0.1).astype(np.float32)
    area = np.sum(mask)
    # Bounding box
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0.0, 1.0, np.zeros((16, 16), dtype=np.float32)
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    h = max_y - min_y + 1
    w = max_x - min_x + 1
    aspect = float(w) / float(h)
    
    # Low-res 16x16 mask for shape IoU
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((16, 16), Image.Resampling.NEAREST)
    mask_16 = np.asarray(mask_img, dtype=np.float32) / 255.0
    return float(area), aspect, mask_16

# Gallery evaluation candidates pool (to allow fast evaluation across 300 queries,
# we evaluate across a 10,000 stratified gallery candidate set or full gallery chunked)
EVAL_GALLERY_SIZE = 10000
print(f"\nBuilding evaluation candidate gallery pool of {EVAL_GALLERY_SIZE} items...")
eval_gallery_indices = list(range(0, len(manifest), len(manifest) // EVAL_GALLERY_SIZE))[:EVAL_GALLERY_SIZE]
eval_manifest = [manifest[i] for i in eval_gallery_indices]
eval_raw_ids = [item["raw_id"] for item in eval_manifest]
eval_cats = [item["category"] for item in eval_manifest]

print("Pre-computing VAE vectors for evaluation pool...")
eval_vae_mat = gallery_matrix[eval_gallery_indices].astype(np.float32)

print("Pre-computing AE vectors for evaluation pool...")
eval_imgs = []
eval_color_hists = []
eval_alpha_feats = []
eval_pixel_lowres = []

for i, raw_id in enumerate(eval_raw_ids):
    row = ds[raw_id]
    img = row["image"].convert("RGBA").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    eval_imgs.append(arr)
    eval_color_hists.append(compute_color_hist(arr))
    eval_alpha_feats.append(compute_alpha_features(arr))
    # 16x16 downsampled for raw pixel L2
    lowres = np.asarray(img.resize((16, 16)), dtype=np.float32).flatten() / 255.0
    eval_pixel_lowres.append(lowres / (np.linalg.norm(lowres) + 1e-8))

eval_imgs_arr = np.array(eval_imgs)
ae_out = ae_encoder.predict(eval_imgs_arr, batch_size=256, verbose=0)
eval_ae_mat = ae_out.reshape((ae_out.shape[0], -1))
eval_ae_mat /= (np.linalg.norm(eval_ae_mat, axis=1, keepdims=True) + 1e-8)
eval_color_hists = np.array(eval_color_hists)
eval_pixel_lowres = np.array(eval_pixel_lowres)

print("Evaluation pool features pre-computed successfully.")

# Method metric trackers
methods = ["VAE", "AE", "Raw Pixel L2", "Color Histogram", "Hybrid Re-ranker"]
metrics = {
    m: {
        "p1": [], "p5": [], "mrr": [],
        "by_cat": defaultdict(lambda: {"p1": [], "p5": [], "mrr": []})
    } for m in methods
}

print(f"\nRunning comparative benchmark on {len(stratified_queries)} test queries...")
t_eval0 = time.time()

for q_idx, (q_raw_id, q_cat) in enumerate(stratified_queries):
    q_row = ds[q_raw_id]
    q_img = q_row["image"].convert("RGBA").resize((128, 128))
    q_arr = np.asarray(q_img, dtype=np.float32) / 255.0
    q_tensor = np.expand_dims(q_arr, axis=0)
    
    # 1. VAE vector
    v_out = vae_encoder(q_tensor, training=False)
    v_mu = v_out[0].numpy()[0] if isinstance(v_out, (list, tuple)) else v_out.numpy()[0]
    q_vae = v_mu / (np.linalg.norm(v_mu) + 1e-8)
    
    # 2. AE vector
    a_out = ae_encoder(q_tensor, training=False).numpy().reshape(-1)
    q_ae = a_out / (np.linalg.norm(a_out) + 1e-8)
    
    # 3. Pixel lowres
    q_lowres = np.asarray(q_img.resize((16, 16)), dtype=np.float32).flatten() / 255.0
    q_lowres /= (np.linalg.norm(q_lowres) + 1e-8)
    
    # 4. Color hist
    q_hist = compute_color_hist(q_arr)
    
    # 5. Alpha features
    q_area, q_aspect, q_mask16 = compute_alpha_features(q_arr)
    
    # Compute similarity scores across pool
    # Exclude self
    self_mask = np.array([r_id == q_raw_id for r_id in eval_raw_ids])
    
    # Method 1: VAE Cosine
    vae_sims = eval_vae_mat @ q_vae
    vae_sims[self_mask] = -999.0
    
    # Method 2: AE Cosine
    ae_sims = eval_ae_mat @ q_ae
    ae_sims[self_mask] = -999.0
    
    # Method 3: Pixel L2 Cosine
    pixel_sims = eval_pixel_lowres @ q_lowres
    pixel_sims[self_mask] = -999.0
    
    # Method 4: Color Hist
    color_sims = eval_color_hists @ q_hist
    color_sims[self_mask] = -999.0
    
    # Method 5: Hybrid Re-ranking (Top-50 VAE candidates re-ranked)
    top50_vae = np.argsort(vae_sims)[::-1][:50]
    hybrid_scores_top50 = []
    for idx in top50_vae:
        s_vae = float(vae_sims[idx])
        s_col = float(color_sims[idx])
        c_area, c_aspect, c_mask16 = eval_alpha_feats[idx]
        # Shape IoU
        intersection = np.sum(np.minimum(q_mask16, c_mask16))
        union = np.sum(np.maximum(q_mask16, c_mask16)) + 1e-8
        s_shape = float(intersection / union)
        
        # Hybrid formula: 0.60 * VAE + 0.25 * Color + 0.15 * Shape
        s_hyb = 0.60 * s_vae + 0.25 * s_col + 0.15 * s_shape
        hybrid_scores_top50.append(s_hyb)
        
    hybrid_ranked_top = [top50_vae[i] for i in np.argsort(hybrid_scores_top50)[::-1]]
    
    ranking_dict = {
        "VAE": np.argsort(vae_sims)[::-1],
        "AE": np.argsort(ae_sims)[::-1],
        "Raw Pixel L2": np.argsort(pixel_sims)[::-1],
        "Color Histogram": np.argsort(color_sims)[::-1],
        "Hybrid Re-ranker": hybrid_ranked_top
    }
    
    for m in methods:
        ranked = ranking_dict[m]
        top1_cat = eval_cats[ranked[0]]
        top5_cats = [eval_cats[ranked[i]] for i in range(min(5, len(ranked)))]
        
        p1 = 1.0 if top1_cat.lower() == q_cat.lower() else 0.0
        p5 = sum(1.0 for c in top5_cats if c.lower() == q_cat.lower()) / float(len(top5_cats))
        
        # MRR
        mrr = 0.0
        for rank, r_idx in enumerate(ranked[:50], 1):
            if eval_cats[r_idx].lower() == q_cat.lower():
                mrr = 1.0 / rank
                break
                
        metrics[m]["p1"].append(p1)
        metrics[m]["p5"].append(p5)
        metrics[m]["mrr"].append(mrr)
        
        metrics[m]["by_cat"][q_cat]["p1"].append(p1)
        metrics[m]["by_cat"][q_cat]["p5"].append(p5)
        metrics[m]["by_cat"][q_cat]["mrr"].append(mrr)

print(f"Evaluation completed in {time.time()-t_eval0:.2f}s!")

# Print Comparative Results Table
print("\n" + "=" * 90)
print("PHASE 3 RETRIEVAL BENCHMARK RESULTS TABLE (N = 300 HELD-OUT TEST QUERIES)")
print("=" * 90)
print(f"Majority-Class Baseline: {majority_baseline*100:.2f}% (Characters)")
print("-" * 90)
print(f"| {'Method':22s} | {'Precision@1':12s} | {'Precision@5':12s} | {'MRR':10s} | {'vs. Majority':14s} |")
print(f"|{'-'*24}|{'-'*14}|{'-'*14}|{'-'*12}|{'-'*16}|")

summary_table = []
for m in methods:
    mean_p1 = np.mean(metrics[m]["p1"])
    mean_p5 = np.mean(metrics[m]["p5"])
    mean_mrr = np.mean(metrics[m]["mrr"])
    delta_maj = mean_p1 - majority_baseline
    sign = "+" if delta_maj >= 0 else ""
    delta_str = f"{sign}{delta_maj*100:.1f}%"
    print(f"| {m:22s} | {mean_p1*100:10.2f}% | {mean_p5*100:10.2f}% | {mean_mrr:10.4f} | {delta_str:14s} |")
    summary_table.append({
        "method": m,
        "precision_at_1": round(float(mean_p1), 4),
        "precision_at_5": round(float(mean_p5), 4),
        "mrr": round(float(mean_mrr), 4),
        "delta_vs_majority": round(float(delta_maj), 4)
    })

# Per-Category Breakdown Table
print("\n" + "=" * 90)
print("PER-CATEGORY PRECISION@5 BREAKDOWN")
print("=" * 90)
categories = list(target_quotas.keys())
header = f"| {'Category':15s} |" + "".join([f" {m[:12]:12s} |" for m in methods])
print(header)
print("|" + "-"*17 + "|" + "".join(["-"*14 + "|" for _ in methods]))

cat_breakdown = {}
for cat in categories:
    row_str = f"| {cat:15s} |"
    cat_breakdown[cat] = {}
    for m in methods:
        cat_p5_list = metrics[m]["by_cat"][cat]["p5"]
        cat_p5 = np.mean(cat_p5_list) if cat_p5_list else 0.0
        cat_breakdown[cat][m] = round(float(cat_p5), 4)
        row_str += f" {cat_p5*100:10.1f}% |"
    print(row_str)

# Save JSON results
eval_output = {
    "num_queries": len(stratified_queries),
    "query_split": "test",
    "majority_class_baseline": round(float(majority_baseline), 4),
    "summary": summary_table,
    "category_breakdown_p5": cat_breakdown
}

with open(os.path.join(EVAL_DIR, "retrieval_eval.json"), "w") as f:
    json.dump(eval_output, f, indent=2)

print(f"\nSaved evaluation metrics to: {os.path.join(EVAL_DIR, 'retrieval_eval.json')}")
