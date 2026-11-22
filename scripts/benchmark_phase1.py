import os
import sys
import time
import json
import psutil
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
print("PHASE 1 BENCHMARK & 50-ROW COSINE VERIFICATION (8-DECIMAL PRECISION)")
print("=" * 80)

# Check file sizes
index_bytes = os.path.getsize(INDEX_PATH)
manifest_bytes = os.path.getsize(MANIFEST_PATH)
print(f"Index File Size on Disk:    {index_bytes / (1024*1024):.2f} MB ({index_bytes:,} bytes)")
print(f"Manifest File Size on Disk: {manifest_bytes / (1024*1024):.2f} MB ({manifest_bytes:,} bytes)")

# Measure Idle RSS before loading
proc = psutil.Process()
rss_before = proc.memory_info().rss / (1024 * 1024)

# Load via np.memmap and create float32 RAM array
t0 = time.time()
fp16_mmap = np.load(INDEX_PATH, mmap_mode="r")
gallery_f32 = fp16_mmap.astype(np.float32)
load_time = time.time() - t0

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

rss_after_load = proc.memory_info().rss / (1024 * 1024)
print(f"Gallery Matrix Shape: {gallery_f32.shape}, dtype: {gallery_f32.dtype}")
print(f"Manifest Records: {len(manifest):,}")
print(f"Load time (mmap + float32 RAM copy): {load_time*1000:.2f} ms")
print(f"Process RSS at idle (with manifest + float32 in RAM): {rss_after_load:.2f} MB (Delta: +{rss_after_load - rss_before:.2f} MB)")

# 1. 50-ROW LIVE VS STORED COSINE CHECK (8 DECIMALS)
print("\n" + "=" * 80)
print("1. 50-ROW LIVE RE-ENCODING VS. STORED FLOAT16 (8-DECIMAL PRECISION)")
print("=" * 80)

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

ENCODER_PATH = os.path.join(MODELS_DIR, "2_VAE_280k_v2", "VAE_280K_Outputs", "encoder_280k_final.keras")
enc = tf.keras.models.load_model(ENCODER_PATH, custom_objects={'Sampling': Sampling})

ds = load_dataset("evilsocket/alucard-sprites")["train"]

# Select 50 deterministic random rows
np.random.seed(42)
sample_indices = np.random.choice(len(manifest), size=50, replace=False)

cosine_table = []
sims = []

for idx in sample_indices:
    item = manifest[idx]
    raw_id = item["raw_id"]
    cat = item["category"]
    split = item["split"]
    txt = item["text"][:35]
    
    # Stored vector: load float16, convert to float32, unit-normalize
    stored_vec = fp16_mmap[idx].astype(np.float32)
    stored_vec /= (np.linalg.norm(stored_vec) + 1e-8)
    
    # Live encode from dataset raw image unquantized
    img = ds[raw_id]["image"].convert("RGBA").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    out = enc(arr, training=False)
    mu = out[0].numpy()[0] if isinstance(out, (list, tuple)) else out.numpy()[0]
    live_vec = mu / (np.linalg.norm(mu) + 1e-8)
    
    cos_sim = float(np.dot(stored_vec, live_vec))
    sims.append(cos_sim)
    cosine_table.append({
        "gallery_idx": int(idx),
        "raw_id": int(raw_id),
        "split": split,
        "category": cat,
        "prompt": txt,
        "cosine_sim": cos_sim
    })

print(f"| {'#':2s} | {'Gallery Idx':11s} | {'Raw ID':8s} | {'Split':10s} | {'Category':12s} | {'Cosine Sim':12s} | {'Prompt Prefix':32s} |")
print(f"|{'-'*4}|{'-'*13}|{'-'*10}|{'-'*12}|{'-'*14}|{'-'*14}|{'-'*34}|")
for i, r in enumerate(cosine_table, 1):
    print(f"| {i:2d} | {r['gallery_idx']:11d} | {r['raw_id']:8d} | {r['split']:10s} | {r['category']:12s} | {r['cosine_sim']:12.8f} | {r['prompt']:32s} |")

print(f"\n50-Row Cosine Summary: Mean = {np.mean(sims):.8f}, Min = {np.min(sims):.8f}, Max = {np.max(sims):.8f}")

# 2. LATENCY & RAM BENCHMARK (100 QUERIES IN FLOAT32 RAM)
print("\n" + "=" * 80)
print("2. SEARCH LATENCY & PEAK RAM BENCHMARK (100 QUERIES IN FLOAT32 RAM)")
print("=" * 80)

# Pre-generate 100 normalized 256D query vectors
np.random.seed(123)
q_vecs = np.random.randn(100, 256).astype(np.float32)
q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)

# Warm up
_ = gallery_f32 @ q_vecs[0]

query_latencies = []
peak_rss = proc.memory_info().rss / (1024 * 1024)

for q_num in range(100):
    q_vec = q_vecs[q_num]
    t_start = time.perf_counter()
    
    # Fast matrix multiplication on in-memory float32 array
    scores = gallery_f32 @ q_vec
    top5_idx = np.argpartition(scores, -5)[-5:]
    
    dt_ms = (time.perf_counter() - t_start) * 1000.0
    query_latencies.append(dt_ms)
    
    cur_rss = proc.memory_info().rss / (1024 * 1024)
    if cur_rss > peak_rss:
        peak_rss = cur_rss

query_latencies = np.array(query_latencies)
median_lat = float(np.median(query_latencies))
p95_lat = float(np.percentile(query_latencies, 95))
min_lat = float(np.min(query_latencies))
max_lat = float(np.max(query_latencies))

print(f"Searches evaluated: {len(query_latencies)}")
print(f"Median Search Latency: {median_lat:.2f} ms")
print(f"P95 Search Latency:    {p95_lat:.2f} ms")
print(f"Min Search Latency:    {min_lat:.2f} ms")
print(f"Max Search Latency:    {max_lat:.2f} ms")
print(f"Peak Server RSS RAM:   {peak_rss:.2f} MB")

# Save Phase 1 results JSON
results = {
    "index_size_mb": round(index_bytes / (1024*1024), 2),
    "manifest_size_mb": round(manifest_bytes / (1024*1024), 2),
    "total_gallery_records": len(manifest),
    "idle_rss_mb": round(rss_after_load, 2),
    "peak_rss_mb": round(peak_rss, 2),
    "latency_median_ms": round(median_lat, 2),
    "latency_p95_ms": round(p95_lat, 2),
    "latency_min_ms": round(min_lat, 2),
    "latency_max_ms": round(max_lat, 2),
    "cosine_check_50_rows": {
        "mean": float(np.mean(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "rows": cosine_table
    }
}

with open(os.path.join(EVAL_DIR, "phase1_benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nPhase 1 benchmark results saved to: {os.path.join(EVAL_DIR, 'phase1_benchmark_results.json')}")
