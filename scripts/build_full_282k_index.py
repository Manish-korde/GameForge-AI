import os
import sys
import time
import json
import re
import gc
import psutil
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

OUTPUT_INDEX_PATH = os.path.join(MODELS_DIR, "latent_gallery_index_282k_fp16.npy")
OUTPUT_MANIFEST_PATH = os.path.join(MODELS_DIR, "latent_gallery_manifest_282k.json")
PROGRESS_PATH = os.path.join(MODELS_DIR, "build_282k_progress.json")

print("=" * 80)
print("BUILDING FULL 282,511 DEDUPLICATED VAE LATENT GALLERY (FLOAT16 MEMMAP)")
print("=" * 80)

# 1. Keras model monkey patch for deserialization compatibility
@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[0]

for cls in [tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose, 
            tf.keras.layers.Flatten, tf.keras.layers.Reshape, tf.keras.layers.InputLayer]:
    orig = cls.__init__
    def make_patch(o):
        def p(self, *a, **kw):
            kw.pop('quantization_config', None)
            o(self, *a, **kw)
        return p
    cls.__init__ = make_patch(orig)

# 2. Load model
ENCODER_PATH = os.path.join(MODELS_DIR, "2_VAE_280k_v2", "VAE_280K_Outputs", "encoder_280k_final.keras")
print(f"Loading VAE Encoder from: {ENCODER_PATH}")
enc = tf.keras.models.load_model(ENCODER_PATH, custom_objects={'Sampling': Sampling})

@tf.function
def encode_batch(x):
    out = enc(x, training=False)
    mu = out[0] if isinstance(out, (list, tuple)) else out
    norm = tf.norm(mu, axis=1, keepdims=True) + 1e-8
    return mu / norm

# Warmup
dummy = np.zeros((1, 128, 128, 4), dtype=np.float32)
_ = encode_batch(dummy)
print("VAE Encoder compiled and warmed up.")

# 3. Load dataset & authentic splits
print("Loading 'evilsocket/alucard-sprites' dataset...")
ds = load_dataset("evilsocket/alucard-sprites")["train"]
total_raw = len(ds)
print(f"Raw dataset rows: {total_raw:,}")

# Load authentic 282,511 deduplicated unique row indices
unique_indices = json.load(open(os.path.join(CONFIG_DIR, "unique_indices.json")))["unique_indices"]
total_unique = len(unique_indices)
print(f"Unique deduplicated sprites to index: {total_unique:,}")

# Split lookups (sets for O(1) membership check)
train_set = set(json.load(open(os.path.join(CONFIG_DIR, "train_indices.json"))))
val_set = set(json.load(open(os.path.join(CONFIG_DIR, "val_indices.json"))))
test_set = set(json.load(open(os.path.join(CONFIG_DIR, "test_indices.json"))))
print(f"Splits verified: Train={len(train_set):,}, Val={len(val_set):,}, Test={len(test_set):,}")

# 4. Standard Paper Category Regex (matching phase1_dataset_analysis.py)
CATEGORY_PATTERNS = [
    ("Character", r"\b(character|hero|knight|wizard|mage|spellcaster|warrior|rogue|ninja|paladin|archer|necromancer|cleric|bard|monk|priest|sorcerer|fighter|adventurer|man|woman|boy|girl|person|human|dwarf|elf|orc|goblin|skeleton|zombie|demon|vampire)\b"),
    ("Enemy", r"\b(enemy|monster|boss|slime|bat|ghost|spider|dragon|golem|beast|creature|zombie|skeleton|demon|vampire|imp|snake|wolf|rat|giant|minotaur)\b"),
    ("Weapon", r"\b(weapon|sword|blade|dagger|bow|crossbow|staff|wand|axe|hammer|spear|shield|mace|scythe|rifle|gun|pistol)\b"),
    ("Item", r"\b(item|potion|elixir|scroll|book|ring|amulet|gem|jewel|crystal|coin|gold|chest|key|food|apple|bread|meat|bottle|flask)\b"),
    ("Prop", r"\b(prop|decor|furniture|table|chair|torch|banner|door|window|statue|column|pillar|barrel|crate|box|sign|fence|gate|grave|tomb)\b"),
    ("Effect", r"\b(effect|spell|magic|fire|ice|lightning|spark|explosion|aura|trail|slash|impact|beam|particle|smoke|flame)\b"),
    ("Tile", r"\b(tile|wall|floor|ground|grass|stone|brick|tree|rock|bush|water|lava|cliff|path|road|background)\b"),
]

def parse_metadata(raw_id, text):
    if not text:
        cat = "Unknown"
        label = "Unknown Sprite"
    else:
        text_lower = text.lower()
        cat = "Unknown"
        label = "Game Sprite"
        for cat_name, pat in CATEGORY_PATTERNS:
            match = re.search(pat, text_lower)
            if match:
                cat = cat_name
                sub = match.group(0).capitalize()
                label = f"{sub} ({cat_name})"
                break

    # Determine authentic training split
    if raw_id in train_set:
        split = "train"
    elif raw_id in val_set:
        split = "validation"
    elif raw_id in test_set:
        split = "test"
    else:
        split = "unknown"

    return cat, label, split

# 5. Initialize or Resume Memory-Mapped Output Matrix
BATCH_SIZE = 512
NUM_WORKERS = 8

# Check if partial build exists
start_idx = 0
if os.path.exists(PROGRESS_PATH) and os.path.exists(OUTPUT_INDEX_PATH):
    try:
        prog = json.load(open(PROGRESS_PATH, "r"))
        start_idx = prog.get("completed_count", 0)
        print(f"Resuming from checkpoint at index {start_idx:,} / {total_unique:,}...")
    except Exception as e:
        start_idx = 0

if start_idx == 0:
    mmap_matrix = np.lib.format.open_memmap(
        OUTPUT_INDEX_PATH, mode="w+", dtype=np.float16, shape=(total_unique, 256)
    )
    manifest = []
else:
    mmap_matrix = np.lib.format.open_memmap(
        OUTPUT_INDEX_PATH, mode="r+", dtype=np.float16, shape=(total_unique, 256)
    )
    manifest = json.load(open(OUTPUT_MANIFEST_PATH, "r"))[:start_idx]

# Helper function to preprocess single image
def get_sprite_tensor(raw_id):
    row = ds[raw_id]
    img = row["image"].convert("RGBA").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    text = row.get("text", "")
    return raw_id, arr, text

print(f"\nProcessing {total_unique - start_idx:,} remaining items in batches of {BATCH_SIZE}...")
t_global_start = time.time()
num_batches = int(np.ceil((total_unique - start_idx) / BATCH_SIZE))

for b in range(num_batches):
    b_start = start_idx + b * BATCH_SIZE
    b_end = min(total_unique, b_start + BATCH_SIZE)
    batch_raw_ids = unique_indices[b_start:b_end]

    t_b0 = time.time()
    # Multi-threaded image loading
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        loaded_items = list(executor.map(get_sprite_tensor, batch_raw_ids))

    batch_tensors = np.empty((len(loaded_items), 128, 128, 4), dtype=np.float32)
    for i, (raw_id, arr, text) in enumerate(loaded_items):
        batch_tensors[i] = arr
        cat, label, split = parse_metadata(raw_id, text)
        manifest.append({
            "index": b_start + i,
            "raw_id": raw_id,
            "category": cat,
            "label": label,
            "split": split,
            "text": text
        })

    # Inference on batch
    mu_norm = encode_batch(batch_tensors).numpy().astype(np.float16)

    # Write directly to memory mapped file
    mmap_matrix[b_start:b_end] = mu_norm

    dt_b = time.time() - t_b0
    completed = b_end
    rate = len(batch_raw_ids) / dt_b
    elapsed = time.time() - t_global_start
    eta = (total_unique - completed) / rate if rate > 0 else 0

    if (b + 1) % 10 == 0 or completed == total_unique:
        mmap_matrix.flush()
        # Update progress checkpoint
        with open(PROGRESS_PATH, "w") as f:
            json.dump({"completed_count": completed, "timestamp": time.time()}, f)
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"[{completed:7,d} / {total_unique:,d}] ({completed/total_unique*100:5.1f}%) | "
              f"Rate: {rate:5.1f} img/s | RSS: {rss_mb:6.1f} MB | Elapsed: {elapsed/60:4.1f}m | ETA: {eta/60:4.1f}m")

# Flush matrix to disk
mmap_matrix.flush()
del mmap_matrix
gc.collect()

print(f"\nFinalizing and saving manifest ({len(manifest):,} records)...")
with open(OUTPUT_MANIFEST_PATH, "w") as f:
    json.dump(manifest, f)

if os.path.exists(PROGRESS_PATH):
    os.remove(PROGRESS_PATH)

total_time = time.time() - t_global_start
print(f"BUILD COMPLETE in {total_time/60:.2f} minutes!")
print(f"Index: {OUTPUT_INDEX_PATH} ({os.path.getsize(OUTPUT_INDEX_PATH)/(1024*1024):.2f} MB)")
print(f"Manifest: {OUTPUT_MANIFEST_PATH} ({os.path.getsize(OUTPUT_MANIFEST_PATH)/(1024*1024):.2f} MB)")
