import os
import json
import re
import numpy as np
import tensorflow as tf
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BUCKETS_DIR = os.path.join(MODELS_DIR, "category_buckets")

for d in [MODELS_DIR, BUCKETS_DIR]:
    os.makedirs(d, exist_ok=True)

print("=== BUILDING HIERARCHICAL VAE LATENT INDEX & CATEGORY BUCKETS ===")

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

layers_to_patch = [
    tf.keras.layers.Dense,
    tf.keras.layers.Conv2D,
    tf.keras.layers.Conv2DTranspose,
    tf.keras.layers.Flatten,
    tf.keras.layers.Reshape,
    tf.keras.layers.InputLayer
]
for layer_cls in layers_to_patch:
    original_init = layer_cls.__init__
    def make_patched_init(orig_init):
        def patched_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            orig_init(self, *args, **kwargs)
        return patched_init
    layer_cls.__init__ = make_patched_init(original_init)

VAE_ENCODER_PATH = os.path.join(MODELS_DIR, "280k model VAE (VAE v2)", "VAE_280K_Outputs", "encoder_280k_final.keras")
print(f"Loading VAE Encoder from {VAE_ENCODER_PATH}...")
vae_encoder = tf.keras.models.load_model(VAE_ENCODER_PATH, custom_objects={'Sampling': Sampling})
vae_encoder(tf.zeros((1, 128, 128, 4), dtype=tf.float32))
print("VAE Encoder initialized.")

print("Loading 'evilsocket/alucard-sprites' dataset...")
ds = load_dataset("evilsocket/alucard-sprites")
raw_data = ds["train"]

CATEGORY_PATTERNS = [
    ("Characters", r"\b(character|hero|knight|wizard|mage|spellcaster|warrior|rogue|ninja|paladin|archer|necromancer|cleric|bard|monk|priest|sorcerer|fighter|adventurer|man|woman|boy|girl|person|human|dwarf|elf|orc|goblin|skeleton|zombie|demon|vampire)\b"),
    ("Enemies", r"\b(enemy|monster|boss|slime|bat|ghost|spider|dragon|golem|beast|creature|zombie|skeleton|demon|vampire|imp|snake|wolf|rat|giant|minotaur)\b"),
    ("Weapons", r"\b(weapon|sword|blade|dagger|bow|crossbow|staff|wand|axe|hammer|spear|shield|mace|scythe|rifle|gun|pistol|shield)\b"),
    ("Items", r"\b(item|potion|elixir|scroll|book|ring|amulet|gem|jewel|crystal|coin|gold|chest|key|food|apple|bread|meat|bottle|flask|chest|armor|helmet|boots)\b"),
    ("Props", r"\b(prop|decor|furniture|table|chair|torch|banner|door|window|statue|column|pillar|barrel|crate|box|sign|fence|gate|grave|tomb)\b"),
    ("Effects", r"\b(effect|spell|magic|fire|ice|lightning|spark|explosion|aura|trail|slash|impact|beam|particle|smoke|flame)\b"),
    ("Tiles", r"\b(tile|wall|floor|ground|grass|stone|brick|tree|rock|bush|water|lava|cliff|path|road|background)\b"),
]

def parse_category(text):
    if not text:
        return "Characters", "Game Character"
    text_lower = text.lower()
    for cat_label, pattern in CATEGORY_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            sub = match.group(0).capitalize()
            return cat_label, f"{sub} ({cat_label[:-1]})"
    return "Characters", "Game Sprite"

INDEX_SIZE = 5000
test_offset = 200000 # Use held-out tail split

print(f"Extracting VAE latent vectors for {INDEX_SIZE} dataset assets...")
batch_size = 256
num_batches = int(np.ceil(INDEX_SIZE / batch_size))

all_latents = []
manifest = []

for b in range(num_batches):
    start_idx = test_offset + b * batch_size
    end_idx = min(test_offset + INDEX_SIZE, start_idx + batch_size)
    
    batch_imgs = []
    for i in range(start_idx, end_idx):
        row = raw_data[i]
        img = row["image"].convert("RGBA").resize((128, 128))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        batch_imgs.append(arr)
        
        cat, label = parse_category(row.get("text", ""))
        manifest.append({
            "id": i,
            "category": cat,
            "label": label,
            "text": row.get("text", "")
        })
        
    X_batch = np.array(batch_imgs)
    out = vae_encoder.predict(X_batch, verbose=0)
    mu = out[0] if isinstance(out, list) else out
    
    # Normalize vectors for fast cosine dot product
    mu_norm = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
    all_latents.append(mu_norm)
    print(f"Processed batch {b+1}/{num_batches} ({len(manifest)} items)...", end="\r")

print("\nCombining feature matrices...")
global_matrix = np.vstack(all_latents) # Shape: (5000, 256)

# Save Global Index & Manifest
global_index_path = os.path.join(MODELS_DIR, "latent_gallery_index.npy")
global_manifest_path = os.path.join(MODELS_DIR, "latent_gallery_manifest.json")

np.save(global_index_path, global_matrix)
with open(global_manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved Global Latent Index ({global_matrix.shape}) -> {global_index_path}")
print(f"Saved Global Manifest ({len(manifest)} items) -> {global_manifest_path}")

# Partition into Category Buckets
print("\nPartitioning into Category Buckets...")
bucket_counts = {}
categories_set = ["Characters", "Weapons", "Items", "Enemies", "Effects", "Props", "Tiles"]

for cat in categories_set:
    indices = [i for i, item in enumerate(manifest) if item["category"] == cat]
    if not indices:
        continue
    
    cat_matrix = global_matrix[indices]
    cat_manifest = [manifest[i] for i in indices]
    
    cat_slug = cat.lower().replace(" ", "_")
    npy_path = os.path.join(BUCKETS_DIR, f"{cat_slug}.npy")
    json_path = os.path.join(BUCKETS_DIR, f"{cat_slug}_manifest.json")
    
    np.save(npy_path, cat_matrix)
    with open(json_path, "w") as f:
        json.dump(cat_manifest, f, indent=2)
        
    bucket_counts[cat] = len(indices)
    print(f"  - Bucket [{cat}]: {len(indices)} vectors -> {cat_slug}.npy")

print("\n=== HIERARCHICAL INDEX BUILDING COMPLETE ===")
print(json.dumps(bucket_counts, indent=2))
