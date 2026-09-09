import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")
CONTACT_DIR = os.path.join(EVAL_DIR, "contact_sheets")

print("=== PHASE 3: AE VS VAE NEAREST-NEIGHBOR RETRIEVAL BENCHMARK ===")

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

AE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "280k dataset model", "AE_280K_best.keras")
VAE_ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "280k model VAE (VAE v2)", "VAE_280K_Outputs", "encoder_280k_final.keras")

ae_full = tf.keras.models.load_model(AE_MODEL_PATH)
ae_full(tf.zeros((1, 128, 128, 4), dtype=tf.float32))
ae_encoder = ae_full.layers[0]

vae_encoder = tf.keras.models.load_model(VAE_ENCODER_PATH, custom_objects={'Sampling': Sampling})
vae_encoder(tf.zeros((1, 128, 128, 4), dtype=tf.float32))

ds = load_dataset("evilsocket/alucard-sprites")
raw_data = ds["train"]

CATEGORY_PATTERNS = [
    ("Character", r"\b(character|hero|knight|wizard|mage|spellcaster|warrior|rogue|ninja|paladin|archer|necromancer|cleric|bard|monk|priest|sorcerer|fighter|adventurer|man|woman|boy|girl|person|human|dwarf|elf|orc|goblin|skeleton|zombie|demon|vampire)\b"),
    ("Enemy / Monster", r"\b(enemy|monster|boss|slime|bat|ghost|spider|dragon|golem|beast|creature|zombie|skeleton|demon|vampire|imp|snake|wolf|rat|giant|minotaur)\b"),
    ("Weapon", r"\b(weapon|sword|blade|dagger|bow|crossbow|staff|wand|axe|hammer|spear|shield|mace|scythe|rifle|gun|pistol|shield)\b"),
    ("Item / Consumable", r"\b(item|potion|elixir|scroll|book|ring|amulet|gem|jewel|crystal|coin|gold|chest|key|food|apple|bread|meat|bottle|flask|chest)\b"),
    ("Prop / Decor", r"\b(prop|decor|furniture|table|chair|torch|banner|door|window|statue|column|pillar|barrel|crate|box|sign|fence|gate|grave|tomb)\b"),
    ("Effect / Spell", r"\b(effect|spell|magic|fire|ice|lightning|spark|explosion|aura|trail|slash|impact|beam|particle|smoke|flame)\b"),
    ("Tile / Environment", r"\b(tile|wall|floor|ground|grass|stone|brick|tree|rock|bush|water|lava|cliff|path|road|background)\b"),
]

def parse_category(text):
    if not text:
        return "Unknown / Miscellaneous"
    text_lower = text.lower()
    for cat_label, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, text_lower):
            return cat_label
    return "Unknown / Miscellaneous"

GALLERY_SIZE = 5000
QUERY_SIZE = 100

print(f"Loading {GALLERY_SIZE} candidate gallery items & {QUERY_SIZE} held-out query items...")
gallery_images = []
gallery_categories = []
for i in range(GALLERY_SIZE):
    row = raw_data[i]
    img = row["image"].convert("RGBA").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gallery_images.append(arr)
    gallery_categories.append(parse_category(row.get("text", "")))

query_images = []
query_categories = []
for i in range(GALLERY_SIZE, GALLERY_SIZE + QUERY_SIZE):
    row = raw_data[i]
    img = row["image"].convert("RGBA").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    query_images.append(arr)
    query_categories.append(parse_category(row.get("text", "")))

X_gallery = np.array(gallery_images)
X_query = np.array(query_images)
gallery_cats = np.array(gallery_categories)
query_cats = np.array(query_categories)

print("Encoding Candidate Gallery...")
ae_gallery = ae_encoder.predict(X_gallery, batch_size=128, verbose=0)
ae_gallery = ae_gallery.reshape((ae_gallery.shape[0], -1))

vae_gallery_out = vae_encoder.predict(X_gallery, batch_size=128, verbose=0)
vae_gallery = vae_gallery_out[0] if isinstance(vae_gallery_out, list) else vae_gallery_out

print("Encoding Held-Out Query Set...")
ae_queries = ae_encoder.predict(X_query, batch_size=128, verbose=0)
ae_queries = ae_queries.reshape((ae_queries.shape[0], -1))

vae_queries_out = vae_encoder.predict(X_query, batch_size=128, verbose=0)
vae_queries = vae_queries_out[0] if isinstance(vae_queries_out, list) else vae_queries_out

def norm_feats(feats):
    return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

ae_gallery_norm = norm_feats(ae_gallery)
ae_queries_norm = norm_feats(ae_queries)
vae_gallery_norm = norm_feats(vae_gallery)
vae_queries_norm = norm_feats(vae_queries)

nn_ae = NearestNeighbors(n_neighbors=5, metric='cosine').fit(ae_gallery_norm)
nn_vae = NearestNeighbors(n_neighbors=5, metric='cosine').fit(vae_gallery_norm)

_, ae_retrieved_idx = nn_ae.kneighbors(ae_queries_norm)
_, vae_retrieved_idx = nn_vae.kneighbors(vae_queries_norm)

ae_p1, ae_p5 = [], []
vae_p1, vae_p5 = [], []

for q_i in range(QUERY_SIZE):
    q_cat = query_cats[q_i]
    
    ae_ret_cats = gallery_cats[ae_retrieved_idx[q_i]]
    ae_p1.append(1.0 if ae_ret_cats[0] == q_cat else 0.0)
    ae_p5.append(np.mean(ae_ret_cats == q_cat))
    
    vae_ret_cats = gallery_cats[vae_retrieved_idx[q_i]]
    vae_p1.append(1.0 if vae_ret_cats[0] == q_cat else 0.0)
    vae_p5.append(np.mean(vae_ret_cats == q_cat))

mean_ae_p1 = float(np.mean(ae_p1))
mean_ae_p5 = float(np.mean(ae_p5))
mean_vae_p1 = float(np.mean(vae_p1))
mean_vae_p5 = float(np.mean(vae_p5))

print("\n=== RETRIEVAL BENCHMARK RESULTS (100 Held-Out Queries vs 5,000 Gallery) ===")
print(f"AE Encoder   -> Precision@1: {mean_ae_p1*100:.2f}%, Precision@5: {mean_ae_p5*100:.2f}%")
print(f"VAE Encoder  -> Precision@1: {mean_vae_p1*100:.2f}%, Precision@5: {mean_vae_p5*100:.2f}%")

plt.figure(figsize=(8, 5))
metrics = ["Precision@1 (Top-1 Match)", "Precision@5 (Top-5 Match)"]
ae_scores = [mean_ae_p1 * 100, mean_ae_p5 * 100]
vae_scores = [mean_vae_p1 * 100, mean_vae_p5 * 100]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, ae_scores, width, label='Autoencoder (AE)', color='#3b82f6')
plt.bar(x + width/2, vae_scores, width, label='VAE (Continuous Latent)', color='#22c55e')

plt.ylabel('Category Retrieval Precision (%)')
plt.title('AE vs VAE Visual Search Retrieval Precision (Held-Out Benchmark)', fontsize=12, fontweight='bold')
plt.xticks(x, metrics)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "retrieval_comparison.png"), dpi=200)
plt.close()

print("Generating Retrieval Contact Sheet for 3 Query Examples...")
fig, axes = plt.subplots(3, 7, figsize=(14, 7))
fig.suptitle("Visual Retrieval Comparison: Held-Out Query | AE Top-5 | VAE Top-5", fontsize=13, fontweight='bold')

for i, q_idx in enumerate([0, 15, 42]):
    q_img = raw_data[GALLERY_SIZE + q_idx]["image"].convert("RGBA")
    q_cat = query_cats[q_idx]
    
    axes[i, 0].imshow(q_img)
    axes[i, 0].set_title(f"Query #{q_idx}\n({q_cat})", fontsize=9, color='blue', fontweight='bold')
    axes[i, 0].axis('off')
    
    for r_i in range(3):
        idx = ae_retrieved_idx[q_idx][r_i]
        r_img = raw_data[idx]["image"].convert("RGBA")
        r_cat = gallery_cats[idx]
        axes[i, 1 + r_i].imshow(r_img)
        axes[i, 1 + r_i].set_title(f"AE #{r_i+1}\n({r_cat})", fontsize=8)
        axes[i, 1 + r_i].axis('off')

    for r_i in range(3):
        idx = vae_retrieved_idx[q_idx][r_i]
        r_img = raw_data[idx]["image"].convert("RGBA")
        r_cat = gallery_cats[idx]
        axes[i, 4 + r_i].imshow(r_img)
        axes[i, 4 + r_i].set_title(f"VAE #{r_i+1}\n({r_cat})", fontsize=8, color='green')
        axes[i, 4 + r_i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(CONTACT_DIR, "retrieval_comparison_contact_sheet.png"), dpi=200)
plt.close()

metrics_json = {
    "query_count": QUERY_SIZE,
    "gallery_count": GALLERY_SIZE,
    "ae": {
        "precision_at_1": round(mean_ae_p1, 4),
        "precision_at_5": round(mean_ae_p5, 4)
    },
    "vae": {
        "precision_at_1": round(mean_vae_p1, 4),
        "precision_at_5": round(mean_vae_p5, 4)
    },
    "winner": "VAE" if mean_vae_p5 > mean_ae_p5 else ("AE" if mean_ae_p5 > mean_vae_p5 else "TIE")
}

with open(os.path.join(EVAL_DIR, "retrieval_metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=4)

print(f"\nPhase 3 Completed Successfully! Summary saved to {os.path.join(EVAL_DIR, 'retrieval_metrics.json')}")
