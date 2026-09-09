import os
import json
import re
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

print("=== EMPIRICAL AE vs VAE RETRIEVAL & CLUSTERING BENCHMARK ===")

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

AE_PATH = os.path.join(MODELS_DIR, "280k dataset model", "AE_280K_best.keras")
VAE_ENCODER_PATH = os.path.join(MODELS_DIR, "280k model VAE (VAE v2)", "VAE_280K_Outputs", "encoder_280k_final.keras")

print(f"Loading AE model from {AE_PATH}...")
ae_full = tf.keras.models.load_model(AE_PATH)
ae_encoder = ae_full.get_layer("encoder")



print(f"Loading VAE Encoder from {VAE_ENCODER_PATH}...")
vae_encoder = tf.keras.models.load_model(VAE_ENCODER_PATH, custom_objects={'Sampling': Sampling})

print("Loading dataset held-out split...")
ds = load_dataset("evilsocket/alucard-sprites")
raw_data = ds["train"]

CATEGORY_PATTERNS = [
    ("Characters", r"\b(character|hero|knight|wizard|mage|spellcaster|warrior|rogue|ninja|paladin|archer|necromancer|cleric|bard|monk|priest|sorcerer|fighter|adventurer|human|dwarf|elf|orc|goblin|skeleton|zombie|demon|vampire)\b"),
    ("Enemies", r"\b(enemy|monster|boss|slime|bat|ghost|spider|dragon|golem|beast|creature|zombie|skeleton|demon|vampire|imp|snake|wolf|rat|giant|minotaur)\b"),
    ("Weapons", r"\b(weapon|sword|blade|dagger|bow|crossbow|staff|wand|axe|hammer|spear|shield|mace|scythe|rifle|gun|pistol)\b"),
    ("Items", r"\b(item|potion|elixir|scroll|book|ring|amulet|gem|jewel|crystal|coin|gold|chest|key|food|apple|bread|meat|bottle|flask|armor|helmet|boots)\b"),
    ("Props", r"\b(prop|decor|furniture|table|chair|torch|banner|door|window|statue|column|pillar|barrel|crate|box|sign|fence|gate|grave|tomb)\b"),
    ("Effects", r"\b(effect|spell|magic|fire|ice|lightning|spark|explosion|aura|trail|slash|impact|beam|particle|smoke|flame)\b"),
    ("Tiles", r"\b(tile|wall|floor|ground|grass|stone|brick|tree|rock|bush|water|lava|cliff|path|road|background)\b"),
]

def parse_category(text):
    if not text:
        return "Characters"
    text_lower = text.lower()
    for cat_label, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, text_lower):
            return cat_label
    return "Characters"

CANDIDATE_SIZE = 1500
QUERY_SIZE = 100
offset = 200000

print(f"Preparing {CANDIDATE_SIZE} candidate assets & ground-truth labels...")
images = []
labels = []

for i in range(offset, offset + CANDIDATE_SIZE):
    row = raw_data[i]
    img = row["image"].convert("RGBA").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    images.append(arr)
    labels.append(parse_category(row.get("text", "")))

X_candidates = np.array(images)
y_candidates = np.array(labels)

print("Encoding candidate assets through AE...")
z_ae = ae_encoder.predict(X_candidates, batch_size=256, verbose=0)
z_ae_flat = z_ae.reshape((len(z_ae), -1))
z_ae_norm = z_ae_flat / (np.linalg.norm(z_ae_flat, axis=1, keepdims=True) + 1e-8)

print("Encoding candidate assets through VAE...")
out_vae = vae_encoder.predict(X_candidates, batch_size=256, verbose=0)
z_vae = out_vae[0] if isinstance(out_vae, list) else out_vae
z_vae_norm = z_vae / (np.linalg.norm(z_vae, axis=1, keepdims=True) + 1e-8)

# 1. Retrieval Benchmark (Precision@1 & Precision@5 with Query Exclusion)
print("\n--- 1. Evaluating Retrieval Quality (Precision@1 & Precision@5) ---")

def evaluate_retrieval(z_matrix, query_indices):
    p1_scores = []
    p5_scores = []
    
    for q_idx in query_indices:
        q_vec = z_matrix[q_idx]
        q_label = y_candidates[q_idx]
        
        # Calculate L2 Latent Distance
        dists = np.linalg.norm(z_matrix - q_vec, axis=1)
        
        # EXCLUDE QUERY IMAGE ITSELF (set distance to infinity for self-index)
        dists[q_idx] = np.inf
        
        top_5_idx = np.argsort(dists)[:5]
        top_5_labels = y_candidates[top_5_idx]
        
        p1 = 1.0 if top_5_labels[0] == q_label else 0.0
        p5 = np.mean([1.0 if l == q_label else 0.0 for l in top_5_labels])
        
        p1_scores.append(p1)
        p5_scores.append(p5)
        
    return float(np.mean(p1_scores)), float(np.mean(p5_scores))

query_sample_indices = np.random.choice(CANDIDATE_SIZE, QUERY_SIZE, replace=False)
ae_p1, ae_p5 = evaluate_retrieval(z_ae_norm, query_sample_indices)
vae_p1, vae_p5 = evaluate_retrieval(z_vae_norm, query_sample_indices)

print(f"Autoencoder (AE)  -> Precision@1: {ae_p1*100:.2f}% | Precision@5: {ae_p5*100:.2f}%")
print(f"Variational AE    -> Precision@1: {vae_p1*100:.2f}% | Precision@5: {vae_p5*100:.2f}%")

# 2. Latent Clustering Evaluation (Silhouette, ARI, NMI)
print("\n--- 2. Evaluating Latent Space Clustering (K-Means K=7) ---")
cat_to_int = {cat: idx for idx, cat in enumerate(set(labels))}
y_true_int = np.array([cat_to_int[l] for l in labels])

# AE K-Means
kmeans_ae = KMeans(n_clusters=7, random_state=42, n_init=10).fit(z_ae_norm)
ae_sil = float(silhouette_score(z_ae_norm, kmeans_ae.labels_))
ae_ari = float(adjusted_rand_score(y_true_int, kmeans_ae.labels_))
ae_nmi = float(normalized_mutual_info_score(y_true_int, kmeans_ae.labels_))

# VAE K-Means
kmeans_vae = KMeans(n_clusters=7, random_state=42, n_init=10).fit(z_vae_norm)
vae_sil = float(silhouette_score(z_vae_norm, kmeans_vae.labels_))
vae_ari = float(adjusted_rand_score(y_true_int, kmeans_vae.labels_))
vae_nmi = float(normalized_mutual_info_score(y_true_int, kmeans_vae.labels_))

print(f"AE Clustering  -> Silhouette: {ae_sil:.4f} | ARI: {ae_ari:.4f} | NMI: {ae_nmi:.4f}")
print(f"VAE Clustering -> Silhouette: {vae_sil:.4f} | ARI: {vae_ari:.4f} | NMI: {vae_nmi:.4f}")

# 3. Empirical Reconstruction MSE Percentiles for Calibrated Anomaly Thresholds
print("\n--- 3. Calculating Empirical Anomaly Score Calibration ---")
recon_ae = ae_full.predict(X_candidates[:500], batch_size=256, verbose=0)
mse_per_sample = np.mean(np.square(X_candidates[:500] - recon_ae), axis=(1, 2, 3))

p50 = float(np.percentile(mse_per_sample, 50))
p90 = float(np.percentile(mse_per_sample, 90))
p95 = float(np.percentile(mse_per_sample, 95))
p99 = float(np.percentile(mse_per_sample, 99))

print(f"Empirical MSE Thresholds: Median(P50)={p50:.6f}, P90={p90:.6f}, P95={p95:.6f}, P99={p99:.6f}")

# Save Benchmark Summary JSON
results = {
    "num_query_evaluations": QUERY_SIZE,
    "candidate_pool_size": CANDIDATE_SIZE,
    "retrieval_benchmark": {
        "ae": {
            "precision_at_1": round(ae_p1, 4),
            "precision_at_5": round(ae_p5, 4)
        },
        "vae": {
            "precision_at_1": round(vae_p1, 4),
            "precision_at_5": round(vae_p5, 4)
        },
        "retrieval_winner": "VAE" if vae_p5 >= ae_p5 else "Autoencoder (AE)"
    },
    "clustering_benchmark": {
        "ae": {
            "silhouette_score": round(ae_sil, 4),
            "ari_score": round(ae_ari, 4),
            "nmi_score": round(ae_nmi, 4)
        },
        "vae": {
            "silhouette_score": round(vae_sil, 4),
            "ari_score": round(vae_ari, 4),
            "nmi_score": round(vae_nmi, 4)
        }
    },
    "empirical_anomaly_calibration": {
        "p50_median_mse": round(p50, 6),
        "p90_mse": round(p90, 6),
        "p95_outlier_threshold": round(p95, 6),
        "p99_severe_threshold": round(p99, 6)
    }
}

benchmark_path = os.path.join(EVAL_DIR, "retrieval_benchmark.json")
with open(benchmark_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved empirical benchmark results -> {benchmark_path}")
print("=== EMPIRICAL BENCHMARK COMPLETE ===")
