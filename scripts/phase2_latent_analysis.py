import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")
CONTACT_DIR = os.path.join(EVAL_DIR, "contact_sheets")

for d in [EVAL_DIR, PLOTS_DIR, CONTACT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=== PHASE 2: LATENT EMBEDDING & CLASSIFICATION / CLUSTERING ANALYSIS ===")

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

print(f"Loading AE model from {AE_MODEL_PATH}...")
ae_full = tf.keras.models.load_model(AE_MODEL_PATH)
ae_full(tf.zeros((1, 128, 128, 4), dtype=tf.float32))
ae_encoder = ae_full.layers[0]
print(f"AE Encoder extracted successfully: {ae_encoder.name}")

print(f"Loading VAE Encoder from {VAE_ENCODER_PATH}...")
vae_encoder = tf.keras.models.load_model(VAE_ENCODER_PATH, custom_objects={'Sampling': Sampling})
vae_encoder(tf.zeros((1, 128, 128, 4), dtype=tf.float32))
print("VAE Encoder loaded successfully.")

print("Loading dataset samples...")
ds = load_dataset("evilsocket/alucard-sprites")
raw_data = ds["train"]

import re
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

SAMPLE_SIZE = 5000
images = []
labels = []

print(f"Extracting features for {SAMPLE_SIZE} dataset samples...")
for i in range(SAMPLE_SIZE):
    row = raw_data[i]
    img = row["image"].convert("RGBA").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    cat = parse_category(row.get("text", ""))
    images.append(arr)
    labels.append(cat)

X_batch = np.array(images)
y_labels = np.array(labels)

print("Computing AE embeddings...")
ae_embeddings = ae_encoder.predict(X_batch, batch_size=128, verbose=0)
ae_embeddings = ae_embeddings.reshape((ae_embeddings.shape[0], -1))

print("Computing VAE embeddings (mean μ)...")
vae_out = vae_encoder.predict(X_batch, batch_size=128, verbose=0)
vae_mu = vae_out[0] if isinstance(vae_out, list) else vae_out

print(f"AE Embedding Shape: {ae_embeddings.shape}")
print(f"VAE μ Embedding Shape: {vae_mu.shape}")

split = int(0.8 * SAMPLE_SIZE)
X_train_ae, X_test_ae = ae_embeddings[:split], ae_embeddings[split:]
X_train_vae, X_test_vae = vae_mu[:split], vae_mu[split:]
y_train, y_test = y_labels[:split], y_labels[split:]

print("\n--- KNN CLASSIFICATION (Category Accuracy) ---")
knn_ae = KNeighborsClassifier(n_neighbors=5)
knn_ae.fit(X_train_ae, y_train)
y_pred_ae = knn_ae.predict(X_test_ae)
acc_ae = accuracy_score(y_test, y_pred_ae)
f1_ae = f1_score(y_test, y_pred_ae, average='macro')

knn_vae = KNeighborsClassifier(n_neighbors=5)
knn_vae.fit(X_train_vae, y_train)
y_pred_vae = knn_vae.predict(X_test_vae)
acc_vae = accuracy_score(y_test, y_pred_vae)
f1_vae = f1_score(y_test, y_pred_vae, average='macro')

print(f"AE KNN (K=5)  -> Accuracy: {acc_ae*100:.2f}%, Macro F1: {f1_ae:.4f}")
print(f"VAE KNN (K=5) -> Accuracy: {acc_vae*100:.2f}%, Macro F1: {f1_vae:.4f}")

n_clusters = len(np.unique(y_labels))
print(f"\n--- K-MEANS CLUSTERING (K={n_clusters}) ---")

kmeans_ae = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels_ae = kmeans_ae.fit_predict(ae_embeddings)
sil_ae = silhouette_score(ae_embeddings, cluster_labels_ae)
ari_ae = adjusted_rand_score(y_labels, cluster_labels_ae)
nmi_ae = normalized_mutual_info_score(y_labels, cluster_labels_ae)

kmeans_vae = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels_vae = kmeans_vae.fit_predict(vae_mu)
sil_vae = silhouette_score(vae_mu, cluster_labels_vae)
ari_vae = adjusted_rand_score(y_labels, cluster_labels_vae)
nmi_vae = normalized_mutual_info_score(y_labels, cluster_labels_vae)

print(f"AE K-Means  -> Silhouette: {sil_ae:.4f}, ARI: {ari_ae:.4f}, NMI: {nmi_ae:.4f}")
print(f"VAE K-Means -> Silhouette: {sil_vae:.4f}, ARI: {ari_vae:.4f}, NMI: {nmi_vae:.4f}")

print("\nGenerating 2D PCA Projections...")
pca_ae = PCA(n_components=2).fit_transform(ae_embeddings)
pca_vae = PCA(n_components=2).fit_transform(vae_mu)

unique_cats = np.unique(y_labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for idx, cat in enumerate(unique_cats):
    mask = y_labels == cat
    ax1.scatter(pca_ae[mask, 0], pca_ae[mask, 1], label=cat, alpha=0.6, s=15)
    ax2.scatter(pca_vae[mask, 0], pca_vae[mask, 1], label=cat, alpha=0.6, s=15)

ax1.set_title("AE Latent Space 2D PCA Projection", fontsize=12, fontweight='bold')
ax1.set_xlabel("PCA Dim 1")
ax1.set_ylabel("PCA Dim 2")
ax1.legend(fontsize=8, loc='best')

ax2.set_title("VAE Latent Space 2D PCA Projection", fontsize=12, fontweight='bold')
ax2.set_xlabel("PCA Dim 1")
ax2.set_ylabel("PCA Dim 2")
ax2.legend(fontsize=8, loc='best')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "latent_pca_comparison.png"), dpi=200)
plt.close()

print("Generating VAE Cluster Contact Sheet...")
cluster_samples = {c: [] for c in range(n_clusters)}
for idx, cl in enumerate(cluster_labels_vae):
    if len(cluster_samples[cl]) < 6:
        cluster_samples[cl].append(idx)

fig, axes = plt.subplots(n_clusters, 6, figsize=(12, n_clusters * 2))
fig.suptitle("VAE Unsupervised Clusters (Sample Assets Per Cluster)", fontsize=14, fontweight='bold')

for cl in range(n_clusters):
    for i in range(6):
        ax = axes[cl, i]
        ax.axis('off')
        if i < len(cluster_samples[cl]):
            img_idx = cluster_samples[cl][i]
            img = raw_data[img_idx]["image"].convert("RGBA")
            ax.imshow(img)
            if i == 0:
                ax.set_title(f"Cluster #{cl+1}", fontsize=10, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(os.path.join(CONTACT_DIR, "vae_cluster_contact_sheet.png"), dpi=200)
plt.close()

metrics_json = {
    "sample_size": SAMPLE_SIZE,
    "ae_embedding_dim": int(ae_embeddings.shape[1]),
    "vae_embedding_dim": int(vae_mu.shape[1]),
    "knn_classification": {
        "ae": {"accuracy": round(float(acc_ae), 4), "macro_f1": round(float(f1_ae), 4)},
        "vae": {"accuracy": round(float(acc_vae), 4), "macro_f1": round(float(f1_vae), 4)}
    },
    "kmeans_clustering": {
        "ae": {"silhouette_score": round(float(sil_ae), 4), "ari": round(float(ari_ae), 4), "nmi": round(float(nmi_ae), 4)},
        "vae": {"silhouette_score": round(float(sil_vae), 4), "ari": round(float(ari_vae), 4), "nmi": round(float(nmi_vae), 4)}
    }
}

with open(os.path.join(EVAL_DIR, "latent_analysis.json"), "w") as f:
    json.dump(metrics_json, f, indent=4)

print(f"\nPhase 2 Completed Successfully! Summary saved to {os.path.join(EVAL_DIR, 'latent_analysis.json')}")
