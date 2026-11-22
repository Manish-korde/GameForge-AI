import os
import random
import json
import numpy as np

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from datasets import load_dataset

index_path = 'models/latent_gallery_index_25k.npy'
manifest_path = 'models/latent_gallery_manifest_25k.json'
gallery = np.load(index_path)
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Monkey patch layers
for cls in [tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose, tf.keras.layers.Flatten, tf.keras.layers.Reshape, tf.keras.layers.InputLayer]:
    orig = cls.__init__
    def make_patch(o):
        def p(self, *a, **kw):
            kw.pop('quantization_config', None)
            o(self, *a, **kw)
        return p
    cls.__init__ = make_patch(orig)

encoder_path = 'models/2_VAE_280k_v2/VAE_280K_Outputs/encoder_280k_final.keras'
encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})

ds = load_dataset('evilsocket/alucard-sprites')['train']

random.seed(42)
test_ids = random.sample(range(200000, 225000), 20)

print("=== COSINE SIMILARITY CHECK (20 RANDOM ROWS) ===")
sims = []
for row_id in test_ids:
    idx = row_id - 200000
    stored_vec = gallery[idx]
    
    row = ds[row_id]
    img = row['image'].convert('RGBA').resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    out = encoder.predict(arr, verbose=0)
    mu = out[0] if isinstance(out, list) else out
    computed_vec = mu[0] / (np.linalg.norm(mu[0]) + 1e-8)
    
    cos_sim = float(np.dot(stored_vec, computed_vec))
    sims.append(cos_sim)
    cat = manifest[idx]['category']
    txt = row.get('text', '')[:35]
    print(f"Row {row_id} (idx {idx:5d}): cos_sim = {cos_sim:.6f} | category = {cat:10s} | text = {txt}...")

print(f"\nSummary: Mean = {np.mean(sims):.6f}, Min = {np.min(sims):.6f}, Max = {np.max(sims):.6f}")
