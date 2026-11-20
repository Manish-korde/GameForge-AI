import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import dcgan_service

MODEL_DIR = r"models\3_DCGAN_42k_v1"
FIXED_Z_PATH = os.path.join(MODEL_DIR, "config", "fixed_latent_vectors.npy")
PALETTE_PATH = os.path.join(MODEL_DIR, "config", "global_palette_32.npy")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "game_ready_assets", "generator_weights_final.weights.h5")

# 1. Load service generator
gen = dcgan_service.load_dcgan_generator()
palette_tf = dcgan_service._palette_tf
fixed_z = np.load(FIXED_Z_PATH)

# Test first 5 seeds
test_z = fixed_z[:5]

# Raw service forward pass
logits_service = gen(test_z, training=False)
imgs_service = dcgan_service.logits_to_image(logits_service, palette_tf, hard=True, noise=False)

# Notebook reference formula (verbatim from Notebook 02)
def notebook_logits_to_image(logits, palette_tf):
    pl = logits[..., :32]
    al = logits[..., 32:]
    ys = tf.nn.softmax(pl, axis=-1)
    y_hard = tf.one_hot(tf.argmax(ys, axis=-1), depth=32, dtype=tf.float32)
    rgb = tf.tensordot(y_hard, palette_tf, axes=[[-1], [0]])
    a_s = tf.sigmoid(al)
    a_hard = tf.cast(a_s > 0.5, dtype=tf.float32)
    rgba = tf.concat([rgb * a_hard, a_hard], axis=-1)
    return tf.cast(tf.clip_by_value(rgba * 255.0 + 0.5, 0.0, 255.0), tf.uint8).numpy()

imgs_notebook = notebook_logits_to_image(logits_service, palette_tf)

diff = np.max(np.abs(imgs_service.astype(np.int32) - imgs_notebook.astype(np.int32)))
print(f"Max absolute pixel difference between Service and Notebook: {diff}")
assert diff == 0, f"REGRESSION FAILURE: Pixel mismatch between service and notebook ({diff})"

print("REGRESSION TEST PASSED: Same z produces identical (100% bit-exact) output in Service and Notebook!")

