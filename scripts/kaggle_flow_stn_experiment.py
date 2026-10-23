"""
GameForge AI — Stage 4: Standalone Kaggle / Local Script for FlowSTN Experiment
Reproducible training and evaluation script for 2D character pose deformation.
"""

import os
import sys
import json
import time
import random
import zipfile
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model

JOINT_NAMES = [
    "head", "neck", "shoulder_left", "shoulder_right",
    "elbow_left", "elbow_right", "wrist_left", "wrist_right",
    "pelvis", "hip_left", "hip_right",
    "knee_left", "knee_right", "ankle_left", "ankle_right", "root"
]

@tf.keras.utils.register_keras_serializable()
class BilinearGridWarp(layers.Layer):
    """Differentiable 2D bilinear grid sampling layer."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        image, flow = inputs
        shape = tf.shape(image)
        batch_size, height, width = shape[0], shape[1], shape[2]
        h_f = tf.cast(height, tf.float32)
        w_f = tf.cast(width, tf.float32)

        x_lin = tf.linspace(0.0, 1.0, width)
        y_lin = tf.linspace(0.0, 1.0, height)
        grid_x, grid_y = tf.meshgrid(x_lin, y_lin)
        base_grid = tf.expand_dims(tf.stack([grid_x, grid_y], axis=-1), axis=0)

        sample_grid = base_grid + flow
        sample_x = tf.clip_by_value(sample_grid[..., 0] * (w_f - 1.0), 0.0, w_f - 1.0)
        sample_y = tf.clip_by_value(sample_grid[..., 1] * (h_f - 1.0), 0.0, h_f - 1.0)

        x0 = tf.floor(sample_x)
        x1 = tf.minimum(x0 + 1.0, w_f - 1.0)
        y0 = tf.floor(sample_y)
        y1 = tf.minimum(y0 + 1.0, h_f - 1.0)

        wa = tf.expand_dims((x1 - sample_x) * (y1 - sample_y), axis=-1)
        wb = tf.expand_dims((x1 - sample_x) * (sample_y - y0), axis=-1)
        wc = tf.expand_dims((sample_x - x0) * (y1 - sample_y), axis=-1)
        wd = tf.expand_dims((sample_x - x0) * (sample_y - y0), axis=-1)

        idx_x0 = tf.cast(x0, tf.int32)
        idx_x1 = tf.cast(x1, tf.int32)
        idx_y0 = tf.cast(y0, tf.int32)
        idx_y1 = tf.cast(y1, tf.int32)

        batch_idx = tf.broadcast_to(tf.range(batch_size)[:, tf.newaxis, tf.newaxis], [batch_size, height, width])

        def gather_channel(y_idx, x_idx):
            indices = tf.stack([batch_idx, y_idx, x_idx], axis=-1)
            return tf.gather_nd(image, indices)

        Ia = gather_channel(idx_y0, idx_x0)
        Ib = gather_channel(idx_y1, idx_x0)
        Ic = gather_channel(idx_y0, idx_x1)
        Id = gather_channel(idx_y1, idx_x1)

        warped = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return tf.clip_by_value(warped, 0.0, 1.0)

@tf.keras.utils.register_keras_serializable()
class FlowScalingLayer(layers.Layer):
    """Scales raw tanh flow field to maximum normalized displacement."""
    def __init__(self, max_displacement: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.max_displacement = max_displacement

    def call(self, inputs):
        return inputs * self.max_displacement

    def get_config(self):
        config = super().get_config()
        config.update({"max_displacement": self.max_displacement})
        return config

def build_flow_stn(image_shape=(128, 128, 4), pose_dim=32, max_displacement=0.25):
    src_input = layers.Input(shape=image_shape, name="source_image")
    pose_input = layers.Input(shape=(pose_dim,), name="pose_delta")

    d1 = layers.Dense(64, activation="relu")(pose_input)
    d2 = layers.Dense(16 * 16 * 16, activation="relu")(d1)
    d_spatial = layers.Reshape((16, 16, 16))(d2)
    d_up = layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(d_spatial)

    c1 = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(src_input)
    c2 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(c1)

    merged = layers.Concatenate()([c2, d_up])
    f1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(merged)
    f2 = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(f1)
    raw_flow = layers.Conv2D(2, (3, 3), padding="same", activation="tanh")(f2)
    flow_field = FlowScalingLayer(max_displacement=max_displacement, name="flow_field")(raw_flow)

    warped_output = BilinearGridWarp(name="bilinear_warp")([src_input, flow_field])
    return Model(inputs=[src_input, pose_input], outputs=[warped_output, flow_field], name="FlowSTN")

def extract_pose_delta(src_skel, tgt_skel):
    sj = src_skel.get("joints", {})
    tj = tgt_skel.get("joints", {})
    deltas = []
    for name in JOINT_NAMES:
        s_pt = sj.get(name, {})
        t_pt = tj.get(name, {})
        deltas.append(t_pt.get("x", 0.5) - s_pt.get("x", 0.5))
        deltas.append(t_pt.get("y", 0.5) - s_pt.get("y", 0.5))
    return np.array(deltas, dtype=np.float32)

def load_split_arrays(manifest, dataset_dir, split_name):
    split_samples = [s for s in manifest["samples"] if s["split"] == split_name]
    x_imgs, x_deltas, y_targets = [], [], []
    for s in split_samples:
        src_p = os.path.join(dataset_dir, s["source_image_path"])
        tgt_p = os.path.join(dataset_dir, s["target_image_path"])
        src_img = np.array(Image.open(src_p).convert("RGBA"), dtype=np.float32) / 255.0
        tgt_img = np.array(Image.open(tgt_p).convert("RGBA"), dtype=np.float32) / 255.0
        delta = extract_pose_delta(s["source_skeleton"], s["target_skeleton"])
        x_imgs.append(src_img)
        x_deltas.append(delta)
        y_targets.append(tgt_img)
    return np.array(x_imgs, dtype=np.float32), np.array(x_deltas, dtype=np.float32), np.array(y_targets, dtype=np.float32)

def main():
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    POSSIBLE_DATASET_PATHS = [
        "/kaggle/input/gameforge-synthetic-poses",
        "/kaggle/input/synthetic-poses",
        "./backend/data/synthetic_poses",
        "backend/data/synthetic_poses",
        "../backend/data/synthetic_poses"
    ]
    DATASET_DIR = next((p for p in POSSIBLE_DATASET_PATHS if os.path.exists(os.path.join(p, "manifest.json"))), None)
    if not DATASET_DIR:
        print("ERROR: Could not locate synthetic dataset directory.")
        sys.exit(1)

    print(f"Dataset path: {DATASET_DIR}")
    with open(os.path.join(DATASET_DIR, "manifest.json"), "r") as f:
        manifest = json.load(f)

    X_train_img, X_train_delta, Y_train_tgt = load_split_arrays(manifest, DATASET_DIR, "train")
    X_val_img, X_val_delta, Y_val_tgt = load_split_arrays(manifest, DATASET_DIR, "val")
    X_test_img, X_test_delta, Y_test_tgt = load_split_arrays(manifest, DATASET_DIR, "test")

    print(f"Data splits loaded: Train={len(X_train_img)}, Val={len(X_val_img)}, Test={len(X_test_img)}")

    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./kaggle_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = build_flow_stn()
    print(f"Model parameters: {model.count_params():,}")
    print("Script ready. Use notebook or run full training when GPUs are available.")

if __name__ == "__main__":
    main()
