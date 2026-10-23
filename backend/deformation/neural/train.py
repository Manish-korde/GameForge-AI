"""
GameForge AI — Neural Deformation Training Pipeline
Stage 3 implementation: Reproducible training script for the Dense Flow STN network
on identity-partitioned synthetic pose pairs.
"""

import os
import json
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Dict, Tuple, List, Optional

from deformation.skeleton import JOINT_NAMES
from deformation.neural.model import build_flow_stn_model


def extract_pose_delta_vector(src_skel_dict: Dict, tgt_skel_dict: Dict) -> np.ndarray:
    """
    Computes a 32-dimensional normalized displacement vector (16 joints * 2)
    delta = [dx_1, dy_1, dx_2, dy_2, ..., dx_16, dy_16]
    """
    src_joints = src_skel_dict.get("joints", {})
    tgt_joints = tgt_skel_dict.get("joints", {})

    deltas = []
    for name in JOINT_NAMES:
        sj = src_joints.get(name, {})
        tj = tgt_joints.get(name, {})
        sx = sj.get("x", 0.5)
        sy = sj.get("y", 0.5)
        tx = tj.get("x", 0.5)
        ty = tj.get("y", 0.5)
        deltas.append(tx - sx)
        deltas.append(ty - sy)

    return np.array(deltas, dtype=np.float32)


def load_dataset_arrays(
    dataset_dir: str,
    split_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads pre-rendered PNGs and skeletons into numpy float32 arrays:
    - X_images: (N, 128, 128, 4) in [0.0, 1.0]
    - X_deltas: (N, 32)
    - Y_targets: (N, 128, 128, 4) in [0.0, 1.0]
    """
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = [s for s in manifest.get("samples", []) if s.get("split") == split_name]

    x_imgs = []
    x_deltas = []
    y_targets = []

    for s in samples:
        src_path = os.path.join(dataset_dir, s["source_image_path"])
        tgt_path = os.path.join(dataset_dir, s["target_image_path"])

        src_arr = np.array(Image.open(src_path).convert("RGBA"), dtype=np.float32) / 255.0
        tgt_arr = np.array(Image.open(tgt_path).convert("RGBA"), dtype=np.float32) / 255.0
        delta = extract_pose_delta_vector(s["source_skeleton"], s["target_skeleton"])

        x_imgs.append(src_arr)
        x_deltas.append(delta)
        y_targets.append(tgt_arr)

    return (
        np.array(x_imgs, dtype=np.float32),
        np.array(x_deltas, dtype=np.float32),
        np.array(y_targets, dtype=np.float32)
    )


class FlowSTNTrainer:
    """Orchestrates controlled neural deformation training."""

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        seed: int = 42
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.dataset_dir = dataset_dir or os.path.join(base_dir, "data", "synthetic_poses")
        self.checkpoint_dir = checkpoint_dir or os.path.join(base_dir, "..", "models", "deformation_neural")
        self.checkpoint_dir = os.path.abspath(self.checkpoint_dir)
        self.seed = seed

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        tv_weight: float = 0.01
    ) -> Dict:
        """
        Executes a controlled training run.
        Uses L1 photometric reconstruction loss + Total Variation flow smoothness.
        """
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        start_time = time.perf_counter()

        # 1. Load Train and Validation Splits
        print(f"Loading synthetic dataset from {self.dataset_dir}...")
        train_imgs, train_deltas, train_tgts = load_dataset_arrays(self.dataset_dir, "train")
        val_imgs, val_deltas, val_tgts = load_dataset_arrays(self.dataset_dir, "val")

        print(f"Loaded {len(train_imgs)} train pairs, {len(val_imgs)} val pairs.")

        # 2. Build Model
        model = build_flow_stn_model(image_shape=(128, 128, 4), pose_dim=32, max_displacement=0.25)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # 3. Custom Training Step with Combined Loss
        @tf.function
        def train_step(src_b, delta_b, tgt_b):
            with tf.GradientTape() as tape:
                pred_img, flow = model([src_b, delta_b], training=True)
                # L1 photometric loss
                l1_loss = tf.reduce_mean(tf.abs(pred_img - tgt_b))
                # Total variation flow smoothness loss
                tv_loss = tf.reduce_mean(tf.image.total_variation(flow))
                total_loss = l1_loss + tv_weight * tv_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss, l1_loss, tv_loss

        @tf.function
        def val_step(src_b, delta_b, tgt_b):
            pred_img, flow = model([src_b, delta_b], training=False)
            l1_loss = tf.reduce_mean(tf.abs(pred_img - tgt_b))
            tv_loss = tf.reduce_mean(tf.image.total_variation(flow))
            total_loss = l1_loss + tv_weight * tv_loss
            return total_loss, l1_loss, tv_loss

        # 4. Training Loop
        history = {
            "epoch": [],
            "train_loss": [],
            "train_l1": [],
            "val_loss": [],
            "val_l1": []
        }

        n_train = len(train_imgs)
        indices = np.arange(n_train)

        print(f"Starting Flow STN training ({epochs} epochs, batch_size={batch_size})...")
        for epoch in range(1, epochs + 1):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            epoch_l1 = 0.0
            batches = 0

            for i in range(0, n_train, batch_size):
                b_idx = indices[i:i + batch_size]
                src_b = tf.constant(train_imgs[b_idx])
                delta_b = tf.constant(train_deltas[b_idx])
                tgt_b = tf.constant(train_tgts[b_idx])

                tot_l, l1_l, _ = train_step(src_b, delta_b, tgt_b)
                epoch_loss += float(tot_l)
                epoch_l1 += float(l1_l)
                batches += 1

            train_loss_avg = round(epoch_loss / max(batches, 1), 5)
            train_l1_avg = round(epoch_l1 / max(batches, 1), 5)

            # Validation pass
            val_src = tf.constant(val_imgs)
            val_delta = tf.constant(val_deltas)
            val_tgt = tf.constant(val_tgts)
            val_tot, val_l1, _ = val_step(val_src, val_delta, val_tgt)

            val_loss_avg = round(float(val_tot), 5)
            val_l1_avg = round(float(val_l1), 5)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss_avg)
            history["train_l1"].append(train_l1_avg)
            history["val_loss"].append(val_loss_avg)
            history["val_l1"].append(val_l1_avg)

            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss_avg} (L1: {train_l1_avg}) | Val Loss: {val_loss_avg} (L1: {val_l1_avg})")

        duration_sec = round(time.perf_counter() - start_time, 2)

        # 5. Save Model Checkpoint & Metadata
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, "flow_stn_model.keras")
        model.save(checkpoint_path)

        meta = {
            "model_architecture": "FlowSTN",
            "parameters": int(model.count_params()),
            "trained_epochs": epochs,
            "training_duration_sec": duration_sec,
            "final_train_l1": history["train_l1"][-1],
            "final_val_l1": history["val_l1"][-1],
            "checkpoint_path": checkpoint_path,
            "history": history
        }

        meta_path = os.path.join(self.checkpoint_dir, "training_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Model saved to {checkpoint_path} in {duration_sec}s.")
        return meta


trainer = FlowSTNTrainer()
