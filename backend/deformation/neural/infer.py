"""
GameForge AI — Neural Deformation Inference Adapter
Stage 3 implementation: Production-ready inference wrapper for the trained FlowSTN model
with automatic graceful CPU fallback to the geometric baseline deformer.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Tuple, Optional

from deformation.schemas import Skeleton2D
from deformation.skeleton import JOINT_NAMES
from deformation.neural.model import BilinearGridWarp, FlowScalingLayer
from deformation.geometry import GeometricDeformer


def extract_pose_delta(src_skel: Skeleton2D, tgt_skel: Skeleton2D) -> np.ndarray:
    """Extracts 32-dimensional joint displacement vector from Skeleton2D instances."""
    deltas = []
    for name in JOINT_NAMES:
        sj = src_skel.get_joint(name)
        tj = tgt_skel.get_joint(name)
        sx = sj.x if sj else 0.5
        sy = sj.y if sj else 0.5
        tx = tj.x if tj else 0.5
        ty = tj.y if tj else 0.5
        deltas.append(tx - sx)
        deltas.append(ty - sy)
    return np.array([deltas], dtype=np.float32)  # Shape (1, 32)


class NeuralDeformationAdapter:
    """
    Adapter executing neural pose deformation inference.
    If the neural checkpoint is unavailable, gracefully falls back to GeometricDeformer.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.checkpoint_path = checkpoint_path or os.path.join(
            base_dir, "..", "models", "deformation_neural", "flow_stn_model.keras"
        )
        self.checkpoint_path = os.path.abspath(self.checkpoint_path)
        self.model = None
        self.fallback_deformer = GeometricDeformer()
        self._load_attempted = False

    def is_available(self) -> bool:
        """Checks whether the trained neural checkpoint exists on disk."""
        return os.path.exists(self.checkpoint_path)

    def load_model(self) -> bool:
        """Loads the trained Keras model with custom BilinearGridWarp layer."""
        if not self.is_available():
            return False
        try:
            self.model = tf.keras.models.load_model(
                self.checkpoint_path,
                custom_objects={
                    "BilinearGridWarp": BilinearGridWarp,
                    "FlowScalingLayer": FlowScalingLayer
                },
                safe_mode=False
            )
            self._load_attempted = True
            return True
        except Exception as e:
            print(f"Warning: Failed to load neural deformation checkpoint: {e}")
            self.model = None
            self._load_attempted = True
            return False


    def deform(
        self,
        image: Image.Image,
        source_skeleton: Skeleton2D,
        target_skeleton: Skeleton2D
    ) -> Image.Image:
        """
        Deforms a character sprite using the learned FlowSTN model.
        Falls back automatically to geometric deformer if neural model is offline.
        """
        if self.model is None and not self._load_attempted:
            self.load_model()

        if self.model is None:
            # Fallback to geometric baseline
            out_img, _ = self.fallback_deformer.deform(image, source_skeleton, target_skeleton)
            return out_img

        try:
            # Prepare inputs
            src_rgba = image.convert("RGBA")
            orig_size = src_rgba.size

            if orig_size != (128, 128):
                proc_img = src_rgba.resize((128, 128), Image.Resampling.NEAREST)
            else:
                proc_img = src_rgba

            img_arr = np.array(proc_img, dtype=np.float32) / 255.0
            img_tensor = np.expand_dims(img_arr, axis=0)  # (1, 128, 128, 4)

            delta_tensor = extract_pose_delta(source_skeleton, target_skeleton)  # (1, 32)

            # Neural Forward Pass
            warped_tensor, _ = self.model([img_tensor, delta_tensor], training=False)
            warped_arr = np.clip(warped_tensor[0].numpy() * 255.0, 0, 255).astype(np.uint8)

            out_img = Image.fromarray(warped_arr, mode="RGBA")
            if orig_size != (128, 128):
                out_img = out_img.resize(orig_size, Image.Resampling.NEAREST)

            return out_img
        except Exception as e:
            print(f"Inference error in neural deformer: {e}. Falling back to geometric baseline.")
            out_img, _ = self.fallback_deformer.deform(image, source_skeleton, target_skeleton)
            return out_img


neural_adapter = NeuralDeformationAdapter()
