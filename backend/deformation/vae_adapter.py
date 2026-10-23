"""
GameForge AI — VAE Pose Feasibility Adapter
Provides an isolated, non-destructive adapter interfacing with the 280K β-VAE
to empirically test latent perturbation, interpolation, and hybrid geometric-VAE deformation.
"""

import os
import io
import time
import base64
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEDeformationAdapter:
    """
    Adapter around the trained 280K β-VAE model checkpoints.
    Executes controlled latent experiments without mutating global backend state.
    """

    def __init__(self, encoder_path: Optional[str] = None, decoder_path: Optional[str] = None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.encoder_path = encoder_path or os.path.join(
            base_dir, "models", "280k model VAE (VAE v2)", "VAE_280K_Outputs", "encoder_280k_final.keras"
        )
        self.decoder_path = decoder_path or os.path.join(
            base_dir, "models", "280k model VAE (VAE v2)", "VAE_280K_Outputs", "decoder_280k_final.keras"
        )

        self.encoder = None
        self.decoder = None
        self.is_loaded = False
        self.load_error = None

    def load_models(self) -> bool:
        """Loads VAE encoder and decoder models lazily."""
        if self.is_loaded:
            return True

        if not os.path.exists(self.encoder_path) or not os.path.exists(self.decoder_path):
            self.load_error = f"VAE checkpoints not found at {self.encoder_path} or {self.decoder_path}"
            return False

        try:
            custom_objs = {"Sampling": Sampling}
            self.encoder = tf.keras.models.load_model(self.encoder_path, custom_objects=custom_objs, compile=False)
            self.decoder = tf.keras.models.load_model(self.decoder_path, custom_objects=custom_objs, compile=False)
            self.is_loaded = True
            self.load_error = None
            return True
        except Exception as e:
            self.load_error = str(e)
            return False

    @staticmethod
    def preprocess_image(image: Image.Image) -> np.ndarray:
        """Preprocesses a PIL Image into a normalized tensor (1, 128, 128, 4) in [0.0, 1.0]."""
        img = image.convert("RGBA").resize((128, 128))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    @staticmethod
    def postprocess_tensor(tensor: np.ndarray, threshold_alpha: bool = True) -> Image.Image:
        """Converts an output tensor (1, 128, 128, 4) back to a PIL RGBA Image."""
        arr = tensor[0] * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if threshold_alpha:
            alpha = arr[:, :, 3]
            alpha[alpha < 35] = 0
            arr[:, :, 3] = alpha
        return Image.fromarray(arr, mode="RGBA")

    def encode(self, image: Image.Image) -> Optional[np.ndarray]:
        """Encodes an image to its 256-d latent mean vector z_mean."""
        if not self.load_models():
            return None
        tensor = self.preprocess_image(image)
        enc_out = self.encoder.predict(tensor, verbose=0)
        z_mean = enc_out[0] if isinstance(enc_out, list) else enc_out
        return z_mean[0]

    def decode(self, latent_vector: np.ndarray) -> Optional[Image.Image]:
        """Decodes a 256-d latent vector into a 128x128 RGBA image."""
        if not self.load_models():
            return None
        z_batch = np.expand_dims(latent_vector, axis=0)
        dec_out = self.decoder.predict(z_batch, verbose=0)
        return self.postprocess_tensor(dec_out)

    def hybrid_refine(
        self,
        geometric_image: Image.Image,
        alpha_blend: float = 0.5
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Applies a Hybrid Geometric + VAE Refinement pipeline:
        1. Encodes the geometrically deformed image into VAE latent space.
        2. Decodes the latent vector back to pixel space.
        3. Blends the high-frequency pixel borders of the geometric output
           with the VAE smoothed output to evaluate reconstruction vs blur.
        """
        start_time = time.perf_counter()
        if not self.load_models():
            return geometric_image, {
                "success": False,
                "error": self.load_error or "VAE models unavailable",
                "method": "geometric_fallback"
            }

        z = self.encode(geometric_image)
        vae_reconstruction = self.decode(z)

        # Blend while strictly preserving geometric alpha mask
        geo_arr = np.array(geometric_image.convert("RGBA"), dtype=np.float32)
        vae_arr = np.array(vae_reconstruction.convert("RGBA"), dtype=np.float32)

        # RGB blend
        blended_rgb = (1.0 - alpha_blend) * geo_arr[:, :, :3] + alpha_blend * vae_arr[:, :, :3]

        # Preserve the crisp discrete geometric alpha channel
        blended = np.zeros_like(geo_arr, dtype=np.uint8)
        blended[:, :, :3] = np.clip(blended_rgb, 0, 255).astype(np.uint8)
        blended[:, :, 3] = geo_arr[:, :, 3].astype(np.uint8)

        result_img = Image.fromarray(blended, mode="RGBA")
        duration = (time.perf_counter() - start_time) * 1000.0

        # Calculate pixel difference (MSE) between geometric and VAE
        norm_geo = geo_arr / 255.0
        norm_vae = vae_arr / 255.0
        mse = float(np.mean(np.square(norm_geo - norm_vae)))

        return result_img, {
            "success": True,
            "method": "hybrid_geometric_vae",
            "duration_ms": round(duration, 2),
            "vae_recon_mse": round(mse, 6),
            "alpha_blend_ratio": alpha_blend,
            "latent_norm": round(float(np.linalg.norm(z)), 4)
        }
