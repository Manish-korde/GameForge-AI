"""
GameForge AI — Character Pose Deformation Service
Orchestrates image validation, skeleton fitting, geometric deformation,
optional VAE hybrid refinement, and automated quality response assembly.
"""

import io
import time
import base64
from PIL import Image
from typing import Optional, Tuple

from .schemas import (
    DeformationRequest,
    DeformationResponse,
    DeformationMethod,
    Skeleton2D,
    BoundingBox,
    ValidationResult
)
from .skeleton import (
    get_default_skeleton,
    get_template_by_name,
    fit_skeleton_to_bbox
)
from .geometry import GeometricDeformer
from .vae_adapter import VAEDeformationAdapter


class PoseDeformationService:
    """High-level service coordinating 2D pose deformation workflows."""

    def __init__(self):
        self.deformer = GeometricDeformer()
        self.vae_adapter = VAEDeformationAdapter()

    @staticmethod
    def decode_base64_image(image_b64: str) -> Image.Image:
        """Parses a Base64 data URI or raw string into a PIL RGBA Image."""
        if not image_b64:
            raise ValueError("No image data provided.")

        clean_b64 = image_b64
        if "data:" in clean_b64 and ";base64," in clean_b64:
            clean_b64 = clean_b64.split(";base64,")[1]

        try:
            image_bytes = base64.b64decode(clean_b64)
            img = Image.open(io.BytesIO(image_bytes))
            return img.convert("RGBA")
        except Exception as e:
            raise ValueError(f"Invalid Base64 image payload: {e}")

    @staticmethod
    def encode_base64_image(image: Image.Image) -> str:
        """Encodes a PIL Image into a PNG Base64 Data URI."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def load_local_image_from_url(image_url: str) -> Image.Image:
        """Safely loads an image from the local alucard_samples directory without arbitrary network calls (SSRF protection)."""
        import os
        import re
        match = re.search(r'alucard_(\d+)\.png', image_url)
        if match:
            sample_idx = int(match.group(1))
            sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "alucard_samples", f"alucard_{sample_idx}.png"))
            if os.path.exists(sample_path):
                img = Image.open(sample_path)
                return img.convert("RGBA")
        raise ValueError("Arbitrary remote image URLs are disallowed for security. Provide a valid local sample or 'image_b64'.")

    def process_request(self, req: DeformationRequest) -> DeformationResponse:
        """Main entrypoint executing end-to-end character pose deformation."""
        start_time = time.perf_counter()

        # 1. Decode & Validate Image
        if req.image_b64:
            source_img = self.decode_base64_image(req.image_b64)
        elif req.image_url:
            source_img = self.load_local_image_from_url(req.image_url)
        else:
            raise ValueError("Must provide either 'image_b64' or a valid local sample 'image_url'.")

        # Enforce size sanity
        if source_img.width > 512 or source_img.height > 512:
            raise ValueError(f"Sprite dimensions ({source_img.width}x{source_img.height}) exceed maximum allowed 512x512.")
        if source_img.width < 16 or source_img.height < 16:
            raise ValueError(f"Sprite dimensions ({source_img.width}x{source_img.height}) are too small for skeleton binding.")

        # Standardize to 128x128 for consistent bone calibration if requested
        if source_img.size != (128, 128) and req.maintain_canvas_size:
            # Resize with nearest-neighbor to retain pixel art integrity
            source_img = source_img.resize((128, 128), Image.Resampling.NEAREST)

        # 2. Detect Bounding Box
        bbox = self.deformer.detect_bounding_box(source_img)
        if bbox.non_transparent_pixels == 0:
            raise ValueError("Uploaded image is completely transparent; cannot deform an empty sprite.")

        # 3. Resolve Source Skeleton
        if req.source_skeleton and len(req.source_skeleton.joints) >= 6:
            src_skel = req.source_skeleton
        else:
            default_skel = get_default_skeleton()
            src_skel = fit_skeleton_to_bbox(default_skel, bbox, source_img.width, source_img.height)

        # 4. Resolve Target Skeleton
        if req.target_skeleton and len(req.target_skeleton.joints) >= 6:
            tgt_skel = req.target_skeleton
        elif req.target_pose:
            template = get_template_by_name(req.target_pose)
            if not template:
                raise ValueError(f"Unknown target pose template: '{req.target_pose}'. Use /deformation/templates to list available templates.")
            tgt_skel = fit_skeleton_to_bbox(template.skeleton, bbox, source_img.width, source_img.height)
        else:
            # Default to idle
            default_template = get_template_by_name("idle")
            tgt_skel = fit_skeleton_to_bbox(default_template.skeleton, bbox, source_img.width, source_img.height)

        # 5. Execute Geometric Deformation
        out_img, val_result = self.deformer.deform(
            image=source_img,
            source_skeleton=src_skel,
            target_skeleton=tgt_skel,
            intensity=req.intensity,
            resampling=req.resampling,
            preserve_pixel_art=req.preserve_pixel_art
        )

        executed_method = "geometric"

        # 6. Optional Experimental VAE or Hybrid Execution
        if req.method in [DeformationMethod.HYBRID, DeformationMethod.VAE_EXPERIMENTAL]:
            out_img, vae_meta = self.vae_adapter.hybrid_refine(out_img, alpha_blend=0.35)
            if vae_meta.get("success"):
                executed_method = "hybrid_geometric_vae"
                val_result.warnings.append(
                    f"Hybrid VAE refinement applied (Recon MSE: {vae_meta.get('vae_recon_mse')})."
                )
            else:
                val_result.warnings.append(
                    f"VAE refinement skipped: {vae_meta.get('error')}. Returned crisp geometric baseline."
                )

        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        out_bbox = self.deformer.detect_bounding_box(out_img)

        return DeformationResponse(
            status="success" if val_result.is_valid else "warning",
            method=executed_method,
            output_image_b64=self.encode_base64_image(out_img),
            source_image_b64=self.encode_base64_image(source_img),
            source_skeleton=src_skel,
            target_skeleton=tgt_skel,
            bounding_box=out_bbox,
            validation=val_result,
            processing_time_ms=duration_ms,
            message=f"Deformed sprite using {executed_method} method in {duration_ms}ms."
        )


deformation_service = PoseDeformationService()
