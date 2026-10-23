"""
GameForge AI — 2D Character Geometric Deformation Engine
Implements 2D skeletal linear blend skinning (LBS), inverse distance bone weighting,
and coordinate mapping with integrated pose constraint and quality validation.
"""

import math
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional

from .schemas import (
    Skeleton2D,
    BoundingBox,
    ValidationResult,
    ResamplingPolicy
)
from .skeleton import (
    BONE_CONNECTIONS,
    get_default_skeleton,
    fit_skeleton_to_bbox
)
from .pose_constraints import (
    validate_skeleton_topology,
    evaluate_pose_constraints
)
from .pixel_art import resample_sprite_grid
from .quality import evaluate_quality_metrics


class GeometricDeformer:
    """
    High-performance 2D character skeletal deformation engine.
    Applies inverse distance-weighted bone coordinate mapping with strict nearest-neighbor
    pixel-art sampling, preserving discrete color palettes and zero background bleed.
    """

    def __init__(self, power: float = 2.0, bone_radius_factor: float = 0.25):
        self.power = power
        self.bone_radius_factor = bone_radius_factor

    @staticmethod
    def detect_bounding_box(image: Image.Image, alpha_threshold: int = 15) -> BoundingBox:
        """
        Detects the non-transparent pixel bounding box of an RGBA image.
        Returns BoundingBox with min/max bounds, dimensions, and fill ratio.
        """
        img_rgba = image.convert("RGBA")
        np_img = np.array(img_rgba)
        alpha = np_img[:, :, 3]

        mask = alpha > alpha_threshold
        non_zero_coords = np.argwhere(mask)

        if non_zero_coords.size == 0:
            return BoundingBox(
                min_x=0, min_y=0, max_x=img_rgba.width, max_y=img_rgba.height,
                width=0, height=0, non_transparent_pixels=0, fill_ratio=0.0
            )

        min_y, min_x = non_zero_coords.min(axis=0)
        max_y, max_x = non_zero_coords.max(axis=0)

        width = int(max_x - min_x + 1)
        height = int(max_y - min_y + 1)
        total_pixels = int(np.sum(mask))
        canvas_pixels = image.width * image.height
        fill_ratio = round(float(total_pixels / canvas_pixels), 4)

        return BoundingBox(
            min_x=int(min_x),
            min_y=int(min_y),
            max_x=int(max_x),
            max_y=int(max_y),
            width=width,
            height=height,
            non_transparent_pixels=total_pixels,
            fill_ratio=fill_ratio
        )

    def auto_detect_skeleton(self, image: Image.Image) -> Skeleton2D:
        """
        Automatically aligns a default canonical skeleton to the sprite's non-transparent bounding box.
        """
        bbox = self.detect_bounding_box(image)
        default_skel = get_default_skeleton()
        return fit_skeleton_to_bbox(default_skel, bbox, image.width, image.height)

    @staticmethod
    def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Computes the perpendicular distance from a grid of 2D points to a line segment AB.
        p: shape (N, 2)
        a: shape (2,)
        b: shape (2,)
        returns: shape (N,) distances
        """
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-8:
            return np.linalg.norm(p - a, axis=1)

        # Projection factor t = ((p - a) . ab) / |ab|^2
        t = np.dot(p - a, ab) / ab_len_sq
        t = np.clip(t, 0.0, 1.0)
        projection = a + t[:, np.newaxis] * ab
        return np.linalg.norm(p - projection, axis=1)

    def deform(
        self,
        image: Image.Image,
        source_skeleton: Skeleton2D,
        target_skeleton: Skeleton2D,
        intensity: float = 1.0,
        resampling: ResamplingPolicy = ResamplingPolicy.NEAREST,
        preserve_pixel_art: bool = True
    ) -> Tuple[Image.Image, ValidationResult]:
        """
        Deforms a source sprite from source_skeleton to target_skeleton using backward skeletal mapping.
        """
        width, height = image.size
        img_rgba = image.convert("RGBA")
        src_array = np.array(img_rgba)

        warnings: List[str] = []

        # 0. Validate Target Skeleton Topology and Pose Constraints
        _, topo_errors = validate_skeleton_topology(target_skeleton)
        if topo_errors:
            warnings.extend(topo_errors)

        _, constraint_warnings = evaluate_pose_constraints(target_skeleton, reference_skeleton=source_skeleton)
        if constraint_warnings:
            warnings.extend(constraint_warnings)

        # 1. Compute Bone Transforms (Source -> Target)
        bone_transforms = []
        max_stretch_ratio = 1.0

        for p1_name, p2_name in BONE_CONNECTIONS:
            s1 = source_skeleton.get_joint(p1_name)
            s2 = source_skeleton.get_joint(p2_name)
            t1 = target_skeleton.get_joint(p1_name)
            t2 = target_skeleton.get_joint(p2_name)

            if not s1 or not s2 or not t1 or not t2:
                continue

            # Convert normalized coordinates to pixel space
            s_a = np.array([s1.x * width, s1.y * height], dtype=np.float32)
            s_b = np.array([s2.x * width, s2.y * height], dtype=np.float32)

            # Apply intensity blend: blended_target = source + intensity * (target - source)
            eff_t1_x = s1.x + intensity * (t1.x - s1.x)
            eff_t1_y = s1.y + intensity * (t1.y - s1.y)
            eff_t2_x = s2.x + intensity * (t2.x - s2.x)
            eff_t2_y = s2.y + intensity * (t2.y - s2.y)

            t_a = np.array([eff_t1_x * width, eff_t1_y * height], dtype=np.float32)
            t_b = np.array([eff_t2_x * width, eff_t2_y * height], dtype=np.float32)

            s_vec = s_b - s_a
            t_vec = t_b - t_a

            s_len = float(np.linalg.norm(s_vec))
            t_len = float(np.linalg.norm(t_vec))

            # Monitor stretching ratio
            if s_len > 1.0:
                stretch = t_len / s_len
                if stretch > max_stretch_ratio:
                    max_stretch_ratio = stretch
                if stretch > 1.70:
                    warnings.append(f"Significant bone elongation ({stretch:.2f}x) on segment {p1_name}-{p2_name}.")
                elif stretch < 0.40:
                    warnings.append(f"Significant bone compression ({stretch:.2f}x) on segment {p1_name}-{p2_name}.")

            # Angles in radians
            theta_s = math.atan2(float(s_vec[1]), float(s_vec[0]))
            theta_t = math.atan2(float(t_vec[1]), float(t_vec[0]))
            delta_theta = theta_t - theta_s

            # Scale along bone
            scale = (t_len / s_len) if s_len > 1e-4 else 1.0

            # Forward 2D Affine: X_t = R(delta_theta) * scale * (X_s - s_a) + t_a
            # Inverse 2D Affine: X_s = R(-delta_theta) * (1/scale) * (X_t - t_a) + s_a
            cos_a = math.cos(-delta_theta)
            sin_a = math.sin(-delta_theta)
            inv_scale = (1.0 / scale) if scale > 1e-4 else 1.0

            # Inverse rotation matrix
            inv_rot = np.array([
                [cos_a * inv_scale, -sin_a * inv_scale],
                [sin_a * inv_scale,  cos_a * inv_scale]
            ], dtype=np.float32)

            bone_transforms.append({
                "t_a": t_a,
                "t_b": t_b,
                "s_a": s_a,
                "inv_rot": inv_rot,
                "name": f"{p1_name}_{p2_name}"
            })

        if not bone_transforms:
            warnings.append("No active bone connections resolved for deformation.")
            return image, ValidationResult(is_valid=True, warnings=warnings)

        # 2. Build 2D Pixel Grid for Canvas
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()]).astype(np.float32)  # (N, 2)
        n_points = points.shape[0]

        # 3. Compute Inverse Distance Weights from Target Bones
        weights = np.zeros((n_points, len(bone_transforms)), dtype=np.float32)

        for i, b in enumerate(bone_transforms):
            dists = self._point_to_segment_distance(points, b["t_a"], b["t_b"])
            # Decay factor with small epsilon to prevent division by zero
            weights[:, i] = 1.0 / (np.power(dists + 2.5, self.power))

        # Normalize weights across all bones
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        norm_weights = weights / (weight_sums + 1e-8)  # (N, B)

        # 4. Backward Coordinate Mapping: Compute Source Coordinates (src_x, src_y)
        src_coords = np.zeros_like(points)  # (N, 2)

        for i, b in enumerate(bone_transforms):
            # Compute inverse transformed point: (points - t_a) @ inv_rot.T + s_a
            rel_t = points - b["t_a"]
            mapped = rel_t @ b["inv_rot"].T + b["s_a"]
            src_coords += mapped * norm_weights[:, i:i+1]

        # 5. Resample from Source Image via Pixel-Art Module
        src_x = src_coords[:, 0].reshape((height, width))
        src_y = src_coords[:, 1].reshape((height, width))

        out_array = resample_sprite_grid(
            src_array=src_array,
            src_x=src_x,
            src_y=src_y,
            width=width,
            height=height,
            resampling=resampling,
            preserve_pixel_art=preserve_pixel_art
        )

        deformed_image = Image.fromarray(out_array, mode="RGBA")

        # 6. Quality Validation & Metrics Audit
        val_result = evaluate_quality_metrics(
            src_array=src_array,
            out_array=out_array,
            max_stretch_ratio=max_stretch_ratio,
            warnings=warnings,
            width=width,
            height=height
        )

        return deformed_image, val_result
