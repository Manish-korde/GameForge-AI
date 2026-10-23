"""
GameForge AI — Deformation Quality Assurance & Distortion Audit Module
Computes silhouette overlap (IoU), alpha mass integrity, canvas border clipping,
and skeletal elongation ratios to detect unnatural distortion.
"""

import numpy as np
from typing import List, Tuple
from .schemas import ValidationResult


def evaluate_quality_metrics(
    src_array: np.ndarray,
    out_array: np.ndarray,
    max_stretch_ratio: float,
    warnings: List[str],
    width: int,
    height: int,
    alpha_threshold: int = 15
) -> ValidationResult:
    """
    Computes rigorous QA metrics on the transformed sprite array relative to source:
    - Silhouette IoU
    - Alpha retention percentage
    - Outer canvas border clipping
    - Bone elongation factor
    """
    diagnostics = list(warnings)

    src_alpha = src_array[:, :, 3] > alpha_threshold
    out_alpha = out_array[:, :, 3] > alpha_threshold

    src_count = int(np.sum(src_alpha))
    out_count = int(np.sum(out_alpha))

    # Alpha integrity: percentage of non-transparent pixel mass retained
    alpha_integrity = (out_count / src_count * 100.0) if src_count > 0 else 100.0
    alpha_integrity = round(min(100.0, max(0.0, alpha_integrity)), 2)

    # Silhouette overlap (Intersection-over-Union)
    intersection = np.sum(src_alpha & out_alpha)
    union = np.sum(src_alpha | out_alpha)
    iou = round(float(intersection / union), 4) if union > 0 else 1.0

    # Clipping detection: non-transparent pixels contacting the canvas boundary
    top_edge = bool(np.any(out_alpha[0, :]))
    bottom_edge = bool(np.any(out_alpha[height - 1, :]))
    left_edge = bool(np.any(out_alpha[:, 0]))
    right_edge = bool(np.any(out_alpha[:, width - 1]))
    clipping_detected = bool(top_edge or bottom_edge or left_edge or right_edge)

    if clipping_detected:
        diagnostics.append("Canvas boundary clipping detected: deformed sprite contacts image borders.")

    if alpha_integrity < 70.0:
        diagnostics.append(f"Low alpha integrity ({alpha_integrity}%): non-trivial pixel loss occurred during deformation.")

    is_valid = bool(alpha_integrity >= 40.0 and out_count > 0)

    return ValidationResult(
        is_valid=is_valid,
        silhouette_overlap_iou=iou,
        alpha_integrity_pct=alpha_integrity,
        clipping_detected=clipping_detected,
        max_bone_stretch_ratio=round(float(max_stretch_ratio), 3),
        warnings=diagnostics
    )
