"""
GameForge AI — Geometric Character Pose Deformation Engine (Backward Compatibility Wrapper)
Re-exports GeometricDeformer from the modular geometry engine.
"""

from .geometry import GeometricDeformer
from .pixel_art import resample_sprite_grid, clean_alpha_channel
from .quality import evaluate_quality_metrics
from .pose_constraints import validate_skeleton_topology, evaluate_pose_constraints

__all__ = [
    "GeometricDeformer",
    "resample_sprite_grid",
    "clean_alpha_channel",
    "evaluate_quality_metrics",
    "validate_skeleton_topology",
    "evaluate_pose_constraints"
]
