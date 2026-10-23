"""
GameForge AI — Character Pose Deformation Package
Provides modular 2D skeletal deformation, pixel-art alpha preservation,
and experimental VAE latent analysis for game character sprites.
"""

from .schemas import (
    Joint2D,
    Skeleton2D,
    PoseTemplate,
    DeformationRequest,
    DeformationResponse,
    BoundingBox,
    ValidationResult,
    KeyframePose,
    NamedAnimationPreset,
    FrameTiming,
    GeneratedFrame,
    AnimationRequest,
    AnimationResponse,
    SpritesheetMetadata,
    SpritesheetRequest,
    SpritesheetResponse
)
from .skeleton import (
    JOINT_NAMES,
    BONE_CONNECTIONS,
    JOINT_CONSTRAINTS,
    get_default_skeleton,
    get_pose_templates,
    get_template_by_name
)
from .geometry import GeometricDeformer
from .service import PoseDeformationService, deformation_service
from .pose_constraints import (
    compute_joint_angle,
    validate_skeleton_topology,
    evaluate_pose_constraints
)
from .pixel_art import resample_sprite_grid, clean_alpha_channel
from .quality import evaluate_quality_metrics
from .vae_adapter import VAEDeformationAdapter
from .animation import (
    interpolate_skeletons,
    get_animation_presets,
    get_animation_preset_by_id,
    generate_interpolated_pose_sequence,
    assemble_spritesheet,
    CharacterAnimationService,
    animation_service
)

__all__ = [
    "Joint2D",
    "Skeleton2D",
    "PoseTemplate",
    "DeformationRequest",
    "DeformationResponse",
    "BoundingBox",
    "ValidationResult",
    "KeyframePose",
    "NamedAnimationPreset",
    "FrameTiming",
    "GeneratedFrame",
    "AnimationRequest",
    "AnimationResponse",
    "SpritesheetMetadata",
    "SpritesheetRequest",
    "SpritesheetResponse",
    "JOINT_NAMES",
    "BONE_CONNECTIONS",
    "JOINT_CONSTRAINTS",
    "get_default_skeleton",
    "get_pose_templates",
    "get_template_by_name",
    "GeometricDeformer",
    "PoseDeformationService",
    "deformation_service",
    "compute_joint_angle",
    "validate_skeleton_topology",
    "evaluate_pose_constraints",
    "resample_sprite_grid",
    "clean_alpha_channel",
    "evaluate_quality_metrics",
    "VAEDeformationAdapter",
    "interpolate_skeletons",
    "get_animation_presets",
    "get_animation_preset_by_id",
    "generate_interpolated_pose_sequence",
    "assemble_spritesheet",
    "CharacterAnimationService",
    "animation_service"
]

