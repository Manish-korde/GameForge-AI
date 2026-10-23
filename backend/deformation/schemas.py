"""
GameForge AI — Deformation Schemas
Pydantic contracts for 2D character skeletons, joints, pose templates,
deformation requests, responses, and validation quality metrics.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ResamplingPolicy(str, Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class DeformationMethod(str, Enum):
    GEOMETRIC = "geometric"
    VAE_EXPERIMENTAL = "vae_experimental"
    HYBRID = "hybrid"


class Joint2D(BaseModel):
    """Normalized 2D joint position within character canvas [0.0, 1.0]."""
    name: str = Field(..., description="Canonical joint identifier (e.g. 'head', 'left_elbow')")
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized horizontal coordinate (0.0=left, 1.0=right)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized vertical coordinate (0.0=top, 1.0=bottom)")
    is_visible: bool = Field(default=True, description="False if joint is occluded or out-of-frame")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection or placement confidence score")

    @field_validator('x', 'y')
    @classmethod
    def round_coords(cls, v: float) -> float:
        return round(float(v), 4)


class BoundingBox(BaseModel):
    """Pixel bounding box of non-transparent alpha content."""
    min_x: int = Field(..., ge=0)
    min_y: int = Field(..., ge=0)
    max_x: int = Field(..., ge=0)
    max_y: int = Field(..., ge=0)
    width: int = Field(..., ge=0)
    height: int = Field(..., ge=0)
    non_transparent_pixels: int = Field(default=0, ge=0)
    fill_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class Skeleton2D(BaseModel):
    """A complete 2D character skeleton composed of named Joint2D instances."""
    joints: Dict[str, Joint2D] = Field(..., description="Map of canonical joint name to Joint2D")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Optional non-transparent bounding box")

    @field_validator('joints', mode='before')
    @classmethod
    def normalize_joints(cls, v):
        """Accepts either a dict of joints or a list of Joint2D / dicts, converting to canonical Dict[str, Joint2D]."""
        if isinstance(v, list):
            normalized = {}
            for item in v:
                if isinstance(item, dict):
                    name = item.get("name")
                    if name:
                        normalized[name] = Joint2D(**item)
                elif hasattr(item, "name"):
                    normalized[item.name] = item
            return normalized
        elif isinstance(v, dict):
            # Ensure any dict values that are raw dicts become Joint2D
            normalized = {}
            for k, val in v.items():
                if isinstance(val, dict):
                    if "name" not in val:
                        val = {**val, "name": k}
                    normalized[k] = Joint2D(**val)
                else:
                    normalized[k] = val
            return normalized
        return v

    def get_joint(self, name: str) -> Optional[Joint2D]:
        return self.joints.get(name)

    @property
    def joint_list(self) -> List[Joint2D]:
        """Provides an ordered list of Joint2D instances for array-based iteration."""
        return list(self.joints.values())

    @property
    def joint_names(self) -> List[str]:
        """Provides the list of canonical joint names present in this skeleton."""
        return list(self.joints.keys())

    def with_updated_joint(self, name: str, x: float, y: float) -> "Skeleton2D":
        """Returns a new Skeleton2D with the specified joint coordinates updated."""
        new_joints = dict(self.joints)
        current = new_joints.get(name)
        new_joints[name] = Joint2D(
            name=name,
            x=x,
            y=y,
            is_visible=current.is_visible if current else True,
            confidence=current.confidence if current else 1.0
        )
        return Skeleton2D(joints=new_joints, bounding_box=self.bounding_box)


class PoseTemplate(BaseModel):
    """Named, pre-tested target pose definition for 2D character deformation."""
    id: str = Field(..., description="Unique template identifier (e.g. 'walk_1', 'jump')")
    name: str = Field(..., description="Human-readable title (e.g. 'Walk Stride 1')")
    category: str = Field(default="locomotion", description="Category: idle, locomotion, combat, acrobatic")
    description: str = Field(..., description="Description of the movement posture")
    skeleton: Skeleton2D = Field(..., description="Target normalized skeleton coordinates")


class DeformationRequest(BaseModel):
    """Request payload for character pose deformation."""
    image_b64: Optional[str] = Field(default=None, description="Base64-encoded source RGBA image (data URI or raw)")
    image_url: Optional[str] = Field(default=None, description="Optional URL to fetch sprite from")
    source_skeleton: Optional[Skeleton2D] = Field(default=None, description="Custom source skeleton; auto-detected if omitted")
    target_pose: Optional[str] = Field(default=None, description="Named pose template ID (e.g. 'walk_1', 'run', 'jump', 'crouch', 'attack_slash')")
    target_skeleton: Optional[Skeleton2D] = Field(default=None, description="Explicit target skeleton joint overrides")
    method: DeformationMethod = Field(default=DeformationMethod.GEOMETRIC, description="Transformation algorithm")
    resampling: ResamplingPolicy = Field(default=ResamplingPolicy.NEAREST, description="Pixel-art resampling policy")
    intensity: float = Field(default=1.0, ge=0.0, le=2.0, description="Interpolation intensity from source to target (0.0=identity, 1.0=full target)")
    preserve_pixel_art: bool = Field(default=True, description="Enforce discrete alpha thresholds and integer grid snapping")
    maintain_canvas_size: bool = Field(default=False, description="Standardize canvas to 128x128 dimensions")


class ValidationResult(BaseModel):
    """Quality assurance and distortion metrics computed on deformed sprite."""
    is_valid: bool = Field(default=True, description="True if no fatal geometric or alpha defects occurred")
    silhouette_overlap_iou: float = Field(default=1.0, ge=0.0, le=1.0, description="Intersection-over-Union with source sprite silhouette")
    alpha_integrity_pct: float = Field(default=100.0, ge=0.0, le=100.0, description="Percentage of original opaque pixels retained")
    clipping_detected: bool = Field(default=False, description="True if transformed pixels hit the canvas boundary")
    max_bone_stretch_ratio: float = Field(default=1.0, description="Maximum elongation ratio of any skeletal bone segment")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic warnings (e.g. extreme rotation, canvas clipping)")


class DeformationResponse(BaseModel):
    """Complete response returned by the pose deformation service."""
    status: str = Field(default="success")
    method: str = Field(..., description="Algorithm executed (geometric, vae_experimental, hybrid)")
    output_image_b64: str = Field(..., description="Base64 Data URI of deformed sprite (data:image/png;base64,...)")
    source_image_b64: str = Field(..., description="Base64 Data URI of preprocessed source sprite")
    source_skeleton: Skeleton2D = Field(..., description="Resolved source skeleton")
    target_skeleton: Skeleton2D = Field(..., description="Resolved target skeleton")
    bounding_box: BoundingBox = Field(..., description="Non-transparent bounding box of output")
    validation: ValidationResult = Field(..., description="Quality verification and distortion audit")
    processing_time_ms: float = Field(..., description="Execution latency in milliseconds")
    message: str = Field(default="Pose deformation completed successfully.")


# =========================================================================
# STAGE 2: CHARACTER ANIMATION & SPRITESHEET SCHEMAS
# =========================================================================

class KeyframePose(BaseModel):
    """A single keyframe pose in an animation sequence with normalized timing."""
    pose_template_id: Optional[str] = Field(default=None, description="Optional named pose template identifier")
    skeleton: Optional[Skeleton2D] = Field(default=None, description="Explicit skeleton coordinates for this keyframe")
    timestamp: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized timestamp along animation timeline [0.0, 1.0]")
    duration_ms: Optional[int] = Field(default=None, ge=10, le=5000, description="Optional explicit hold duration in milliseconds")


class NamedAnimationPreset(BaseModel):
    """Definition and metadata for a pre-configured character animation sequence."""
    id: str = Field(..., description="Preset identifier (e.g. 'walk', 'idle', 'run', 'jump', 'attack')")
    name: str = Field(..., description="Human-readable title (e.g. 'Walking Gait Cycle')")
    description: str = Field(..., description="Detailed description of the motion cycle")
    category: str = Field(default="locomotion", description="Category: locomotion, combat, idle, acrobatic")
    default_frame_count: int = Field(default=8, ge=2, le=32, description="Standard frame count recommended for this preset")
    default_fps: int = Field(default=10, ge=1, le=30, description="Recommended playback frames per second")
    is_looping: bool = Field(default=True, description="Whether the animation loops back seamlessly to frame 0")
    keyframe_template_ids: List[str] = Field(default_factory=list, description="Ordered sequence of pose template IDs forming this motion")


class FrameTiming(BaseModel):
    """Timing parameters for a single animation frame."""
    frame_index: int = Field(..., ge=0)
    timestamp_ms: float = Field(..., ge=0.0)
    duration_ms: float = Field(..., ge=0.0)


class GeneratedFrame(BaseModel):
    """A rendered animation frame with associated skeleton, timing, and quality metrics."""
    frame_index: int = Field(..., ge=0, description="Zero-indexed position in sequence")
    timestamp_ms: float = Field(..., ge=0.0, description="Elapsed timestamp in sequence milliseconds")
    duration_ms: float = Field(..., ge=0.0, description="Display duration for this frame")
    image_b64: str = Field(..., description="Base64 Data URI of rendered RGBA frame")
    width: int = Field(..., ge=16, le=512)
    height: int = Field(..., ge=16, le=512)
    skeleton: Skeleton2D = Field(..., description="Interpolated skeletal pose for this frame")
    validation: ValidationResult = Field(..., description="Quality verification metrics for this frame")


class AnimationRequest(BaseModel):
    """Request payload for multi-frame 2D character animation generation."""
    image_b64: Optional[str] = Field(default=None, description="Base64 source sprite data URI")
    image_url: Optional[str] = Field(default=None, description="Local sample URL for sprite source")
    preset_id: Optional[str] = Field(default="walk", description="Named preset ID (idle, walk, run, jump, attack)")
    source_skeleton: Optional[Skeleton2D] = Field(default=None, description="Optional custom source skeleton override")
    custom_keyframes: Optional[List[KeyframePose]] = Field(default=None, description="Optional explicit sequence of custom keyframes")
    frame_count: int = Field(default=8, ge=2, le=32, description="Total number of frames to generate (2-32)")
    fps: int = Field(default=10, ge=1, le=30, description="Playback speed in frames per second (1-30)")
    is_looping: bool = Field(default=True, description="Whether to generate a seamless looping cycle")
    method: DeformationMethod = Field(default=DeformationMethod.GEOMETRIC, description="Deformation algorithm")
    resampling: ResamplingPolicy = Field(default=ResamplingPolicy.NEAREST, description="Pixel-art resampling policy")
    preserve_pixel_art: bool = Field(default=True, description="Enforce discrete alpha thresholds and integer grid snapping")
    maintain_canvas_size: bool = Field(default=True, description="Preserve uniform canvas dimensions across all frames")


class AnimationResponse(BaseModel):
    """Response returned upon generating a complete character animation sequence."""
    status: str = Field(default="success")
    preset_id: Optional[str] = Field(default=None, description="Applied preset ID, if any")
    frame_count: int = Field(..., ge=1, description="Number of frames generated")
    fps: int = Field(..., ge=1, le=30, description="Playback rate in frames per second")
    duration_ms: float = Field(..., ge=0.0, description="Total animation duration in milliseconds")
    is_looping: bool = Field(default=True, description="Whether animation is configured to loop")
    frames: List[GeneratedFrame] = Field(..., description="Ordered list of generated frames")
    overall_validation: ValidationResult = Field(..., description="Aggregate quality metrics across all frames")
    processing_time_ms: float = Field(..., description="Total execution latency in milliseconds")
    message: str = Field(default="Character animation generated successfully.")


class SpritesheetMetadata(BaseModel):
    """Metadata describing the layout and frame structure of an assembled spritesheet."""
    frame_width: int = Field(..., ge=1, description="Width of an individual frame in pixels")
    frame_height: int = Field(..., ge=1, description="Height of an individual frame in pixels")
    total_width: int = Field(..., ge=1, description="Total spritesheet image width in pixels")
    total_height: int = Field(..., ge=1, description="Total spritesheet image height in pixels")
    columns: int = Field(..., ge=1, description="Number of columns in the grid layout")
    rows: int = Field(..., ge=1, description="Number of rows in the grid layout")
    frame_count: int = Field(..., ge=1, description="Total number of frames packed")
    frame_order: List[int] = Field(default_factory=list, description="Zero-indexed frame order (row-major)")
    fps: int = Field(default=10, ge=1, le=30, description="Target playback speed in frames per second")
    is_looping: bool = Field(default=True, description="Whether the animation loops")


class SpritesheetRequest(BaseModel):
    """Request payload to assemble frames into an exportable spritesheet grid."""
    image_b64: Optional[str] = Field(default=None, description="Base64 source sprite to animate and pack")
    image_url: Optional[str] = Field(default=None, description="Local sample URL to animate and pack")
    preset_id: Optional[str] = Field(default="walk", description="Preset to generate before packing, if frames not provided")
    frame_count: Optional[int] = Field(default=8, ge=2, le=32, description="Number of frames if generating from preset")
    fps: Optional[int] = Field(default=10, ge=1, le=30, description="Target animation FPS")
    columns: Optional[int] = Field(default=None, ge=1, le=16, description="Optional column count override; auto-calculated if None")
    custom_frames_b64: Optional[List[str]] = Field(default=None, description="Pre-generated base64 PNG frames to pack directly")


class SpritesheetResponse(BaseModel):
    """Response returned upon packing an animation into a transparent spritesheet."""
    status: str = Field(default="success")
    spritesheet_image_b64: str = Field(..., description="Base64 Data URI of transparent assembled spritesheet PNG")
    metadata: SpritesheetMetadata = Field(..., description="Layout grid and frame coordinate metadata")
    processing_time_ms: float = Field(..., description="Execution latency in milliseconds")
    message: str = Field(default="Spritesheet assembled and exported successfully.")

