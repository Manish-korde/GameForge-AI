"""
GameForge AI — Character Animation Generation & Spritesheet Assembler
Stage 2 implementation providing multi-frame pose interpolation, named motion presets
(idle, walk, run, jump, attack), geometric frame rendering, quality metrics, and
transparent spritesheet grid packaging.
"""

import io
import math
import time
import base64
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional

from .schemas import (
    Skeleton2D,
    Joint2D,
    BoundingBox,
    ValidationResult,
    NamedAnimationPreset,
    KeyframePose,
    GeneratedFrame,
    AnimationRequest,
    AnimationResponse,
    SpritesheetMetadata,
    SpritesheetRequest,
    SpritesheetResponse,
    DeformationMethod,
    ResamplingPolicy
)
from .skeleton import (
    get_default_skeleton,
    get_pose_templates,
    get_template_by_name,
    fit_skeleton_to_bbox,
    create_skeleton_from_dict,
    JOINT_NAMES,
    BONE_CONNECTIONS
)
from .pose_constraints import (
    validate_skeleton_topology,
    evaluate_pose_constraints
)
from .geometry import GeometricDeformer
from .pixel_art import clean_alpha_channel


# =========================================================================
# 1. POSE INTERPOLATION
# =========================================================================

def interpolate_skeletons(
    skel_a: Skeleton2D,
    skel_b: Skeleton2D,
    t: float,
    use_smoothstep: bool = True
) -> Skeleton2D:
    """
    Interpolates joint positions between two canonical Skeleton2D instances at parameter t in [0.0, 1.0].
    
    Mathematical Formulation:
      smooth_t = (1 - cos(pi * t)) / 2  if use_smoothstep else t
      x_interp = (1 - smooth_t) * x_a + smooth_t * x_b
      y_interp = (1 - smooth_t) * y_a + smooth_t * y_b
    
    Coordinates are clamped to [0.01, 0.99] to prevent zero-length boundary anomalies.
    Joint identity and 16-joint topology are strictly preserved.
    """
    clamped_t = max(0.0, min(1.0, float(t)))
    factor = (1.0 - math.cos(math.pi * clamped_t)) / 2.0 if use_smoothstep else clamped_t

    interpolated_joints: Dict[str, Joint2D] = {}

    for name in JOINT_NAMES:
        ja = skel_a.get_joint(name)
        jb = skel_b.get_joint(name)

        if ja and jb:
            new_x = (1.0 - factor) * ja.x + factor * jb.x
            new_y = (1.0 - factor) * ja.y + factor * jb.y
            vis = ja.is_visible and jb.is_visible
            conf = min(ja.confidence, jb.confidence)
        elif ja:
            new_x, new_y, vis, conf = ja.x, ja.y, ja.is_visible, ja.confidence
        elif jb:
            new_x, new_y, vis, conf = jb.x, jb.y, jb.is_visible, jb.confidence
        else:
            # Fallback to neutral default joint position
            def_joint = get_default_skeleton().get_joint(name)
            new_x, new_y, vis, conf = def_joint.x, def_joint.y, True, 1.0

        clamped_x = round(float(np.clip(new_x, 0.01, 0.99)), 4)
        clamped_y = round(float(np.clip(new_y, 0.01, 0.99)), 4)

        interpolated_joints[name] = Joint2D(
            name=name,
            x=clamped_x,
            y=clamped_y,
            is_visible=vis,
            confidence=round(conf, 2)
        )

    # Interpolate bounding boxes if both exist
    interp_bbox = None
    if skel_a.bounding_box and skel_b.bounding_box:
        ba = skel_a.bounding_box
        bb = skel_b.bounding_box
        interp_bbox = BoundingBox(
            min_x=int(round((1.0 - factor) * ba.min_x + factor * bb.min_x)),
            min_y=int(round((1.0 - factor) * ba.min_y + factor * bb.min_y)),
            max_x=int(round((1.0 - factor) * ba.max_x + factor * bb.max_x)),
            max_y=int(round((1.0 - factor) * ba.max_y + factor * bb.max_y)),
            width=int(round((1.0 - factor) * ba.width + factor * bb.width)),
            height=int(round((1.0 - factor) * ba.height + factor * bb.height)),
            non_transparent_pixels=int(round((1.0 - factor) * ba.non_transparent_pixels + factor * bb.non_transparent_pixels)),
            fill_ratio=round((1.0 - factor) * ba.fill_ratio + factor * bb.fill_ratio, 4)
        )

    return Skeleton2D(joints=interpolated_joints, bounding_box=interp_bbox)


# =========================================================================
# 2. ANIMATION PRESETS CATALOG
# =========================================================================

def _get_preset_definitions() -> List[NamedAnimationPreset]:
    """Defines the 5 canonical character motion presets with timing and keyframes."""
    return [
        NamedAnimationPreset(
            id="idle",
            name="Idle Breathing Oscillation",
            description="Subtle rhythmic stance cycle simulating organic character respiration and ready poise.",
            category="idle",
            default_frame_count=4,
            default_fps=8,
            is_looping=True,
            keyframe_template_ids=["idle", "idle_inhale", "idle"]
        ),
        NamedAnimationPreset(
            id="walk",
            name="Walking Gait Cycle",
            description="Classic 4-phase contact-pass locomotion cycle with alternating arm and leg extensions.",
            category="locomotion",
            default_frame_count=8,
            default_fps=10,
            is_looping=True,
            keyframe_template_ids=["walk_1", "idle", "walk_2", "idle", "walk_1"]
        ),
        NamedAnimationPreset(
            id="run",
            name="Sprint Locomotion",
            description="Dynamic high-velocity running cycle featuring pronounced torso tilt and wide limb swings.",
            category="locomotion",
            default_frame_count=6,
            default_fps=12,
            is_looping=True,
            keyframe_template_ids=["run", "idle", "run_opposite", "idle", "run"]
        ),
        NamedAnimationPreset(
            id="jump",
            name="Airborne Jump Arc",
            description="Acrobatic jump trajectory: crouch anticipation -> upward launch -> apex tuck -> descent -> landing compression -> recovery.",
            category="acrobatic",
            default_frame_count=6,
            default_fps=10,
            is_looping=False,
            keyframe_template_ids=["crouch", "idle", "jump", "idle", "crouch", "idle"]
        ),
        NamedAnimationPreset(
            id="attack",
            name="Overhead Weapon Strike",
            description="Combat attack combo: backswing windup -> overhead slash impact -> forward follow-through -> defensive recovery -> idle.",
            category="combat",
            default_frame_count=5,
            default_fps=12,
            is_looping=False,
            keyframe_template_ids=["idle", "attack_slash", "walk_1", "crouch", "idle"]
        ),
    ]


def get_animation_presets() -> List[NamedAnimationPreset]:
    """Returns the verified catalog of supported character animation presets."""
    return _get_preset_definitions()


def get_animation_preset_by_id(preset_id: str) -> Optional[NamedAnimationPreset]:
    """Retrieves an animation preset by identifier (case-insensitive)."""
    clean_id = preset_id.strip().lower()
    for preset in _get_preset_definitions():
        if preset.id == clean_id:
            return preset
    return None


def _resolve_template_skeleton(template_id: str) -> Skeleton2D:
    """Helper to resolve a pose template by ID or create synthesized keyframe poses."""
    # Special synthesized variants
    if template_id == "idle_inhale":
        base = get_template_by_name("idle")
        if base:
            # Raise chest and head slightly, widen elbows
            inhale_skel = base.skeleton.with_updated_joint("head", 0.50, 0.17)
            inhale_skel = inhale_skel.with_updated_joint("neck", 0.50, 0.27)
            inhale_skel = inhale_skel.with_updated_joint("torso", 0.50, 0.43)
            inhale_skel = inhale_skel.with_updated_joint("left_elbow", 0.33, 0.44)
            inhale_skel = inhale_skel.with_updated_joint("right_elbow", 0.67, 0.44)
            return inhale_skel
    elif template_id == "run_opposite":
        base = get_template_by_name("run")
        if base:
            # Mirrored run stride
            j = base.skeleton.joints
            mirrored_coords = {
                "head": (0.46, 0.20),
                "neck": (0.47, 0.30),
                "torso": (0.49, 0.45),
                "pelvis": (0.52, 0.58),
                "left_shoulder": (0.38, 0.32),
                "left_elbow": (0.30, 0.46),
                "left_wrist": (0.26, 0.60),
                "right_shoulder": (0.58, 0.32),
                "right_elbow": (0.70, 0.38),
                "right_wrist": (0.74, 0.30),
                "left_hip": (0.46, 0.60),
                "left_knee": (0.34, 0.70),
                "left_ankle": (0.24, 0.80),
                "right_hip": (0.56, 0.60),
                "right_knee": (0.68, 0.70),
                "right_ankle": (0.76, 0.84)
            }
            return create_skeleton_from_dict(mirrored_coords)

    t = get_template_by_name(template_id)
    if t:
        return t.skeleton
    return get_default_skeleton()


# =========================================================================
# 3. TIMELINE INTERPOLATION & POSE SEQUENCE GENERATION
# =========================================================================

def build_preset_keyframes(
    preset_id: str,
    bbox: Optional[BoundingBox] = None,
    canvas_w: int = 128,
    canvas_h: int = 128
) -> List[Tuple[float, Skeleton2D]]:
    """
    Constructs normalized (timestamp, Skeleton2D) keyframes for a named preset,
    scaling all keyframes to the sprite's bounding box.
    """
    preset = get_animation_preset_by_id(preset_id)
    if not preset:
        raise ValueError(f"Unknown animation preset: '{preset_id}'. Available: idle, walk, run, jump, attack.")

    template_ids = preset.keyframe_template_ids
    n_keys = len(template_ids)
    if n_keys < 2:
        raise ValueError(f"Preset '{preset_id}' must contain at least 2 keyframe templates.")

    keyframes: List[Tuple[float, Skeleton2D]] = []
    for idx, t_id in enumerate(template_ids):
        timestamp = float(idx) / float(n_keys - 1)
        raw_skel = _resolve_template_skeleton(t_id)
        if bbox:
            fitted_skel = fit_skeleton_to_bbox(raw_skel, bbox, canvas_w, canvas_h)
        else:
            fitted_skel = raw_skel
        keyframes.append((timestamp, fitted_skel))

    return keyframes


def sample_pose_at_time(
    keyframes: List[Tuple[float, Skeleton2D]],
    tau: float,
    is_looping: bool = True
) -> Skeleton2D:
    """
    Samples an interpolated skeleton at normalized time tau in [0.0, 1.0].
    Finds the adjacent keyframe interval and performs smoothstep interpolation.
    """
    if not keyframes:
        return get_default_skeleton()
    if len(keyframes) == 1:
        return keyframes[0][1]

    # Normalize tau
    if is_looping:
        t = tau % 1.0
    else:
        t = max(0.0, min(1.0, tau))

    # Find bounding keyframes
    # keyframes is sorted by timestamp [0.0 ... 1.0]
    for i in range(len(keyframes) - 1):
        t0, skel0 = keyframes[i]
        t1, skel1 = keyframes[i + 1]

        if t0 <= t <= t1:
            segment_duration = max(t1 - t0, 1e-6)
            local_t = (t - t0) / segment_duration
            return interpolate_skeletons(skel0, skel1, local_t, use_smoothstep=True)

    # Edge boundary cases
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]
    return keyframes[0][1]


def generate_interpolated_pose_sequence(
    keyframes: List[Tuple[float, Skeleton2D]],
    frame_count: int,
    is_looping: bool = True
) -> List[Skeleton2D]:
    """
    Generates an ordered list of frame_count interpolated skeletons along the timeline.
    For looping motions: frame k is sampled at tau = k / frame_count (wraps seamlessly).
    For non-looping motions: frame k is sampled at tau = k / (frame_count - 1).
    """
    if frame_count < 2:
        raise ValueError(f"frame_count must be at least 2, got {frame_count}.")
    if frame_count > 32:
        raise ValueError(f"frame_count ({frame_count}) exceeds maximum limit of 32 frames.")

    skeletons: List[Skeleton2D] = []

    for k in range(frame_count):
        if is_looping:
            tau = float(k) / float(frame_count)
        else:
            tau = float(k) / float(frame_count - 1) if frame_count > 1 else 0.0

        skel = sample_pose_at_time(keyframes, tau, is_looping=is_looping)
        skeletons.append(skel)

    return skeletons


# =========================================================================
# 4. SPRITESHEET GRID PACKER
# =========================================================================

def calculate_spritesheet_grid(frame_count: int, columns: Optional[int] = None) -> Tuple[int, int]:
    """
    Calculates the optimal (columns, rows) grid structure for a given frame count.
    If columns is unspecified:
      - <= 4 frames: single horizontal row (cols = frame_count, rows = 1)
      - <= 8 frames: 4 columns x 2 rows
      - > 8 frames: square-proportional grid ceil(sqrt(N)) columns
    """
    if frame_count <= 0:
        return 1, 1

    if columns is not None and columns > 0:
        cols = min(columns, frame_count)
        rows = int(math.ceil(frame_count / float(cols)))
        return cols, rows

    if frame_count <= 4:
        return frame_count, 1
    elif frame_count <= 8:
        cols = 4
        rows = int(math.ceil(frame_count / 4.0))
        return cols, rows
    else:
        cols = int(math.ceil(math.sqrt(frame_count)))
        rows = int(math.ceil(frame_count / float(cols)))
        return cols, rows


def assemble_spritesheet(
    frames: List[Image.Image],
    columns: Optional[int] = None,
    fps: int = 10,
    is_looping: bool = True
) -> Tuple[Image.Image, SpritesheetMetadata]:
    """
    Packs an ordered sequence of PIL RGBA frames into a transparent PNG spritesheet grid.
    
    Validation & Safety:
      - All frames must have positive identical dimensions.
      - Total packed dimensions must not exceed 2048x2048 pixels.
      - Incomplete final rows are cleanly padded with 100% transparent pixels.
    """
    if not frames:
        raise ValueError("Cannot assemble an empty list of frames into a spritesheet.")

    frame_count = len(frames)
    first_w, first_h = frames[0].size

    # Validate dimension uniformity
    for idx, f in enumerate(frames):
        if f.size != (first_w, first_h):
            raise ValueError(
                f"Frame {idx} dimension mismatch: expected {first_w}x{first_h}, got {f.size[0]}x{f.size[1]}."
            )

    cols, rows = calculate_spritesheet_grid(frame_count, columns=columns)
    total_w = cols * first_w
    total_h = rows * first_h

    # Memory and dimension limits
    if total_w > 2048 or total_h > 2048:
        raise ValueError(
            f"Assembled spritesheet dimensions ({total_w}x{total_h}) exceed maximum allowed 2048x2048 pixels."
        )

    # Initialize transparent canvas
    sheet = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))

    frame_order = []
    for idx, frame in enumerate(frames):
        c = idx % cols
        r = idx // cols
        x_offset = c * first_w
        y_offset = r * first_h

        rgba_frame = frame.convert("RGBA")
        sheet.paste(rgba_frame, (x_offset, y_offset), mask=rgba_frame)
        frame_order.append(idx)

    metadata = SpritesheetMetadata(
        frame_width=first_w,
        frame_height=first_h,
        total_width=total_w,
        total_height=total_h,
        columns=cols,
        rows=rows,
        frame_count=frame_count,
        frame_order=frame_order,
        fps=fps,
        is_looping=is_looping
    )

    return sheet, metadata


# =========================================================================
# 5. HIGH-LEVEL CHARACTER ANIMATION SERVICE
# =========================================================================

class CharacterAnimationService:
    """
    Orchestration service for generating multi-frame character animations and spritesheets.
    Coordinates keyframe interpolation, geometric skinning, quality verification, and export.
    """

    def __init__(self):
        self.deformer = GeometricDeformer()

    @staticmethod
    def decode_base64_image(image_b64: str) -> Image.Image:
        """Parses Base64 data URI into a PIL RGBA Image."""
        if not image_b64:
            raise ValueError("No image payload provided.")
        clean = image_b64.split(";base64,")[1] if ";base64," in image_b64 else image_b64
        try:
            raw_bytes = base64.b64decode(clean)
            return Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
        except Exception as e:
            raise ValueError(f"Invalid Base64 image payload: {e}")

    @staticmethod
    def encode_base64_image(image: Image.Image) -> str:
        """Encodes a PIL Image into a Base64 PNG Data URI."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def load_local_image_from_url(image_url: str) -> Image.Image:
        """SSRF-safe local sample loader for alucard_samples."""
        import os
        import re
        match = re.search(r'alucard_(\d+)\.png', image_url)
        if match:
            sample_idx = int(match.group(1))
            sample_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "alucard_samples", f"alucard_{sample_idx}.png")
            )
            if os.path.exists(sample_path):
                return Image.open(sample_path).convert("RGBA")
        raise ValueError("Arbitrary remote image URLs are disallowed. Provide a local sample or 'image_b64'.")

    def generate_animation(self, req: AnimationRequest) -> AnimationResponse:
        """
        Generates an animation sequence of deformed character frames from an input sprite.
        """
        start_time = time.perf_counter()

        # 1. Decode & Validate Source Image
        if req.image_b64:
            source_img = self.decode_base64_image(req.image_b64)
        elif req.image_url:
            source_img = self.load_local_image_from_url(req.image_url)
        else:
            raise ValueError("Must provide either 'image_b64' or a valid local 'image_url'.")

        if source_img.width > 512 or source_img.height > 512:
            raise ValueError(f"Image dimensions ({source_img.width}x{source_img.height}) exceed maximum 512x512.")
        if source_img.width < 16 or source_img.height < 16:
            raise ValueError(f"Image dimensions ({source_img.width}x{source_img.height}) are too small.")

        if req.maintain_canvas_size and source_img.size != (128, 128):
            source_img = source_img.resize((128, 128), Image.Resampling.NEAREST)

        # 2. Detect Bounding Box
        bbox = self.deformer.detect_bounding_box(source_img)
        if bbox.non_transparent_pixels == 0:
            raise ValueError("Image is completely transparent; cannot animate an empty canvas.")

        # 3. Resolve Source Skeleton
        if req.source_skeleton and len(req.source_skeleton.joints) >= 6:
            src_skel = req.source_skeleton
        else:
            default_skel = get_default_skeleton()
            src_skel = fit_skeleton_to_bbox(default_skel, bbox, source_img.width, source_img.height)

        # 4. Resolve Keyframes
        preset_id = req.preset_id or "walk"
        preset = get_animation_preset_by_id(preset_id)

        if req.custom_keyframes and len(req.custom_keyframes) >= 2:
            keyframes = []
            for kf in req.custom_keyframes:
                if kf.skeleton:
                    kf_skel = fit_skeleton_to_bbox(kf.skeleton, bbox, source_img.width, source_img.height)
                elif kf.pose_template_id:
                    t = get_template_by_name(kf.pose_template_id)
                    kf_skel = fit_skeleton_to_bbox(t.skeleton, bbox, source_img.width, source_img.height) if t else src_skel
                else:
                    kf_skel = src_skel
                keyframes.append((kf.timestamp, kf_skel))
            # Sort by timestamp
            keyframes.sort(key=lambda x: x[0])
            is_looping = req.is_looping
        else:
            if not preset:
                raise ValueError(f"Unknown animation preset '{preset_id}'. Available: idle, walk, run, jump, attack.")
            keyframes = build_preset_keyframes(preset_id, bbox, source_img.width, source_img.height)
            is_looping = preset.is_looping if req.is_looping is None else req.is_looping

        # 5. Generate Interpolated Skeletal Sequence
        frame_count = req.frame_count if req.frame_count is not None else (preset.default_frame_count if preset else 8)
        fps = req.fps if req.fps is not None else (preset.default_fps if preset else 10)

        interpolated_skeletons = generate_interpolated_pose_sequence(
            keyframes=keyframes,
            frame_count=frame_count,
            is_looping=is_looping
        )

        # 6. Render Frames via Geometric Skinning
        frame_duration_ms = round(1000.0 / float(fps), 2)
        generated_frames: List[GeneratedFrame] = []
        all_warnings: List[str] = []
        sum_iou = 0.0
        min_alpha_pct = 100.0
        max_stretch_overall = 1.0

        for idx, target_skel in enumerate(interpolated_skeletons):
            frame_img, val_result = self.deformer.deform(
                image=source_img,
                source_skeleton=src_skel,
                target_skeleton=target_skel,
                intensity=1.0,
                resampling=req.resampling,
                preserve_pixel_art=req.preserve_pixel_art
            )

            sum_iou += val_result.silhouette_overlap_iou
            if val_result.alpha_integrity_pct < min_alpha_pct:
                min_alpha_pct = val_result.alpha_integrity_pct
            if val_result.max_bone_stretch_ratio > max_stretch_overall:
                max_stretch_overall = val_result.max_bone_stretch_ratio

            for w in val_result.warnings:
                if w not in all_warnings:
                    all_warnings.append(f"Frame {idx}: {w}")

            timestamp_ms = round(idx * frame_duration_ms, 2)
            frame_b64 = self.encode_base64_image(frame_img)

            generated_frames.append(GeneratedFrame(
                frame_index=idx,
                timestamp_ms=timestamp_ms,
                duration_ms=frame_duration_ms,
                image_b64=frame_b64,
                width=frame_img.width,
                height=frame_img.height,
                skeleton=target_skel,
                validation=val_result
            ))

        total_duration_ms = round(frame_count * frame_duration_ms, 2)
        proc_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        avg_iou = round(sum_iou / max(frame_count, 1), 4)

        overall_val = ValidationResult(
            is_valid=True,
            silhouette_overlap_iou=avg_iou,
            alpha_integrity_pct=round(min_alpha_pct, 2),
            clipping_detected=any(f.validation.clipping_detected for f in generated_frames),
            max_bone_stretch_ratio=round(max_stretch_overall, 2),
            warnings=all_warnings
        )

        return AnimationResponse(
            status="success",
            preset_id=preset_id,
            frame_count=frame_count,
            fps=fps,
            duration_ms=total_duration_ms,
            is_looping=is_looping,
            frames=generated_frames,
            overall_validation=overall_val,
            processing_time_ms=proc_time_ms,
            message=f"Generated {frame_count}-frame '{preset_id}' animation ({fps} FPS) in {proc_time_ms}ms."
        )

    def assemble_spritesheet_from_request(self, req: SpritesheetRequest) -> SpritesheetResponse:
        """
        Assembles frames into a transparent spritesheet.
        Accepts pre-generated base64 frames or generates from preset if frames not provided.
        """
        start_time = time.perf_counter()

        # Case A: User provided pre-generated base64 PNG frames
        if req.custom_frames_b64 and len(req.custom_frames_b64) > 0:
            pil_frames = [self.decode_base64_image(b64) for b64 in req.custom_frames_b64]
            fps = req.fps or 10
            is_looping = True
        # Case B: Generate animation first from preset
        else:
            if not req.image_b64 and not req.image_url:
                raise ValueError("Must provide either 'custom_frames_b64' or source sprite ('image_b64' / 'image_url').")
            anim_req = AnimationRequest(
                image_b64=req.image_b64,
                image_url=req.image_url,
                preset_id=req.preset_id or "walk",
                frame_count=req.frame_count or 8,
                fps=req.fps or 10
            )
            anim_res = self.generate_animation(anim_req)
            pil_frames = [self.decode_base64_image(f.image_b64) for f in anim_res.frames]
            fps = anim_res.fps
            is_looping = anim_res.is_looping

        # Pack frames into grid
        sheet_img, metadata = assemble_spritesheet(
            frames=pil_frames,
            columns=req.columns,
            fps=fps,
            is_looping=is_looping
        )

        sheet_b64 = self.encode_base64_image(sheet_img)
        proc_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return SpritesheetResponse(
            status="success",
            spritesheet_image_b64=sheet_b64,
            metadata=metadata,
            processing_time_ms=proc_ms,
            message=f"Packed {metadata.frame_count} frames into {metadata.columns}x{metadata.rows} grid ({metadata.total_width}x{metadata.total_height}px) in {proc_ms}ms."
        )


animation_service = CharacterAnimationService()
