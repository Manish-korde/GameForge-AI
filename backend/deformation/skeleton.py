"""
GameForge AI — 2D Character Skeleton & Pose Templates
Defines canonical 2D joint hierarchies, bone connections, biomechanical limits,
and tested named pose templates (idle, walk, run, jump, crouch, attack).
"""

from typing import Dict, List, Tuple, Optional
from .schemas import Joint2D, Skeleton2D, PoseTemplate, BoundingBox

JOINT_NAMES: List[str] = [
    "head",
    "neck",
    "torso",
    "pelvis",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle"
]

BONE_CONNECTIONS: List[Tuple[str, str]] = [
    ("head", "neck"),
    ("neck", "torso"),
    ("torso", "pelvis"),
    # Arms
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Legs
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle")
]

from .pose_constraints import JOINT_CONSTRAINTS


def create_skeleton_from_dict(coords: Dict[str, Tuple[float, float]]) -> Skeleton2D:
    """Builds a Skeleton2D instance from a mapping of joint names to (x, y) normalized coordinates."""
    joints = {}
    for name, (x, y) in coords.items():
        joints[name] = Joint2D(name=name, x=x, y=y, is_visible=True, confidence=1.0)
    return Skeleton2D(joints=joints)


def get_default_skeleton() -> Skeleton2D:
    """Default neutral standing pose (centered inside 128x128 canvas)."""
    coords = {
        "head": (0.50, 0.18),
        "neck": (0.50, 0.28),
        "torso": (0.50, 0.44),
        "pelvis": (0.50, 0.58),
        # Left Arm (viewer's left)
        "left_shoulder": (0.40, 0.31),
        "left_elbow": (0.35, 0.44),
        "left_wrist": (0.33, 0.56),
        # Right Arm (viewer's right)
        "right_shoulder": (0.60, 0.31),
        "right_elbow": (0.65, 0.44),
        "right_wrist": (0.67, 0.56),
        # Left Leg
        "left_hip": (0.44, 0.60),
        "left_knee": (0.43, 0.74),
        "left_ankle": (0.42, 0.88),
        # Right Leg
        "right_hip": (0.56, 0.60),
        "right_knee": (0.57, 0.74),
        "right_ankle": (0.58, 0.88)
    }
    return create_skeleton_from_dict(coords)


def get_pose_templates() -> List[PoseTemplate]:
    """Returns the verified, tested catalog of 2D character target pose templates."""
    templates = []

    # 1. Idle / Ready Stance
    idle_coords = {
        "head": (0.50, 0.19),
        "neck": (0.50, 0.29),
        "torso": (0.50, 0.45),
        "pelvis": (0.50, 0.59),
        "left_shoulder": (0.39, 0.32),
        "left_elbow": (0.34, 0.45),
        "left_wrist": (0.35, 0.56),
        "right_shoulder": (0.61, 0.32),
        "right_elbow": (0.66, 0.45),
        "right_wrist": (0.65, 0.56),
        "left_hip": (0.43, 0.60),
        "left_knee": (0.41, 0.74),
        "left_ankle": (0.40, 0.88),
        "right_hip": (0.57, 0.60),
        "right_knee": (0.59, 0.74),
        "right_ankle": (0.60, 0.88)
    }
    templates.append(PoseTemplate(
        id="idle",
        name="Idle Breathing Stance",
        category="idle",
        description="Relaxed combat-ready stance with slightly angled arms and balanced feet.",
        skeleton=create_skeleton_from_dict(idle_coords)
    ))

    # 2. Walk Stride 1 (Left forward, right back)
    walk1_coords = {
        "head": (0.50, 0.17),
        "neck": (0.50, 0.27),
        "torso": (0.51, 0.43),
        "pelvis": (0.50, 0.57),
        # Arms counter-swing
        "left_shoulder": (0.40, 0.30),
        "left_elbow": (0.32, 0.40),
        "left_wrist": (0.28, 0.49),
        "right_shoulder": (0.60, 0.30),
        "right_elbow": (0.68, 0.42),
        "right_wrist": (0.72, 0.54),
        # Legs stride
        "left_hip": (0.44, 0.59),
        "left_knee": (0.38, 0.72),
        "left_ankle": (0.34, 0.87),
        "right_hip": (0.56, 0.59),
        "right_knee": (0.62, 0.73),
        "right_ankle": (0.66, 0.87)
    }
    templates.append(PoseTemplate(
        id="walk_1",
        name="Walk Stride A",
        category="locomotion",
        description="Forward walking stride with left leg and right arm extended.",
        skeleton=create_skeleton_from_dict(walk1_coords)
    ))

    # 3. Walk Stride 2 (Right forward, left back)
    walk2_coords = {
        "head": (0.50, 0.17),
        "neck": (0.50, 0.27),
        "torso": (0.49, 0.43),
        "pelvis": (0.50, 0.57),
        # Arms counter-swing reversed
        "left_shoulder": (0.40, 0.30),
        "left_elbow": (0.32, 0.42),
        "left_wrist": (0.28, 0.54),
        "right_shoulder": (0.60, 0.30),
        "right_elbow": (0.68, 0.40),
        "right_wrist": (0.72, 0.49),
        # Legs stride reversed
        "left_hip": (0.44, 0.59),
        "left_knee": (0.50, 0.73),
        "left_ankle": (0.58, 0.87),
        "right_hip": (0.56, 0.59),
        "right_knee": (0.50, 0.72),
        "right_ankle": (0.42, 0.87)
    }
    templates.append(PoseTemplate(
        id="walk_2",
        name="Walk Stride B",
        category="locomotion",
        description="Opposing walking stride with right leg and left arm extended.",
        skeleton=create_skeleton_from_dict(walk2_coords)
    ))

    # 4. Run / Sprint (Dynamic forward lean)
    run_coords = {
        "head": (0.54, 0.20),
        "neck": (0.53, 0.30),
        "torso": (0.51, 0.45),
        "pelvis": (0.48, 0.58),
        # High arm swing
        "left_shoulder": (0.42, 0.32),
        "left_elbow": (0.30, 0.38),
        "left_wrist": (0.26, 0.30),
        "right_shoulder": (0.62, 0.32),
        "right_elbow": (0.70, 0.46),
        "right_wrist": (0.74, 0.60),
        # Deep leg separation
        "left_hip": (0.44, 0.60),
        "left_knee": (0.32, 0.70),
        "left_ankle": (0.24, 0.84),
        "right_hip": (0.54, 0.60),
        "right_knee": (0.66, 0.70),
        "right_ankle": (0.76, 0.80)
    }
    templates.append(PoseTemplate(
        id="run",
        name="Dynamic Sprint",
        category="locomotion",
        description="High-velocity forward run with tilted torso and pronounced stride.",
        skeleton=create_skeleton_from_dict(run_coords)
    ))

    # 5. Jump / Aerial Apex
    jump_coords = {
        "head": (0.50, 0.12),
        "neck": (0.50, 0.22),
        "torso": (0.50, 0.36),
        "pelvis": (0.50, 0.48),
        # Raised arms
        "left_shoulder": (0.38, 0.24),
        "left_elbow": (0.28, 0.20),
        "left_wrist": (0.24, 0.12),
        "right_shoulder": (0.62, 0.24),
        "right_elbow": (0.72, 0.20),
        "right_wrist": (0.76, 0.12),
        # Tucked knees
        "left_hip": (0.43, 0.50),
        "left_knee": (0.36, 0.60),
        "left_ankle": (0.40, 0.72),
        "right_hip": (0.57, 0.50),
        "right_knee": (0.64, 0.60),
        "right_ankle": (0.60, 0.72)
    }
    templates.append(PoseTemplate(
        id="jump",
        name="Airborne Jump",
        category="acrobatic",
        description="Mid-air leap with raised arms and tucked legs.",
        skeleton=create_skeleton_from_dict(jump_coords)
    ))

    # 6. Crouch / Duck
    crouch_coords = {
        "head": (0.50, 0.36),
        "neck": (0.50, 0.45),
        "torso": (0.50, 0.56),
        "pelvis": (0.50, 0.68),
        # Guarding arms
        "left_shoulder": (0.38, 0.46),
        "left_elbow": (0.32, 0.56),
        "left_wrist": (0.36, 0.66),
        "right_shoulder": (0.62, 0.46),
        "right_elbow": (0.68, 0.56),
        "right_wrist": (0.64, 0.66),
        # Deeply bent knees
        "left_hip": (0.42, 0.70),
        "left_knee": (0.32, 0.78),
        "left_ankle": (0.38, 0.88),
        "right_hip": (0.58, 0.70),
        "right_knee": (0.68, 0.78),
        "right_ankle": (0.62, 0.88)
    }
    templates.append(PoseTemplate(
        id="crouch",
        name="Defensive Crouch",
        category="combat",
        description="Low center of gravity with folded legs and guarded torso.",
        skeleton=create_skeleton_from_dict(crouch_coords)
    ))

    # 7. Attack / Sword Slash
    attack_coords = {
        "head": (0.48, 0.17),
        "neck": (0.48, 0.27),
        "torso": (0.49, 0.43),
        "pelvis": (0.49, 0.57),
        # Raised striking arm (right) and balancing arm (left)
        "left_shoulder": (0.38, 0.32),
        "left_elbow": (0.28, 0.42),
        "left_wrist": (0.26, 0.54),
        "right_shoulder": (0.58, 0.28),
        "right_elbow": (0.72, 0.20),
        "right_wrist": (0.82, 0.12),
        # Braced wide stance
        "left_hip": (0.43, 0.59),
        "left_knee": (0.36, 0.73),
        "left_ankle": (0.32, 0.87),
        "right_hip": (0.56, 0.59),
        "right_knee": (0.64, 0.73),
        "right_ankle": (0.70, 0.87)
    }
    templates.append(PoseTemplate(
        id="attack_slash",
        name="Overhead Weapon Strike",
        category="combat",
        description="Dynamic combat slash with right arm extended high and braced footwork.",
        skeleton=create_skeleton_from_dict(attack_coords)
    ))

    return templates


def get_template_by_name(name_or_id: str) -> Optional[PoseTemplate]:
    """Retrieves a PoseTemplate by its ID or case-insensitive title."""
    clean = name_or_id.strip().lower()
    for t in get_pose_templates():
        if t.id == clean or t.name.lower() == clean:
            return t
    return None


def fit_skeleton_to_bbox(skeleton: Skeleton2D, bbox: BoundingBox, canvas_w: int = 128, canvas_h: int = 128) -> Skeleton2D:
    """
    Scales and shifts a normalized skeleton so its vertical span and horizontal center
    align naturally with the actual non-transparent bounding box of the sprite.
    """
    if bbox.width == 0 or bbox.height == 0:
        return skeleton

    # Compute bounding box normalized coordinates
    norm_min_x = bbox.min_x / canvas_w
    norm_max_x = bbox.max_x / canvas_w
    norm_min_y = bbox.min_y / canvas_h
    norm_max_y = bbox.max_y / canvas_h

    bbox_center_x = (norm_min_x + norm_max_x) / 2.0
    bbox_h = norm_max_y - norm_min_y

    # Find the current y-span of the skeleton (head to ankles)
    y_vals = [j.y for j in skeleton.joints.values()]
    x_vals = [j.x for j in skeleton.joints.values()]
    skel_min_y, skel_max_y = min(y_vals), max(y_vals)
    skel_center_x = (min(x_vals) + max(x_vals)) / 2.0
    skel_h = max(skel_max_y - skel_min_y, 0.05)

    scale_y = (bbox_h * 0.90) / skel_h

    fitted_joints = {}
    for name, joint in skeleton.joints.items():
        # Scale relative to head/pelvis and reposition
        rel_y = joint.y - skel_min_y
        new_y = norm_min_y + (bbox_h * 0.05) + (rel_y * scale_y)
        rel_x = joint.x - skel_center_x
        new_x = bbox_center_x + (rel_x * scale_y)

        clamped_x = max(0.01, min(0.99, new_x))
        clamped_y = max(0.01, min(0.99, new_y))
        fitted_joints[name] = Joint2D(
            name=name,
            x=clamped_x,
            y=clamped_y,
            is_visible=joint.is_visible,
            confidence=joint.confidence
        )

    return Skeleton2D(joints=fitted_joints, bounding_box=bbox)
