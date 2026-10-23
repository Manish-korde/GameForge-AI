"""
GameForge AI — 2D Character Pose Constraints & Biomechanical Validation
Validates 2D joint coordinates, finite bounds, topology completeness,
and enforces anatomical angular and displacement limits for game character skeletons.
"""

import math
from typing import Dict, List, Tuple, Optional
from .schemas import Skeleton2D, Joint2D

# Canonical angular and displacement limits
JOINT_CONSTRAINTS: Dict[str, Dict[str, float]] = {
    "left_elbow": {"min_angle_deg": 10.0, "max_angle_deg": 180.0},
    "right_elbow": {"min_angle_deg": 10.0, "max_angle_deg": 180.0},
    "left_knee": {"min_angle_deg": 15.0, "max_angle_deg": 180.0},
    "right_knee": {"min_angle_deg": 15.0, "max_angle_deg": 180.0},
    "neck": {"max_displacement": 0.12},
    "pelvis": {"max_displacement": 0.20}
}

# Hinge joint triplet definitions: (proximal_joint, hinge_joint, distal_joint)
HINGE_TRIPLETS: Dict[str, Tuple[str, str, str]] = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle")
}

REQUIRED_CANONICAL_JOINTS = [
    "head", "neck", "torso", "pelvis",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle"
]


def compute_joint_angle(p1: Joint2D, p_mid: Joint2D, p2: Joint2D) -> float:
    """
    Computes the interior angle in degrees at vertex p_mid between segments (p1 - p_mid) and (p2 - p_mid).
    Returns value in range [0.0, 180.0] degrees.
    """
    v1_x = p1.x - p_mid.x
    v1_y = p1.y - p_mid.y
    v2_x = p2.x - p_mid.x
    v2_y = p2.y - p_mid.y

    len1 = math.hypot(v1_x, v1_y)
    len2 = math.hypot(v2_x, v2_y)

    if len1 < 1e-6 or len2 < 1e-6:
        return 180.0

    dot = v1_x * v2_x + v1_y * v2_y
    cos_theta = dot / (len1 * len2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return round(math.degrees(math.acos(cos_theta)), 2)


def validate_skeleton_topology(skeleton: Skeleton2D) -> Tuple[bool, List[str]]:
    """
    Validates structural integrity of a Skeleton2D:
    - Presence of canonical joints.
    - Finite coordinate values (no NaN or Inf).
    - Normalized coordinate bounds [0.0, 1.0].
    """
    errors: List[str] = []

    if not skeleton or not skeleton.joints:
        return False, ["Skeleton has no joint definitions."]

    # Check for missing canonical joints
    missing = [name for name in REQUIRED_CANONICAL_JOINTS if name not in skeleton.joints]
    if missing:
        errors.append(f"Missing required canonical joints: {', '.join(missing)}")

    # Check coordinates
    for name, joint in skeleton.joints.items():
        if math.isnan(joint.x) or math.isnan(joint.y) or math.isinf(joint.x) or math.isinf(joint.y):
            errors.append(f"Joint '{name}' has non-finite coordinates: ({joint.x}, {joint.y})")
        elif joint.x < 0.0 or joint.x > 1.0 or joint.y < 0.0 or joint.y > 1.0:
            errors.append(f"Joint '{name}' coordinates ({joint.x}, {joint.y}) are outside normalized canvas [0.0, 1.0].")

    return (len(errors) == 0), errors


def evaluate_pose_constraints(
    skeleton: Skeleton2D,
    reference_skeleton: Optional[Skeleton2D] = None
) -> Tuple[bool, List[str]]:
    """
    Evaluates biomechanical plausibility of joint angles and displacements.
    Returns (is_compliant, warnings).
    """
    warnings: List[str] = []

    # 1. Check hinge angle limits (elbows and knees)
    for hinge_name, (prox, mid, dist) in HINGE_TRIPLETS.items():
        j_prox = skeleton.get_joint(prox)
        j_mid = skeleton.get_joint(mid)
        j_dist = skeleton.get_joint(dist)

        if not j_prox or not j_mid or not j_dist:
            continue

        angle = compute_joint_angle(j_prox, j_mid, j_dist)
        limits = JOINT_CONSTRAINTS.get(hinge_name, {})
        min_angle = limits.get("min_angle_deg", 0.0)
        max_angle = limits.get("max_angle_deg", 180.0)

        if angle < min_angle:
            warnings.append(
                f"Joint '{hinge_name}' acute angle violation: {angle}° < minimum allowable {min_angle}°."
            )
        elif angle > max_angle:
            warnings.append(
                f"Joint '{hinge_name}' hyper-extension violation: {angle}° > maximum allowable {max_angle}°."
            )

    # 2. Check maximum displacement for anchor joints (neck and pelvis) if reference is provided
    if reference_skeleton:
        for anchor_name in ["neck", "pelvis"]:
            j_cur = skeleton.get_joint(anchor_name)
            j_ref = reference_skeleton.get_joint(anchor_name)
            if j_cur and j_ref:
                disp = math.hypot(j_cur.x - j_ref.x, j_cur.y - j_ref.y)
                max_disp = JOINT_CONSTRAINTS.get(anchor_name, {}).get("max_displacement", 0.25)
                if disp > max_disp:
                    warnings.append(
                        f"Excessive displacement on '{anchor_name}': {disp:.3f} > maximum threshold {max_disp:.3f}."
                    )

    is_compliant = (len(warnings) == 0)
    return is_compliant, warnings
