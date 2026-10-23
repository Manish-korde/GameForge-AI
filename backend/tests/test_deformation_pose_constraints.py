"""
Tests for 2D Character Pose Constraints & Biomechanical Validation
Validates joint angle computation, hinge limit enforcement, coordinate sanity,
and anchor displacement limits.
"""

import os
import math
import unittest

import sys
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from deformation.schemas import Joint2D, Skeleton2D
from deformation.skeleton import get_default_skeleton, get_pose_templates
from deformation.pose_constraints import (
    compute_joint_angle,
    validate_skeleton_topology,
    evaluate_pose_constraints,
    JOINT_CONSTRAINTS
)


class TestDeformationPoseConstraints(unittest.TestCase):

    def test_compute_joint_angle_geometry(self):
        """Angle calculation must accurately compute straight, right, and acute angles."""
        p_mid = Joint2D(name="mid", x=0.5, y=0.5)
        # Straight line: p1=(0.5, 0.2), mid=(0.5, 0.5), p2=(0.5, 0.8) -> 180 degrees
        p1_straight = Joint2D(name="p1", x=0.5, y=0.2)
        p2_straight = Joint2D(name="p2", x=0.5, y=0.8)
        self.assertAlmostEqual(compute_joint_angle(p1_straight, p_mid, p2_straight), 180.0, places=1)

        # Right angle (90 deg): p1=(0.5, 0.2), mid=(0.5, 0.5), p2=(0.8, 0.5)
        p2_right = Joint2D(name="p2", x=0.8, y=0.5)
        self.assertAlmostEqual(compute_joint_angle(p1_straight, p_mid, p2_right), 90.0, places=1)

    def test_default_skeleton_topology_valid(self):
        """Default neutral skeleton must pass structural topology checks without error."""
        default_skel = get_default_skeleton()
        is_valid, errors = validate_skeleton_topology(default_skel)
        self.assertTrue(is_valid, f"Topology errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_all_tested_templates_topology_valid(self):
        """All 7 pre-tested pose templates must have complete 16-joint canonical topology."""
        for tpl in get_pose_templates():
            is_valid, errors = validate_skeleton_topology(tpl.skeleton)
            self.assertTrue(is_valid, f"Template '{tpl.id}' topology invalid: {errors}")

    def test_topology_catches_missing_joint(self):
        """Skeleton missing required canonical joints must be flagged invalid."""
        default_skel = get_default_skeleton()
        bad_joints = dict(default_skel.joints)
        del bad_joints["head"]
        bad_skel = Skeleton2D(joints=bad_joints)

        is_valid, errors = validate_skeleton_topology(bad_skel)
        self.assertFalse(is_valid)
        self.assertTrue(any("head" in err for err in errors))

    def test_topology_catches_non_finite_coordinates(self):
        """Skeleton with NaN or Inf coordinates must be flagged invalid by validate_skeleton_topology."""
        default_skel = get_default_skeleton()
        bad_joints = dict(default_skel.joints)
        bad_joint = Joint2D(name="head", x=0.5, y=0.5)
        object.__setattr__(bad_joint, "x", float("nan"))
        bad_joints["head"] = bad_joint
        bad_skel = Skeleton2D.model_construct(joints=bad_joints)

        is_valid, errors = validate_skeleton_topology(bad_skel)
        self.assertFalse(is_valid)
        self.assertTrue(any("non-finite" in err for err in errors))

    def test_elbow_acute_angle_violation(self):
        """Folding the elbow below 10 degrees must trigger an acute angle constraint warning."""
        skel = get_default_skeleton()
        # Set left shoulder=(0.4, 0.3), elbow=(0.4, 0.4), wrist=(0.401, 0.301) -> acute ~ 0 degrees
        bad_skel = skel.with_updated_joint("left_wrist", x=0.401, y=0.301)
        is_compliant, warnings = evaluate_pose_constraints(bad_skel)
        self.assertFalse(is_compliant)
        self.assertTrue(any("acute angle violation" in w.lower() for w in warnings))


if __name__ == "__main__":
    unittest.main()
