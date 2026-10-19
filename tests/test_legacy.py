"""Tests for the legacy compatibility layer.

These tests load the original legacy modules directly from their folders via
importlib and compare behavior with the armlab.legacy adapters. One test
documents the legacy IK bug that the corrected implementation fixes.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from armlab.config import RobotParams
from armlab.kinematics import RobotKinematics
from armlab.legacy import Robot as LegacyRobotAdapter

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_legacy_robot():
    path = REPO_ROOT / "Analytical_Inverse_Kinematics" / "robot.py"
    spec = importlib.util.spec_from_file_location("legacy_robot_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["legacy_robot_module"] = module
    spec.loader.exec_module(module)
    return module.Robot


LegacyRobot = _load_legacy_robot()


class TestFKEquivalence:
    def test_fk_matches_legacy(self):
        legacy = LegacyRobot([5, 10, 8])
        adapter = LegacyRobotAdapter([5, 10, 8])
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.uniform(-np.pi, np.pi, size=3)
            pts_legacy = legacy.forward_kinematics(q)
            pts_adapter = adapter.forward_kinematics(q)
            for pl, pa in zip(pts_legacy, pts_adapter):
                np.testing.assert_allclose(pl, pa, atol=1e-12)


class TestIKDivergence:
    def test_legacy_ik_round_trip_is_broken(self):
        """Documents the legacy IK bug: FK measures theta2 from vertical,
        but the legacy IK computes the plane angle from horizontal, so the
        round trip FK(IK(FK(q))) does not return to the target."""
        legacy = LegacyRobot([5, 10, 8])
        q = np.array([0.3, 0.5, -0.4])
        target = legacy.forward_kinematics(q)[3]
        sol = legacy.inverse_kinematics(target, q)
        assert sol is not None
        residual = np.linalg.norm(np.array(legacy.forward_kinematics(sol)[3]) - target)
        # Residual is O(arm size) - far beyond any acceptable tolerance.
        assert residual > 1.0

    def test_corrected_ik_round_trips(self):
        adapter = LegacyRobotAdapter([5, 10, 8])
        rng = np.random.default_rng(1)
        for _ in range(50):
            q = rng.uniform(-np.pi, np.pi, size=3)
            target = np.array(adapter.forward_kinematics(q)[3])
            sol = adapter.inverse_kinematics(target, q)
            assert sol is not None
            residual = np.linalg.norm(np.array(adapter.forward_kinematics(sol)[3]) - target)
            assert residual < 1e-6


class TestLegacyAPI:
    def test_total_reach(self):
        assert LegacyRobotAdapter([5, 10, 8]).total_reach == 18
        assert LegacyRobot([5, 10, 8]).total_reach == 18

    def test_length_validation(self):
        with pytest.raises(ValueError):
            LegacyRobotAdapter([1, 2])
        with pytest.raises(ValueError):
            LegacyRobot([1, 2])  # legacy raises the same way

    def test_unreachable_returns_none(self):
        adapter = LegacyRobotAdapter([5, 10, 8])
        assert adapter.inverse_kinematics([100, 0, 0], [0, 0, 0]) is None
        legacy = LegacyRobot([5, 10, 8])
        assert legacy.inverse_kinematics([100, 0, 0], [0, 0, 0]) is None

    def test_state_tracking(self):
        adapter = LegacyRobotAdapter([5, 10, 8])
        q = [0.1, 0.2, 0.3]
        adapter.forward_kinematics(q)
        assert adapter.thetas == [0.1, 0.2, 0.3]


class TestLegacyFKConsistencyWithKinematics:
    def test_kinematics_matches_legacy_fk(self):
        kin = RobotKinematics(RobotParams(link_lengths=(5.0, 10.0, 8.0)))
        legacy = LegacyRobot([5, 10, 8])
        rng = np.random.default_rng(2)
        for _ in range(30):
            q = rng.uniform(-np.pi, np.pi, size=3)
            pts_legacy = legacy.forward_kinematics(q)
            pts_new = kin.link_positions(q)
            np.testing.assert_allclose(pts_new, np.stack(pts_legacy), atol=1e-12)
