"""Tests for armlab.kinematics: FK, IK round trip, Jacobian, DLS."""

import numpy as np
import pytest

from armlab.config import RobotParams
from armlab.kinematics import RobotKinematics, wrap_angle, wrap_angles


@pytest.fixture
def kin():
    return RobotKinematics(RobotParams())


class TestFK:
    def test_home_pose(self, kin):
        pts = kin.link_positions(np.zeros(3))
        assert pts.shape == (4, 3)
        np.testing.assert_allclose(pts[0], [0, 0, 0])
        np.testing.assert_allclose(pts[1], [0, 0, kin.l1])
        # theta2=0: link 2 points straight up; theta3=0: fully extended.
        np.testing.assert_allclose(pts[3], [0, 0, kin.l1 + kin.l2 + kin.l3])

    def test_link_lengths_preserved(self, kin):
        q = np.array([0.7, -0.4, 1.2])
        pts = kin.link_positions(q)
        np.testing.assert_allclose(np.linalg.norm(pts[2] - pts[1]), kin.l2)
        np.testing.assert_allclose(np.linalg.norm(pts[3] - pts[2]), kin.l3)

    def test_nonfinite_rejected(self, kin):
        with pytest.raises(ValueError):
            kin.end_effector(np.array([np.nan, 0.0, 0.0]))


class TestIK:
    def test_round_trip_random(self, kin):
        rng = np.random.default_rng(42)
        worst = 0.0
        for _ in range(300):
            q = rng.uniform([-np.pi, -np.pi / 2, -np.pi], [np.pi, np.pi / 2, np.pi])
            target = kin.end_effector(q)
            res = kin.inverse_kinematics(target, q_current=q)
            assert res.reachable
            worst = max(worst, res.residual)
        assert worst < 1e-6

    def test_unreachable(self, kin):
        res = kin.inverse_kinematics(np.array([100.0, 0.0, 0.0]))
        assert not res.reachable
        assert res.solution is None
        assert np.isnan(res.residual)

    def test_branch_forced(self, kin):
        target = np.array([10.0, 2.0, 12.0])
        up = kin.inverse_kinematics(target, elbow="elbow_up")
        down = kin.inverse_kinematics(target, elbow="elbow_down")
        assert up.branch == "elbow_up"
        assert down.branch == "elbow_down"
        assert up.residual < 1e-6 and down.residual < 1e-6
        # Different branches genuinely differ in elbow sign.
        assert up.solution[2] > 0 > down.solution[2]

    def test_closest_branch_selection(self, kin):
        target = np.array([9.0, 3.0, 11.0])
        down = kin.inverse_kinematics(target, elbow="elbow_down")
        closest = kin.inverse_kinematics(target, q_current=down.solution)
        assert closest.branch == "elbow_down"

    def test_singularity_keeps_waist(self, kin):
        # Target on the z-axis: theta1 must be preserved, not reset to 0.
        q_current = np.array([1.234, 0.3, 0.2])
        target = np.array([0.0, 0.0, kin.l1 + 10.0])
        res = kin.inverse_kinematics(target, q_current=q_current)
        assert res.reachable
        assert res.solution[0] == pytest.approx(1.234)

    def test_bad_elbow_arg(self, kin):
        with pytest.raises(ValueError):
            kin.inverse_kinematics(np.array([10, 0, 10]), elbow="sideways")


class TestJacobian:
    def test_matches_finite_differences(self, kin):
        rng = np.random.default_rng(7)
        eps = 1e-7
        for _ in range(50):
            q = rng.uniform(-np.pi, np.pi, size=3)
            J = kin.jacobian(q)
            Jfd = np.zeros((3, 3))
            for i in range(3):
                dq = np.zeros(3)
                dq[i] = eps
                Jfd[:, i] = (kin.end_effector(q + dq) - kin.end_effector(q - dq)) / (2 * eps)
            np.testing.assert_allclose(J, Jfd, atol=1e-6)


class TestDLS:
    def test_converges_to_reachable_target(self, kin):
        target = kin.end_effector(np.array([0.5, 0.2, -0.3]))
        q = np.zeros(3)
        for _ in range(80):
            dx = target - kin.end_effector(q)
            q = q + kin.damped_least_squares(q, dx, damping=0.05)
        assert np.linalg.norm(kin.end_effector(q) - target) < 1e-4

    def test_single_step_reduces_error(self, kin):
        target = np.array([10.0, 1.0, 12.0])
        q = np.array([0.1, 0.2, 0.3])
        err0 = np.linalg.norm(kin.end_effector(q) - target)
        # DLS is a local linearization: use a bounded per-step displacement,
        # as resolved-rate control does in practice.
        dx = 0.5 * (target - kin.end_effector(q)) / err0
        q2 = q + kin.damped_least_squares(q, dx, damping=0.05)
        err1 = np.linalg.norm(kin.end_effector(q2) - target)
        assert err1 < err0

    def test_negative_damping_rejected(self, kin):
        with pytest.raises(ValueError):
            kin.damped_least_squares(np.zeros(3), np.ones(3), damping=-0.1)


class TestWrap:
    def test_wrap_angle(self):
        assert wrap_angle(0.0) == 0.0
        assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
        assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)

    def test_wrap_angles_array(self):
        arr = wrap_angles(np.array([0.0, 3 * np.pi, -3 * np.pi]))
        np.testing.assert_allclose(arr, [0.0, -np.pi, -np.pi], atol=1e-12)
