"""Tests for classical controllers and the discrete action adapter."""

import numpy as np
import pytest

from armlab.config import RobotParams
from armlab.controllers import (
    CartesianDLSController,
    DiscreteActionAdapter,
    JointPDController,
    JointPositionController,
    ShootingMPCController,
    SphereObstacle,
    discrete_argmax,
)
from armlab.kinematics import RobotKinematics


@pytest.fixture
def kin():
    return RobotKinematics(RobotParams())


def _distance(kin, q, target):
    return float(np.linalg.norm(kin.end_effector(np.asarray(q)) - target))


class TestDiscreteArgmax:
    def test_picks_correct_joint_direction(self):
        step = np.deg2rad(5)
        assert discrete_argmax(np.array([0.1, 0.0, 0.0]), step) == 0   # joint 0 +
        assert discrete_argmax(np.array([-0.1, 0.0, 0.0]), step) == 1  # joint 0 -
        assert discrete_argmax(np.array([0.0, 0.0, 0.1]), step) == 4   # joint 2 +

    def test_largest_magnitude_wins(self):
        step = np.deg2rad(5)
        assert discrete_argmax(np.array([0.1, 0.5, -0.2]), step) == 2  # joint 1 +


class TestJointPositionController:
    def test_reduces_distance(self, kin):
        ctrl = JointPositionController(kin, kp=0.4)
        target = np.array([10.0, 2.0, 12.0])
        q = np.zeros(3)
        d0 = _distance(kin, q, target)
        for _ in range(40):
            q = q + ctrl.act(q, target)
        assert _distance(kin, q, target) < d0

    def test_reaches_target(self, kin):
        ctrl = JointPositionController(kin, kp=0.5, max_dq=0.3)
        target = np.array([9.0, -3.0, 13.0])
        q = np.zeros(3)
        for _ in range(120):
            q = q + ctrl.act(q, target)
        assert _distance(kin, q, target) < 1e-2

    def test_unreachable_returns_zero(self, kin):
        ctrl = JointPositionController(kin)
        q = np.zeros(3)
        np.testing.assert_allclose(ctrl.act(q, np.array([100.0, 0, 0])), np.zeros(3))

    def test_invalid_params(self, kin):
        with pytest.raises(ValueError):
            JointPositionController(kin, kp=-1.0)
        with pytest.raises(ValueError):
            JointPositionController(kin, max_dq=0.0)


class TestJointPDController:
    def test_reaches_target(self, kin):
        ctrl = JointPDController(kin, kp=0.5, kd=0.1, dt=0.1)
        target = np.array([8.0, 4.0, 14.0])
        q = np.zeros(3)
        for _ in range(150):
            q = q + ctrl.act(q, target)
        assert _distance(kin, q, target) < 0.5

    def test_reset_clears_state(self, kin):
        ctrl = JointPDController(kin)
        ctrl.act(np.zeros(3), np.array([10.0, 0, 10.0]))
        assert ctrl._q_prev is not None
        ctrl.reset()
        assert ctrl._q_prev is None


class TestCartesianDLS:
    def test_reaches_target(self, kin):
        ctrl = CartesianDLSController(kin, kp=0.5, damping=0.05, max_dq=0.3)
        target = np.array([11.0, 1.0, 10.0])
        q = np.zeros(3)
        for _ in range(150):
            q = q + ctrl.act(q, target)
        assert _distance(kin, q, target) < 1e-2

    def test_obstacle_repulsion_pushes_away(self, kin):
        # Obstacle directly between the start EE and the target.
        obstacle = SphereObstacle(center=np.array([8.0, 0.0, 10.0]), radius=1.0)
        ctrl_free = CartesianDLSController(kin, kp=0.5, damping=0.05, max_dq=0.3)
        ctrl_obs = CartesianDLSController(
            kin, kp=0.5, damping=0.05, max_dq=0.3,
            obstacles=[obstacle], repulse_gain=3.0, influence_radius=5.0,
        )
        target = np.array([14.0, 0.0, 10.0])
        q_free = np.zeros(3)
        q_obs = np.zeros(3)
        min_dist_free = np.inf
        min_dist_obs = np.inf
        for _ in range(60):
            q_free = q_free + ctrl_free.act(q_free, target)
            q_obs = q_obs + ctrl_obs.act(q_obs, target)
            ee_f = kin.end_effector(q_free)
            ee_o = kin.end_effector(q_obs)
            min_dist_free = min(min_dist_free, np.linalg.norm(ee_f - obstacle.center))
            min_dist_obs = min(min_dist_obs, np.linalg.norm(ee_o - obstacle.center))
        # With repulsion the EE keeps farther from the obstacle center.
        assert min_dist_obs > min_dist_free

    def test_invalid_params(self, kin):
        with pytest.raises(ValueError):
            CartesianDLSController(kin, kp=-1.0)
        with pytest.raises(ValueError):
            SphereObstacle(center=np.zeros(3), radius=-1.0)


class TestShootingMPC:
    def test_deterministic(self, kin):
        ctrl1 = ShootingMPCController(kin, seed=3, horizon=4, n_samples=20)
        ctrl2 = ShootingMPCController(kin, seed=3, horizon=4, n_samples=20)
        q = np.array([0.2, 0.3, -0.1])
        target = np.array([10.0, 2.0, 12.0])
        np.testing.assert_allclose(ctrl1.act(q, target), ctrl2.act(q, target))

    def test_reduces_distance(self, kin):
        ctrl = ShootingMPCController(kin, seed=1, horizon=5, n_samples=30)
        target = kin.end_effector(np.array([0.4, -0.3, 0.5]))
        q = np.zeros(3)
        d0 = _distance(kin, q, target)
        for _ in range(60):
            q = q + ctrl.act(q, target)
        assert _distance(kin, q, target) < d0

    def test_invalid_params(self, kin):
        with pytest.raises(ValueError):
            ShootingMPCController(kin, horizon=0)
        with pytest.raises(ValueError):
            ShootingMPCController(kin, n_samples=0)


class TestAdapter:
    def test_adapter_produces_discrete_action(self, kin):
        ctrl = JointPositionController(kin)
        adapter = DiscreteActionAdapter(ctrl, step_size=np.deg2rad(5))
        action = adapter.act(np.zeros(3), np.array([10.0, 0.0, 12.0]))
        assert action in range(6)
