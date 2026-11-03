"""Tests for collision primitives and the joint-space RRT planner."""

import numpy as np
import pytest

from armlab.config import RobotParams
from armlab.kinematics import RobotKinematics
from armlab.planning import (
    BoxObstacle,
    JointSpaceRRT,
    SphereObstacle,
    collides,
    min_clearance,
    segment_point_distance,
)


class TestCollisionPrimitives:
    def test_segment_point_distance(self):
        a, b = np.zeros(3), np.array([1.0, 0, 0])
        assert segment_point_distance(a, b, np.array([0.5, 1.0, 0])) == pytest.approx(1.0)
        assert segment_point_distance(a, b, np.array([-1.0, 0, 0])) == pytest.approx(1.0)
        assert segment_point_distance(a, a, np.array([3.0, 4.0, 0])) == pytest.approx(5.0)

    def test_sphere_clearance(self):
        obs = SphereObstacle(center=np.array([0.5, 0.5, 0.0]), radius=0.2)
        seg = [np.array([0.0, 0, 0]), np.array([1.0, 0, 0])]
        # Distance from center to segment is 0.5; surface clearance = 0.3.
        assert min_clearance(seg, obs) == pytest.approx(0.3)
        penetrating = [np.array([0.4, 0.5, 0]), np.array([0.6, 0.5, 0])]
        assert min_clearance(penetrating, obs) < 0

    def test_box_clearance(self):
        box = BoxObstacle(low=np.array([0.4, -0.1, -0.1]), high=np.array([0.6, 0.1, 0.1]))
        seg = [np.array([0.0, 0, 0]), np.array([1.0, 0, 0])]
        assert min_clearance(seg, box) == pytest.approx(0.0)  # passes through
        above = [np.array([0.0, 0.5, 0]), np.array([1.0, 0.5, 0])]
        assert min_clearance(above, box) == pytest.approx(0.4)

    def test_collides_with_radius(self):
        obs = SphereObstacle(center=np.array([0.5, 0.5, 0.0]), radius=0.1)
        seg = [np.array([0.0, 0, 0]), np.array([1.0, 0, 0])]
        # Distance from center to segment is 0.5; with link radius 0.3 no collision...
        assert not collides(seg, [obs], link_radius=0.3)
        # ...but with radius 0.5 there is.
        assert collides(seg, [obs], link_radius=0.5)

    def test_invalid_obstacles(self):
        with pytest.raises(ValueError):
            SphereObstacle(center=np.zeros(3), radius=0.0)
        with pytest.raises(ValueError):
            BoxObstacle(low=np.ones(3), high=np.ones(3))

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            min_clearance([np.zeros(3), np.ones(3)], "not-an-obstacle")


class TestRRT:
    @pytest.fixture
    def kin(self):
        return RobotKinematics(RobotParams())

    def test_plans_without_obstacles(self, kin):
        planner = JointSpaceRRT(kin, seed=1)
        q_start = np.zeros(3)
        q_goal = np.array([1.0, 0.5, -0.5])
        result = planner.plan(q_start, q_goal)
        assert result.success
        assert result.path.shape[1] == 3
        np.testing.assert_allclose(result.path[0], q_start, atol=1e-9)
        np.testing.assert_allclose(result.path[-1], q_goal, atol=1e-9)

    def test_path_is_collision_free(self, kin):
        obstacle = SphereObstacle(center=np.array([10.0, 0.0, 12.0]), radius=1.5)
        planner = JointSpaceRRT(kin, obstacles=[obstacle], seed=2, max_iterations=1500)
        q_start = np.array([0.0, 0.6, -0.4])  # EE around (0, 0, 17)
        q_goal = np.array([np.pi, 0.6, -0.4])  # EE mirrored across z-axis
        result = planner.plan(q_start, q_goal)
        assert result.success, result.reason
        for q in result.path:
            assert planner._config_free(q)

    def test_start_in_collision(self, kin):
        obstacle = SphereObstacle(center=np.array([0.0, 0.0, kin.l1 + kin.l2 + kin.l3]), radius=2.0)
        planner = JointSpaceRRT(kin, obstacles=[obstacle], seed=3)
        q_start = np.zeros(3)  # EE exactly at obstacle center
        result = planner.plan(q_start, np.array([0.5, 0.5, 0.5]))
        assert not result.success
        assert result.reason == "start_in_collision"

    def test_goal_in_collision(self, kin):
        q_goal = np.zeros(3)
        ee = kin.end_effector(q_goal)
        obstacle = SphereObstacle(center=ee, radius=1.0)
        planner = JointSpaceRRT(kin, obstacles=[obstacle], seed=4)
        result = planner.plan(np.array([1.0, 0.3, 0.2]), q_goal)
        assert result.reason == "goal_in_collision"

    def test_reproducible(self, kin):
        def run():
            planner = JointSpaceRRT(kin, seed=9, max_iterations=400)
            return planner.plan(np.zeros(3), np.array([0.8, 0.4, -0.6]))

        r1, r2 = run(), run()
        assert r1.success == r2.success
        assert r1.iterations == r2.iterations
        if r1.success:
            np.testing.assert_allclose(r1.path, r2.path)

    def test_invalid_params(self, kin):
        with pytest.raises(ValueError):
            JointSpaceRRT(kin, step_size=0.0)
        with pytest.raises(ValueError):
            JointSpaceRRT(kin, goal_bias=1.5)
        with pytest.raises(ValueError):
            JointSpaceRRT(kin, max_iterations=0)
