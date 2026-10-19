"""Tests for the evaluation harness: metrics, runner, aggregation."""

import numpy as np
import pytest

from armlab.config import EnvParams, RewardParams, RobotParams
from armlab.controllers import JointPositionController
from armlab.envs import KinematicReachingEnv
from armlab.evaluation import (
    EpisodeRecord,
    aggregate,
    compute_metrics,
    controller_policy,
    evaluate_controller,
    rollout_controller,
)
from armlab.kinematics import RobotKinematics
from armlab.planning import SphereObstacle


class TestComputeMetrics:
    def _record(self, success=True):
        # EE travels toward target over 5 steps.
        ee = np.array(
            [[0, 0, 5], [2, 0, 5], [4, 0, 5], [6, 0, 5], [6.5, 0, 5], [6.8, 0, 5]], dtype=float
        )
        target_dist = np.array([10, 8, 6, 4, 2, 0.1 if success else 5.0])
        return EpisodeRecord(
            ee_positions=ee,
            distances=target_dist,
            rewards=[-1.0] * 5,
            q_positions=np.zeros((6, 3)),
            success_threshold=1.0,
            steps=5,
        )

    def test_success_metrics(self):
        m = compute_metrics(self._record(success=True))
        assert m.success
        assert m.time_to_reach == 5
        assert m.path_length == pytest.approx(
            np.sum(np.linalg.norm(np.diff(self._record().ee_positions, axis=0), axis=1))
        )

    def test_failure_metrics(self):
        m = compute_metrics(self._record(success=False))
        assert not m.success
        assert np.isinf(m.time_to_reach)

    def test_smoothness_nonnegative(self):
        m = compute_metrics(self._record())
        assert m.smoothness >= 0.0

    def test_total_reward(self):
        m = compute_metrics(self._record())
        assert m.total_reward == -5.0


class TestAggregate:
    def test_aggregate_fields(self):
        records = [
            compute_metrics(
                EpisodeRecord(
                    ee_positions=np.zeros((3, 3)),
                    distances=np.array([2.0, 1.0, 0.5]),
                    rewards=[-1, -1],
                    q_positions=np.zeros((3, 3)),
                    success_threshold=1.0,
                )
            )
            for _ in range(4)
        ]
        report = aggregate(records)
        assert report["n_episodes"] == 4
        assert report["success_rate"] == 1.0
        assert "time_to_reach" in report and "path_length" in report

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate([])


class TestRollout:
    def test_rollout_records_and_succeeds(self):
        kin = RobotKinematics(RobotParams())
        ctrl = JointPositionController(kin, kp=0.5, max_dq=0.3)
        env = KinematicReachingEnv(
            env_params=EnvParams(max_episode_steps=250),
            reward_params=RewardParams(success_threshold=0.5),
        )
        record, metrics = rollout_controller(env, controller_policy(ctrl, np.deg2rad(5)), seed=0)
        assert record.ee_positions.shape[0] == record.steps + 1
        assert record.q_positions.shape == (record.steps + 1, 3)
        assert record.link_positions.shape == (record.steps + 1, 4, 3)
        # The P controller with 5-degree discrete steps should reach a
        # reachable target within 250 steps in most cases; seed 0 verified
        # during development. We assert success for this deterministic case.
        assert metrics.success, f"IK controller failed to reach: {metrics}"

    def test_rollout_deterministic(self):
        kin = RobotKinematics(RobotParams())
        ctrl = JointPositionController(kin, kp=0.5, max_dq=0.3)
        env = KinematicReachingEnv()
        _, m1 = rollout_controller(env, controller_policy(ctrl, np.deg2rad(5)), seed=3)
        _, m2 = rollout_controller(env, controller_policy(ctrl, np.deg2rad(5)), seed=3)
        assert m1.as_dict() == m2.as_dict()

    def test_evaluate_controller_report(self):
        kin = RobotKinematics(RobotParams())
        ctrl = JointPositionController(kin, kp=0.5, max_dq=0.3)
        env = KinematicReachingEnv(env_params=EnvParams(max_episode_steps=250))
        report = evaluate_controller(
            env, controller_policy(ctrl, np.deg2rad(5)), n_episodes=5, base_seed=0
        )
        assert report["n_episodes"] == 5
        assert 0.0 <= report["success_rate"] <= 1.0

    def test_obstacle_metrics(self):
        kin = RobotKinematics(RobotParams())
        ctrl = JointPositionController(kin, kp=0.5, max_dq=0.3)
        env = KinematicReachingEnv()
        # Obstacle far away: no collisions, positive clearance.
        obstacle = SphereObstacle(center=np.array([30.0, 30.0, 30.0]), radius=1.0)
        record, metrics = rollout_controller(
            env, controller_policy(ctrl, np.deg2rad(5)), seed=1, obstacles=[obstacle]
        )
        assert metrics.collisions == 0
        assert metrics.min_clearance is not None and metrics.min_clearance > 0
