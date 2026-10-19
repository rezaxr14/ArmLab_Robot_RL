"""Tests for armlab.config validation and serialization."""

import pytest

from armlab.config import EnvParams, PhysicsParams, RewardParams, RLConfig, RobotParams


class TestRobotParams:
    def test_defaults_valid(self):
        p = RobotParams()
        assert p.link_lengths == (5.0, 10.0, 8.0)
        assert p.max_reach == 18.0
        assert p.base_height == 5.0

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            RobotParams(link_lengths=(1.0, 2.0))
        with pytest.raises(ValueError):
            RobotParams(link_lengths=(1.0, 2.0, 3.0, 4.0))

    def test_non_positive_length_rejected(self):
        with pytest.raises(ValueError):
            RobotParams(link_lengths=(1.0, 0.0, 3.0))
        with pytest.raises(ValueError):
            RobotParams(link_lengths=(1.0, -2.0, 3.0))

    def test_limits_order_rejected(self):
        with pytest.raises(ValueError):
            RobotParams(joint_limits_low=(1.0, -1.0, -1.0), joint_limits_high=(-1.0, 1.0, 1.0))

    def test_as_dict_round_trip(self):
        p = RobotParams(link_lengths=(1.0, 2.0, 3.0))
        p2 = RobotParams(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in p.as_dict().items()}
        )
        assert p2.link_lengths == (1.0, 2.0, 3.0)


class TestRewardParams:
    def test_valid(self):
        r = RewardParams(success_threshold=0.5)
        assert r.success_threshold == 0.5

    def test_negative_scale_rejected(self):
        with pytest.raises(ValueError):
            RewardParams(distance_penalty_scale=-0.1)

    def test_zero_threshold_rejected(self):
        with pytest.raises(ValueError):
            RewardParams(success_threshold=0.0)


class TestEnvParams:
    def test_defaults(self):
        e = EnvParams()
        assert e.control_mode == "discrete"
        assert e.max_episode_steps == 250
        assert e.action_repeat == 1

    def test_bad_control_mode(self):
        with pytest.raises(ValueError):
            EnvParams(control_mode="torque")

    def test_bad_ranges(self):
        with pytest.raises(ValueError):
            EnvParams(target_min_radius=10.0, target_max_radius=5.0)
        with pytest.raises(ValueError):
            EnvParams(target_min_height=3.0, target_max_height=1.0)

    def test_bad_action_repeat(self):
        with pytest.raises(ValueError):
            EnvParams(action_repeat=0)


class TestRLConfig:
    def test_epsilon_schedule(self):
        c = RLConfig(epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.5)
        assert c.epsilon_at(0) == 1.0
        assert c.epsilon_at(1) == 0.5
        assert c.epsilon_at(10) == 0.1  # clamped at min

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            RLConfig(gamma=1.5)

    def test_invalid_batch_buffer(self):
        with pytest.raises(ValueError):
            RLConfig(batch_size=1000, buffer_size=100)

    def test_invalid_target_mode(self):
        with pytest.raises(ValueError):
            RLConfig(target_update_mode="sometimes")

    def test_invalid_epsilon_order(self):
        with pytest.raises(ValueError):
            RLConfig(epsilon_start=0.01, epsilon_min=0.5)

    def test_from_dict_round_trip(self):
        c = RLConfig(hidden_sizes=(64, 64), lr=1e-3)
        c2 = RLConfig.from_dict(c.as_dict())
        assert c2.hidden_sizes == (64, 64)
        assert c2.lr == 1e-3


class TestPhysicsParams:
    def test_control_dt(self):
        p = PhysicsParams(sim_dt=0.002, control_decimation=10)
        assert p.control_dt == pytest.approx(0.02)

    def test_invalid(self):
        with pytest.raises(ValueError):
            PhysicsParams(sim_dt=0.0)
        with pytest.raises(ValueError):
            PhysicsParams(control_decimation=0)
