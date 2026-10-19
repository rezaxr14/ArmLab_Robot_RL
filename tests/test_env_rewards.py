"""Tests for reward components, termination reasons and observation bounds."""

import numpy as np
import pytest

from armlab.config import EnvParams, RewardParams, RobotParams
from armlab.envs import KinematicReachingEnv
from armlab.kinematics import RobotKinematics


class TestRewardComponents:
    def test_progress_rewards_approaching(self):
        rp = RewardParams(
            progress_scale=1.0,
            distance_penalty_scale=0.0,
            success_threshold=1e-6,
            success_terminates=False,
        )
        env = KinematicReachingEnv(
            env_params=EnvParams(control_mode="continuous"), reward_params=rp
        )
        env.reset(seed=0)
        target = env.unwrapped.target.copy()
        # Aim the arm roughly at the target so distance decreases.
        q = env.unwrapped.kin.inverse_kinematics(target).solution
        env.unwrapped.q = q * 0.5
        before = env.unwrapped._distance()
        _, reward, _, _, info = env.step(
            np.clip((q - env.unwrapped.q) / env.env_params.continuous_action_scale, -1, 1)
        )
        after = info["distance"]
        assert after < before
        assert info["reward_components"]["progress"] > 0.0
        assert reward > 0.0

    def test_success_component_and_termination(self):
        rp = RewardParams(success_threshold=0.5, success_bonus=50.0)
        env = KinematicReachingEnv(
            env_params=EnvParams(control_mode="continuous"), reward_params=rp
        )
        env.reset(seed=0)
        q = env.unwrapped.kin.inverse_kinematics(env.unwrapped.target).solution
        env.unwrapped.q = q
        _, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert terminated is True and truncated is False
        assert info["success"] is True
        assert info["reward_components"]["success"] == pytest.approx(50.0)
        assert info["termination_reason"] == "success"

    def test_time_limit_truncation_reason(self):
        env = KinematicReachingEnv(env_params=EnvParams(max_episode_steps=3))
        env.reset(seed=0)
        info = {}
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(0)
        assert truncated and not terminated
        assert info["termination_reason"] == "time_limit"
        assert info["success"] is False

    def test_components_present_every_step(self):
        env = KinematicReachingEnv()
        env.reset(seed=1)
        _, _, _, _, info = env.step(0)
        for key in ("progress", "distance", "action_cost", "time_cost", "limit_penalty", "success"):
            assert key in info["reward_components"]

    def test_joint_limit_hit_flag(self):
        env = KinematicReachingEnv()
        env.reset(seed=0)
        env.unwrapped.q = np.full(3, np.pi)
        _, _, _, _, info = env.step(0)
        assert info["joint_limit_hit"] is True


class TestObservationBounds:
    def test_random_actions_stay_in_bounds(self):
        env = KinematicReachingEnv()
        obs, _ = env.reset(seed=2)
        rng = np.random.default_rng(0)
        for _ in range(200):
            assert env.observation_space.contains(obs)
            assert np.all(np.isfinite(obs))
            obs, _, terminated, truncated, _ = env.step(int(rng.integers(0, 6)))
            if terminated or truncated:
                obs, _ = env.reset(seed=int(rng.integers(0, 10_000)))

    def test_invalid_discrete_action_raises(self):
        env = KinematicReachingEnv()
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(9)

    def test_observation_version_present(self):
        assert KinematicReachingEnv.observation_version == "1.0"
        assert RobotKinematics(RobotParams()) is not None
