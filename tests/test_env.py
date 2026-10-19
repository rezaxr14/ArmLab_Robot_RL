"""Tests for the Gymnasium kinematic environments and wrappers."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env

from armlab.config import EnvParams, RewardParams, RobotParams
from armlab.envs import KinematicReachingEnv, MovingTargetEnv, register_envs
from armlab.envs.wrappers import ActionRepeat, StackObservations


class TestKinematicEnv:
    def test_gymnasium_api_compliance(self):
        env = KinematicReachingEnv()
        check_env(env, skip_render_check=True)

    def test_registration_and_make(self):
        register_envs()
        env = gym.make("ArmLabKinematic-v0")
        obs, info = env.reset(seed=1)
        assert obs.shape == (9,)
        env.close()

    def test_reset_determinism(self):
        env = KinematicReachingEnv()
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_allclose(obs1, obs2)
        # Different seeds give different targets with overwhelming likelihood.
        obs3, _ = env.reset(seed=456)
        assert not np.allclose(obs1[6:9], obs3[6:9])

    def test_step_determinism(self):
        def rollout(seed):
            env = KinematicReachingEnv()
            env.reset(seed=7)
            traj = []
            for a in [0, 2, 4, 1, 3, 5, 0]:
                obs, r, term, trunc, info = env.step(a)
                traj.append((obs.copy(), r, term, trunc))
            return traj

        t1, t2 = rollout(7), rollout(7)
        for (o1, r1, _, _), (o2, r2, _, _) in zip(t1, t2):
            np.testing.assert_allclose(o1, o2)
            assert r1 == r2

    def test_discrete_action_semantics(self):
        env = KinematicReachingEnv()
        env.reset(seed=0)
        env.unwrapped.q = np.zeros(3)
        env.step(0)  # joint 0, +
        assert env.unwrapped.q[0] == pytest.approx(np.deg2rad(5))
        env.step(1)  # joint 0, -
        assert env.unwrapped.q[0] == pytest.approx(0.0)

    def test_joint_limits_enforced(self):
        env = KinematicReachingEnv()
        env.reset(seed=0)
        env.unwrapped.q = np.full(3, np.pi)
        env.step(0)
        assert env.unwrapped.q[0] <= np.pi

    def test_success_terminates_with_bonus(self):
        rp = RewardParams(success_threshold=0.5, success_bonus=100.0)
        env = KinematicReachingEnv(
            env_params=EnvParams(control_mode="continuous"), reward_params=rp
        )
        env.reset(seed=0)
        # Place the arm exactly at the IK solution, then command zero motion.
        target = env.unwrapped.target.copy()
        res = env.unwrapped.kin.inverse_kinematics(target)
        env.unwrapped.q = res.solution
        obs, r, term, trunc, info = env.step(np.zeros(3))
        assert term is True
        assert r >= 100.0 - 1.0  # bonus minus tiny distance/time penalties

    def test_max_episode_steps_truncation(self):
        env = KinematicReachingEnv(env_params=EnvParams(max_episode_steps=5))
        env.reset(seed=0)
        term = trunc = False
        for _ in range(5):
            _, _, term, trunc, _ = env.step(0)
        assert trunc and not term

    def test_continuous_control_mode(self):
        env = KinematicReachingEnv(env_params=EnvParams(control_mode="continuous"))
        assert isinstance(env.action_space, gym.spaces.Box)
        check_env(env, skip_render_check=True)
        env.reset(seed=3)
        obs, r, term, trunc, info = env.step(np.array([0.5, -0.5, 0.0]))
        assert obs.shape == (9,)

    def test_relative_observation(self):
        env = KinematicReachingEnv(env_params=EnvParams(observation_kind="relative"))
        obs, _ = env.reset(seed=2)
        assert obs.shape == (6,)

    def test_action_repeat_param(self):
        env = KinematicReachingEnv(env_params=EnvParams(action_repeat=3))
        env.reset(seed=0)
        env.unwrapped.q = np.zeros(3)
        env.step(0)
        assert env.unwrapped.q[0] == pytest.approx(3 * np.deg2rad(5))

    def test_info_fields(self):
        env = KinematicReachingEnv()
        _, info = env.reset(seed=5)
        for key in ("distance", "ee_position", "target_position", "q"):
            assert key in info


class TestMovingTargetEnv:
    def test_determinism(self):
        def rollout(seed):
            env = MovingTargetEnv()
            env.reset(seed=seed)
            return [env.step(a)[0].copy() for a in [0, 2, 4]]

        assert all(np.allclose(a, b) for a, b in zip(rollout(11), rollout(11)))

    def test_target_moves(self):
        env = MovingTargetEnv()
        env.reset(seed=1)
        t0 = env.unwrapped.target.copy()
        for _ in range(10):
            env.step(0)
        assert not np.allclose(t0, env.unwrapped.target)


class TestWrappers:
    def test_action_repeat(self):
        env = ActionRepeat(KinematicReachingEnv(), 4)
        env.reset(seed=0)
        env.unwrapped.q = np.zeros(3)
        env.step(0)
        assert env.unwrapped.q[0] == pytest.approx(4 * np.deg2rad(5))

    def test_action_repeat_invalid(self):
        with pytest.raises(ValueError):
            ActionRepeat(KinematicReachingEnv(), 0)

    def test_stack_observations(self):
        env = StackObservations(KinematicReachingEnv(), 3)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3, 9)
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (3, 9)
        # After reset all stacked rows are identical.
        obs, _ = env.reset(seed=0)
        np.testing.assert_allclose(obs[0], obs[1])
        np.testing.assert_allclose(obs[1], obs[2])
