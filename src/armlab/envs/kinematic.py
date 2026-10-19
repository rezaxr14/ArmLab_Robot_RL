"""Kinematic reaching environment (Gymnasium API).

The environment simulates joint-space kinematics only (no dynamics, no
contacts): actions change joint angles directly, and the end effector moves
according to analytical forward kinematics. This is the benchmark's
lowest-dependency task environment and the base for the RL experiments.

Observations (``observation_kind="joint_ee_target"``):
    [q(3), ee(3), target(3)] -> shape (9,)
Observations (``observation_kind="relative"``):
    [q(3), target - ee(3)] -> shape (6,)

Actions:
    Discrete(6): joint a//2 by +/- action_step_size (default 5 deg).
    Continuous: Box(-1, 1, (3,)) scaled by continuous_action_scale.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from armlab.config import EnvParams, RewardParams, RobotParams
from armlab.kinematics import RobotKinematics
from armlab.targets import sample_uniform_reachable

__all__ = ["KinematicReachingEnv"]


class KinematicReachingEnv(gym.Env):
    """3-link arm reaching task on the kinematic model.

    Deterministic for a fixed (state, action, seed): transitions are pure
    functions of the joint angles and the action; targets are sampled with
    the env's seeded ``numpy.random.Generator``.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        robot_params: Optional[RobotParams] = None,
        env_params: Optional[EnvParams] = None,
        reward_params: Optional[RewardParams] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.robot_params = robot_params or RobotParams()
        self.env_params = env_params or EnvParams()
        self.reward_params = reward_params or RewardParams()
        self.render_mode = render_mode
        self.kin = RobotKinematics(self.robot_params)

        if self.env_params.control_mode == "discrete":
            self.action_space = spaces.Discrete(6)
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )

        low_lim = np.asarray(self.robot_params.joint_limits_low, dtype=np.float32)
        high_lim = np.asarray(self.robot_params.joint_limits_high, dtype=np.float32)
        reach = np.float32(self.robot_params.base_height + self.robot_params.max_reach)
        if self.env_params.observation_kind == "joint_ee_target":
            obs_low = np.concatenate([low_lim, [-reach] * 3, [-reach] * 3]).astype(np.float32)
            obs_high = np.concatenate([high_lim, [reach] * 3, [reach] * 3]).astype(np.float32)
        else:  # relative
            obs_low = np.concatenate([low_lim, [-2 * reach] * 3]).astype(np.float32)
            obs_high = np.concatenate([high_lim, [2 * reach] * 3]).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.q = np.zeros(3, dtype=np.float64)
        self.target = np.zeros(3, dtype=np.float64)
        self.elapsed_steps = 0
        self._visualizer = None

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        ee = self.kin.end_effector(self.q)
        if self.env_params.observation_kind == "joint_ee_target":
            obs = np.concatenate([self.q, ee, self.target])
        else:
            obs = np.concatenate([self.q, self.target - ee])
        return obs.astype(np.float32)

    def _apply_action(self, action: np.ndarray | int) -> None:
        """Apply one action to the joint angles (with limits enforced)."""
        low = np.asarray(self.robot_params.joint_limits_low)
        high = np.asarray(self.robot_params.joint_limits_high)
        if isinstance(action, (int, np.integer)):
            joint = int(action) // 2
            sign = 1.0 if int(action) % 2 == 0 else -1.0
            dq = np.zeros(3)
            dq[joint] = sign * self.env_params.action_step_size
        else:
            a = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
            dq = a * self.env_params.continuous_action_scale
        self.q = np.clip(self.q + dq, low, high)

    def _reward(self, action: np.ndarray | int, dist: float) -> Tuple[float, bool, bool]:
        rp = self.reward_params
        reward = -rp.distance_penalty_scale * dist
        if rp.action_penalty:
            magnitude = 1.0 if isinstance(action, (int, np.integer)) else float(
                np.linalg.norm(np.asarray(action, dtype=float))
            )
            reward -= rp.action_penalty * magnitude
        reward -= rp.time_penalty

        terminated = bool(dist < rp.success_threshold and rp.success_terminates)
        if terminated:
            reward += rp.success_bonus
        truncated = bool(self.elapsed_steps >= self.env_params.max_episode_steps)
        return float(reward), terminated, truncated

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.q = np.zeros(3, dtype=np.float64)
        if options and "target" in options:
            self.target = np.asarray(options["target"], dtype=np.float64).reshape(3)
        else:
            self.target = sample_uniform_reachable(
                self.robot_params, self.env_params, self.np_random, kinematics=self.kin
            )
        self.elapsed_steps = 0
        return self._get_obs(), self._info(0.0)

    def step(self, action):
        for _ in range(self.env_params.action_repeat):
            self._apply_action(action)
            self.elapsed_steps += 1
        ee = self.kin.end_effector(self.q)
        dist = float(np.linalg.norm(ee - self.target))
        reward, terminated, truncated = self._reward(action, dist)
        obs = self._get_obs()
        info = self._info(dist)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _info(self, dist: float) -> Dict[str, Any]:
        return {
            "distance": dist,
            "ee_position": self.kin.end_effector(self.q).copy(),
            "target_position": self.target.copy(),
            "q": self.q.copy(),
        }

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return None
        if self._visualizer is None:
            from armlab.legacy.visualizer_matplotlib import RobotVisualizer

            self._visualizer = RobotVisualizer(self.robot_params.link_lengths)
        self._visualizer.plot_robot(self.kin.link_positions(self.q), self.target)
        return None

    def close(self):
        if self._visualizer is not None:
            self._visualizer.close()
            self._visualizer = None
