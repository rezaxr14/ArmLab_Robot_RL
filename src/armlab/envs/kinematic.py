"""Kinematic reaching environment (Gymnasium API).

The environment simulates joint-space kinematics only (no dynamics, no
contacts): actions change joint angles directly, and the end effector moves
according to analytical forward kinematics. This is the benchmark's
lowest-dependency task environment and the base for the RL experiments.

Observation contract (version ``1.0``, ``observation_kind="joint_ee_target"``):
    [q(3), ee(3), target(3)] -> shape (9,)
``observation_kind="relative"``:
    [q(3), target - ee(3)] -> shape (6,)

Actions:
    Discrete(6): joint ``a // 2`` moves by ``+/- action_step_size``.
    Continuous: ``Box(-1, 1, (3,))`` scaled by ``continuous_action_scale``.

Termination semantics (implementation plan C5):
    ``terminated`` is True only for task success (when ``success_terminates``);
    ``truncated`` is True when the episode hit ``max_episode_steps``. Reward
    components are returned in ``info["reward_components"]``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from armlab.config import EnvParams, RewardParams, RobotParams
from armlab.kinematics import RobotKinematics
from armlab.targets import sample_uniform_reachable

__all__ = ["KinematicReachingEnv"]

OBSERVATION_VERSION = "1.0"


class KinematicReachingEnv(gym.Env):
    """3-link arm reaching task on the kinematic model.

    Deterministic for a fixed (state, action, seed): transitions are pure
    functions of the joint angles and the action; targets are sampled with
    the env's seeded ``numpy.random.Generator``.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}
    observation_version = OBSERVATION_VERSION

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        env_params: EnvParams | None = None,
        reward_params: RewardParams | None = None,
        render_mode: str | None = None,
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
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

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

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.robot_params.joint_limits_low, dtype=float),
            np.asarray(self.robot_params.joint_limits_high, dtype=float),
        )

    def _action_delta(self, action: np.ndarray | int) -> np.ndarray:
        if isinstance(action, (int, np.integer)):
            joint = int(action) // 2
            if not 0 <= joint < 3:
                raise ValueError(f"Discrete action must be in [0, 5], got {action}.")
            sign = 1.0 if int(action) % 2 == 0 else -1.0
            dq = np.zeros(3)
            dq[joint] = sign * self.env_params.action_step_size
            return dq
        a = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
        return a * self.env_params.continuous_action_scale

    def _apply_action(self, action: np.ndarray | int) -> bool:
        """Apply one action; return True if a joint limit clipped the motion."""
        low, high = self._joint_limits()
        dq = self._action_delta(action)
        raw = self.q + dq
        self.q = np.clip(raw, low, high)
        return bool(np.any(raw > high + 1e-12) or np.any(raw < low - 1e-12))

    def _reward_components(
        self, action: np.ndarray | int, prev_dist: float, dist: float, limit_hit: bool
    ) -> dict[str, float]:
        rp = self.reward_params
        if isinstance(action, (int, np.integer)):
            magnitude = 1.0
        else:
            magnitude = float(np.linalg.norm(np.asarray(action, dtype=float)))
        success = dist < rp.success_threshold
        return {
            "progress": float(rp.progress_scale * (prev_dist - dist)),
            "distance": float(-rp.distance_penalty_scale * dist),
            "action_cost": float(-rp.action_penalty * magnitude),
            "time_cost": float(-rp.time_penalty),
            "limit_penalty": float(-rp.limit_penalty) if limit_hit else 0.0,
            "success": float(rp.success_bonus) if success else 0.0,
        }

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.q = np.zeros(3, dtype=np.float64)
        if options and "target" in options:
            self.target = np.asarray(options["target"], dtype=np.float64).reshape(3)
        else:
            self.target = sample_uniform_reachable(
                self.robot_params, self.env_params, self.np_random, kinematics=self.kin
            )
        self.elapsed_steps = 0
        return self._get_obs(), self._info(dist=self._distance(), termination_reason="reset")

    def step(self, action):
        prev_dist = self._distance()
        limit_hit = False
        for _ in range(self.env_params.action_repeat):
            limit_hit = self._apply_action(action) or limit_hit
            self.elapsed_steps += 1
        dist = self._distance()

        components = self._reward_components(action, prev_dist, dist, limit_hit)
        reward = float(sum(components.values()))
        success = dist < self.reward_params.success_threshold
        terminated = bool(success and self.reward_params.success_terminates)
        truncated = bool(self.elapsed_steps >= self.env_params.max_episode_steps)

        if terminated:
            reason = "success"
        elif truncated:
            reason = "time_limit"
        else:
            reason = "running"

        info = self._info(dist=dist, termination_reason=reason)
        info.update(
            {
                "success": bool(success),
                "joint_limit_hit": bool(limit_hit),
                "collision": False,
                "reward_components": components,
            }
        )
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def _distance(self) -> float:
        return float(np.linalg.norm(self.kin.end_effector(self.q) - self.target))

    def _info(self, dist: float, termination_reason: str = "running") -> dict[str, Any]:
        return {
            "distance": dist,
            "distance_to_target": dist,
            "ee_position": self.kin.end_effector(self.q).copy(),
            "target_position": self.target.copy(),
            "q": self.q.copy(),
            "episode_steps": int(self.elapsed_steps),
            "termination_reason": termination_reason,
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
