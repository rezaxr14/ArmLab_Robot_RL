"""Moving-target reaching environment (Phase F: moving_target).

Extends the static kinematic env with a seeded smooth 3D target trajectory.
Success requires the end effector to come within ``success_threshold`` of the
moving target at any point during the episode.
"""

from __future__ import annotations

import numpy as np

from armlab.config import EnvParams, RewardParams, RobotParams
from armlab.envs.kinematic import KinematicReachingEnv
from armlab.targets import MovingTargetTrajectory

__all__ = ["MovingTargetEnv"]


class MovingTargetEnv(KinematicReachingEnv):
    """Kinematic reaching with a sinusoidal moving target.

    The trajectory is rebuilt on every ``reset`` from the env seed, so fixed
    seeds reproduce identical episodes.
    """

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        env_params: EnvParams | None = None,
        reward_params: RewardParams | None = None,
        render_mode: str | None = None,
        trajectory_center: np.ndarray | None = None,
        amp_range: float = 2.0,
        freq_range: float = 0.4,
        dt: float = 0.05,
    ) -> None:
        super().__init__(robot_params, env_params, reward_params, render_mode)
        self.trajectory_center = trajectory_center
        self.amp_range = float(amp_range)
        self.freq_range = float(freq_range)
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        self._traj: MovingTargetTrajectory | None = None

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = super().reset(seed=seed, options={"target": np.zeros(3)})
        self._traj = MovingTargetTrajectory(
            self.robot_params,
            self.np_random,
            center=self.trajectory_center,
            freq_range=self.freq_range,
            amp_range=self.amp_range,
        )
        self.target = self._traj.position(0.0)
        return self._get_obs(), self._info(0.0)

    def step(self, action):
        t = self.elapsed_steps * self.dt
        self.target = self._traj.position(t + self.dt)
        return super().step(action)
