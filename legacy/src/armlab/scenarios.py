"""Named scenarios with declarative configs (Phase F).

Scenario registry:
- ``static_reaching``: random reachable static target, no obstacles.
- ``boundary_reaching``: targets near the workspace boundary.
- ``moving_target``: seeded smooth 3D target trajectory.
- ``collision_avoidance``: static target with spherical obstacles placed
  between a fixed start posture and the target.

Each scenario returns a ``ScenarioBundle`` containing a ready-to-use env, an
obstacle list for metrics, and metadata needed for reproducible evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from armlab.config import EnvParams, RewardParams, RobotParams
from armlab.envs.kinematic import KinematicReachingEnv
from armlab.envs.moving import MovingTargetEnv
from armlab.planning.collision import Obstacle, SphereObstacle
from armlab.targets import sample_boundary_reachable

__all__ = ["ScenarioBundle", "SCENARIOS", "make_scenario", "list_scenarios"]


@dataclass
class ScenarioBundle:
    """Everything needed to run and evaluate one scenario."""

    name: str
    env: KinematicReachingEnv
    obstacles: list[Obstacle] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _make_static(robot: RobotParams, seed: int, **overrides) -> ScenarioBundle:
    env = KinematicReachingEnv(
        robot_params=robot,
        env_params=overrides.get("env_params") or EnvParams(),
        reward_params=overrides.get("reward_params") or RewardParams(),
    )
    return ScenarioBundle("static_reaching", env, [], {"seed": seed})


def _make_boundary(robot: RobotParams, seed: int, **overrides) -> ScenarioBundle:
    env_params = overrides.get("env_params") or EnvParams()
    reward = overrides.get("reward_params") or RewardParams()

    class _BoundaryEnv(KinematicReachingEnv):
        def reset(self, *, seed=None, options=None):
            obs, info = super().reset(seed=seed, options={"target": np.zeros(3)})
            self.target = sample_boundary_reachable(
                self.robot_params, self.env_params, self.np_random, kinematics=self.kin
            )
            return self._get_obs(), self._info(0.0)

    env = _BoundaryEnv(robot_params=robot, env_params=env_params, reward_params=reward)
    return ScenarioBundle("boundary_reaching", env, [], {"seed": seed, "margin": 0.1})


def _make_moving(robot: RobotParams, seed: int, **overrides) -> ScenarioBundle:
    env = MovingTargetEnv(
        robot_params=robot,
        env_params=overrides.get("env_params") or EnvParams(max_episode_steps=300),
        reward_params=overrides.get("reward_params") or RewardParams(success_terminates=False),
        amp_range=overrides.get("amp_range", 2.0),
        freq_range=overrides.get("freq_range", 0.4),
    )
    return ScenarioBundle(
        "moving_target", env, [], {"seed": seed, "amp_range": 2.0, "freq_range": 0.4}
    )


def _make_collision(robot: RobotParams, seed: int, **overrides) -> ScenarioBundle:
    env_params = overrides.get("env_params") or EnvParams()
    reward = overrides.get("reward_params") or RewardParams()
    env = KinematicReachingEnv(robot_params=robot, env_params=env_params, reward_params=reward)
    # Deterministic obstacle layout: two spheres between the default home
    # posture's workspace and a seeded target region in the +x half-space.
    obstacles: list[Obstacle] = [
        SphereObstacle(center=np.array([9.0, 1.0, 12.0]), radius=1.5),
        SphereObstacle(center=np.array([11.0, -2.0, 10.0]), radius=2.0),
    ]
    return ScenarioBundle("collision_avoidance", env, obstacles, {"seed": seed, "link_radius": 0.3})


SCENARIOS: dict[str, Callable[..., ScenarioBundle]] = {
    "static_reaching": _make_static,
    "boundary_reaching": _make_boundary,
    "moving_target": _make_moving,
    "collision_avoidance": _make_collision,
}


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS)


def make_scenario(
    name: str,
    seed: int = 0,
    robot: RobotParams | None = None,
    **overrides,
) -> ScenarioBundle:
    """Build a scenario bundle by name.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}. Available: {list_scenarios()}.")
    return SCENARIOS[name](robot or RobotParams(), seed, **overrides)
