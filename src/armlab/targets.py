"""Reachable target generators for the reaching environments.

Uniform reachable target generation must not depend on any visualizer
(implementation plan Phase B acceptance criteria). All generators accept an
``numpy.random.Generator`` so that seeded environment resets are reproducible.
"""

from __future__ import annotations

import numpy as np

from armlab.config import EnvParams, RobotParams
from armlab.kinematics import RobotKinematics

__all__ = [
    "sample_uniform_reachable",
    "sample_boundary_reachable",
    "MovingTargetTrajectory",
]


def sample_uniform_reachable(
    robot: RobotParams,
    env: EnvParams,
    rng: np.random.Generator,
    kinematics: RobotKinematics | None = None,
    max_attempts: int = 100,
) -> np.ndarray:
    """Sample a target inside the reachable workspace by rejection sampling.

    A target is reachable iff the distance from the shoulder joint
    ``[0, 0, l1]`` lies within ``[|l2 - l3|, l2 + l3]``. Sampling uses the
    configured axis-aligned bounding region (radius/height ranges from
    ``EnvParams``) and rejects points outside the reachable annulus.

    Returns:
        Target position (3,).

    Raises:
        RuntimeError: if no reachable point is found within ``max_attempts``
            (indicates contradictory configuration ranges).
    """
    l2, l3 = robot.link_lengths[1], robot.link_lengths[2]
    margin = float(getattr(env, "target_annulus_margin", 0.0))
    span = (l2 + l3) - abs(l2 - l3)
    d_min = abs(l2 - l3) + margin * span
    d_max = (l2 + l3) - margin * span
    shoulder = np.array([0.0, 0.0, robot.base_height])

    for _ in range(max_attempts):
        radius = rng.uniform(env.target_min_radius, env.target_max_radius)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        height = rng.uniform(env.target_min_height, env.target_max_height)
        target = np.array([radius * np.cos(angle), radius * np.sin(angle), height])
        dist = np.linalg.norm(target - shoulder)
        if d_min <= dist <= d_max:
            return target
    raise RuntimeError(
        "Could not sample a reachable target; check EnvParams radius/height "
        "ranges against the arm geometry (reachable distance from shoulder: "
        f"[{d_min:.3f}, {d_max:.3f}])."
    )


def sample_boundary_reachable(
    robot: RobotParams,
    env: EnvParams,
    rng: np.random.Generator,
    kinematics: RobotKinematics | None = None,
    margin: float = 0.1,
    max_attempts: int = 200,
) -> np.ndarray:
    """Sample a target near the workspace boundary (Phase F policy).

    Targets are drawn at distance ``(1 - margin) * max_reach`` from the
    shoulder along random directions with elevation in [-0.9, 0.9] rad, so
    they sit close to the workspace boundary where singularities live.
    """
    l2, l3 = robot.link_lengths[1], robot.link_lengths[2]
    d_max = l2 + l3
    shoulder = np.array([0.0, 0.0, robot.base_height])
    desired = (1.0 - margin) * d_max
    if desired <= abs(l2 - l3):
        raise ValueError("margin too large: desired boundary distance below min reach.")

    for _ in range(max_attempts):
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        elevation = rng.uniform(-0.9, 0.9)
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        target = shoulder + desired * direction
        if np.linalg.norm(target - shoulder) <= d_max:
            return target
    raise RuntimeError("Could not sample a boundary target; this should not happen.")


class MovingTargetTrajectory:
    """Deterministic smooth 3D target trajectory (Phase F: moving_target).

    p(t) = center + amps * sin(freqs * t + phases).

    All sinusoid parameters are drawn from a seeded ``Generator`` so the
    trajectory is exactly reproducible for a given seed. Construction
    validates that the trajectory's bounding box intersects the reachable
    annulus; individual instants may still fall outside it.
    """

    def __init__(
        self,
        robot: RobotParams,
        rng: np.random.Generator,
        center: np.ndarray | None = None,
        freq_range: float = 0.4,
        amp_range: float = 2.0,
    ) -> None:
        if amp_range < 0 or freq_range < 0:
            raise ValueError("freq_range and amp_range must be non-negative.")
        self.robot = robot
        if center is None:
            center = np.array([10.0, 0.0, 12.0])
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.amps = rng.uniform(0.0, amp_range, size=3)
        self.freqs = rng.uniform(0.05, max(freq_range, 0.05), size=3)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=3)

        l2, l3 = robot.link_lengths[1], robot.link_lengths[2]
        shoulder = np.array([0.0, 0.0, robot.base_height])
        corners = [
            self.center + np.array([sx * self.amps[0], sy * self.amps[1], sz * self.amps[2]])
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        dists = [float(np.linalg.norm(c - shoulder)) for c in corners]
        if min(dists) > l2 + l3 or max(dists) < abs(l2 - l3):
            raise ValueError(
                "MovingTargetTrajectory bounding box is entirely unreachable: "
                f"distances from shoulder range [{min(dists):.2f}, {max(dists):.2f}], "
                f"reachable annulus [{abs(l2 - l3):.2f}, {l2 + l3:.2f}]."
            )

    def position(self, t: float) -> np.ndarray:
        """Target position at time ``t`` seconds."""
        phase = self.freqs * t + self.phases
        return self.center + self.amps * np.sin(phase)
