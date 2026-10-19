"""Cartesian-space controllers: damped least squares resolved-rate tracking.

The Cartesian DLS controller computes joint displacements that move the end
effector toward the target along a straight line in task space:

    dq = J^T (J J^T + lambda^2 I)^-1 * kp * (target - ee(q))

An optional obstacle-repulsion term adds a Cartesian velocity that pushes the
end effector away from spherical obstacles (Phase F: collision avoidance via
navigation-function-free artificial potential field). This is a reactive
heuristic, not a completeness-guaranteed planner; see armlab.planning for the
planner with completeness (RRT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from armlab.controllers.base import Controller
from armlab.kinematics import RobotKinematics

__all__ = ["SphereObstacle", "CartesianDLSController"]


@dataclass(frozen=True)
class SphereObstacle:
    """Spherical obstacle for reactive avoidance."""

    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float).reshape(3))
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}.")


class CartesianDLSController(Controller):
    """Damped least-squares resolved-rate controller with optional obstacles.

    Args:
        kinematics: robot kinematics.
        kp: Cartesian tracking gain (>= 0). At 0 the controller is idle.
        damping: DLS lambda (>= 0).
        max_dq: per-step joint displacement clip (>= 0).
        obstacles: optional spherical obstacles for repulsive velocities.
        repulse_gain: gain for the repulsive field (>= 0).
        influence_radius: obstacles only repel within this distance of the
            end effector (>= obstacle radius).
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        kp: float = 0.6,
        damping: float = 0.08,
        max_dq: float = 0.25,
        obstacles: Optional[Sequence[SphereObstacle]] = None,
        repulse_gain: float = 0.8,
        influence_radius: float = 3.0,
    ) -> None:
        if kp < 0 or damping < 0 or max_dq < 0:
            raise ValueError("kp, damping, and max_dq must be non-negative.")
        if repulse_gain < 0 or influence_radius < 0:
            raise ValueError("repulse_gain and influence_radius must be non-negative.")
        self.kin = kinematics
        self.kp = float(kp)
        self.damping = float(damping)
        self.max_dq = float(max_dq)
        self.obstacles = list(obstacles) if obstacles else []
        self.repulse_gain = float(repulse_gain)
        self.influence_radius = float(influence_radius)

    def act(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)
        ee = self.kin.end_effector(q)
        v_track = self.kp * (target - ee)
        v_repulse = self._repulsive_velocity(ee)
        dx = v_track + v_repulse
        dq = self.kin.damped_least_squares(q, dx, damping=self.damping)
        return np.clip(dq, -self.max_dq, self.max_dq)

    def _repulsive_velocity(self, ee: np.ndarray) -> np.ndarray:
        """Sum of repulsive Cartesian velocities from nearby obstacles."""
        v = np.zeros(3)
        for obs in self.obstacles:
            delta = ee - obs.center
            dist = float(np.linalg.norm(delta))
            # Distance from the obstacle surface.
            surface_dist = dist - obs.radius
            if dist < 1e-9 or surface_dist >= self.influence_radius:
                continue
            # Clamp to avoid division by zero at/inside the surface.
            effective = max(surface_dist, 1e-3)
            magnitude = self.repulse_gain * (1.0 / effective - 1.0 / self.influence_radius)
            v += magnitude * (delta / dist)
        return v
