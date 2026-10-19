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

from collections.abc import Sequence
from dataclasses import dataclass

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
        obstacles: Sequence[SphereObstacle] | None = None,
        repulse_gain: float = 0.8,
        influence_radius: float = 3.0,
        joint_limits: Sequence[Sequence[float]] | None = None,
        limit_epsilon: float = 1e-6,
        limit_avoidance_gain: float = 0.0,
        limit_avoidance_band: float = 0.2,
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
        if joint_limits is None:
            self.joint_limits = None
        else:
            low = np.asarray(joint_limits[0], dtype=float).reshape(3)
            high = np.asarray(joint_limits[1], dtype=float).reshape(3)
            if np.any(low >= high):
                raise ValueError("joint_limits low must be strictly less than high.")
            self.joint_limits = (low, high)
        self.limit_epsilon = float(limit_epsilon)
        if limit_avoidance_gain < 0 or limit_avoidance_band <= 0:
            raise ValueError("limit_avoidance_gain must be >= 0 and band > 0.")
        self.limit_avoidance_gain = float(limit_avoidance_gain)
        self.limit_avoidance_band = float(limit_avoidance_band)

    def _dls(self, q: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Damped least squares, projecting out joints locked at their limits.

        Clamping the commanded joint velocity at a limit is not enough: the
        local solver keeps pushing a saturated joint and stalls (a well-known
        resolved-rate failure). Locked-at-limit joints are removed from the
        Jacobian and the remaining joints are re-solved, which lets the
        controller route motion through the free joints instead.
        """
        if self.joint_limits is None:
            return self.kin.damped_least_squares(q, dx, damping=self.damping)

        low, high = self.joint_limits
        free = np.ones(3, dtype=bool)
        dq = np.zeros(3)
        for _ in range(3):
            J = self.kin.jacobian(q)[:, free]
            A = J @ J.T + (self.damping**2) * np.eye(3)
            dq_free = J.T @ np.linalg.solve(A, dx)
            dq = np.zeros(3)
            dq[free] = dq_free
            locked = self._locked_joints(q, dq, low, high)
            locked = locked & free  # only newly lock a currently free joint
            if not np.any(locked):
                break
            free = free & ~locked
            if not np.any(free):
                return np.zeros(3)
        return dq

    def _locked_joints(
        self, q: np.ndarray, dq: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        eps = self.limit_epsilon
        return ((q <= low + eps) & (dq < 0.0)) | ((q >= high - eps) & (dq > 0.0))

    def act(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)
        ee = self.kin.end_effector(q)
        v_track = self.kp * (target - ee)
        v_repulse = self._repulsive_velocity(ee)
        dx = v_track + v_repulse
        dq = self._dls(q, dx)
        if self.joint_limits is not None and self.limit_avoidance_gain > 0:
            dq = dq + self._limit_avoidance(q)
        return np.clip(dq, -self.max_dq, self.max_dq)

    def _limit_avoidance(self, q: np.ndarray) -> np.ndarray:
        """Small joint-space bias away from joint limits.

        When the local task needs a joint that is already saturated, this bias
        nudges the arm off the limit so the solver can re-plan through the free
        joints on the next step instead of stalling at the boundary.
        """
        low, high = self.joint_limits  # type: ignore[misc]
        band = self.limit_avoidance_band
        bias = np.zeros(3)
        for j in range(3):
            d_low = q[j] - low[j]
            d_high = high[j] - q[j]
            if d_low < band:
                bias[j] += self.limit_avoidance_gain * (band - d_low)
            if d_high < band:
                bias[j] -= self.limit_avoidance_gain * (band - d_high)
        return bias

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
