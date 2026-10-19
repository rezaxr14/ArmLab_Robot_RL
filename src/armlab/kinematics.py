"""Analytical kinematics for the 3-link spatial arm.

Conventions (radians, meters):
- Joint order: q = [theta1, theta2, theta3].
- theta1: waist rotation about world +z.
- theta2: shoulder angle from vertical (+z); positive tilts toward +x at theta1=0.
- theta3: elbow angle relative to link 2 (0 = fully extended).

The shoulder joint sits at [0, 0, l1]; links 2 and 3 have lengths l2, l3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armlab.config import RobotParams

__all__ = [
    "RobotKinematics",
    "IKResult",
    "wrap_angle",
    "wrap_angles",
]


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Wrap an array of angles elementwise to [-pi, pi)."""
    arr = np.asarray(angles, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class IKResult:
    """Result of an analytical inverse kinematics query.

    Attributes:
        solution: joint angles [theta1, theta2, theta3] in radians, or None
            when the target is unreachable.
        reachable: True iff ``solution`` is not None.
        branch: "elbow_up", "elbow_down", or None.
        residual: distance between FK(solution) and the requested target
            (0.0 for exact solutions, NaN when unreachable).
    """

    solution: np.ndarray | None
    reachable: bool
    branch: str | None
    residual: float


class RobotKinematics:
    """Forward/inverse kinematics, geometric Jacobian, and damped least squares
    for the 3-link arm defined by ``RobotParams``.

    Stateless with respect to joint configuration: pass ``q`` explicitly.
    """

    def __init__(self, params: RobotParams) -> None:
        self.params = params
        self.l1, self.l2, self.l3 = params.link_lengths

    # ------------------------------------------------------------------ FK
    def link_positions(self, q: np.ndarray) -> np.ndarray:
        """Positions of [base, shoulder, elbow, ee], shape (4, 3)."""
        q = self._validate_q(q)
        t1, t2, t3 = q
        s2, c2 = np.sin(t2), np.cos(t2)
        s23, c23 = np.sin(t2 + t3), np.cos(t2 + t3)
        c1, s1 = np.cos(t1), np.sin(t1)

        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.0, self.l1])
        p2 = np.array([self.l2 * c1 * s2, self.l2 * s1 * s2, self.l1 + self.l2 * c2])
        p3 = np.array(
            [
                c1 * (self.l2 * s2 + self.l3 * s23),
                s1 * (self.l2 * s2 + self.l3 * s23),
                self.l1 + self.l2 * c2 + self.l3 * c23,
            ]
        )
        return np.stack([p0, p1, p2, p3])

    def end_effector(self, q: np.ndarray) -> np.ndarray:
        """End-effector position, shape (3,)."""
        return self.link_positions(q)[3]

    ee_position = end_effector  # alias

    # ------------------------------------------------------------------ IK
    def inverse_kinematics(
        self,
        target: np.ndarray,
        q_current: np.ndarray | None = None,
        elbow: str | None = None,
    ) -> IKResult:
        """Analytical IK with solution selection.

        Args:
            target: [x, y, z] desired end-effector position.
            q_current: current joint angles used to pick the closest branch
                when ``elbow`` is None. Defaults to zeros.
            elbow: force "elbow_up", "elbow_down", or None for closest.

        Returns:
            IKResult; ``reachable`` False when the target lies outside the
            reachable annulus of the shoulder joint.
        """
        target = self._as_triple(target, "target")
        if q_current is None:
            q_current = np.zeros(3)
        q_current = self._validate_q(q_current)
        if elbow not in (None, "elbow_up", "elbow_down"):
            raise ValueError(f"elbow must be None, 'elbow_up', or 'elbow_down'; got {elbow!r}.")

        x, y, z = target
        r_xy = float(np.hypot(x, y))

        # Waist singularity: target on the z-axis -> keep current theta1.
        if r_xy < 1e-9:
            theta1 = float(q_current[0])
        else:
            theta1 = float(np.arctan2(y, x))

        r = r_xy
        z_prime = z - self.l1
        D = float(np.hypot(r, z_prime))

        l2, l3 = self.l2, self.l3
        if D > l2 + l3 or D < abs(l2 - l3):
            return IKResult(solution=None, reachable=False, branch=None, residual=np.nan)

        # Deliberate singular case: the target coincides with the shoulder.
        # This is only reachable when l2 == l3; the arm folds back on itself,
        # so theta2 is free and the elbow is fully bent. Preserve the current
        # shoulder angle instead of dividing by D = 0.
        if D < 1e-12:
            chosen = wrap_angles(np.array([theta1, float(q_current[1]), np.pi]))
            residual = float(np.linalg.norm(self.end_effector(chosen) - target))
            return IKResult(
                solution=chosen,
                reachable=True,
                branch="elbow_up" if elbow != "elbow_down" else "elbow_down",
                residual=residual,
            )

        # Guard tiny negative values from floating point before arccos.
        cos_t3 = np.clip((D**2 - l2**2 - l3**2) / (2.0 * l2 * l3), -1.0, 1.0)
        cos_alpha = np.clip((D**2 + l2**2 - l3**2) / (2.0 * D * l2), -1.0, 1.0)
        # theta2 is measured from the VERTICAL (+z) axis in FK, so the target
        # angle in the vertical plane must also be measured from vertical:
        # gamma = atan2(horizontal_distance, vertical_distance).
        # (The legacy implementation used atan2(z_prime, r), an axis swap that
        # breaks the FK/IK round trip; see execution log.)
        gamma = np.arctan2(r, z_prime)
        alpha = np.arccos(cos_alpha)
        t3_mag = np.arccos(cos_t3)

        sol_up = np.array([theta1, gamma - alpha, t3_mag])
        sol_down = np.array([theta1, gamma + alpha, -t3_mag])

        if elbow == "elbow_up":
            chosen, branch = sol_up, "elbow_up"
        elif elbow == "elbow_down":
            chosen, branch = sol_down, "elbow_down"
        else:
            cost_up = self._config_distance(sol_up, q_current)
            cost_down = self._config_distance(sol_down, q_current)
            if cost_up <= cost_down:
                chosen, branch = sol_up, "elbow_up"
            else:
                chosen, branch = sol_down, "elbow_down"

        chosen = wrap_angles(chosen)
        residual = float(np.linalg.norm(self.end_effector(chosen) - target))
        return IKResult(solution=chosen, reachable=True, branch=branch, residual=residual)

    # ------------------------------------------------------------- Jacobian
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Position Jacobian (3x3): d(ee)/d(q), analytic form."""
        q = self._validate_q(q)
        t1, t2, t3 = q
        s1, c1 = np.sin(t1), np.cos(t1)
        s2, c2 = np.sin(t2), np.cos(t2)
        s23, c23 = np.sin(t2 + t3), np.cos(t2 + t3)

        r = self.l2 * s2 + self.l3 * s23  # horizontal reach
        dr_dtheta2 = self.l2 * c2 + self.l3 * c23
        dr_dtheta3 = self.l3 * c23
        dz_dtheta2 = -(self.l2 * s2 + self.l3 * s23)
        dz_dtheta3 = -self.l3 * s23

        J = np.array(
            [
                [-s1 * r, c1 * dr_dtheta2, c1 * dr_dtheta3],
                [c1 * r, s1 * dr_dtheta2, s1 * dr_dtheta3],
                [0.0, dz_dtheta2, dz_dtheta3],
            ]
        )
        return J

    def damped_least_squares(
        self,
        q: np.ndarray,
        dx: np.ndarray,
        damping: float = 0.05,
    ) -> np.ndarray:
        """One damped least-squares step: dq = J^T (J J^T + lambda^2 I)^-1 dx.

        Args:
            q: current joint angles (3,).
            dx: desired end-effector displacement (3,) in world frame.
            damping: lambda >= 0; larger values trade accuracy for
                smoothness near singularities.

        Returns:
            Joint displacement dq (3,).
        """
        if damping < 0:
            raise ValueError(f"damping must be >= 0, got {damping}.")
        q = self._validate_q(q)
        dx = self._as_triple(dx, "dx")
        J = self.jacobian(q)
        A = J @ J.T + (damping**2) * np.eye(3)
        return J.T @ np.linalg.solve(A, dx)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _config_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Weighted sum of squared wrapped angle differences."""
        diff = wrap_angles(np.asarray(a) - np.asarray(b))
        return float(np.sum(diff**2))

    @staticmethod
    def _as_triple(value: np.ndarray, name: str) -> np.ndarray:
        """Validate that ``value`` is a finite vector with exact shape ``(3,)``."""
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        return arr

    def _validate_q(self, q: np.ndarray) -> np.ndarray:
        return self._as_triple(q, "Joint angles")
