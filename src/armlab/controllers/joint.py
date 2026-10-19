"""Joint-space controllers: proportional and PD.

These controllers operate on analytical IK targets: they compute the desired
configuration from the target position (via ``RobotKinematics``) and command
joint displacements proportional to the configuration error.
"""

from __future__ import annotations

import numpy as np

from armlab.controllers.base import Controller
from armlab.kinematics import RobotKinematics, wrap_angles

__all__ = ["JointPositionController", "JointPDController"]


class JointPositionController(Controller):
    """P controller in joint space around an analytical IK solution.

    dq = kp * wrap(IK(target) - q), clipped to ``max_dq``.

    Falls back to zero displacement when the target is unreachable.
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        kp: float = 0.4,
        max_dq: float = 0.3,
    ) -> None:
        if kp <= 0:
            raise ValueError(f"kp must be positive, got {kp}.")
        if max_dq <= 0:
            raise ValueError(f"max_dq must be positive, got {max_dq}.")
        self.kin = kinematics
        self.kp = float(kp)
        self.max_dq = float(max_dq)

    def act(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(3)
        result = self.kin.inverse_kinematics(target, q_current=q)
        if not result.reachable:
            return np.zeros(3)
        dq = self.kp * wrap_angles(result.solution - q)
        return np.clip(dq, -self.max_dq, self.max_dq)


class JointPDController(JointPositionController):
    """PD controller in joint space.

    Extends the P controller with a derivative term on the measured joint
    velocity: dq = kp * e + kd * (dq_desired_extra - qdot), clipped.

    For the kinematic environment, qdot is estimated from successive calls as
    (q - q_prev) / dt, where dt is provided at construction.
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        kp: float = 0.4,
        kd: float = 0.1,
        dt: float = 0.02,
        max_dq: float = 0.3,
    ) -> None:
        super().__init__(kinematics, kp=kp, max_dq=max_dq)
        if kd < 0:
            raise ValueError(f"kd must be >= 0, got {kd}.")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")
        self.kd = float(kd)
        self.dt = float(dt)
        self._q_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._q_prev = None

    def act(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(3)
        dq_pos = super().act(q, target)
        if self._q_prev is None:
            qdot = np.zeros(3)
        else:
            qdot = (q - self._q_prev) / self.dt
        self._q_prev = q.copy()
        dq = dq_pos - self.kd * qdot * self.dt
        return np.clip(dq, -self.max_dq, self.max_dq)
