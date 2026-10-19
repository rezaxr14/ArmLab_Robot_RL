"""Compatibility wrapper for ``projects/analytical_inverse_kinematics/robot.py``.

Preserves the legacy class/method names and calling conventions:

- ``Robot(link_lengths)`` with ``forward_kinematics(thetas)`` returning a
  list of 4 position arrays and ``inverse_kinematics(target, current_thetas)``
  returning a list of 3 angles or None.
- ``total_reach`` property.

Internally delegates to ``armlab.kinematics.RobotKinematics`` and preserves
the original demo's vertical-plane IK convention.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from armlab.config import RobotParams
from armlab.kinematics import RobotKinematics

__all__ = ["Robot"]


class Robot:
    """Legacy-compatible 3-link robot facade over RobotKinematics."""

    def __init__(self, link_lengths: Sequence[float]):
        if len(link_lengths) != 3:
            raise ValueError("Please provide a list of 3 link lengths.")
        self._kin = RobotKinematics(RobotParams(link_lengths=tuple(link_lengths)))
        self.l1, self.l2, self.l3 = self._kin.l1, self._kin.l2, self._kin.l3
        self.thetas = [0.0, 0.0, 0.0]

    def forward_kinematics(self, thetas) -> list[np.ndarray]:
        """Return [base, shoulder, elbow, ee] positions as a list."""
        self.thetas = list(thetas)
        return list(self._kin.link_positions(np.asarray(thetas, dtype=float)))

    def inverse_kinematics(self, target_pos, current_thetas) -> list[float] | None:
        """Return closest-branch IK solution as a list, or None."""
        result = self._kin.inverse_kinematics(
            np.asarray(target_pos, dtype=float),
            q_current=np.asarray(current_thetas, dtype=float),
        )
        if not result.reachable:
            return None
        return list(result.solution)

    @property
    def total_reach(self) -> float:
        return self._kin.l2 + self._kin.l3
