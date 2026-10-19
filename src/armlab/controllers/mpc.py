"""Model-predictive controller via random shooting on the kinematic model.

MPC samples candidate action sequences over a finite horizon, rolls them out
with the deterministic kinematic model, and executes the first action of the
best sequence according to a distance-based cost. Candidate sampling is
seeded, making the controller deterministic for a given seed and state.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from armlab.controllers.base import Controller
from armlab.kinematics import RobotKinematics, wrap_angles

__all__ = ["ShootingMPCController"]


class ShootingMPCController(Controller):
    """Random-shooting MPC on the kinematic model with joint clipping.

    Args:
        kinematics: robot kinematics used as the internal model.
        horizon: rollout length in steps (>= 1).
        n_samples: candidate sequences per call (>= 1).
        action_magnitude: max |dq| per joint per step (>= 0).
        seed: base RNG seed; each ``act`` call re-seeds a child RNG with
            ``seed + call_count`` so behavior is reproducible.
        joint_limits: optional (low, high) arrays for clipping; defaults to
            no limits.
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        horizon: int = 6,
        n_samples: int = 40,
        action_magnitude: float = 0.2,
        seed: int = 0,
        joint_limits: Optional[tuple] = None,
    ) -> None:
        if horizon < 1 or n_samples < 1:
            raise ValueError("horizon and n_samples must be >= 1.")
        if action_magnitude < 0:
            raise ValueError("action_magnitude must be non-negative.")
        self.kin = kinematics
        self.horizon = int(horizon)
        self.n_samples = int(n_samples)
        self.action_magnitude = float(action_magnitude)
        self.seed = int(seed)
        self._calls = 0
        if joint_limits is None:
            self.low = None
            self.high = None
        else:
            self.low = np.asarray(joint_limits[0], dtype=float)
            self.high = np.asarray(joint_limits[1], dtype=float)

    def reset(self) -> None:
        self._calls = 0

    def act(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)
        rng = np.random.default_rng(self.seed + self._calls)
        self._calls += 1

        # Candidate sequences: uniform in [-action_magnitude, action_magnitude].
        sequences = rng.uniform(
            -self.action_magnitude,
            self.action_magnitude,
            size=(self.n_samples, self.horizon, 3),
        )
        # Always include a greedy descent candidate (toward IK solution).
        result = self.kin.inverse_kinematics(target, q_current=q)
        if result.reachable:
            greedy_dq = np.clip(wrap_angles(result.solution - q), -self.action_magnitude,
                                self.action_magnitude)
            sequences[0, :, :] = greedy_dq  # repeated greedy step

        best_cost = np.inf
        best_first = np.zeros(3)
        for seq in sequences:
            q_sim = q.copy()
            cost = 0.0
            for t in range(self.horizon):
                q_sim = q_sim + seq[t]
                if self.low is not None:
                    q_sim = np.clip(q_sim, self.low, self.high)
                err = np.linalg.norm(self.kin.end_effector(q_sim) - target)
                cost += err
                if err < 1e-6:
                    break
            if cost < best_cost:
                best_cost = cost
                best_first = seq[0]
        return best_first
