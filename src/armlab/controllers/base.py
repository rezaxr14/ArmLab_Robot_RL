"""Base controller interface shared by all classical controllers.

Controllers map an observation to an action. For the kinematic environment
the observation contains joint angles; the exact contract is:

    act(q, target) -> command

where ``q`` is the current joint configuration (3,) and ``target`` is the
desired end-effector position (3,). Commands are joint displacements
(``dq``) that the environment converts to its action space via
``armlab.envs.wrappers.discrete_argmax`` or applies directly in continuous
mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Controller", "DiscreteActionAdapter"]


class Controller(ABC):
    """Abstract base for classical controllers producing joint displacements."""

    @abstractmethod
    def act(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return a joint displacement command (3,) for the current state."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Reset any internal state (e.g. integrators, MPC warm starts)."""


def discrete_argmax(dq: np.ndarray, step_size: float) -> int:
    """Map a continuous joint displacement to the best discrete action.

    Discrete action ``a`` moves joint ``a // 2`` by ``step_size`` radians in
    direction ``+1`` (a even) or ``-1`` (a odd). The mapping picks the action
    whose displacement has the largest absolute projection onto ``dq``.
    """
    dq = np.asarray(dq, dtype=float).reshape(3)
    best_action, best_score = 0, -np.inf
    for joint in range(3):
        for sign_idx, sign in ((0, 1.0), (1, -1.0)):
            action = joint * 2 + sign_idx
            score = sign * dq[joint]
            if score > best_score:
                best_action, best_score = action, score
    return best_action


class DiscreteActionAdapter(Controller):
    """Wrap a continuous controller to emit discrete environment actions."""

    def __init__(self, inner: Controller, step_size: float) -> None:
        self.inner = inner
        self.step_size = float(step_size)

    def act(self, q: np.ndarray, target: np.ndarray) -> int:
        dq = self.inner.act(q, target)
        return discrete_argmax(dq, self.step_size)

    def reset(self) -> None:
        self.inner.reset()
