"""Joint-space RRT planner with obstacle checking and shortcut smoothing.

The planner operates in the 3D joint-angle space with wrapped angular
distances, validates edges by discretizing them and running capsule collision
checks, and finally shortcut-smooths the path. It is a sampling planner:
with enough iterations it finds a path when one exists in the sampled
resolution, but it does not guarantee completeness; the API reports
success/failure explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from armlab.kinematics import RobotKinematics, wrap_angles
from armlab.planning.collision import Obstacle, collides

__all__ = ["PlannerResult", "JointSpaceRRT"]


@dataclass(frozen=True)
class PlannerResult:
    """Outcome of a planning query.

    path: (N, 3) joint configurations from start to goal, or None on failure.
    success: True iff a collision-free path was found.
    iterations: RRT iterations executed.
    reason: "reached_goal", "start_in_collision", "goal_in_collision",
        or "max_iterations".
    """

    success: bool
    path: Optional[np.ndarray]
    iterations: int
    reason: str


class JointSpaceRRT:
    """RRT in joint space with capsule collision checking.

    Args:
        kinematics: used for FK-based collision checks.
        obstacles: static obstacles.
        link_radius: capsule radius for links (>= 0).
        step_size: max extension length per iteration (radians, > 0).
        goal_bias: probability of sampling the goal in [0, 1].
        max_iterations: RRT iteration budget (>= 1).
        edge_check_steps: discretization of edges for collision checks.
        seed: RNG seed for reproducible trees.
        joint_limits: (low, high) arrays; defaults to (-pi, pi)^3.
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        obstacles: Optional[Sequence[Obstacle]] = None,
        link_radius: float = 0.3,
        step_size: float = 0.25,
        goal_bias: float = 0.1,
        max_iterations: int = 800,
        edge_check_steps: int = 6,
        seed: int = 0,
        joint_limits: Optional[tuple] = None,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if not (0.0 <= goal_bias <= 1.0):
            raise ValueError("goal_bias must be in [0, 1].")
        if max_iterations < 1 or edge_check_steps < 1:
            raise ValueError("max_iterations and edge_check_steps must be >= 1.")
        self.kin = kinematics
        self.obstacles = list(obstacles) if obstacles else []
        self.link_radius = float(link_radius)
        self.step_size = float(step_size)
        self.goal_bias = float(goal_bias)
        self.max_iterations = int(max_iterations)
        self.edge_check_steps = int(edge_check_steps)
        self.rng = np.random.default_rng(seed)
        if joint_limits is None:
            self.low = np.full(3, -np.pi)
            self.high = np.full(3, np.pi)
        else:
            self.low = np.asarray(joint_limits[0], float)
            self.high = np.asarray(joint_limits[1], float)

    def _config_free(self, q: np.ndarray) -> bool:
        pts = self.kin.link_positions(q)
        return not collides(pts, self.obstacles, self.link_radius)

    def _edge_free(self, qa: np.ndarray, qb: np.ndarray) -> bool:
        for t in np.linspace(0.0, 1.0, self.edge_check_steps):
            q = qa + t * (qb - qa)
            if not self._config_free(q):
                return False
        return True

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlannerResult:
        """Plan a collision-free joint path from ``q_start`` to ``q_goal``."""
        q_start = np.asarray(q_start, float).reshape(3)
        q_goal = np.asarray(q_goal, float).reshape(3)

        if not self._config_free(q_start):
            return PlannerResult(False, None, 0, "start_in_collision")
        if not self._config_free(q_goal):
            return PlannerResult(False, None, 0, "goal_in_collision")

        nodes: List[np.ndarray] = [q_start.copy()]
        parents: List[int] = [-1]

        for it in range(self.max_iterations):
            if self.rng.random() < self.goal_bias:
                q_sample = q_goal.copy()
            else:
                q_sample = self.rng.uniform(self.low, self.high)

            # Nearest by wrapped squared distance.
            dists = [float(np.sum(wrap_angles(n - q_sample) ** 2)) for n in nodes]
            nearest_idx = int(np.argmin(dists))
            q_nearest = nodes[nearest_idx]

            diff = wrap_angles(q_sample - q_nearest)
            norm = float(np.linalg.norm(diff))
            if norm < 1e-9:
                continue
            q_new = q_nearest + diff * min(1.0, self.step_size / norm)
            q_new = np.clip(q_new, self.low, self.high)

            if not self._edge_free(q_nearest, q_new):
                continue

            nodes.append(q_new)
            parents.append(nearest_idx)

            # Try to connect straight to the goal from the newest node.
            if float(np.linalg.norm(wrap_angles(q_goal - q_new))) <= self.step_size:
                if self._edge_free(q_new, q_goal):
                    nodes.append(q_goal.copy())
                    parents.append(len(nodes) - 2)
                    path = self._extract_path(nodes, parents)
                    path = self._shortcut(path)
                    return PlannerResult(True, path, it + 1, "reached_goal")

        return PlannerResult(False, None, self.max_iterations, "max_iterations")

    @staticmethod
    def _extract_path(nodes: List[np.ndarray], parents: List[int]) -> np.ndarray:
        path = []
        idx = len(nodes) - 1
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        return np.stack(path[::-1])

    def _shortcut(self, path: np.ndarray, attempts: int = 120) -> np.ndarray:
        """Greedy shortcut smoothing: repeatedly connect non-adjacent nodes."""
        pts = [path[i] for i in range(len(path))]
        for _ in range(attempts):
            if len(pts) <= 2:
                break
            i = int(self.rng.integers(0, len(pts) - 1))
            j = int(self.rng.integers(i + 1, len(pts)))
            if j - i < 2:
                continue
            if self._edge_free(pts[i], pts[j]):
                pts = pts[: i + 1] + pts[j:]
        return np.stack(pts)

