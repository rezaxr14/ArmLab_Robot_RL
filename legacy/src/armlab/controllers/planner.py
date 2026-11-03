"""Planner-based reference controller.

The RRT planner supplies a collision-free joint path; this controller tracks
that path with a bounded proportional joint controller. This keeps the plan
(supplies a path) and the controller (tracks it) distinct, as required by
implementation plan F3.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from armlab.controllers.base import Controller
from armlab.controllers.joint import JointPositionController
from armlab.kinematics import RobotKinematics, wrap_angles
from armlab.planning.collision import Obstacle
from armlab.planning.rrt import JointSpaceRRT

__all__ = ["RRTPlannerController"]


class RRTPlannerController(Controller):
    """Plan once to an IK goal, then track the planned waypoints.

    Falls back to analytical joint-position control when the planner fails, so
    the controller always returns a bounded, finite command.
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        obstacles: Sequence[Obstacle] | None = None,
        link_radius: float = 0.3,
        kp: float = 0.5,
        max_dq: float = 0.3,
        step_size: float = 0.4,
        max_iterations: int = 2000,
        seed: int = 0,
        waypoint_tolerance: float = 0.05,
        joint_limits: tuple | None = None,
        replan_distance: float = 1.0,
    ) -> None:
        if max_dq <= 0:
            raise ValueError("max_dq must be positive.")
        self.kin = kinematics
        self.obstacles = list(obstacles) if obstacles else []
        self.link_radius = float(link_radius)
        self.kp = float(kp)
        self.max_dq = float(max_dq)
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.seed = int(seed)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.replan_distance = float(replan_distance)
        self.joint_limits = joint_limits
        self._fallback = JointPositionController(kinematics, kp=kp, max_dq=max_dq)
        self._path: np.ndarray | None = None
        self._index = 0
        self._planned_target: np.ndarray | None = None
        self.diagnostics = {"planned": False, "reason": "not_started"}

    def reset(self) -> None:
        self._path = None
        self._index = 0
        self._planned_target = None
        self._fallback.reset()
        self.diagnostics = {"planned": False, "reason": "reset"}

    def _plan(self, q: np.ndarray, target: np.ndarray) -> None:
        ik = self.kin.inverse_kinematics(target, q_current=q)
        if not ik.reachable:
            self._path = None
            self.diagnostics = {"planned": False, "reason": "target_unreachable"}
            return
        planner = JointSpaceRRT(
            self.kin,
            obstacles=self.obstacles,
            link_radius=self.link_radius,
            step_size=self.step_size,
            max_iterations=self.max_iterations,
            seed=self.seed,
            joint_limits=self.joint_limits,
        )
        result = planner.plan(q, ik.solution)
        self._planned_target = np.asarray(target, dtype=float).copy()
        if result.success and result.path is not None:
            self._path = result.path
            self._index = 0
            self.diagnostics = {
                "planned": True,
                "reason": result.reason,
                "iterations": result.iterations,
            }
        else:
            self._path = None
            self.diagnostics = {"planned": False, "reason": result.reason}

    def act(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)

        need_plan = self._path is None
        if self._planned_target is None:
            need_plan = True
        elif float(np.linalg.norm(target - self._planned_target)) > self.replan_distance:
            need_plan = True
        if need_plan:
            self._plan(q, target)

        if self._path is None:
            return self._fallback.act(q, target)

        waypoint = self._path[min(self._index, len(self._path) - 1)]
        if float(np.linalg.norm(wrap_angles(waypoint - q))) < self.waypoint_tolerance:
            self._index = min(self._index + 1, len(self._path) - 1)
            waypoint = self._path[self._index]

        dq = self.kp * wrap_angles(waypoint - q)
        return np.clip(dq, -self.max_dq, self.max_dq)
