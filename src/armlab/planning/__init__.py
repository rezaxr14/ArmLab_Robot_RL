"""Joint-space motion planning with obstacle awareness.

- ``collision``: capsule/sphere/box distance primitives.
- ``JointSpaceRRT``: sampling-based planner with shortcut smoothing.
"""

from armlab.planning.collision import (
    BoxObstacle,
    SphereObstacle,
    collides,
    min_clearance,
    segment_point_distance,
)
from armlab.planning.rrt import JointSpaceRRT, PlannerResult

__all__ = [
    "BoxObstacle",
    "SphereObstacle",
    "collides",
    "min_clearance",
    "segment_point_distance",
    "JointSpaceRRT",
    "PlannerResult",
]
