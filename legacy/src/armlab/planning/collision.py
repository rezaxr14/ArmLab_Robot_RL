"""Collision primitives for the kinematic planner.

Links are modeled as capsules (a segment with a radius); obstacles are
spheres and axis-aligned boxes (AABB). The segment–sphere distance is exact;
the segment–box distance uses point sampling along the segment with a
documented resolution, which is conservative only up to that resolution —
the planner's clearance claims are approximate and this is documented in
docstrings and docs/conventions.md. Physical contact behavior in simulation
is the authority for real collisions (see armlab.physics).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SphereObstacle",
    "BoxObstacle",
    "Obstacle",
    "segment_point_distance",
    "min_clearance",
    "collides",
]


@dataclass(frozen=True)
class SphereObstacle:
    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, float).reshape(3))
        if self.radius <= 0:
            raise ValueError("radius must be positive.")


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned box given by min/max corners."""

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = np.asarray(self.low, float).reshape(3)
        high = np.asarray(self.high, float).reshape(3)
        if np.any(low >= high):
            raise ValueError(f"BoxObstacle requires low < high, got {low} / {high}.")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


Obstacle = object  # union type alias: SphereObstacle | BoxObstacle


def segment_point_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Minimum distance between segment ab and point p."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    p = np.asarray(p, float)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip((p - a) @ ab / denom, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _segment_box_distance(
    a: np.ndarray, b: np.ndarray, box: BoxObstacle, samples: int = 12
) -> float:
    """Approximate segment-to-AABB distance by sampling interior points.

    Sampled points along the segment are clipped to the box and the minimum
    distance is returned. Resolution: ``samples`` interior points; this can
    overestimate the true distance for grazing contacts, which the planner
    mitigates by checking edge midpoints as well.
    """
    best = np.inf
    for t in np.linspace(0.0, 1.0, samples):
        p = a + t * (b - a)
        q = np.clip(p, box.low, box.high)
        best = min(best, float(np.linalg.norm(p - q)))
    return best


def min_clearance(points: Sequence[np.ndarray], obstacle: Obstacle) -> float:
    """Minimum surface clearance between the polyline ``points`` and obstacle.

    Negative values indicate penetration depth for spheres.
    """
    pts = [np.asarray(p, float) for p in points]
    if isinstance(obstacle, SphereObstacle):
        d = min(
            segment_point_distance(pts[i], pts[i + 1], obstacle.center) for i in range(len(pts) - 1)
        )
        return d - obstacle.radius
    if isinstance(obstacle, BoxObstacle):
        return min(_segment_box_distance(pts[i], pts[i + 1], obstacle) for i in range(len(pts) - 1))
    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")


def collides(
    points: Sequence[np.ndarray],
    obstacles: Sequence[Obstacle],
    link_radius: float = 0.3,
) -> bool:
    """True if any capsule (segment + link_radius) intersects any obstacle."""
    for obs in obstacles:
        if min_clearance(points, obs) < link_radius:
            return True
    return False
