"""Metrics for controller and policy evaluation.

All metrics are computed from recorded per-step arrays so that any policy
(discrete agent or continuous controller) is evaluated identically:

- success_rate: fraction of episodes whose minimum end-effector distance to
  the target is below the success threshold.
- time_to_reach: steps until first success (inf when never successful).
- path_length: total end-effector travel in meters.
- smoothness: mean squared end-effector acceleration (m^2/s^4 with unit
  step); lower is smoother.
- collisions / min_clearance: kinematic obstacle checks when obstacle
  geometry is provided (approximate capsule checks, see armlab.planning).

No metric is fabricated: every number derives from the recorded rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from armlab.planning.collision import Obstacle, min_clearance

__all__ = ["EpisodeRecord", "EpisodeMetrics", "aggregate", "compute_metrics"]


@dataclass
class EpisodeRecord:
    """Raw data recorded during one episode rollout."""

    ee_positions: np.ndarray        # (T+1, 3)
    distances: np.ndarray           # (T+1,)
    rewards: List[float]
    q_positions: np.ndarray         # (T+1, 3)
    success_threshold: float
    link_positions: Optional[np.ndarray] = None  # (T+1, 4, 3) if available
    steps: int = 0
    terminated: bool = False
    truncated: bool = False


@dataclass
class EpisodeMetrics:
    """Metrics derived from one EpisodeRecord."""

    success: bool
    time_to_reach: float            # steps; np.inf when never successful
    path_length: float
    smoothness: float
    total_reward: float
    min_clearance: Optional[float] = None
    collisions: int = 0

    def as_dict(self) -> dict:
        d = {
            "success": bool(self.success),
            "time_to_reach": float(self.time_to_reach),
            "path_length": float(self.path_length),
            "smoothness": float(self.smoothness),
            "total_reward": float(self.total_reward),
        }
        if self.min_clearance is not None:
            d["min_clearance"] = float(self.min_clearance)
            d["collisions"] = int(self.collisions)
        return d


def compute_metrics(
    record: EpisodeRecord,
    obstacles: Optional[List[Obstacle]] = None,
    link_radius: float = 0.3,
) -> EpisodeMetrics:
    """Compute metrics from a recorded episode.

    Collision counting samples each recorded configuration and reports the
    number of configurations whose capsule chain penetrates any obstacle
    (approximate kinematic check; see armlab.planning.collision).
    """
    ee = record.ee_positions
    dists = record.distances

    successes = np.nonzero(dists <= record.success_threshold)[0]
    success = bool(successes.size > 0)
    time_to_reach = float(successes[0]) if success else float(np.inf)

    if ee.shape[0] > 1:
        seg = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        path_length = float(seg.sum())
    else:
        path_length = 0.0

    if ee.shape[0] >= 3:
        acc = ee[2:] - 2 * ee[1:-1] + ee[:-2]
        smoothness = float(np.mean(np.sum(acc**2, axis=1)))
    else:
        smoothness = 0.0

    min_clear = None
    collisions = 0
    if obstacles and record.link_positions is not None:
        clears = []
        for links in record.link_positions:  # (4, 3) base->shoulder->elbow->ee
            clears.append(
                min(min_clearance(list(links), obs) for obs in obstacles)
            )
        min_clear = float(np.min(clears))
        collisions = int(np.sum(np.array(clears) < link_radius))

    return EpisodeMetrics(
        success=success,
        time_to_reach=time_to_reach,
        path_length=path_length,
        smoothness=smoothness,
        total_reward=float(np.sum(record.rewards)),
        min_clearance=min_clear,
        collisions=collisions,
    )


def aggregate(metrics: List[EpisodeMetrics]) -> dict:
    """Aggregate episode metrics into a report with mean/std/median."""
    if not metrics:
        raise ValueError("aggregate() requires at least one EpisodeMetrics.")
    succ = np.array([m.success for m in metrics], dtype=float)
    ttr = np.array([m.time_to_reach for m in metrics])
    finite_ttr = ttr[np.isfinite(ttr)]
    path = np.array([m.path_length for m in metrics])
    smooth = np.array([m.smoothness for m in metrics])
    reward = np.array([m.total_reward for m in metrics])

    def _stats(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"mean": None, "std": None, "median": None}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
        }

    report = {
        "n_episodes": len(metrics),
        "success_rate": float(np.mean(succ)),
        "success_rate_std": float(np.std(succ)),
        "time_to_reach": _stats(finite_ttr),
        "path_length": _stats(path),
        "smoothness": _stats(smooth),
        "total_reward": _stats(reward),
    }
    clears = [m.min_clearance for m in metrics if m.min_clearance is not None]
    if clears:
        report["min_clearance"] = _stats(np.array(clears))
        report["collisions_total"] = int(sum(m.collisions for m in metrics))
    return report
