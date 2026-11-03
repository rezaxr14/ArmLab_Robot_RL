"""Goal-conditioned reward that works on numpy arrays *and* torch tensors (for HER relabelling)."""

from __future__ import annotations

import numpy as np


def goal_distance(achieved, desired):
    """Euclidean distance over the last axis (numpy or torch)."""
    return ((achieved - desired) ** 2).sum(-1) ** 0.5


def goal_reward(distance, threshold: float, mode: str = "sparse"):
    """``sparse``: -1 until within ``threshold`` of the goal, then 0.  ``dense``: -distance."""
    if mode == "sparse":
        miss = distance > threshold
        return -(miss.float() if hasattr(miss, "float") else np.asarray(miss, dtype=np.float32))
    if mode == "dense":
        return -distance
    raise ValueError(f"unknown reward mode {mode!r} (use 'sparse' or 'dense')")
