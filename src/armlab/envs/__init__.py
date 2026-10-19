"""Gymnasium environments and registration.

Importing this package registers:
- ``ArmLabKinematic-v0``: static-target kinematic reaching.
- ``ArmLabMovingTarget-v0``: moving-target kinematic reaching.

Physics environments (``ArmLabPhysics-v0``) are registered by
``armlab.physics`` when MuJoCo is installed.
"""

from __future__ import annotations

import gymnasium as gym

from armlab.envs.kinematic import KinematicReachingEnv
from armlab.envs.moving import MovingTargetEnv

__all__ = ["KinematicReachingEnv", "MovingTargetEnv", "register_envs"]

_ENV_IDS = {
    "ArmLabKinematic-v0": KinematicReachingEnv,
    "ArmLabMovingTarget-v0": MovingTargetEnv,
}


def register_envs() -> None:
    """Idempotently register all kinematic environments with Gymnasium."""
    for env_id, entry in _ENV_IDS.items():
        try:
            gym.register(id=env_id, entry_point=entry)
        except Exception:
            # Already registered (e.g. module re-imported); keep going.
            pass


register_envs()
