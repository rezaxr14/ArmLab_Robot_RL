"""Physics simulation via MuJoCo (optional ``physics`` extra).

Importing this package does NOT require MuJoCo; ``register_physics_envs``
raises a helpful ImportError only when called without MuJoCo installed.
After registration, ``gym.make("ArmLabPhysics-v0")`` works.
"""

from __future__ import annotations

import gymnasium as gym

from armlab.physics.env import MUJOCO_AVAILABLE, PhysicsReachingEnv

__all__ = ["PhysicsReachingEnv", "MUJOCO_AVAILABLE", "register_physics_envs"]

_ENV_ID = "ArmLabPhysics-v0"


def register_physics_envs() -> None:
    """Register the physics env with Gymnasium (requires MuJoCo)."""
    if not MUJOCO_AVAILABLE:
        raise ImportError(
            "Registering ArmLabPhysics-v0 requires MuJoCo. "
            "Install with: pip install -e '.[physics]' (or pip install mujoco)."
        )
    try:
        gym.register(id=_ENV_ID, entry_point=PhysicsReachingEnv)
    except Exception:
        pass  # already registered
