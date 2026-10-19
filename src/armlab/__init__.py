"""ArmLab: a reproducible robotics control, simulation, and RL benchmark.

Public entry points:
- ``armlab.config``: validated configuration dataclasses.
- ``armlab.kinematics``: forward/inverse kinematics, Jacobian, damped least squares.
- ``armlab.targets``: reachable target generators.
- ``armlab.controllers``: classical joint/Cartesian controllers.
- ``armlab.planning``: joint-space planner with obstacle checks.
- ``armlab.envs``: Gymnasium environments (kinematic and, optionally, physics).
- ``armlab.rl``: Dueling Double DQN agent (requires torch).
- ``armlab.evaluation``: controller/policy evaluation and metrics.
- ``armlab.cli``: command-line interface (``armlab``).

Conventions (see docs/conventions.md):
- Units: meters, radians, seconds.
- Joint order: [waist (z-axis), shoulder, elbow]; the shoulder angle is measured
  from the vertical (+z) axis and the elbow angle relative to the second link.
"""

from armlab.config import EnvParams, PhysicsParams, RLConfig, RewardParams, RobotParams

__version__ = "0.1.0"

__all__ = [
    "EnvParams",
    "PhysicsParams",
    "RLConfig",
    "RewardParams",
    "RobotParams",
    "__version__",
]
