"""Backward-compatible adapters for the original demo scripts.

The demo projects (``projects/analytical_inverse_kinematics/`` and
``projects/rl_double_dueling_dqn/``) remain runnable. These adapters expose the
same public class names on top of the tested ``armlab`` implementation so new
code can import a single namespace. The original IK and the adapter use the
same vertical-plane convention, so both round-trip reachable targets.
"""

from armlab.legacy.analytical_ik import Robot
from armlab.legacy.visualizer_matplotlib import RobotVisualizer

__all__ = ["Robot", "RobotVisualizer"]
