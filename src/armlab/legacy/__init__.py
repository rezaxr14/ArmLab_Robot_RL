"""Backward-compatible adapters for the legacy scripts.

The legacy folders (``Analytical_Inverse_Kinematics/`` and
``RL_Double_Dueling_DQN/``) remain runnable as before. These adapters expose
the same public class names on top of the tested ``armlab`` implementation so
new code can import a single namespace.

Known intentional deviation from legacy behavior:
- ``Analytical_Inverse_Kinematics/robot.py`` computes the IK plane angle as
  ``atan2(z', r)`` while its own FK measures the shoulder angle from the
  vertical axis; this makes the legacy FK/IK round trip fail with residuals
  on the order of the arm size (verified numerically during migration). The
  corrected formula ``atan2(r, z')`` is used here. See execution log in
  implementation_plan.md.
"""

from armlab.legacy.analytical_ik import Robot
from armlab.legacy.visualizer_matplotlib import RobotVisualizer

__all__ = ["Robot", "RobotVisualizer"]
