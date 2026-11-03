"""ArmForge: UR5e + Robotiq 2F-85 tabletop manipulation in native MuJoCo, with SAC/TQC + HER.

Headless rendering needs an OpenGL backend. On a Linux machine without a display we default to
EGL (GPU, works on Colab); set ``MUJOCO_GL=osmesa`` for software rendering. This must be decided
before ``mujoco`` is first imported, hence it lives here.
"""

import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "1.0.0"
