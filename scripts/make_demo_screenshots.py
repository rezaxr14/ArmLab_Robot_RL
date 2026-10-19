"""Regenerate the original analytical-IK demo screenshots.

The two images in ``projects/screenshots`` are captured from the original
matplotlib visualizer. This script runs the corrected IK solver headlessly so
the README media always matches the code.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "projects" / "analytical_inverse_kinematics"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "projects" / "screenshots",
        help="directory for inv_kin_Figure_1.png / inv_kin_Figure_2.png",
    )
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    robot_mod = _load("demo_robot", DEMO / "robot.py")
    viz_mod = _load("demo_visualizer", DEMO / "visualizer.py")
    robot = robot_mod.Robot([5, 10, 8])

    # Two reachable poses showing different shoulder/elbow branches.
    cases = [
        (np.array([12.0, 8.0, 10.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([-8.0, 10.0, 14.0]), np.array([2.0, 0.4, -0.4])),
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    for index, (target, current) in enumerate(cases, start=1):
        solution = robot.inverse_kinematics(target, current)
        if solution is None:
            raise SystemExit(f"Unreachable target: {target}")
        joints = robot.forward_kinematics(solution)
        residual = float(np.linalg.norm(np.asarray(joints[-1]) - target))
        if residual > 1e-6:
            raise SystemExit(f"IK residual {residual:.3e} for target {target}")

        viz = viz_mod.RobotVisualizer(robot)
        viz.plot_robot(joints, target)
        path = args.out / f"inv_kin_Figure_{index}.png"
        viz.fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(viz.fig)
        print(f"wrote {path} (residual {residual:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
