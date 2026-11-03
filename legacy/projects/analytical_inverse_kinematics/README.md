# Analytical inverse kinematics demo

The original 3-link arm demo. `robot.py` implements forward kinematics and a
closed-form IK solver; `main.py` drives the arm toward a moving 3D target and
`visualizer.py` renders it with matplotlib.

```bash
cd projects/analytical_inverse_kinematics
python main.py
```

The IK uses the same vertical-plane convention as `forward_kinematics`, so a
reachable target round-trips to numerical precision (see
[`tests/test_legacy.py`](../../tests/test_legacy.py)). The integrated and
hardened version is `armlab.kinematics.RobotKinematics`; the same API is exposed
as `armlab.legacy.Robot`.
