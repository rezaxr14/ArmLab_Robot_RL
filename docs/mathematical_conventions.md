# Mathematical conventions

This document is the single reference for the coordinate frames, joint
conventions, units and equations used by ArmLab. Everything in
`armlab.kinematics` and the derived controllers follows it.

## Units

ArmLab is explicit about units on every public interface:

- **Angles:** radians everywhere in code and config. Degrees appear only in
  comments/plots.
- **Lengths:** the *kinematic* model stores numbers in **legacy units**, the
  same scale the original demo used, where `link_lengths = [5, 10, 8]`. These
  numbers behave as metres inside the code (`numpy`/`gymnasium` request no unit
  conversion), so distances reported by the environment and metrics are in
  "legacy units".
- **Declared scale:** `configs/robot/three_link.yaml` records
  `legacy_unit_scale_m: 0.1`, i.e. the intended SI mapping is
  `SI length [m] = legacy length * 0.1` (base column 0.5 m, shoulder link
  1.0 m, forearm 0.8 m).
- **Physics:** the MuJoCo model (`armlab.physics.model`) is generated from the
  same numbers and therefore also uses them directly as metres. The simulator
  and the analytic kinematics are mutually consistent (a Phase E test compares
  the MuJoCo end-effector site to `RobotKinematics`), but **neither applies the
  0.1 scale**. This is a deliberate, documented deviation from the plan's
  "use SI in physics" goal: rescaling would silently change the meaning of the
  existing `[5, 10, 8]` defaults and every published number, so the legacy
  numbers are preserved and the intended scale lives in config metadata only.

## Robot geometry

The arm is a 3-link spatial manipulator with one base yaw joint and two
in-plane joints:

| index | name | symbol | description |
|:--|:--|:--|:--|
| 0 | waist | `theta1` | rotation about world `+z` at the origin |
| 1 | shoulder | `theta2` | angle from the local vertical `+z`; positive tilts the second link toward `+x` when `theta1 = 0` |
| 2 | elbow | `theta3` | angle of the third link relative to the second (`0` = fully extended) |

World is a right-handed frame with `+z` up. The shoulder joint sits at
`[0, 0, l1]`, where `l1`, `l2`, `l3 = link_lengths`.

## Forward kinematics

For `q = [theta1, theta2, theta3]`, with shorthands `c1 = cos(theta1)`,
`s1 = sin(theta1)`, `s2 = sin(theta2)`, `c2 = cos(theta2)`,
`s23 = sin(theta2 + theta3)`, `c23 = cos(theta2 + theta3)`:

```
base     p0 = [0,                       0,                       0        ]
shoulder p1 = [0,                       0,                       l1       ]
elbow    p2 = [l2 * c1 * s2,            l2 * s1 * s2,            l1 + l2*c2 ]
ee       p3 = [c1*(l2*s2 + l3*s23),     s1*(l2*s2 + l3*s23),     l1 + l2*c2 + l3*c23]
```

`RobotKinematics.link_positions` returns these stacked as a `(4, 3)` array
`[p0, p1, p2, p3]`; `end_effector` returns `p3`.

Key consequences:

- The waist joint rotates `x/y` without changing height, exactly as a yaw about
  `+z` should.
- At `theta2 = 0` with `theta3 = 0` the arm points straight up and the
  end-effector is at `[0, 0, l1 + l2 + l3]`.
- Endpoint distances `|p2 - p1| = l2` and `|p3 - p2| = l3`.

## Inverse kinematics

Given a target `[x, y, z]` and a reference configuration `q_current`:

1. `r_xy = hypot(x, y)`. If `r_xy < 1e-9` the target is on the base axis, so
   `theta1` is preserved from `q_current`; otherwise `theta1 = atan2(y, x)`.
2. Work in the vertical plane: `z' = z - l1`, `D = hypot(r_xy, z')`.
3. **Reachability annulus:** the target must satisfy
   `|l2 - l3| <= D <= l2 + l3`. Outside this range the result is
   `reachable=False` with `solution=None` and `residual=nan`.
4. **Shoulder-coincident singularity:** if `D < 1e-12` (only possible when
   `l2 == l3`) the arm folds back on itself. The elbow is set fully bent
   (`theta3 = pi`), `theta2` is preserved, and `theta1` is taken from step 1.
5. Otherwise the cosine law gives

   ```
   cos(theta3) = (D^2 - l2^2 - l3^2) / (2 * l2 * l3)      ->  t3_mag = acos(...)
   cos(alpha)  = (D^2 + l2^2 - l3^2) / (2 * D * l2)        ->  alpha  = acos(...)
   gamma       = atan2(r_xy, z')
   ```

   `gamma` is measured from the **vertical** (`+z`), matching how `theta2` is
   defined in the FK. Two branches are produced:

   ```
   elbow_up   : [theta1, gamma - alpha,   t3_mag]
   elbow_down : [theta1, gamma + alpha,  -t3_mag]
   ```

   Both cosine arguments are clipped to `[-1, 1]` after the reachability check
   to absorb floating-point noise.

6. **Solution selection:** `elbow` may force a branch. When it is `None`, the
   branch whose wrapped joint-space distance to `q_current` is smallest is
   returned. The chosen solution is wrapped to `[-pi, pi)`.

The result is an `IKResult(solution, reachable, branch, residual)`. `residual`
is the FK reconstruction error and is `0` for exact solutions. No code path
returns `NaN`/`Inf` angles.

> Regression note. The legacy `Analytical_Inverse_Kinematics/robot.py` computed
> the plane angle as `atan2(z', r_xy)` — the complement of `gamma` — which
> breaks the FK/IK round trip. That file is kept as the historical buggy
> reference; `tests/test_legacy.py::test_legacy_ik_round_trip_is_broken`
> documents the residual (> 1.0) and `armlab.legacy.Robot` provides the
> corrected behavior behind the same API.

## Geometric Jacobian

The position Jacobian `J = d(ee)/d(q)` is `3 x 3` and maps joint rates to
end-effector linear velocity (units: length per radian). With
`r = l2*s2 + l3*s23`:

```
J = [ -s1*r,  c1*(l2*c2 + l3*c23),  c1*(l3*c23) ]
    [  c1*r,  s1*(l2*c2 + l3*c23),  s1*(l3*c23) ]
    [  0,     -(l2*s2 + l3*s23),    -(l3*s23)   ]
```

`tests/test_kinematics.py` validates each column against central finite
differences. Columns 2 and 3 are controlled by the shoulder/elbow geometry;
column 1 vanishes when `r = 0` (the shoulder/elbow workspace is on the base
axis), which is the classic wrist singularity.

### Damped least squares

Differential IK uses the damped pseudoinverse

```
dq = J^T (J J^T + lambda^2 I)^-1 dx
```

with configurable `lambda` (`damping`). Larger damping trades Cartesian
accuracy for smooth, bounded joint motion near singular configurations.
`CartesianDLSController` additionally clips each joint step to `max_dq` and
projects onto the joint limits.

## Joint limits

Joint limits live in `RobotParams` (`joint_limits_low`, `joint_limits_high`)
and per-config in `configs/robot/three_link.yaml`. Defaults are
`[-pi, -pi/2, -pi] .. [pi, pi/2, pi]`. Physical limits are **not** the same as
angle wrapping: a limited physical joint is clipped, not wrapped around its
stops. `wrap_angle`/`wrap_angles` apply only where `theta` and `theta + 2*pi`
are genuinely equivalent (for example solution selection).

## Reward components

The reaching reward is a sum of explicit, logged components
(`info["reward_components"]`):

| component | formula |
|:--|:--|
| `progress` | `progress_scale * (prev_distance - distance)` |
| `distance` | `-distance_penalty_scale * distance` |
| `action_cost` | `-action_penalty * mean(abs(action))` |
| `time_cost` | `-time_penalty` |
| `limit_penalty` | `-limit_penalty` per step at a joint limit |
| `success` | `+success_bonus` when `distance <= success_threshold` (terminal) |

Weights come from `RewardParams` / `configs/env/reaching.yaml`.
