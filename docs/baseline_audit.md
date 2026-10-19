# Baseline audit

This records the state of the repository at the start of the ArmLab
implementation, the defects confirmed against the checked-out code, and the
commands that were actually run. It is intentionally candid: it lists what was
broken, not just what was fixed.

## Environment capture

| item | value |
|:--|:--|
| OS | Windows (win32) |
| Python | 3.12.3 |
| numpy | 2.2.6 |
| gymnasium | 1.2.0 |
| matplotlib | 3.9.0 |
| PyYAML | 6.0.2 |
| torch | 2.14.1+cpu |
| mujoco | 3.16.0 |
| stable_baselines3 | 2.9.0 |
| pytest | 8.2.1 |
| GPU | none (CPU-only torch build) |
| missing optional | `dash`, `plotly`, `coverage` (not installed locally) |

Command used: `python -m armlab doctor` (after the CLI existed) plus manual
`pip` inspection during setup.

## Starting repository state

- The repo was a working tree on Git with two legacy subprojects and an
  already-present `src/armlab` package (Phases A–E of the plan had been
  implemented in prior work). The worktree was dirty relative to `origin`.
- Existing test suite at inspection time: **141 passed**. After the remaining
  phases in this execution it grew to **201 passed, 1 skipped** (see the
  execution log; the skip is the dashboard test, which requires the optional
  `ui` extra).
- The legacy subprojects remained runnable and are preserved as the historical
  reference.

## Confirmed defects

These were reproduced against the checked-out code, with a test or a command,
before being addressed:

1. **Analytical IK angle-convention bug (resolved).**
   The original demo computed the in-plane angle as `atan2(z', r_xy)`, but its
   own FK measures `theta2` from the vertical, for which the correct angle is
   `atan2(r_xy, z')`. The round trip `FK(IK(FK(q)))` missed the target by
   `> 1.0` length units. The demo now uses the correct convention and
   `tests/test_legacy.py::TestIKRoundTrip` verifies a sub-`1e-6` residual for
   both the original demo and `armlab.legacy.Robot`.

2. **IK degenerate / boundary handling gaps.**
   The legacy solver divided by the target distance without guarding a
   near-zero denominator and did not clamp the cosine-law argument. The new
   solver checks the reachability annulus, clamps `acos` inputs, handles the
   waist singularity and adds a deliberate shoulder-coincident (`D ~ 0`) path.

3. **Duplicated forward kinematics.**
   FK existed in the analytical robot, in the RL environment, and in the
   visualizers. Consolidated into `RobotKinematics`; the legacy adapters now
   delegate to it.

4. **Unreachable / degenerate targets in the RL task.**
   Targets were sampled by independent radius/height ranges, some of which the
   arm cannot reach. Targets are now sampled from valid configurations via FK
   (`armlab.targets`), with boundary/singularity scenarios handled separately.

5. **DQN target-network initialization.**
   The original agent built local and target networks independently without
   copying weights. The current `DQNAgent` copies local weights into the target
   immediately and freezes target parameters.

6. **Checkpoint completeness.**
   The original only saved Q-network weights. Checkpoints now include both
   networks, optimizer state, step counters, epsilon, config, and version
   metadata.

7. **Global RNG usage / reproducibility.**
   Target generation and training used global `numpy`/`random` state. All
   randomness now flows through explicit seeded `numpy.random.Generator`s or
   `env.reset(seed=...)`.

8. **Thin experiment records.**
   No standard result schema existed. Phase G added the schema-1.0 run
   directories described in `docs/architecture.md`.

## Headless vs GUI

A missing display was never treated as a robotics bug. All math, the kinematic
environment, controllers and the benchmark run headless; `--headless` is
available on demos and rendering is decoupled from `step()`. MuJoCo model
validation and short rollouts run without a viewer.

## Commands attempted (representative)

```text
python -m armlab doctor
python -m armlab demo ik --headless --trials 5
python -m armlab evaluate --scenario static_reaching --controller ik --episodes 5
python -m armlab benchmark --suite default --config configs/benchmark/default.yaml
python -m armlab train dqn --episodes 20 --max-steps 2000 --eval-every 5 --seed 7
python -m armlab train sac --total-timesteps 2000 --seed 7
python -m pytest
```

Results and pass counts for these are recorded in `docs/experiments.md` and in
the plan's execution log.
