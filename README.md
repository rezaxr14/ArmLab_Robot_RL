# ArmLab

**A reproducible 3-link robot-arm benchmark that puts classical control
(analytical IK, PD, damped-least-squares Jacobian, MPC, RRT) and learned
policies (Dueling Double DQN, SAC) on the same task, robot model and evaluation
schema — in kinematics and, optionally, MuJoCo physics.**

ArmLab grew out of a 3D robot-arm demo that could solve inverse kinematics and
train a DQN. The engineering goal here is narrower and more useful: make the
math correct and tested, make training reproducible and checkpointed, and make
every controller comparable through one measured, versioned results format.

> **Honesty note:** the reported table below is a **20-episode smoke suite** on
> the kinematic task, not a converged multi-seed benchmark. RL smoke runs are
> labelled as such and are not evidence of convergence. See
> [`docs/experiments.md`](docs/experiments.md) for protocol and failure
> analysis.

---

## Demo

The original demonstration still runs and shows analytical IK tracking a moving
target and a trained DQN reaching goals:

[![Robot arm demo](Screenshots/inv_kin_Figure_2.png)](https://youtu.be/OU_E-vsCCfU "Robot arm demo")

| Analytical IK tracking | RL reaching |
|:--:|:--:|
| ![IK tracking](Screenshots/inv_kin_Figure_2.png) | ![RL success](Screenshots/RL_Figure_1.png) |
| ![IK tracking pose](Screenshots/inv_kin_Figure_1.png) | ![RL target](Screenshots/RL_Figure_4.png) |

Reproducible headless runs also generate their own plots and trajectory
artifacts (see the results section).

---

## What is implemented

- **Correct, tested kinematics.** One `RobotKinematics` source of truth for FK,
  analytical IK (both elbow branches, closest-branch selection, joint limits,
  reachability, singular cases), the geometric Jacobian and damped least
  squares.
- **Classical controllers.** Analytical IK + joint P/PD tracking, resolved-rate
  damped-least-squares Cartesian control, random-shooting MPC, and an
  RRT-Connect path planner tracked by a low-level controller.
- **Gymnasium environments.** Discrete (`ArmLabKinematic-v0`) and continuous
  (`ArmLabKinematicContinuous-v0`) reaching, a moving-target environment, and
  seeded, reachable target sampling. Seeded resets, joint-limit containment,
  `terminated` vs `truncated` semantics, reward components in `info`, and
  `check_env` coverage.
- **Dueling Double DQN from scratch.** Target-network initialization/freezing,
  Double-DQN target selection, seeded replay buffer, soft/hard target updates,
  gradient clipping, full resumable checkpoints and CSV/JSON training metrics.
- **SAC baseline via Stable-Baselines3** on the continuous-action task, with
  saved model + evaluation metrics and deterministic evaluation.
- **MuJoCo physics (optional).** Programmatically generated 3-link MJCF,
  velocity actuators, joint limits, contact detection, deterministic reset, and
  a test that the MuJoCo end-effector site agrees with analytical FK.
- **Named scenarios.** `static_reaching`, `boundary_reaching`, `moving_target`,
  `collision_avoidance`, all seeded and held out from training.
- **Standardized evaluation + artifacts.** Schema-1.0 run directories with
  `metadata.json`, `episodes.csv`, `summary.json`, `metrics.csv` and NPZ
  trajectories; plots and reports generated only from saved data.
- **CLI, replay and optional dashboard.** `python -m armlab` with `doctor`,
  `demo`, `train`, `evaluate`, `benchmark`, `report`, `replay` and `dashboard`.
- **Tests and CI.** 200+ deterministic tests that run on CPU without a
  display; GitHub Actions for base + optional physics.

### Not implemented (do not assume it exists)

- A **final** multi-seed benchmark (≥5 training seeds × ≥100 held-out episodes).
  The current results are explicitly a smoke suite.
- Robustness scenarios beyond held-out targets: `perturbed_dynamics`,
  `sensor_noise_delay`, `unseen_targets` are not wired into the scenario
  registry yet.
- Six-DOF/UR5e models, grippers, pick-and-place, ROS 2 or hardware control, and
  any sim-to-real claim.
- Vectorized/multi-process training and hosted experiment tracking.
- A browser-hosted simulator.

---

## Architecture

```text
configs/*.yaml ──► configio ──► frozen dataclasses (config.py)
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
             kinematics.py                        controllers/  (classical)
        (FK / IK / Jacobian / DLS)                        │
                    │                                     │
                    ▼                                     ▼
             envs/ (Gymnasium) ◄──────────── rl/ (DQN, SAC)
                    │
                    ▼
     evaluation/runner + metrics + benchmark
                    │
                    ▼
     results/<run-id>/{metadata,episodes,summary,metrics,trajectory}
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   evaluation/plots.py   visualization/replay, dashboard
```

Every environment, controller, planner, metric and visualizer uses the same
`RobotKinematics`; no module re-derives link positions. Optional dependencies
(MuJoCo, Stable-Baselines3, Dash/Plotly) are imported lazily. Full details:
[`docs/architecture.md`](docs/architecture.md).

---

## Installation

```bash
git clone https://github.com/rezaxr14/3D-Robot-Control-RL-Analytical.git
cd 3D-Robot-Control-RL-Analytical
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[rl]"        # torch + Stable-Baselines3 (SAC)
python -m pip install -e ".[physics]"   # MuJoCo
python -m pip install -e ".[ui]"        # Dash + Plotly dashboard
```

The base install needs no GPU, display, MuJoCo or Dash.

---

## Quick start (commands tested on this checkout)

```bash
# Environment / optional-dependency report
python -m armlab doctor

# Headless demos
python -m armlab demo ik --headless --trials 5
python -m armlab demo physics --headless --steps 20   # requires physics extra

# Evaluate a controller; writes a run directory under results/
python -m armlab evaluate --scenario static_reaching --controller ik --episodes 5
python -m armlab evaluate --scenario static_reaching --controller jacobian --episodes 5

# Smoke training (not convergence)
python -m armlab train dqn --episodes 20 --max-steps 2000 --eval-every 5 --seed 7
python -m armlab train sac --total-timesteps 2000 --seed 7

# Evaluate a saved RL checkpoint
python -m armlab evaluate --scenario static_reaching --controller dqn \
  --checkpoint results/train_dqn_seed7/latest.pt --episodes 5

# Benchmark + reports + replay
python -m armlab benchmark --suite default --config configs/benchmark/default.yaml
python -m armlab report --run results/<run-id>
python -m armlab replay --run results/<run-id> --episode 0

# Optional dashboard
python -m armlab dashboard --results results
```

All paths are relative to the current working directory.

---

## Controllers

| name | type | summary |
|:--|:--|:--|
| `ik` | classical | analytical IK branch + bounded joint P control |
| `pd` | classical | analytical IK target + joint PD tracking |
| `jacobian` | classical | resolved-rate damped least squares on Cartesian error; clips steps and projects onto joint limits |
| `mpc` | classical | random-shooting MPC over a short horizon on the kinematic model |
| `rrt` | classical | RRT-Connect joint-space path, then tracked by a low-level controller |
| `dqn` | learned | from-scratch Dueling Double DQN, discrete joint increments |
| `sac` | learned | Stable-Baselines3 SAC, continuous bounded joint increments |

Classical controllers are evaluated in the **continuous** environment so their
true bounded commands are used. DQN and SAC keep their native action spaces.

Math reference (frames, FK/IK equations, Jacobian, damping, reward):
[`docs/mathematical_conventions.md`](docs/mathematical_conventions.md).

---

## Measured results (smoke suite — not a final benchmark)

Command:

```bash
python -m armlab benchmark --suite default \
  --config configs/benchmark/default.yaml \
  --output-dir results/smoke-default
```

Kinematic task, 20 held-out episodes per cell, `base_eval_seed = 100000`,
success threshold `1.0` legacy unit. `mean final error` is in the same legacy
units.

| scenario | controller | success | mean final error |
|:--|:--|--:|--:|
| static_reaching | ik | 1.00 | 0.635 |
| static_reaching | jacobian | **0.50** | 2.803 |
| static_reaching | pd | 1.00 | 0.684 |
| static_reaching | mpc | 1.00 | 0.774 |
| boundary_reaching | ik | 1.00 | 1.045 |
| boundary_reaching | jacobian | 1.00 | 2.905 |
| boundary_reaching | pd | 1.00 | 1.152 |
| boundary_reaching | mpc | 1.00 | 0.693 |
| moving_target | ik | 1.00 | 0.022 |
| moving_target | jacobian | 1.00 | 1.086 |
| moving_target | pd | 1.00 | 0.024 |
| moving_target | mpc | 1.00 | 0.013 |
| collision_avoidance | ik | 1.00 | 0.635 |
| collision_avoidance | jacobian | **0.50** | 2.803 |
| collision_avoidance | pd | 1.00 | 0.684 |
| collision_avoidance | mpc | 1.00 | 0.774 |

Generated plots: `results/smoke-default/benchmark_success.png` and
`results/smoke-default/benchmark_errors.png`.

### Failure case worth showing

The Jacobian/DLS controller succeeds only half the time on static and
collision-avoidance seeds. It is a local method: a joint can reach a limit while
the Cartesian error still points through that limit, so the projected update
stalls just outside tolerance. The collision-avoidance run also records 3
approximate obstacle penetrations. This is reported rather than tuned away; see
[`docs/experiments.md`](docs/experiments.md) for the analysis and possible
fixes (limit-avoidance/null-space term, stall re-seeding).

RL smoke runs (2,000-step budgets) reached 0.00 eval success — expected for
undertrained agents. They are reported to prove the train → checkpoint → load →
evaluate path, not to claim learning. Full details in
[`docs/experiments.md`](docs/experiments.md).

---

## Reproduce the results

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m armlab benchmark --suite default \
  --config configs/benchmark/default.yaml --output-dir results/smoke-default
python -m armlab report --results results/smoke-default
```

A real benchmark requires editing `configs/benchmark/default.yaml`
(`smoke: false`, more episodes/seeds) and substantially more compute.

---

## Limitations

- Lengths use the legacy unit system (`link_lengths = [5, 10, 8]` treated as
  metres). `legacy_unit_scale_m: 0.1` documents the intended SI mapping but is
  **not applied**; changing it would silently alter every existing number.
- Physics parameters (masses, inertia, damping) are illustrative, not measured.
- Collision checks in the kinematic benchmark are approximate capsule tests,
  not MuJoCo contacts.
- The DLS controller has the documented local-minimum failure; MPC and IK/PD do
  not on these seeds.
- The dashboard requires the `ui` extra and is not covered by the local test
  run (its test skips when Dash/Plotly are absent).
- No sim-to-real, hardware, grasping or six-DOF claims.

---

## Project structure

```text
src/armlab/           # package: kinematics, controllers, envs, physics,
                      # planning, rl, evaluation, visualization, cli, dashboard
configs/              # validated robot/env/train/benchmark YAML
tests/                # deterministic unit + integration + regression tests
docs/                 # architecture, math conventions, experiments,
                      # troubleshooting, baseline audit
Analytical_Inverse_Kinematics/   # original demo + historical IK bug reference
RL_Double_Dueling_DQN/           # original demo
Screenshots/          # README media
```

---

## Testing and contributing

```bash
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, conventions and how to add
controllers/scenarios. Tests that need optional dependencies skip cleanly when
those packages are absent.

---

## License and attribution

The project is distributed under the MIT license (`pyproject.toml`). MuJoCo,
PyTorch, Stable-Baselines3, Gymnasium and Dash/Plotly remain under their own
licenses. No third-party robot meshes are redistributed; the MuJoCo model is
generated from configuration.
