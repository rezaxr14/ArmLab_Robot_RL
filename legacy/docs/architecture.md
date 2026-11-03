# Architecture

ArmLab is organised around one idea: **classical controllers and learned
policies solve the same task against the same robot model, evaluated through one
interface that writes one artifact schema.** This document explains how the code
is layered and where each responsibility lives.

## Package layout (as implemented)

```text
src/armlab/
├── __init__.py            # version
├── __main__.py            # `python -m armlab` -> cli.main
├── cli.py                 # argparse CLI, all subcommands
├── config.py              # frozen, validated dataclasses (RobotParams, ...)
├── configio.py            # YAML loading, validation, resolved-config saving
├── kinematics.py          # FK, analytical IK, Jacobian, damped least squares
├── scenarios.py           # named scenario registry -> ScenarioBundle
├── targets.py             # seeded reachable target samplers
├── controllers/
│   ├── base.py            # Controller protocol + discrete adapter
│   ├── joint.py           # JointPositionController, JointPDController
│   ├── cartesian.py       # CartesianDLSController (resolved-rate DLS)
│   ├── mpc.py             # ShootingMPCController
│   └── planner.py         # RRTPlannerController (path -> tracked commands)
├── envs/
│   ├── kinematic.py       # KinematicReachingEnv (Gymnasium)
│   ├── moving.py          # MovingTargetEnv
│   └── wrappers.py        # ActionRepeat, StackObservations
├── physics/
│   ├── model.py           # programmatic MJCF builder (importable without MuJoCo)
│   └── env.py             # PhysicsReachingEnv (lazy MuJoCo import)
├── planning/
│   ├── collision.py       # Obstacle / SphereObstacle / BoxObstacle, clearance
│   └── rrt.py             # RRT-Connect with edge validation
├── rl/
│   ├── networks.py        # Dueling Q-network
│   ├── replay.py          # seeded replay buffer
│   ├── agent.py           # DQNAgent (Double DQN targets, soft/hard updates)
│   ├── training.py        # train_dqn loop + metrics/checkpoints
│   └── sac.py             # Stable-Baselines3 SAC wrapper (optional)
├── evaluation/
│   ├── metrics.py         # EpisodeRecord / EpisodeMetrics / aggregate
│   ├── runner.py          # rollout helpers + controller_policy adapters
│   ├── benchmark.py       # evaluate_run / run_benchmark / controller registry
│   ├── artifacts.py       # run directories, schema 1.0, load/discover
│   └── plots.py           # training curves, comparisons, reports
├── visualization/
│   └── replay.py          # load + replay saved trajectories, optional GIF
├── dashboard/
│   └── app.py             # optional Dash UI (lazy dash/plotly import)
└── legacy/                # compatibility facades for the original demo
    ├── analytical_ik.py
    └── visualizer_matplotlib.py
```

The original demos are grouped under `projects/`
(`analytical_inverse_kinematics/`, `rl_double_dueling_dqn/`) and kept runnable
for provenance. New code never duplicates their logic: `armlab.legacy.Robot`
delegates to `RobotKinematics`, and the original IK is verified by
`tests/test_legacy.py`.

## Layering

```text
             configs/*.yaml
                  │  (configio validates)
                  ▼
  config.py  ──────────────┐
   (frozen dataclasses)    │
                           ▼
  kinematics.py  ◄── controllers/ (classical)   rl/ (learned)
        ▲                  │                      │
        │                  ▼                      ▼
        │             envs/ (Gymnasium)  ◄── training.py / sac.py
        │                  │
        │                  ▼
        └────────── evaluation/ (runner, metrics)
                           │
                           ▼
              evaluation/artifacts.py (results/<run-id>/…)
                           │
             ┌─────────────┴──────────────┐
             ▼                            ▼
     evaluation/plots.py          dashboard/app.py, visualization/replay.py
```

Rules enforced by the layout:

- **One FK source of truth.** Every env, controller, planner, visualizer and
  metric calls `RobotKinematics`. No module re-derives link positions.
- **Kinematics does not depend on environments.** `kinematics.py`, `config.py`
  and `controllers/` are usable headless with no Gym/MuJoCo import.
- **Optional dependencies are lazy.** `physics/__init__.py` and `rl/sac.py`
  import MuJoCo / Stable-Baselines3 only when used; `dashboard/app.py` imports
  Dash only when launched. Base install + math tests need none of them.
- **Evaluation is separate from training.** `evaluate_run` loads a policy
  callable and rolls it out; it never trains.

## Configuration

All tunables live in typed frozen dataclasses in `config.py`, loaded (and
rejected on unknown keys) by `configio.py`:

| file | dataclass |
|:--|:--|
| `configs/robot/three_link.yaml` | `RobotParams` |
| `configs/env/reaching.yaml`, `obstacle_reaching.yaml` | `EnvParams` + `RewardParams` |
| `configs/train/dqn.yaml` | loop settings + `RLConfig` |
| `configs/train/sac.yaml` | loop settings + SB3 kwargs |
| `configs/benchmark/default.yaml` | `BenchmarkConfig` |

Every constructed dataclass validates ranges in `__post_init__` and raises a
`ValueError` with the offending field, rather than silently clamping.

## Environments

`KinematicReachingEnv` is the fast, headless task used for tests and the
classical benchmark. It has a **discrete** action variant (six actions:
`joint_index x direction`, for DQN) and a **continuous** variant
(`Box(-1, 1, (3,))` scaled to bounded joint increments, for SAC and the
classical controllers). Action semantics do not change with `render_mode`.

Observations carry a version string (`info["observation_version"]`) so an
incompatible checkpoint can be detected. Terminal states and time-limit
truncation are distinct: `terminated=True` on success/failure, `truncated=True`
on horizon exhaustion.

`MovingTargetEnv` subclasses the reaching env and drives the target along a
seeded smooth trajectory. `PhysicsReachingEnv` (optional) is a MuJoCo adapter
with the same task concept and separate actuator/contact semantics.

## Scenarios

`scenarios.make_scenario(name, seed, robot)` returns a `ScenarioBundle`
(env + obstacles + metadata) for one of:

- `static_reaching` — random reachable target, no obstacles.
- `boundary_reaching` — targets near the workspace boundary.
- `moving_target` — seeded 3D target trajectory.
- `collision_avoidance` — fixed spheres between a home posture and the target.

Evaluation uses `base_eval_seed + i` for episode `i`, so held-out seeds are
identical across every controller.

## Evaluation and artifacts

`evaluation/benchmark.py` is the single entry point. `CONTROLLERS` is the
registry (`ik`, `pd`, `jacobian`, `mpc`, `rrt`, `dqn`, `sac`). Classical
controllers are evaluated in the **continuous** env so their real bounded joint
commands are used; `dqn`/`sac` keep their native action spaces.

Each run writes a self-describing directory (schema `1.0`):

```text
results/<timestamp>_<scenario>_<controller>_seed-<seed>/
  metadata.json     # git revision/dirty, platform, dependency versions, config hash, timestamps
  episodes.csv      # one row per episode
  summary.json      # aggregate metrics + aggregation method
  metrics.csv       # flattened aggregates
  trajectory_seed-*.npz
  plots/
```

`load_run` / `discover_runs` read these back, so reports and the dashboard never
recompute different definitions from the CLI.

## CLI

`python -m armlab <command>` (also installed as the `armlab` console script):

`doctor`, `demo ik|physics`, `train dqn|sac`, `evaluate`, `benchmark`,
`report`, `replay`, `dashboard`. All paths are relative to the working
directory; optional features fail with an install hint instead of a traceback.
