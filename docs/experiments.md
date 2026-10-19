# Experiments and measured results

This document describes the evaluation protocol and reports results that were
**actually produced by the code in this repository**. Every number below comes
from a run directory under `results/` (gitignored). If a result is a smoke /
development run, it is labelled as such.

## Protocol

- **Task:** kinematic three-link reaching in the legacy unit system
  (`docs/mathematical_conventions.md`). No physics is involved in the table
  below.
- **Held-out seeds:** episode `i` uses `base_eval_seed + i`. The default
  `base_eval_seed` is `100000`, so every controller sees the same 20 target
  seeds. Training is never run on these seeds.
- **Episodes:** 20 per (scenario, controller) cell unless stated otherwise.
- **Controllers:** `ik`, `pd`, `jacobian`, `mpc` are classical; they are
  evaluated in the **continuous** action environment so their real bounded
  joint commands are applied. `dqn` and `sac` are learned policies evaluated in
  their native action spaces.
- **Success:** an episode succeeds if the minimum end-effector distance ever
  drops below `success_threshold` (`1.0` legacy unit). For the moving-target
  scenario `success_terminates=False`, so success means "ever entered
  tolerance"; `final_error` then reflects tracking at the horizon.
- **Aggregation:** metrics are computed per episode and then averaged
  (`mean`/`std`/`median` in `summary.json`). `success_rate` is the episode
  fraction, not an average over training seeds.

## Benchmark — classical controllers (smoke suite)

Command actually run:

```bash
python -m armlab benchmark --suite default \
  --config configs/benchmark/default.yaml \
  --output-dir results/smoke-default
```

The suite config has `smoke: true`. **These are development results from 20
episodes per cell, not a final multi-seed benchmark.**

| scenario | controller | episodes | success rate | mean final error | mean min error | collision configs |
|:--|:--|--:|--:|--:|--:|--:|
| static_reaching | ik | 20 | 1.00 | 0.635 | 0.635 | 0 |
| static_reaching | jacobian | 20 | 0.50 | 2.803 | 2.803 | 0 |
| static_reaching | pd | 20 | 1.00 | 0.684 | 0.684 | 0 |
| static_reaching | mpc | 20 | 1.00 | 0.774 | 0.774 | 0 |
| boundary_reaching | ik | 20 | 1.00 | 1.045 | 0.000 | 0 |
| boundary_reaching | jacobian | 20 | 1.00 | 2.905 | 0.000 | 0 |
| boundary_reaching | pd | 20 | 1.00 | 1.152 | 0.000 | 0 |
| boundary_reaching | mpc | 20 | 1.00 | 0.693 | 0.000 | 0 |
| moving_target | ik | 20 | 1.00 | 0.022 | 0.000 | 0 |
| moving_target | jacobian | 20 | 1.00 | 1.086 | 0.000 | 0 |
| moving_target | pd | 20 | 1.00 | 0.024 | 0.000 | 0 |
| moving_target | mpc | 20 | 1.00 | 0.013 | 0.000 | 0 |
| collision_avoidance | ik | 20 | 1.00 | 0.635 | 0.635 | 0 |
| collision_avoidance | jacobian | 20 | 0.50 | 2.803 | 2.803 | 3 |
| collision_avoidance | pd | 20 | 1.00 | 0.684 | 0.684 | 0 |
| collision_avoidance | mpc | 20 | 1.00 | 0.774 | 0.774 | 0 |

Notes on interpretation:

- **Boundary-reaching `final_error` can exceed `min_error`.** These episodes
  reach the boundary target (min error `0.000`) and then oscillate/drift at the
  horizon; success is defined by the minimum distance, as documented.
- **Jacobian DLS is the weak cell (0.50).** Resolved-rate DLS is a local method:
  on half the static/collision seeds it settles in a local minimum or against a
  joint limit before entering tolerance. The `collision_avoidance` run also
  records 3 configurations inside an obstacle sphere, which is why this cell is
  the honest failure case discussed below. IK/PD/MPC do not exhibit it on these
  seeds.
- **Moving-target errors are small for all controllers** because the targets
  move slowly (amplitude ~2, frequency ~0.4) relative to the control step.
- **Collisions** are approximate kinematic capsule checks
  (`armlab.planning.collision`), not MuJoCo contacts.

Plots generated from these artifacts:

```text
results/smoke-default/benchmark_success.png
results/smoke-default/benchmark_errors.png
```

## Learning curves / RL smoke runs

Commands actually run (CPU):

```bash
python -m armlab train dqn --episodes 20 --max-steps 2000 --eval-every 5 --seed 7 \
  --output-dir results/smoke-rl
python -m armlab train sac --total-timesteps 2000 --seed 7 \
  --output-dir results/smoke-rl
```

| run | budget | eval success | note |
|:--|:--|--:|:--|
| DQN (Dueling Double DQN) | 2,056 env steps | 0.00 | smoke; not converged |
| SAC (Stable-Baselines3) | 2,000 timesteps | 0.00 | smoke; not converged |

Both checkpoints were then loaded through the **standard evaluation interface**
to prove the full path works:

```bash
python -m armlab evaluate --scenario static_reaching --controller dqn \
  --checkpoint results/smoke-rl/train_dqn_seed7/latest.pt \
  --episodes 5 --output-dir results/smoke-rl-eval
python -m armlab evaluate --scenario static_reaching --controller sac \
  --checkpoint results/smoke-rl/sac_ArmLabKinematicContinuous-v0_seed7.zip \
  --episodes 5 --output-dir results/smoke-rl-eval
```

| policy | episodes | success | interpretation |
|:--|--:|--:|:--|
| DQN checkpoint | 5 | 0.20 (1/5) | chance-level on 5 episodes; smoke only |
| SAC checkpoint | 5 | 0.00 (0/5) | undertrained; smoke only |

**No convergence claim is made.** Thousands to tens of thousands of steps are
needed before these policies are competitive; the full configs specify 300
episodes / 50,000 timesteps. The runs above exist to prove that training,
checkpointing, loading and standardized evaluation work end to end.

## Failure analysis: DLS local minima

The `jacobian` controller computes `dq = J^T (J J^T + λ²I)^-1 dx`, clips each
step, and projects onto joint limits. It has no global model of the target
manifold, so when a joint hits a limit while the Cartesian error still points
"through" the limit, the projected update cannot reduce the error and the
episode stalls slightly outside tolerance. This is the expected trade-off of
resolved-rate control, not a numerical bug: the same controller succeeds on the
boundary and moving-target scenarios and every IK/PD/MPC cell. A future
improvement is a null-space/limit-avoidance term or a re-seeding strategy after
a detected stall; the planner and MPC already avoid this by evaluating multiple
candidate motions.

## Not run / limitations

- No final benchmark: the plan asks for ≥5 training seeds and ≥100 held-out
  episodes for reported numbers. The suite above is a 20-episode smoke suite.
- No GPU runs; all training is CPU smoke.
- The dashboard test is skipped locally because `dash`/`plotly` are not
  installed in the development environment. The base package and CLI do not
  require them.
- MuJoCo is validated locally (`tests/test_physics.py`), but the physics job is
  a separate optional CI job.

## Reproducing

```bash
python -m pip install -e ".[dev,rl,physics]"
python -m pytest
python -m armlab demo ik --headless
python -m armlab benchmark --suite default --config configs/benchmark/default.yaml \
  --output-dir results/smoke-default
python -m armlab report --results results/smoke-default
```
