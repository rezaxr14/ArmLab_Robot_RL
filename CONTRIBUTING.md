# Contributing to ArmLab

Thanks for helping improve ArmLab. This guide covers the development workflow
and the conventions that keep the math, environments and results consistent.

## Setup

```bash
git clone https://github.com/rezaxr14/ArmLab_Robot_RL.git
cd ArmLab_Robot_RL
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional extras, installed only when you need them:

```bash
python -m pip install -e ".[rl]"        # torch + Stable-Baselines3 (SAC)
python -m pip install -e ".[physics]"   # MuJoCo
python -m pip install -e ".[ui]"        # Dash + Plotly dashboard
```

## Development workflow

```bash
python -m pytest                 # full suite (fast; no GPU/GUI required)
python -m pytest -q tests/test_kinematics.py
python -m ruff check src tests
python -m ruff format --check src tests   # formatting (optional locally)
python -m armlab doctor          # environment/dependency report
```

The base test suite must pass **without** GPU, display, MuJoCo, Dash or
network access. Tests that need an optional dependency use `pytest.importorskip`
and are skipped (not failed) when it is absent.

## Conventions

- **One source of truth for kinematics.** Never re-derive link positions or
  Jacobians outside `armlab.kinematics`. Envs, controllers, planners, metrics
  and visualizers must call it.
- **Units.** Angles in radians; lengths in the legacy units documented in
  `docs/mathematical_conventions.md`. Read that document before touching
  geometry.
- **Configuration.** Add tunables to the typed dataclasses in
  `src/armlab/config.py` and to the matching YAML in `configs/`. Validate in
  `__post_init__`; fail loudly on bad values.
- **Determinism.** Use explicit seeded `numpy.random.Generator`s or
  `env.reset(seed=...)`. Never rely on global RNG state.
- **Results are measured.** Never hardcode success rates, curves or benchmark
  numbers. Everything under `results/` is generated; the README table cites the
  command that produced it.
- **Optional imports are lazy.** Keep `mujoco`, `stable_baselines3`, `dash` and
  `plotly` imports inside functions or guarded modules so a base install stays
  usable.
- **Style.** Ruff with `line-length = 100`, rules `E, F, I, UP, B, W`. Public
  functions and dataclasses are typed and documented.

## Adding a controller

1. Implement a callable `policy(obs) -> action` factory in
   `evaluation/benchmark.py::make_policy` and add the name to `CONTROLLERS`.
2. Decide its action space: classical controllers run in the continuous env;
   discrete learners keep their native space.
3. Add a unit test for the controller and an integration test through
   `evaluate_run`.
4. Update `docs/architecture.md` and the README controller list.

## Adding a scenario

1. Add a builder to `scenarios.py` returning a `ScenarioBundle`.
2. Register it in `SCENARIOS`.
3. Ensure its sampler is seeded and only emits valid targets (or documents why
   it may emit hard ones).
4. Add tests and list it in `configs/benchmark/default.yaml`.

## Artifacts

Runs use the schema-1.0 layout described in `docs/architecture.md`. If you add a
field, keep `EPISODE_FIELDS`/`METRIC_FIELDS` in `evaluation/artifacts.py` and
`SCHEMA_VERSION` in sync, and extend the plots/dashboard to read the real data.

## Pull requests

- Keep changes focused and reviewable.
- Include tests for behavior changes.
- Run `pytest` and `ruff check` before opening the PR.
- Do not commit generated runs (`results/`), checkpoints or large media; they
  are gitignored on purpose.
