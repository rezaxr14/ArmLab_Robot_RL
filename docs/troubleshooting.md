# Troubleshooting

## Installation

### `ModuleNotFoundError: No module named 'armlab'`

Install the package (editable for development):

```bash
python -m pip install -e ".[dev]"
```

`pip` needs to see `pyproject.toml`; run the command from the repository root.

### `pip install -e ".[physics]"` fails on MuJoCo

MuJoCo wheels require a recent `pip`. Upgrade first:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[physics]"
```

If you do not need the simulator, stay with the base install. The kinematic
environment, controllers, evaluation and tests need no MuJoCo.

### MuJoCo rendering / OpenGL errors on a headless machine

Pass `--headless` to `demo physics`; the physics env advances without a viewer.
Headless offscreen rendering may still be unavailable depending on the OpenGL
setup — that only affects video capture, not simulation or evaluation.

### SAC reports missing dependencies

`train sac` prints an install hint and exits with code `3` when `torch` /
`stable-baselines3` are absent:

```bash
python -m pip install -e ".[rl]"
```

### Dashboard exits with code `3`

The dashboard needs Dash and Plotly:

```bash
python -m pip install -e ".[ui]"
python -m armlab dashboard --results results
```

## Units

Lengths in the kinematic model are in **legacy units** (`link_lengths =
[5, 10, 8]`), which the code treats as metres. `configs/robot/three_link.yaml`
documents the intended `legacy_unit_scale_m: 0.1` mapping, but the scale is not
applied anywhere. This is intentional so existing numbers keep their meaning —
see `docs/mathematical_conventions.md` before changing the geometry.

If you change link lengths, update the robot config **and** re-check the target
sampling ranges (`target_min_radius`, `target_max_height`, …), otherwise the
sampler may produce unreachable targets.

## Configuration errors

```
ValueError: Unknown key(s) ['...'] in configs/...; allowed keys are [...]
```

`configio` rejects unknown keys so typos fail loudly. Either fix the key name or
add the field to the corresponding dataclass in `src/armlab/config.py`.

```
ValueError: <field> must be a positive finite number, got ...
```

A dataclass received an invalid value. Limits and dimensionality are validated
in `__post_init__`; the message names the field.

## Environment behavior

### `check_env` complains about observation bounds

Observations are normalized to the declared space. If you add a new observation
field, extend `observation_space` and the version string
(`info["observation_version"]`) together, then re-run
`python -m pytest tests/test_env.py`.

### Targets look unreachable

The default scenario samples targets from valid configurations via FK. The
`boundary_reaching` scenario deliberately samples near the boundary and may
produce hard targets. Unreachable IK results are reported (`reachable=False`),
never returned as `NaN` angles.

## Training

### DQN/SAC success rate is ~0 after a short run

Short runs (`--max-steps`/`--total-timesteps` in the low thousands) are
**smoke tests**, not converged policies. The CLI prints an explicit note. Use
the full config values for a real run and expect it to take substantially
longer.

### Replay GIF is not written

`replay --gif` needs Pillow. Without it the NPZ trajectory is still written and
is the authoritative artifact; the command returns non-zero with a message.

```bash
python -m pip install pillow
```

### Dashboard shows an empty state

That is expected when `results/` has no runs. Produce one first:

```bash
python -m armlab evaluate --scenario static_reaching --controller ik --episodes 5
```

The dashboard shows a helpful empty state rather than fabricated sample
metrics.

### Results look non-deterministic

Classical controllers are deterministic for a fixed config and seed; re-running
an identical `evaluate` command reproduces metrics within numerical tolerance.
RL policies depend on the checkpoint and the recorded seed. GPU training can be
non-deterministic; the CI and smoke workflows use CPU.
