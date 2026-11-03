# Original demo projects

This folder contains the two historical demos that ArmLab grew out of. They are
kept runnable and intentionally small; the integrated, tested implementation is
in [`../src/armlab/`](../src/armlab/).

| # | project | demonstrates | entry point |
|:--|:--|:--|:--|
| 1 | [`analytical_inverse_kinematics/`](analytical_inverse_kinematics/) | closed-form IK for a 3-link arm tracking a moving 3D target | `python main.py` |
| 2 | [`rl_double_dueling_dqn/`](rl_double_dueling_dqn/) | a Dueling Double DQN agent learning discrete joint moves | `python train.py`, then `python test.py` |

Media from these demos is in [`screenshots/`](screenshots/).

## Running a demo

The demos use flat imports (`from robot import Robot`), so run them from their
own folder:

```bash
cd projects/analytical_inverse_kinematics
python main.py
```

```bash
cd projects/rl_double_dueling_dqn
python train.py     # writes qnetwork.pth
python test.py      # loads qnetwork.pth and renders the policy
```

Requirements: `numpy`, `matplotlib`, `gymnasium` and, for the DQN demo,
`torch`. Install the repository extras with `python -m pip install -e ".[rl]"`.

## Relationship to the integrated package

The original scripts duplicate pieces of kinematics, environments and
visualisation. In `src/armlab` those concerns are consolidated:

- `armlab.kinematics.RobotKinematics` is the single FK/IK/Jacobian source of
  truth.
- `armlab.legacy.Robot` keeps the original class/method names but delegates to
  the tested implementation.
- `armlab.rl` contains the from-scratch Dueling Double DQN; SAC is provided
  through Stable-Baselines3.
- `armlab.evaluation` writes the versioned run artifacts used by the README
  figures.

For the full benchmark, controller comparison and RL results, start at the
[top-level README](../README.md).
