# ArmForge design notes

## Why these choices

**Native MuJoCo, no Gymnasium.** The environment is a plain class with `reset / step / compute_reward / render`.
Goal-conditioned learning needs `compute_reward(achieved, desired)` to be vectorised so replay can relabel goals; a
thin custom API makes that the first-class path instead of an afterthought wrapper.

**UR5e + Robotiq 2F-85 from Menagerie, merged as text.** `scene.py` merges the two Menagerie XML files with
ElementTree (gripper names, meshes, materials and default classes get an `rq_` prefix, meshes become absolute paths,
the gripper is mounted at the UR5e `attachment_site`). The result is ordinary MJCF that is unit-tested without the
simulator and can be opened in the MuJoCo viewer (`python -m mujoco.viewer --mjcf <file>`; files are written to
`~/.cache/armforge/scenes/`). The assets are fetched at a pinned commit, not vendored (45 MB).

**Cartesian control through the real servos.** The policy commands a pinch-point displacement; damped-least-squares
IK turns it into joint position targets for the UR5e's force-limited PD actuators. The tool orientation is held
pointing down. Menagerie's servos are heavily over-damped (kd/kp = 0.2 s, so only ~18 % of a commanded joint move
happens in one 40 ms control step), which makes a hand that can never keep up with a scripted or learned controller.
Two things fix that without teleporting anything: the servo `kd` is scaled by `physics.servo_damping_scale` (default
0.25, a ~50 ms response, closer to a real UR joint controller), and the joint step requested by the IK is multiplied by a
lead gain derived from the model (`1 / fraction-of-a-step-achieved`, override with `physics.servo_lead_gain`). The
command is *not* persistent: it is the measured joints plus a lead of at most 0.15 rad, so a blocked arm cannot wind
up (an integrator wrapped around the servo lag oscillates, which an earlier version of this code did). Arm and gripper
bodies are gravity-compensated, as the UR controller does on hardware; the cube is not.

**Friction.** Elliptic cones with `impratio` 10, `condim` 4 on pads, cube and table (torsional friction), small rolling
friction. MuJoCo combines two geoms' friction with the per-component **max** unless their `priority` differs; the
Robotiq pad boxes have `priority 1`, so pad parameters win against the cube. Table-vs-cube therefore slides at
`max(table, cube)` sliding friction.

**TQC-SAC + HER.** Sparse `{-1, 0}` rewards need hindsight relabelling. TQC (pooled quantile critics, drop the top
quantiles) reduces over-estimation. Bootstrapping is never cut at the time limit (fixed-length episodes, truncation
only), and targets are clipped to the reachable return range. A success-gated curriculum widens the goal range;
evaluation always uses the full range, so reported success is never curriculum-inflated.

**DQN is deliberately not used.** Six coupled joints with continuous Cartesian commands would need a large discretised
action set; actor-critic methods are the standard fit. The original Dueling Double DQN stays in `legacy/`.

## If something does not work on first run

Run these in order; each isolates one layer.

1. `armforge doctor` - mujoco, torch/CUDA, Menagerie download, one render.
2. `python -m pytest -rs` - pure tests always run; simulator/torch tests run when those packages are installed.
3. `armforge expert --config configs/pickplace.yaml --episodes 20` - the scripted expert must succeed. If it does not,
   the problem is in the scene/controller (grasp height, pad friction, gripper force), not in learning.
4. `armforge train --config configs/smoke.yaml` - exercises the whole loop in seconds on CPU (does not learn).
5. `armforge train --config configs/reach.yaml` - the simplest real task; success should rise within the first
   ~100k steps. If it does not, inspect `runs/<name>/metrics.csv` (`alpha`, `critic_loss`, `q_mean`).

Knobs that matter most: `env.task.max_ee_step` (action scale), `env.task.success_threshold`, `algo.her_prob`,
`algo.updates_per_step` (update-to-data ratio), `train.curriculum.*`, `train.demos.*` (pick-and-place), and
`env.physics.pad_friction` / gripper force (grasp reliability). Step budgets in `configs/` are estimates, not tuned.

## What the tests check

* Pure Python (always run): config round-trip and typo rejection, MJCF composition invariants against the real
  Menagerie files (unique names, resolved references, existing meshes, namespacing), a numpy FK oracle for the composed
  arm (including DLS IK reachability of the whole workspace with the tool pointing down), media/GIF tooling, README
  generation (legacy README embedded verbatim, links re-pointed), notebook syntax/imports/undefined-name checks.
* Simulator (`mujoco`): model structure, friction values reach the model, determinism, controller tracking, no gravity
  drift, cube resting on the table, gripper open/close, domain randomisation restore, friction changes a slide
  distance, rendering, and scripted-expert success per task.
* Learning (`torch`): HER relabelling uses true future achieved goals, tanh log-prob, quantile Huber loss, twin-critic
  target equals the pairwise minimum, a critic converges to a known value, behaviour cloning fits an expert, policy
  save/load, and a full trainer smoke run with resume.

**CPU forward-kinematics cross-check.** `armforge fk-check` (and notebook cell 3b) compares two independent forward-kinematics
implementations on the same joint angles: `kinematics.XmlKinematics`, a pure-numpy walk of the generated MJCF tree that shares no code
with MuJoCo, and MuJoCo's own `mj_kinematics`. It covers random joint vectors within the limits, every joint state a scripted-expert
episode visits, and the environment's own `ee_pos`. Both are exact rigid-body chains, so they should agree to ~1e-9 m; a wrong
attachment frame, joint axis or base placement shows up as a millimetre-scale disagreement. The numpy walker is a checker, not a fast
path (it is slower per call than MuJoCo).
