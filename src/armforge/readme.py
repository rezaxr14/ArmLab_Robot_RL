"""Generates the new top section of README.md from the code's own defaults and published media.

The physics and hyper-parameter tables are read from the config dataclasses (so they cannot drift
from the code), and every result number/figure comes from ``docs/media/<task>/``. If no run has
been published yet, the Results section says so instead of showing anything made up.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import AlgoConfig, EnvConfig, TaskConfig, TrainConfig
from .report import legacy_section

REPO = "rezaxr14/ArmLab_Robot_RL"
BRANCH = "main"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/notebooks/armforge_control_panel.ipynb"

TASK_BLURBS = {
    "reach": (
        "Reach",
        "Move the Robotiq pinch point to a random 3D goal in the air. The warm-up task: it proves the "
        "controller, the contact-free dynamics and the learning stack work before objects are involved.",
    ),
    "push": (
        "Push",
        "Slide a free 40 mm cube to a goal on the table with the **closed** gripper held 2 cm above the surface. "
        "The cube is moved only by contact friction, so sliding, torsional and rolling friction decide whether "
        "a push goes straight, spins the cube or tips it.",
    ),
    "pickplace": (
        "Pick and place",
        "Grasp the cube with the 2F-85, lift it, and put it at a goal on the table, in the air, or (in the "
        "cluttered variant) inside a walled tray among loose distractor cubes. Needs a real grasp: pad friction, "
        "finger linkage compliance and grip force all matter.",
    ),
    "pickplace_clutter": (
        "Pick and place (cluttered)",
        "The same grasp-and-place task in the hard scene: a walled tray as the place target, two loose distractor "
        "cubes that can be knocked around, and full domain randomisation.",
    ),
}


def _fmt_tuple(t) -> str:
    return "(" + ", ".join(f"{v:g}" for v in t) + ")"


def physics_table() -> str:
    e = EnvConfig()
    p, t = e.physics, e.task
    rows = [
        ("Simulator / integrator", f"MuJoCo, `{p.integrator}`, dt = {p.timestep * 1000:g} ms, {p.substeps} substeps per control step -> {1 / (p.timestep * p.substeps):g} Hz policy"),
        ("Friction cone", f"`{p.cone}`, `impratio` {p.impratio:g} (no linearised-pyramid friction artefacts)"),
        ("Table friction (sliding, torsional, rolling)", f"{_fmt_tuple(p.table_friction)}"),
        ("Cube friction (sliding, torsional, rolling)", f"{_fmt_tuple(p.object_friction)}; MuJoCo uses the per-component **max** of two touching geoms, so cube-on-table slides at mu = {max(p.table_friction[0], p.object_friction[0]):g}"),
        ("Gripper pad friction", f"{p.pad_friction:g} sliding, contact dimension 4 (adds torsional friction), pad priority overrides the cube's values"),
        ("Cube", f"{2 * p.object_half_size * 1000:g} mm, {p.object_mass * 1000:g} g, `condim` {p.object_condim}"),
        ("Contact softness", f"`solref` {_fmt_tuple(p.solref)} (pads {_fmt_tuple(p.pad_solref)}), `solimp` {_fmt_tuple(p.solimp)}"),
        ("Arm actuators", f"UR5e position servos from Menagerie: kp 2000 / 500 N m/rad, force limits 150 / 28 N m; servo damping kd scaled x{p.servo_damping_scale:g} (the stock servo is over-damped: 0.2 s time constant)"),
        ("Arm joint damping + Coulomb friction", f"{p.arm_joint_damping:g} N m s/rad + {p.arm_frictionloss:g} N m `frictionloss` on every joint (added on top of the model's armature)"),
        ("Gravity", f"{p.gravity:g} m/s^2; the arm and gripper bodies are gravity-compensated (`arm_gravity_comp`, as the UR controller does on hardware) - the cube is not"),
        ("Gripper", "Robotiq 2F-85: 4-bar linkage with equality constraints, spring links, tendon-coupled actuator, 5 N force limit"),
        ("Domain randomisation (optional)", f"friction x{_fmt_tuple(p.randomize.friction_scale)}, cube mass x{_fmt_tuple(p.randomize.mass_scale)}, joint damping x{_fmt_tuple(p.randomize.damping_scale)}, actuator kp x{_fmt_tuple(p.randomize.kp_scale)}"),
        ("Episode", f"{t.horizon} control steps = {t.horizon * p.timestep * p.substeps:g} s of simulated time; success radius {t.success_threshold * 100:g} cm"),
    ]
    return "| quantity | value |\n|:--|:--|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def algo_table() -> str:
    a = AlgoConfig()
    rows = [
        ("Algorithm", "SAC with truncated quantile critics (TQC); `critic: twin` gives classic clipped double-Q SAC for ablations"),
        ("Critic", f"{a.critic_ensemble} critics x {a.n_quantiles} quantiles, drop top {a.top_quantiles_to_drop} per critic, LayerNorm = {a.layer_norm}"),
        ("Networks", f"MLP {_fmt_tuple(a.hidden)} for actor and critics, tanh-Gaussian policy, automatic temperature"),
        ("Hindsight replay", f"`future` strategy, relabel probability {a.her_prob:g}, sparse reward {{-1, 0}}"),
        ("Bootstrap", f"gamma {a.gamma:g}, targets clipped to the reachable return range, no bootstrap cut at the time limit"),
        ("Optimisation", f"Adam {a.actor_lr:g}, batch {a.batch_size}, {a.updates_per_step:g} gradient steps per environment step, soft target update tau {a.tau:g}"),
        ("Curriculum", "goals start near the object and spread out only as the rolling success rate rises; **evaluation always uses the full goal range**"),
        ("Demonstrations (optional)", "scripted experts seed the replay buffer and warm-start the actor by behaviour cloning"),
    ]
    return "| component | setting |\n|:--|:--|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _results_for(task: str, media_root: Path) -> str:
    title, blurb = TASK_BLURBS[task]
    base = media_root / task
    manifest = _load(base / "manifest.json")
    out = [f"### {title}", "", blurb, ""]
    if manifest is None:
        out += [f"> *No published run for `{task}` yet - run the notebook's **Publish** cell to generate the media here.*", ""]
        return "\n".join(out)
    rel = f"docs/media/{task}"
    stages = manifest["stages"]
    first, last = stages[0], stages[-1]
    out += [
        f"**Untrained policy: {first['success_rate']:.0%} success. After {last['step']:,} steps: {last['success_rate']:.0%}** "
        f"(deterministic policy, {last['episodes']} held-out episodes at the *full* goal range, mean final goal distance "
        f"{last['mean_final_distance'] * 100:.1f} cm).",
        "",
        f"![{task} progression]({rel}/progression.gif)",
        "",
        "*One continuous clip: the same start/goal seeds replayed by policy snapshots from step 0 to the end of training.*",
        "",
        f"![{task} stages, end of episode]({rel}/stages_end.png)",
        "",
        f"![{task} stages, mid episode]({rel}/stages_mid.png)",
        "",
        "| snapshot (env steps) | stage | success | mean final distance |",
        "|--:|:--|--:|--:|",
    ]
    for s in stages:
        out.append(f"| {s['step']:,} | {s['label']} | {s['success_rate']:.0%} | {s['mean_final_distance'] * 100:.1f} cm |")
    n = last["episodes"]
    se = (last["success_rate"] * (1 - last["success_rate"]) / max(n, 1)) ** 0.5
    out += ["", f"*Each snapshot row is {n} held-out episodes: one success rate carries about +/-{se * 100:.0f} points (1 standard error).*", ""]
    out += ["", f"![{task} learning curve]({rel}/learning_curve.png)", "", f"![{task} snapshot success]({rel}/stage_success.png)", ""]
    summary = _load(base / "summary.json")
    if summary:
        out += [
            f"Run: {summary['total_steps']:,} environment steps ({summary['episodes']:,} episodes) in {summary['wall_time_s'] / 60:.0f} min on `{summary['device']}`"
            + (f", plus {summary['demo_steps']:,} scripted demonstration steps." if summary.get("demo_steps") else "."),
            "",
        ]
    out += ["<details><summary>Per-snapshot clips</summary>", ""]
    for s in stages:
        for cam, f in s["files"].items():
            out += [f"**{s['step']:,} steps ({s['label']})** - `{cam}` camera, {s['success_rate']:.0%} success", "", f"![]({rel}/{f})", ""]
    out += ["</details>", ""]
    return "\n".join(out)


def top_section(media_root: str | Path = "docs/media") -> str:
    media_root = Path(media_root)
    gallery = media_root / "scene_gallery.png"
    have_any = any((media_root / t / "manifest.json").exists() for t in TASK_BLURBS)
    hero = ""
    for t in ("pickplace", "push", "reach"):
        if (media_root / t / "progression.gif").exists():
            hero = f"![ArmForge hero](docs/media/{t}/progression.gif)\n"
            break
    t, tr = TaskConfig(), TrainConfig()
    parts = [
        "# ArmForge: UR5e + Robotiq 2F-85 manipulation in native MuJoCo, learned with TQC-SAC + HER",
        "",
        "**A real 6-DoF robot arm (Universal Robots UR5e from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)) "
        "with a real two-finger gripper (Robotiq 2F-85), simulated in a custom MuJoCo environment with friction cones, joint damping, "
        "force-limited position servos and a 4-bar gripper linkage - and a from-scratch PyTorch RL stack that learns reaching, pushing "
        "and pick-and-place from sparse rewards.** No Gymnasium, no toy kinematics: contacts and actuator limits are in the loop.",
        "",
        hero,
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})",
        "",
        "The whole project is driven from one notebook - [`notebooks/armforge_control_panel.ipynb`](notebooks/armforge_control_panel.ipynb): "
        "configure, inspect the scene, train, monitor, evaluate, render the *failing -> learning -> solved* media, and regenerate this README.",
        "",
        "## Contents",
        "",
        "1. [Results](#results)  2. [The robot and the physics](#the-robot-and-the-physics)  3. [How it learns](#how-it-learns)  "
        "4. [Run it](#run-it)  5. [Repository layout](#repository-layout)  6. [Limitations](#limitations)  7. [Licence](#license-and-third-party-assets)  8. [Legacy project](#legacy-armlab-rl-original-project)",
        "",
        "## Results",
        "",
    ]
    if not have_any:
        parts += [
            "> **No training run has been published in this checkout yet.** The sections below fill in automatically - with real GIFs of early "
            "failures, mid-training attempts and the final policy, learning curves and success tables - when you run the notebook's "
            "*Publish* cell (or `armforge publish`). Nothing here is a mock-up.",
            "",
        ]
    for task in ("reach", "push", "pickplace"):
        parts.append(_results_for(task, media_root))
    clutter = media_root / "pickplace_clutter"
    if (clutter / "manifest.json").exists():
        parts.append(_results_for("pickplace_clutter", media_root))
    parts += [
        "## The robot and the physics",
        "",
        (f"![scene gallery](docs/media/scene_gallery.png)\n" if gallery.exists() else "*Scene gallery: written to `docs/media/scene_gallery.png` by the notebook's Media cell or `armforge scene`.*\n"),
        "The scene is generated programmatically ([`scene.py`](src/armforge/scene.py)) from the pinned Menagerie models: the UR5e and 2F-85 XML are "
        "merged (names, classes, meshes and contact exclusions are namespaced), the gripper is mounted on the flange, and a table, optional walled "
        "tray, free cubes, a mocap goal marker and five cameras are added. The meshes are downloaded once into a cache (about 45 MB), not vendored.",
        "",
        physics_table(),
        "",
        f"**Control.** The policy outputs a Cartesian displacement of the gripper's pinch point (up to {t.max_ee_step * 100:g} cm per step) and, for pick-and-place, "
        "a gripper command. Damped-least-squares differential IK holds the tool pointing straight down and turns that into joint targets for the arm's *actual* "
        "PD servos - so actuator lag, torque saturation, gravity and contact loads act on every motion; nothing is teleported.",
        "",
        "**Observation** (goal-conditioned): pinch position and velocity, gripper opening, six joint angles and velocities, and - for object tasks - cube position, "
        "cube relative to the hand, orientation, linear/angular velocity, and left/right pad contact flags. The desired and achieved goals are separate vectors, "
        "which is what hindsight relabelling needs.",
        "",
        "## How it learns",
        "",
        algo_table(),
        "",
        "```text",
        "  N parallel MuJoCo workers  --episodes-->  episode buffer (GPU)  --(s, a, goal)-->  TQC-SAC",
        "        ^                                       |  HER: replace goal with an achieved future state, recompute reward",
        "        |                                       v",
        "   policy (actor)  <------------------ gradient updates (UTD ratio)      curriculum widens goals on success",
        "```",
        "",
        "Why these pieces: sparse rewards give no signal until the arm happens to succeed, so **HER** turns every failed episode into a success for "
        "the goal it actually reached; **TQC** curbs the value over-estimation that makes sparse-reward off-policy RL diverge; the **curriculum** keeps goals "
        "near enough to hit by accident at first; **scripted demonstrations** (optional) give pick-and-place a first successful grasp to learn from.",
        "",
        "## Run it",
        "",
        f"**Colab (GPU):** open the badge above, `Runtime -> Change runtime type -> GPU`, then *Run all* and edit the form fields in the *Configure* cell.",
        "",
        "**Local:**",
        "",
        "```bash",
        "git clone https://github.com/" + REPO + ".git && cd ArmLab_Robot_RL",
        "python -m pip install -e \".[dev]\"          # numpy, mujoco, torch, imageio, matplotlib, ...",
        "armforge doctor                               # checks mujoco, torch/CUDA, rendering and the Menagerie download",
        "armforge scene --task pickplace --out docs/media/scene_gallery.png",
        "armforge pack --run runs/reach-demo           # zip a run's checkpoints + media",
        "armforge fk-check                             # CPU (numpy) forward kinematics vs MuJoCo's FK: should agree to ~1e-9 m",
        "armforge train --config configs/reach.yaml --name reach-demo",
        "armforge publish --run runs/reach-demo        # GIFs, stills, curves -> docs/media/ ; then: armforge readme",
        "```",
        "",
        "Config presets live in [`configs/`](configs/) (`reach`, `push`, `pickplace`, `pickplace_clutter`, `smoke`); any field can be overridden with "
        "`--set train.total_steps=500000 env.physics.pad_friction=0.8`. Rendering needs an OpenGL backend: EGL by default on headless Linux "
        "(`MUJOCO_GL=osmesa` for software rendering).",
        "",
        "## Repository layout",
        "",
        "```text",
        "src/armforge/   config, scene (MJCF composition), kinematics oracle + fkcheck (CPU FK vs MuJoCo), env/ (MuJoCo env, tasks, controller, vec),",
        "                rl/ (agent, networks, HER buffer, trainer, experts), media, report, readme, cli",
        "configs/        run presets",
        "notebooks/      armforge_control_panel.ipynb - the control panel",
        "tests/          pure-python tests + simulator/torch tests (skipped when those packages are absent)",
        "docs/media/     generated GIFs, stills and plots shown above",
        "legacy/         the previous ArmLab project, moved here unchanged (see the section at the bottom)",
        "```",
        "",
        "## Limitations",
        "",
        "- Simulation only; no sim-to-real claim. Domain randomisation is provided, but transfer has not been tested.",
        "- The tool orientation is held fixed (pointing down), so grasps are top-down; no wrist-yaw control, so cubes are spawned within +/-0.35 rad of the gripper axis for pick-and-place.",
        "- Success is the cube/pinch within a radius of the goal (position only, not orientation).",
        "- Results are single-seed unless you run more seeds; one seed is not a benchmark.",
        f"- Defaults ({tr.total_steps:,}-step budget, 4 workers) are sized for a single Colab GPU session and have not been tuned.",
        "",
        "## License and third-party assets",
        "",
        "ArmForge is MIT-licensed (see `pyproject.toml`). The UR5e and Robotiq 2F-85 models come from "
        "[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (both are BSD-style licensed from ROS-Industrial; each model "
        "folder carries its own `LICENSE`). They are **downloaded at first use, pinned to a commit, and not redistributed** in this repository. "
        "MuJoCo is Apache-2.0; PyTorch, NumPy, Pillow, Matplotlib and imageio keep their own licences. Design notes: [`docs/design.md`](docs/design.md).",
        "",
    ]
    return "\n".join(p for p in parts if p is not None)


def build_readme(legacy_readme: str, media_root: str | Path = "docs/media") -> str:
    return top_section(media_root).rstrip() + "\n\n" + legacy_section(legacy_readme).rstrip() + "\n"
