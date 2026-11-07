"""Turn run directories into README media (GIFs, stills, plots) and a regenerated README.

Everything shown is produced from saved policy snapshots and ``metrics.csv`` - no hand-drawn
figures. ``build_readme`` stitches the new top section together with the untouched legacy README.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .config import EVAL_SEED_BASE, RunConfig, load_config
from .media import (
    caption,
    filmstrip,
    hold,
    load_policy,
    make_env,
    pick_frames,
    record_episode,
    write_gif,
    write_mp4,
)

# Reference categorical palette (validated: first three slots pass all-pairs CVD/normal-vision gates).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ------------------------------------------------------------------------------ stage media
def stage_label(step: int, total: int) -> str:
    if step == 0:
        return "untrained"
    return "final" if step >= total else f"{step / total:.0%} of training"


def render_task_media(run_dir: str | Path, out_dir: str | Path, cameras=("hero",), clips_per_stage: int = 2, stats_episodes: int = 30, size=(480, 360), render_every: int = 2, device: str = "cpu", log=print) -> dict:
    """Render captioned GIFs/stills for every saved milestone of one run. Returns the manifest."""
    from .rl.evaluate import evaluate_policy

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cfg = load_config(run_dir / "config.yaml")
    total = cfg.train.total_steps
    snaps = sorted((run_dir / "milestones").glob("step_*.pt"))
    if not snaps:
        raise FileNotFoundError(f"no milestone snapshots in {run_dir / 'milestones'}")
    # The last snapshot in the directory is the final published policy (normally the end of
    # training; for a "best checkpoint" publish run it is the best evaluated snapshot).
    final_step = int(snaps[-1].stem.split("_")[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(cfg)
    task = cfg.env.task.name
    thr = cfg.env.task.success_threshold
    manifest: dict = {"task": task, "run": run_dir.name, "total_steps": total, "stages": []}
    progression: list[np.ndarray] = []
    finals, mids = [], []
    try:
        for snap in snaps:
            step = int(snap.stem.split("_")[1])
            agent = load_policy(snap, cfg, device)
            act = lambda o, a=agent: a.act(o["observation"], o["desired_goal"], deterministic=True)  # noqa: E731
            stats = evaluate_policy(cfg.env, act, stats_episodes, seed0=EVAL_SEED_BASE, goal_scale=1.0, env=env)
            label = "final" if step == final_step else stage_label(step, total)
            title = f"{task} | {step:,} steps ({label})"
            sub = f"eval success {stats['success_rate']:.0%}"
            log(f"[media] {task} step {step:,}: success {stats['success_rate']:.2f} over {stats_episodes} episodes")
            stage_files = {}
            for cam in cameras:
                stage_frames: list[np.ndarray] = []
                for k in range(clips_per_stage):
                    clip = record_episode(env, act, EVAL_SEED_BASE + k, cam, size, render_every)
                    cap = [caption(f, title, sub, d, s, thr) for f, d, s in zip(clip["frames"], clip["dist"], clip["success"], strict=False)]
                    stage_frames += hold(cap, 4)
                    if k == 0 and cam == cameras[0]:
                        finals.append((cap[-1], f"{step:,} steps - {stats['success_rate']:.0%}"))
                        mids.append((pick_frames(cap, 3)[1], f"{step:,} steps"))
                name = f"stage_{step:09d}_{cam}"
                write_gif(stage_frames, out_dir / f"{name}.gif", fps=1000 / (40 * render_every))
                write_mp4(stage_frames, run_dir / "videos" / f"{name}.mp4", fps=1000 / (40 * render_every))  # not committed
                stage_files[cam] = f"{name}.gif"
                if cam == cameras[0]:
                    progression += stage_frames
            manifest["stages"].append({"step": step, "label": label, "success_rate": stats["success_rate"], "mean_final_distance": stats["mean_final_distance"], "episodes": stats_episodes, "files": stage_files})
    finally:
        env.close()
    write_gif(progression, out_dir / "progression.gif", fps=1000 / (40 * render_every), colors=80)
    filmstrip([f for f, _ in finals], [t for _, t in finals], out_dir / "stages_end.png")
    filmstrip([f for f, _ in mids], [t for _, t in mids], out_dir / "stages_mid.png")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ------------------------------------------------------------------------------ plots
def read_metrics(path: str | Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if v not in ("", None) else None) for k, v in r.items()})
    return rows


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def plot_learning_curves(run_dir: str | Path, out_png: str | Path, milestones: list[int] | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    rows = read_metrics(run_dir / "metrics.csv")
    cfg = load_config(run_dir / "config.yaml")
    steps = lambda key: ([r["steps"] for r in rows if r.get(key) is not None], [r[key] for r in rows if r.get(key) is not None])  # noqa: E731
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True, facecolor=SURFACE, gridspec_kw={"height_ratios": [1.5, 1, 0.8]})

    ax = axes[0]
    _style(ax)
    x, y = steps("eval_success")
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=4, label="evaluation (full difficulty)")
    x2, y2 = steps("train_success")
    ax.plot(x2, y2, color=ORANGE, lw=2, label="training (curriculum goals)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("success rate", color=MUTED)
    ax.set_title(f"{cfg.env.task.name}: learning curve (TQC-SAC + HER)", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=MUTED)
    if x:
        ax.annotate(f"{y[-1]:.0%}", (x[-1], y[-1]), textcoords="offset points", xytext=(6, -3), color=MUTED, fontsize=9)

    ax = axes[1]
    _style(ax)
    x, y = steps("eval_final_distance")
    ax.plot(x, [v * 100 for v in y], color=BLUE, lw=2, marker="o", ms=4)
    ax.axhline(cfg.env.task.success_threshold * 100, color=MUTED, lw=1)
    ax.text(0.01, cfg.env.task.success_threshold * 100, " success threshold", color=MUTED, fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
    ax.set_ylabel("final goal distance (cm)", color=MUTED)

    ax = axes[2]
    _style(ax)
    x, y = steps("goal_scale")
    ax.step(x, y, where="post", color=AQUA, lw=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("curriculum", color=MUTED)
    ax.set_xlabel("environment steps", color=MUTED)
    if x:
        ax.annotate("goal range", (x[-1], y[-1]), textcoords="offset points", xytext=(-4, 6), color=MUTED, fontsize=8, ha="right")

    for m in milestones or []:
        for a in axes:
            a.axvline(m, color=GRID, lw=1, zorder=0)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def plot_stage_success(manifest: dict, out_png: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = manifest["stages"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
    _style(ax)
    xs = np.arange(len(stages))
    vals = [s["success_rate"] for s in stages]
    ax.bar(xs, vals, width=0.5, color=BLUE)
    for x, v in zip(xs, vals, strict=False):
        ax.text(x, v + 0.025, f"{v:.0%}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(xs, [f"{s['step'] / 1000:g}k" if s["step"] else "0" for s in stages])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("success rate", color=MUTED)
    ax.set_xlabel("environment steps at snapshot", color=MUTED)
    ax.set_title(f"{manifest['task']}: policy snapshots, {stages[0]['episodes']} held-out episodes each", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------------------ README
_LINK = re.compile(r"(\]\()(?!https?://|#|mailto:|/)([^)\s]+)")


def relocate_links(markdown: str, prefix: str) -> str:
    """Prefix relative markdown link/image targets so the moved README still resolves."""
    return _LINK.sub(lambda m: f"{m.group(1)}{prefix}{m.group(2)}", markdown)


LEGACY_NOTE = (
    "> **Everything below is the original project, kept intact.** The code moved from the repository\n"
    "> root into [`legacy/`](legacy/); relative links below were re-pointed there, but the commands in\n"
    "> code blocks are unchanged - run them from inside `legacy/` (e.g. `cd legacy && pip install -e .`).\n"
    "> The legacy README is reproduced verbatim after this line.\n"
)


def legacy_section(legacy_readme: str) -> str:
    return "\n---\n\n# Legacy: ArmLab RL (original project)\n\n" + LEGACY_NOTE + "\n" + relocate_links(legacy_readme, "legacy/")


# ------------------------------------------------------------------------------ extras
def render_scene_gallery(cfg: RunConfig, out_png: str | Path, cameras=("front", "side", "top", "close"), size=(400, 300)) -> Path:
    """Stills of the generated scene from several cameras (proof of what the physics world is)."""
    env = make_env(cfg)
    try:
        env.reset(seed=EVAL_SEED_BASE)
        imgs = [env.render(c, *size) for c in cameras]
    finally:
        env.close()
    return filmstrip(imgs, [f"camera: {c}" for c in cameras], out_png)


def plot_comparison(run_dirs: list[str | Path], labels: list[str], out_png: str | Path, title: str = "ablation") -> Path:
    """Evaluation success vs steps for several runs on one axis (one colour per run, fixed order)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [BLUE, ORANGE, AQUA]
    if len(run_dirs) > len(colors):
        raise ValueError("plot_comparison supports at most 3 runs (validated categorical slots)")
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    _style(ax)
    for rd, lab, col in zip(run_dirs, labels, colors, strict=False):
        rows = [r for r in read_metrics(Path(rd) / "metrics.csv") if r.get("eval_success") is not None]
        ax.plot([r["steps"] for r in rows], [r["eval_success"] for r in rows], color=col, lw=2, label=lab)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("environment steps", color=MUTED)
    ax.set_ylabel("evaluation success", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def publish_task(run_dir: str | Path, repo_root: str | Path, cameras=("hero",), clips_per_stage: int = 2, stats_episodes: int = 30, out_name: str | None = None, log=print) -> dict:
    """Render media + plots for one run into ``docs/media/<task>/`` and return its manifest.

    ``out_name`` overrides the directory (used by the cluttered variant, whose task is still
    ``pickplace`` but whose results belong in ``docs/media/pickplace_clutter``).
    """
    run_dir, repo_root = Path(run_dir), Path(repo_root)
    cfg = load_config(run_dir / "config.yaml")
    task = cfg.env.task.name
    out = repo_root / "docs" / "media" / (out_name or task)
    manifest = render_task_media(run_dir, out, cameras=cameras, clips_per_stage=clips_per_stage, stats_episodes=stats_episodes, log=log)
    plot_learning_curves(run_dir, out / "learning_curve.png", milestones=[s["step"] for s in manifest["stages"]])
    plot_stage_success(manifest, out / "stage_success.png")
    summary = run_dir / "summary.json"
    if summary.exists():
        (out / "summary.json").write_text(summary.read_text())
    (out / "config.yaml").write_text((run_dir / "config.yaml").read_text())
    return manifest
