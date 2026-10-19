"""Generate curated README images from *measured* run artifacts.

This script never invents numbers: it reads ``training_metrics.json``,
``sac_metrics_seed*.json``, ``trajectory_*.npz`` and run summaries produced by
``python -m armlab train ...`` / ``evaluate`` and renders PNGs.

Examples
--------
    python scripts/make_readme_assets.py \
        --dqn-run results/dqn/dqn_seed7 \
        --sac-metrics results/sac/sac_metrics_seed7.json \
        --trajectory-run results/<run-id> \
        --out docs/assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _robot_params():
    from armlab.config import RobotParams
    from armlab.configio import load_robot_params

    for candidate in ("configs/robot/three_link.yaml",):
        path = Path(candidate)
        if path.is_file():
            return load_robot_params(path)
    return RobotParams()


def plot_dqn_curve(run_dir: Path, out: Path) -> Path | None:
    from armlab.evaluation.plots import plot_training_curves

    return plot_training_curves(run_dir, out)


def plot_sac_curve(metrics_path: Path, out: Path) -> Path:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = payload.get("training_history") or []
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if history:
        steps = [h["timesteps"] for h in history]
        rewards = [h["mean_reward"] for h in history]
        ax.plot(steps, rewards, color="C2", label="mean episode reward (rolling 50)")
    ax.set_xlabel("environment timesteps")
    ax.set_ylabel("reward")
    ax.set_title(
        f"SAC training (seed {payload.get('seed')}, {payload.get('total_timesteps'):,} timesteps)"
    )
    ax.grid(alpha=0.3)
    if history:
        ax.legend()
    eval_metrics = payload.get("eval", {})
    if eval_metrics:
        ax.text(
            0.02,
            0.97,
            f"final eval success: {eval_metrics.get('success_rate', float('nan')):.2f}",
            transform=ax.transAxes,
            va="top",
        )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _draw_pose(ax, links: np.ndarray, color: str, label: str | None = None) -> None:
    ax.plot(links[:, 0], links[:, 1], links[:, 2], "-o", lw=3, ms=5, color=color, label=label)


def plot_scene(run_dir: Path, out: Path, episode: int = 0) -> Path:
    """Three-panel start/middle/final view of one measured reaching episode."""
    from armlab.kinematics import RobotKinematics

    files = sorted(run_dir.glob("trajectory_*.npz"))
    if not files:
        raise FileNotFoundError(f"No trajectory_*.npz in {run_dir}")
    index = min(max(episode, 0), len(files) - 1)
    with np.load(files[index]) as data:
        ee = np.asarray(data["ee_positions"], dtype=float)
        q = np.asarray(data["q_positions"], dtype=float)
        target = np.asarray(data["target"], dtype=float) if "target" in data else None

    kin = RobotKinematics(_robot_params())
    poses = np.stack([kin.link_positions(qi) for qi in q])
    n = len(q)
    picks = [0, n // 2, n - 1]

    plt = _plt()
    fig = plt.figure(figsize=(13, 4.6))
    all_pts = np.vstack([poses.reshape(-1, 3), target[None, :] if target is not None else ee])
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    pad = 0.08 * max(1.0, float(np.max(hi - lo)))
    for panel, idx in enumerate(picks, start=1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="0.6", lw=1, ls="--", label="ee path")
        ax.plot(
            ee[: idx + 1, 0],
            ee[: idx + 1, 1],
            ee[: idx + 1, 2],
            color="C0",
            lw=2,
            label="progress",
        )
        _draw_pose(ax, poses[idx], "C0", "arm")
        ax.scatter(*poses[idx][0], color="k", s=30, label="base")
        if target is not None:
            ax.scatter(*target, marker="*", s=260, color="C3", label="target")
            ax.plot(
                [poses[idx][3, 0], target[0]],
                [poses[idx][3, 1], target[1]],
                [poses[idx][3, 2], target[2]],
                color="C3",
                ls=":",
                lw=1,
            )
        err = float(np.linalg.norm(poses[idx][3] - target)) if target is not None else float("nan")
        ax.set_title(f"step {idx} / {n - 1}   err={err:.3f}")
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_zlim(lo[2] - pad, hi[2] + pad)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Measured reaching rollout: arm, end-effector path and target")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_benchmark(runs_root: Path, out: Path) -> Path | None:
    from armlab.evaluation.artifacts import discover_runs
    from armlab.evaluation.plots import plot_controller_comparison

    runs = discover_runs(runs_root)
    if not runs:
        return None
    return plot_controller_comparison(runs, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dqn-run", type=Path, default=None)
    parser.add_argument("--sac-metrics", type=Path, default=None)
    parser.add_argument("--trajectory-run", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/assets"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    if args.dqn_run:
        path = plot_dqn_curve(args.dqn_run, args.out / "training_dqn.png")
        if path:
            produced.append(path)
    if args.sac_metrics:
        produced.append(plot_sac_curve(args.sac_metrics, args.out / "training_sac.png"))
    if args.trajectory_run:
        produced.append(plot_scene(args.trajectory_run, args.out / "arm_reaching.png"))
    if args.benchmark_root:
        path = plot_benchmark(args.benchmark_root, args.out / "benchmark_success.png")
        if path:
            produced.append(path)

    for path in produced:
        print(f"wrote {path}")
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
