"""Plot generation from saved artifacts (implementation plan G4).

Every plot loads measured data from run artifacts; no values are hardcoded.
Functions use the non-interactive Agg backend and close their figures, so
batch report generation never needs a display.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "plot_training_curves",
    "plot_controller_comparison",
    "plot_error_distributions",
    "plot_trajectory_3d",
    "plot_joint_curves",
    "plot_tracking_error",
    "generate_run_report",
]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_training_curves(
    source: str | Path,
    output_path: str | Path | None = None,
) -> Path | None:
    """Plot DQN/SAC training progress from ``training_metrics.json``.

    ``source`` may be a training output directory, a DQN run directory
    (containing ``training_metrics.json``) or the JSON file itself.
    """
    source = Path(source)
    data = source if source.suffix == ".json" else source / "training_metrics.json"
    payload = _load_json(data)
    if payload is None:
        return None
    metrics = payload.get("metrics", {})
    scores = np.asarray(metrics.get("episode_scores", []), dtype=float)
    eval_steps = np.asarray(metrics.get("eval_steps", []), dtype=float)
    eval_sr = np.asarray(metrics.get("eval_success_rate", []), dtype=float)
    if scores.size == 0 and eval_sr.size == 0:
        return None

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if scores.size:
        window = min(50, max(1, scores.size // 10 or 1))
        rolling = np.convolve(scores, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(1, scores.size + 1), scores, alpha=0.3, label="episode return")
        axes[0].plot(
            np.arange(window, scores.size + 1),
            rolling,
            color="C1",
            label=f"rolling mean ({window})",
        )
        axes[0].set_xlabel("training episode")
        axes[0].set_ylabel("episode return")
        axes[0].legend()
    axes[0].set_title("Training return")

    if eval_sr.size:
        axes[1].plot(eval_steps, eval_sr, "o-", label="greedy eval success rate")
        axes[1].set_xlabel("environment steps")
        axes[1].set_ylabel("success rate")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].legend()
    axes[1].set_title("Evaluation success")

    fig.suptitle(f"Training progress (seed {payload.get('seed')}, smoke/partial unless stated)")
    fig.tight_layout()
    if output_path is None:
        output_path = (source.parent if source.suffix == ".json" else source) / "training_curve.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _summaries_of(runs: Sequence[Any]) -> list[dict[str, Any]]:
    from armlab.evaluation.artifacts import load_run

    out: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run)
        if (run_dir / "summary.json").is_file():
            loaded = load_run(run_dir)
            if loaded["summary"]:
                out.append({"run_dir": str(run_dir), **loaded["summary"]})
        elif isinstance(run, dict):
            out.append(run)
    return out


def plot_controller_comparison(
    runs: Sequence[Any],
    output_path: str | Path = "plots/success_comparison.png",
    metric: str = "success_rate",
) -> Path:
    """Bar chart comparing a measured metric across run summaries."""
    summaries = _summaries_of(runs)
    if not summaries:
        raise ValueError("No run summaries found for comparison.")
    labels = [f"{s.get('scenario')}\n{s.get('controller')}" for s in summaries]
    values = [float(s.get(metric, np.nan)) for s in summaries]

    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4.5))
    ax.bar(range(len(labels)), values, color="C0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(1.0, float(np.nanmax(values)) * 1.15) if values else 1.0)
    ax.set_title(f"{metric} by scenario/controller (measured)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_error_distributions(
    runs: Sequence[Any],
    output_path: str | Path = "plots/final_error_distribution.png",
) -> Path:
    """Box plot of per-episode final errors from ``episodes.csv`` records."""
    from armlab.evaluation.artifacts import load_run

    labels: list[str] = []
    distributions: list[np.ndarray] = []
    for run in runs:
        run_dir = Path(run)
        if not (run_dir / "summary.json").is_file():
            continue
        loaded = load_run(run_dir)
        errs = []
        for row in loaded["episodes"]:
            try:
                errs.append(float(row["final_error"]))
            except (KeyError, ValueError):
                continue
        if errs:
            labels.append(
                f"{loaded['summary'].get('scenario')}\n{loaded['summary'].get('controller')}"
            )
            distributions.append(np.asarray(errs, dtype=float))
    if not distributions:
        raise ValueError("No episode error data found.")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4.5))
    ax.boxplot(distributions, labels=labels)
    ax.set_ylabel("final end-effector error")
    ax.set_title("Final error distribution (per episode, measured)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _load_trajectory(run_dir: Path, episode_index: int = 0) -> dict[str, np.ndarray]:
    files = sorted(run_dir.glob("trajectory_*.npz"))
    if not files:
        raise FileNotFoundError(f"No trajectory_*.npz files in {run_dir}.")
    index = min(max(episode_index, 0), len(files) - 1)
    with np.load(files[index]) as data:
        return {k: data[k] for k in data.files}


def plot_trajectory_3d(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    episode_index: int = 0,
) -> Path:
    """3D end-effector trajectory against the recorded target."""
    run_dir = Path(run_dir)
    traj = _load_trajectory(run_dir, episode_index)
    ee = traj["ee_positions"]
    output_path = Path(output_path) if output_path else run_dir / "plots" / "trajectory_3d.png"

    plt = _plt()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-o", ms=3, color="C0", label="end effector")
    if "target" in traj:
        target = traj["target"]
        ax.scatter(
            [target[0]], [target[1]], [target[2]], marker="*", s=180, color="C3", label="target"
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("End-effector trajectory (measured)")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_joint_curves(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    episode_index: int = 0,
) -> Path:
    """Joint positions and distance-to-target over time for one episode."""
    run_dir = Path(run_dir)
    traj = _load_trajectory(run_dir, episode_index)
    q = traj["q_positions"]
    dists = traj["distances"]
    output_path = Path(output_path) if output_path else run_dir / "plots" / "joint_curves.png"

    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for j, name in enumerate(("waist", "shoulder", "elbow")):
        axes[0].plot(q[:, j], label=name)
    axes[0].set_ylabel("joint angle [rad]")
    axes[0].legend()
    axes[1].plot(dists, color="C3")
    axes[1].set_ylabel("distance to target")
    axes[1].set_xlabel("step")
    fig.suptitle("Saved episode: joint state and tracking error")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_tracking_error(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Mean tracking-error curve across all saved episodes in a run."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("trajectory_*.npz"))
    if not files:
        raise FileNotFoundError(f"No trajectory files in {run_dir}.")
    curves = []
    for path in files:
        with np.load(path) as data:
            curves.append(np.asarray(data["distances"], dtype=float))
    length = min(len(c) for c in curves)
    stacked = np.stack([c[:length] for c in curves])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    output_path = Path(output_path) if output_path else run_dir / "plots" / "tracking_error.png"

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = np.arange(length)
    ax.plot(steps, mean, color="C0", label="mean error")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color="C0", label="+/- 1 std")
    ax.set_xlabel("step")
    ax.set_ylabel("distance to target")
    ax.set_title(f"Tracking error across {len(curves)} episodes (measured)")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def generate_run_report(run_dir: str | Path) -> list[Path]:
    """Generate all plots for one run directory from its saved artifacts."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    produced: list[Path] = []
    for func in (plot_trajectory_3d, plot_joint_curves, plot_tracking_error):
        try:
            produced.append(func(run_dir))
        except (FileNotFoundError, ValueError):
            continue
    return produced
