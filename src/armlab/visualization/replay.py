"""Trajectory replay from saved NPZ artifacts (implementation plan H1).

Replay is deterministic and requires no policy inference or retraining. The
matplotlib backend is selected lazily so importing this module is headless-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Trajectory", "load_trajectory", "list_trajectories", "replay_episode", "render_gif"]


@dataclass
class Trajectory:
    """One saved episode trajectory."""

    episode_index: int
    path: Path
    ee_positions: np.ndarray
    q_positions: np.ndarray
    distances: np.ndarray
    rewards: np.ndarray
    target: np.ndarray | None = None

    @property
    def steps(self) -> int:
        return max(0, self.ee_positions.shape[0] - 1)

    @property
    def final_error(self) -> float:
        return float(self.distances[-1]) if self.distances.size else float("nan")

    @property
    def min_error(self) -> float:
        return float(np.min(self.distances)) if self.distances.size else float("nan")


def list_trajectories(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return sorted(run_dir.glob("trajectory_*.npz"))


def load_trajectory(run_dir: str | Path, episode_index: int = 0) -> Trajectory:
    """Load trajectory ``episode_index`` from a run directory."""
    files = list_trajectories(run_dir)
    if not files:
        raise FileNotFoundError(f"No trajectory_*.npz files in {run_dir}.")
    index = min(max(episode_index, 0), len(files) - 1)
    path = files[index]
    with np.load(path) as data:
        payload: dict[str, np.ndarray] = {k: data[k] for k in data.files}
    return Trajectory(
        episode_index=index,
        path=path,
        ee_positions=payload["ee_positions"],
        q_positions=payload["q_positions"],
        distances=payload["distances"],
        rewards=payload.get("rewards", np.zeros(0)),
        target=payload.get("target"),
    )


def replay_episode(
    run_dir: str | Path,
    episode_index: int = 0,
    show: bool = False,
    output_html: str | Path | None = None,
):
    """Replay the end-effector path; returns the loaded :class:`Trajectory`.

    When ``show`` is False (the default, and always in headless CI) no GUI is
    opened and no figure is created.
    """
    traj = load_trajectory(run_dir, episode_index)
    if not show:
        return traj

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ee = traj.ee_positions
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-", color="C0", label="end effector")
    if traj.target is not None:
        ax.scatter(
            [traj.target[0]], [traj.target[1]], [traj.target[2]], marker="*", s=180, color="C3"
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Replay episode {traj.episode_index}: final error {traj.final_error:.3f}")
    if output_html:
        fig.savefig(str(output_html))
    plt.close(fig)
    return traj


def render_gif(
    run_dir: str | Path,
    episode_index: int = 0,
    output_path: str | Path | None = None,
    fps: int = 20,
) -> Path | None:
    """Render an animated GIF of the end-effector path if Pillow is available.

    Returns ``None`` when the optional writer is unavailable; the raw NPZ
    trajectory remains the authoritative artifact either way.
    """
    traj = load_trajectory(run_dir, episode_index)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter
    except Exception:
        return None

    run_dir = Path(run_dir)
    output_path = (
        Path(output_path) if output_path else run_dir / "plots" / f"replay_ep{episode_index}.gif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ee = traj.ee_positions
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    pad = 1.0
    ax.set_xlim(ee[:, 0].min() - pad, ee[:, 0].max() + pad)
    ax.set_ylim(ee[:, 1].min() - pad, ee[:, 1].max() + pad)
    ax.set_zlim(max(0.0, ee[:, 2].min() - pad), ee[:, 2].max() + pad)
    (line,) = ax.plot([], [], [], "-", color="C0")
    (point,) = ax.plot([], [], [], "o", color="C0")
    if traj.target is not None:
        ax.scatter(
            [traj.target[0]], [traj.target[1]], [traj.target[2]], marker="*", s=120, color="C3"
        )

    def update(frame: int):
        line.set_data(ee[: frame + 1, 0], ee[: frame + 1, 1])
        line.set_3d_properties(ee[: frame + 1, 2])
        point.set_data([ee[frame, 0]], [ee[frame, 1]])
        point.set_3d_properties([ee[frame, 2]])
        return line, point

    try:
        writer = PillowWriter(fps=fps)
        anim = __import__("matplotlib.animation", fromlist=["FuncAnimation"]).FuncAnimation(
            fig, update, frames=len(ee), interval=1000 / fps
        )
        anim.save(str(output_path), writer=writer)
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return output_path
