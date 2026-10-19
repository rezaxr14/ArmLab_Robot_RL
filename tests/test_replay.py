"""Tests for deterministic trajectory replay."""

from pathlib import Path

import numpy as np
import pytest

from armlab.evaluation.artifacts import create_run_dir, save_trajectory
from armlab.visualization.replay import (
    list_trajectories,
    load_trajectory,
    render_gif,
    replay_episode,
)


def _write_run(tmp_path: Path) -> Path:
    artifacts = create_run_dir(tmp_path, "static_reaching", "ik", seed=0)
    for seed in (100000, 100001):
        ee = np.linspace(0, 1, 12).reshape(4, 3)
        q = np.zeros((4, 3))
        save_trajectory(artifacts, seed, ee, q, np.linspace(3, 1, 4), [0.0, 1.0, 1.0])
    return artifacts.run_dir


class TestReplay:
    def test_load_and_properties(self, tmp_path):
        run_dir = _write_run(tmp_path)
        traj = load_trajectory(run_dir, 0)
        assert traj.steps == 3
        assert traj.ee_positions.shape == (4, 3)
        assert list_trajectories(run_dir)  # non-empty

    def test_episode_index_clamped(self, tmp_path):
        run_dir = _write_run(tmp_path)
        traj = load_trajectory(run_dir, 999)
        assert traj.episode_index == 1

    def test_headless_replay_no_figure(self, tmp_path):
        run_dir = _write_run(tmp_path)
        traj = replay_episode(run_dir, 0, show=False)
        assert isinstance(traj.ee_positions, np.ndarray)

    def test_missing_run_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trajectory(tmp_path / "nope")

    def test_gif_optional(self, tmp_path):
        run_dir = _write_run(tmp_path)
        path = render_gif(run_dir, 0, output_path=tmp_path / "replay.gif")
        if path is not None:
            assert Path(path).exists()
