"""Tests for standardized run artifacts (metadata/episodes/summary)."""

import numpy as np
import pytest

from armlab.evaluation.artifacts import (
    SCHEMA_VERSION,
    create_run_dir,
    discover_runs,
    load_run,
    save_trajectory,
    write_episodes_csv,
    write_metrics_csv,
)


def _sample_episode(success=True):
    return {
        "episode": 0,
        "seed": 100000,
        "success": success,
        "final_error": 0.3 if success else 5.0,
        "min_error": 0.2,
        "length": 40,
        "steps_to_success": 40 if success else -1,
        "total_return": 80.0,
        "collisions": 0,
        "termination_reason": "success" if success else "time_limit",
    }


class TestRunArtifacts:
    def test_create_and_write(self, tmp_path):
        artifacts = create_run_dir(tmp_path, "static_reaching", "ik", seed=7)
        assert artifacts.run_dir.is_dir()
        assert artifacts.plots_dir.is_dir()
        write_episodes_csv(artifacts, [_sample_episode()])
        artifacts.write_summary({"success_rate": 1.0, "n_episodes": 1})
        write_metrics_csv(artifacts, {"success_rate": 1.0, "n_episodes": 1})
        artifacts.write_metadata({"scenario": "static_reaching"})

        loaded = load_run(artifacts.run_dir)
        assert loaded["summary"]["success_rate"] == 1.0
        assert loaded["metadata"]["schema_version"] == SCHEMA_VERSION
        assert len(loaded["episodes"]) == 1
        assert loaded["episodes"][0]["termination_reason"] == "success"

    def test_unique_directories(self, tmp_path):
        a = create_run_dir(tmp_path, "s", "ik", seed=0, timestamp="2026-01-01_000000")
        b = create_run_dir(tmp_path, "s", "ik", seed=0, timestamp="2026-01-01_000000")
        assert a.run_dir != b.run_dir

    def test_trajectory_round_trip(self, tmp_path):
        artifacts = create_run_dir(tmp_path, "s", "ik", seed=0)
        ee = np.arange(9, dtype=float).reshape(3, 3)
        q = np.zeros((3, 3))
        dists = np.array([3.0, 2.0, 1.0])
        path = save_trajectory(artifacts, 5, ee, q, dists, [1.0, 1.0], target=np.zeros(3))
        assert path.suffix == ".npz"
        loaded = load_run(artifacts.run_dir)
        assert len(loaded["trajectories"]) == 1

    def test_discover_runs(self, tmp_path):
        artifacts = create_run_dir(tmp_path, "s", "ik", seed=0)
        artifacts.write_summary({"success_rate": 0.0})
        assert discover_runs(tmp_path) == [artifacts.run_dir]

    def test_malformed_json_raises(self, tmp_path):
        artifacts = create_run_dir(tmp_path, "s", "ik", seed=0)
        (artifacts.run_dir / "summary.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_run(artifacts.run_dir)

    def test_missing_run_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_run(tmp_path / "nope")

    def test_inf_serialized_as_string(self, tmp_path):
        artifacts = create_run_dir(tmp_path, "s", "ik", seed=0)
        write_episodes_csv(artifacts, [dict(_sample_episode(), final_error=float("inf"))])
        text = (artifacts.run_dir / "episodes.csv").read_text(encoding="utf-8")
        assert "inf" in text
