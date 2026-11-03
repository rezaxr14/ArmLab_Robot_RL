"""Tests for the standardized evaluation/benchmark entry point."""

from pathlib import Path

import numpy as np
import pytest

from armlab.config import RobotParams
from armlab.configio import BenchmarkConfig
from armlab.evaluation.artifacts import discover_runs, load_run
from armlab.evaluation.benchmark import (
    CONTROLLERS,
    evaluate_run,
    list_controllers,
    make_policy,
    run_benchmark,
)
from armlab.kinematics import RobotKinematics


class TestRegistry:
    def test_known_controllers(self):
        assert set(list_controllers()) == set(CONTROLLERS)
        assert "ik" in CONTROLLERS and "sac" in CONTROLLERS

    def test_unknown_controller(self):
        kin = RobotKinematics(RobotParams())
        with pytest.raises(ValueError):
            make_policy("teleport", kin)


class TestEvaluateRun:
    def test_produces_documented_artifacts(self, tmp_path):
        result = evaluate_run(
            "static_reaching",
            "ik",
            seed=3,
            n_episodes=3,
            base_eval_seed=100000,
            output_dir=tmp_path,
            save_trajectories=True,
        )
        run_dir = Path(result["run_dir"])
        for name in ("metadata.json", "summary.json", "episodes.csv", "metrics.csv"):
            assert (run_dir / name).is_file(), name
        assert len(list(run_dir.glob("trajectory_*.npz"))) == 3
        summary = result["summary"]
        assert summary["scenario"] == "static_reaching"
        assert summary["controller"] == "ik"
        assert summary["n_episodes"] == 3
        assert 0.0 <= summary["success_rate"] <= 1.0
        assert summary["aggregation"]

    def test_deterministic_across_runs(self, tmp_path):
        r1 = evaluate_run("static_reaching", "ik", seed=0, n_episodes=3, output_dir=tmp_path / "a")
        r2 = evaluate_run("static_reaching", "ik", seed=0, n_episodes=3, output_dir=tmp_path / "b")
        assert r1["summary"]["success_rate"] == r2["summary"]["success_rate"]
        assert r1["episodes"] == r2["episodes"]

    def test_episode_rows_have_termination_reason(self, tmp_path):
        result = evaluate_run("static_reaching", "ik", n_episodes=2, output_dir=tmp_path)
        reasons = {row["termination_reason"] for row in result["episodes"]}
        assert reasons <= {"success", "time_limit", "failure", "ended"}
        assert np.isfinite([row["final_error"] for row in result["episodes"]]).all()

    def test_unknown_scenario(self, tmp_path):
        with pytest.raises(ValueError):
            evaluate_run("warp_drive", "ik", output_dir=tmp_path)


class TestBenchmarkSuite:
    def test_smoke_benchmark_writes_runs(self, tmp_path):
        config = BenchmarkConfig(
            scenarios=("static_reaching",),
            controllers=("ik", "pd"),
            n_eval_episodes=2,
            smoke=True,
        )
        result = run_benchmark(config, output_dir=tmp_path)
        assert result["n_runs"] == 2
        assert result["failures"] == []
        runs = discover_runs(tmp_path)
        assert len(runs) == 2
        for run in runs:
            loaded = load_run(run)
            assert loaded["summary"]["smoke"] is True

    def test_missing_checkpoint_is_reported_not_crashed(self, tmp_path):
        config = BenchmarkConfig(
            scenarios=("static_reaching",), controllers=("dqn",), n_eval_episodes=1
        )
        result = run_benchmark(config, output_dir=tmp_path)
        assert result["n_runs"] == 0
        assert result["failures"][0]["reason"] == "missing checkpoint"
