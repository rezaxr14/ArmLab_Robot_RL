"""Tests for report/plot generation from saved artifacts."""

import pytest

from armlab.evaluation.benchmark import evaluate_run
from armlab.evaluation.plots import (
    generate_run_report,
    plot_controller_comparison,
    plot_error_distributions,
    plot_training_curves,
)


def _make_run(tmp_path):
    return evaluate_run(
        "static_reaching",
        "ik",
        seed=0,
        n_episodes=2,
        output_dir=tmp_path,
    )["run_dir"]


class TestRunReport:
    def test_generates_plots_from_artifacts(self, tmp_path):
        run_dir = _make_run(tmp_path)
        produced = generate_run_report(run_dir)
        assert produced, "expected at least one plot"
        for path in produced:
            assert path.is_file()
            assert path.stat().st_size > 0

    def test_comparison_and_error_plots(self, tmp_path):
        run_a = _make_run(tmp_path / "a")
        run_b = evaluate_run(
            "static_reaching", "pd", seed=0, n_episodes=2, output_dir=tmp_path / "b"
        )["run_dir"]
        comp = plot_controller_comparison([run_a, run_b], output_path=tmp_path / "cmp.png")
        errs = plot_error_distributions([run_a, run_b], output_path=tmp_path / "err.png")
        assert comp.is_file() and errs.is_file()

    def test_missing_artifacts_raises(self, tmp_path):
        with pytest.raises(ValueError):
            plot_controller_comparison([tmp_path / "empty"], output_path=tmp_path / "x.png")

    def test_training_curve_missing_returns_none(self, tmp_path):
        assert plot_training_curves(tmp_path) is None
