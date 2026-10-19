"""Dashboard tests (skipped when the optional ui extra is unavailable)."""

import pytest

pytest.importorskip("dash")
pytest.importorskip("plotly")

from armlab.dashboard.app import UI_AVAILABLE, build_app, collect_runs


@pytest.mark.skipif(not UI_AVAILABLE, reason="dash/plotly not installed")
class TestDashboard:
    def test_empty_results_no_fabrication(self, tmp_path):
        runs = collect_runs(tmp_path)
        assert runs == []
        app = build_app(tmp_path)
        # The empty state must mention the results path, not sample numbers.
        text = str(app.layout)
        assert "no results" in text.lower()

    def test_reads_real_run(self, tmp_path):
        from armlab.evaluation.benchmark import evaluate_run

        evaluate_run("static_reaching", "ik", n_episodes=2, output_dir=tmp_path)
        runs = collect_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["summary"]["controller"] == "ik"
        app = build_app(tmp_path)
        assert app is not None
