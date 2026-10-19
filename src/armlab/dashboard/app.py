"""Optional Streamlit-free Dash dashboard (implementation plan H2).

The dashboard reads existing run artifacts only; it never starts training and
never fabricates sample metrics. If no runs are found it shows an empty state
with the command needed to produce results. Dash/Plotly are imported lazily so
the base package works without the ``ui`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from armlab.evaluation.artifacts import discover_runs, load_run

__all__ = ["build_app", "main", "collect_runs", "UI_AVAILABLE"]

try:  # pragma: no cover - exercised only with the ui extra installed
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, dcc, html

    UI_AVAILABLE = True
except Exception:  # pragma: no cover
    Dash = None
    Input = Output = dcc = html = None
    go = None
    UI_AVAILABLE = False


def _require_ui() -> None:
    if not UI_AVAILABLE:
        raise ImportError(
            "The dashboard requires dash and plotly. Install with: pip install -e '.[ui]'"
        )


def collect_runs(results_root: str | Path) -> list[dict[str, Any]]:
    """Load all run summaries under ``results_root`` (never raises on empty)."""
    runs: list[dict[str, Any]] = []
    for run_dir in discover_runs(results_root):
        try:
            loaded = load_run(run_dir)
        except (ValueError, OSError):
            continue
        if loaded.get("summary"):
            runs.append(
                {
                    "dir": str(run_dir),
                    "summary": loaded["summary"],
                    "episodes": loaded["episodes"],
                    "metadata": loaded.get("metadata") or {},
                    "trajectories": [str(p) for p in loaded["trajectories"]],
                }
            )
    return runs


def _load_trajectory(path: str) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    except (OSError, ValueError):
        return None


def build_app(results_root: str | Path = "results"):
    """Build the Dash app reading artifacts from ``results_root``."""
    _require_ui()
    results_root = Path(results_root)
    app = Dash(__name__)
    app.title = "ArmLab results"

    runs = collect_runs(results_root)
    if not runs:
        app.layout = html.Div(
            [
                html.H2("ArmLab — no results found"),
                html.P(f"Looked in: {results_root.resolve()}"),
                html.P("Generate results first, for example:"),
                html.Code(
                    "python -m armlab benchmark --suite default --config configs/benchmark/default.yaml"
                ),
            ],
            style={"padding": "2rem", "fontFamily": "sans-serif"},
        )
        return app

    options = [
        {
            "label": f"{r['summary'].get('scenario')} / {r['summary'].get('controller')}",
            "value": r["dir"],
        }
        for r in runs
    ]
    app.layout = html.Div(
        [
            html.H2("ArmLab results dashboard"),
            html.Label("Run"),
            dcc.Dropdown(id="run-select", options=options, value=options[0]["value"]),
            html.Div(id="overview"),
            html.H3("Controller comparison"),
            dcc.Graph(id="comparison", figure=_comparison_figure(runs)),
            html.H3("Trajectory explorer"),
            dcc.Graph(id="trajectory"),
            html.H3("Episode details"),
            dcc.Graph(id="episode"),
        ],
        style={"padding": "1.5rem", "fontFamily": "sans-serif", "maxWidth": "1100px"},
    )

    @app.callback(
        Output("overview", "children"),
        Output("trajectory", "figure"),
        Output("episode", "figure"),
        Input("run-select", "value"),
    )
    def _update(selected_dir: str):  # pragma: no cover - UI callback
        run = next((r for r in runs if r["dir"] == selected_dir), runs[0])
        summary = run["summary"]
        overview = html.Ul(
            [
                html.Li(f"scenario: {summary.get('scenario')}"),
                html.Li(f"controller: {summary.get('controller')}"),
                html.Li(f"episodes: {summary.get('n_episodes')}"),
                html.Li(f"success rate: {summary.get('success_rate')}"),
                html.Li(f"smoke result: {summary.get('smoke')}"),
            ]
        )
        traj_fig = go.Figure()
        if run["trajectories"]:
            data = _load_trajectory(run["trajectories"][0])
            if data is not None:
                ee = data["ee_positions"]
                traj_fig.add_scatter3d(
                    x=ee[:, 0], y=ee[:, 1], z=ee[:, 2], mode="lines+markers", name="end effector"
                )
                if "target" in data:
                    t = data["target"]
                    traj_fig.add_scatter3d(
                        x=[t[0]], y=[t[1]], z=[t[2]], mode="markers", name="target"
                    )
        traj_fig.update_layout(title="Saved end-effector trajectory")
        ep_fig = go.Figure()
        if run["trajectories"]:
            data = _load_trajectory(run["trajectories"][0])
            if data is not None:
                ep_fig.add_scatter(y=data["distances"], name="distance to target")
        ep_fig.update_layout(title="Saved episode: tracking error", xaxis_title="step")
        return overview, traj_fig, ep_fig

    return app


def _comparison_figure(runs: list[dict[str, Any]]):  # pragma: no cover - UI
    labels = [f"{r['summary'].get('scenario')}/{r['summary'].get('controller')}" for r in runs]
    values = [r["summary"].get("success_rate", 0.0) for r in runs]
    fig = go.Figure(go.Bar(x=labels, y=values))
    fig.update_layout(title="Success rate (measured)", yaxis_title="success rate")
    return fig


def main(results_root: str | Path = "results", port: int = 8050) -> None:  # pragma: no cover - UI
    app = build_app(results_root)
    app.run(port=port)
