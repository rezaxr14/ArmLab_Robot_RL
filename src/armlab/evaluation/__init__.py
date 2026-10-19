"""Evaluation harness: metrics, rollout runner, artifacts, and benchmark.

- ``metrics``: EpisodeRecord/EpisodeMetrics and aggregation.
- ``runner``: unified rollout for policies and classical controllers.
- ``artifacts``: standardized run directories (metadata/episodes/summary).
- ``benchmark``: one entry point evaluating controllers on named scenarios.
- ``plots``: report figures generated from saved artifacts only.
"""

from armlab.evaluation.artifacts import (
    SCHEMA_VERSION,
    RunArtifacts,
    create_run_dir,
    discover_runs,
    load_run,
)
from armlab.evaluation.benchmark import evaluate_run, list_controllers, run_benchmark
from armlab.evaluation.metrics import (
    EpisodeMetrics,
    EpisodeRecord,
    aggregate,
    compute_metrics,
)
from armlab.evaluation.runner import (
    controller_policy,
    evaluate_controller,
    rollout_controller,
)

__all__ = [
    "SCHEMA_VERSION",
    "RunArtifacts",
    "EpisodeMetrics",
    "EpisodeRecord",
    "aggregate",
    "compute_metrics",
    "controller_policy",
    "evaluate_controller",
    "rollout_controller",
    "create_run_dir",
    "discover_runs",
    "load_run",
    "evaluate_run",
    "list_controllers",
    "run_benchmark",
]
