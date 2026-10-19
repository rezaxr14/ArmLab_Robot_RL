"""Evaluation harness: metrics, rollout runner, reports.

- ``metrics``: EpisodeRecord/EpisodeMetrics and aggregation.
- ``runner``: unified rollout for policies and classical controllers.
"""

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
    "EpisodeMetrics",
    "EpisodeRecord",
    "aggregate",
    "compute_metrics",
    "controller_policy",
    "evaluate_controller",
    "rollout_controller",
]
