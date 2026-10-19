"""Standardized run artifacts (implementation plan Phase G3).

Every evaluation run gets a unique directory::

    results/<timestamp>_<scenario>_<controller>_seed-<seed>/
        config.resolved.yaml
        metadata.json
        episodes.csv
        metrics.csv
        summary.json
        trajectory_seed-<seed>.npz
        plots/

``metadata.json`` records provenance (git revision/dirty state, platform,
dependency versions, config hash, seed, schema version, timestamps).
``episodes.csv`` has one row per episode and ``summary.json`` stores aggregate
metrics plus the aggregation method. Trajectories are stored in NPZ (not JSON
floats). All numeric values come from real rollouts.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "RunArtifacts",
    "create_run_dir",
    "collect_metadata",
    "write_episodes_csv",
    "write_metrics_csv",
    "save_trajectory",
    "load_run",
    "discover_runs",
]

SCHEMA_VERSION = "1.0"

EPISODE_FIELDS = [
    "episode",
    "seed",
    "success",
    "final_error",
    "min_error",
    "length",
    "steps_to_success",
    "total_return",
    "collisions",
    "termination_reason",
]

METRIC_FIELDS = ["metric", "mean", "std", "median", "n"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> dict[str, Any]:
    """Return the current git revision and dirty state if available."""
    info: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        root = Path(__file__).resolve().parents[3]
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL
        ).decode()
        info["dirty"] = bool(status.strip())
    except Exception:
        # Not a git checkout (or git unavailable): provenance is simply absent.
        pass
    return info


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "gymnasium", "matplotlib", "torch", "mujoco", "stable_baselines3"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def collect_metadata(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Collect reproducibility provenance for a run."""
    try:
        import armlab

        armlab_version = armlab.__version__
    except Exception:  # pragma: no cover - only during partially imported states
        armlab_version = None
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "armlab_version": armlab_version,
        "git": git_revision(),
        "dependencies": _dependency_versions(),
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


@dataclass
class RunArtifacts:
    """Handle for a run directory; writes are explicit and fail loudly."""

    run_dir: Path
    schema_version: str = SCHEMA_VERSION

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    def ensure(self) -> RunArtifacts:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        return self

    def write_metadata(self, extra: Mapping[str, Any]) -> Path:
        payload = collect_metadata(extra)
        return self._write_json("metadata.json", payload)

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        return self._write_json("summary.json", {"schema_version": SCHEMA_VERSION, **summary})

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        self.ensure()
        path = self.run_dir / name
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(_jsonable(payload), handle, indent=2, sort_keys=False)
        except OSError as exc:  # pragma: no cover - filesystem failure path
            raise OSError(f"Failed to write run artifact {path}: {exc}") from exc
        return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def create_run_dir(
    root: str | Path,
    scenario: str,
    controller: str,
    seed: int,
    timestamp: str | None = None,
) -> RunArtifacts:
    """Create a uniquely named run directory under ``root``."""
    root = Path(root)
    stamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = f"{stamp}_{scenario}_{controller}_seed-{int(seed):04d}"
    run_dir = root / base
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{base}_{suffix}"
        suffix += 1
    return RunArtifacts(run_dir).ensure()


def write_episodes_csv(artifacts: RunArtifacts, episodes: Sequence[Mapping[str, Any]]) -> Path:
    artifacts.ensure()
    path = artifacts.run_dir / "episodes.csv"
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            for row in episodes:
                writer.writerow({k: _csv_value(row.get(k)) for k in EPISODE_FIELDS})
    except OSError as exc:  # pragma: no cover
        raise OSError(f"Failed to write episodes CSV {path}: {exc}") from exc
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def write_metrics_csv(artifacts: RunArtifacts, summary: Mapping[str, Any]) -> Path:
    """Flatten numeric aggregate metrics into ``metrics.csv``."""
    artifacts.ensure()
    path = artifacts.run_dir / "metrics.csv"
    rows: list[dict[str, Any]] = []

    def _emit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping) and {"mean", "std", "median"} & set(value):
            rows.append(
                {
                    "metric": prefix,
                    "mean": _csv_value(value.get("mean")),
                    "std": _csv_value(value.get("std")),
                    "median": _csv_value(value.get("median")),
                    "n": value.get("n", ""),
                }
            )
        elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, bool
        ):
            rows.append(
                {
                    "metric": prefix,
                    "mean": _csv_value(float(value)),
                    "std": "",
                    "median": "",
                    "n": "",
                }
            )

    for key, value in summary.items():
        if key in {"schema_version", "aggregation", "scenario", "controller", "smoke"}:
            continue
        _emit(key, value)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_trajectory(
    artifacts: RunArtifacts,
    seed: int,
    ee_positions: np.ndarray,
    q_positions: np.ndarray,
    distances: np.ndarray,
    rewards: Sequence[float],
    target: np.ndarray | None = None,
) -> Path:
    """Save one episode trajectory as NPZ (authoritative replay artifact)."""
    artifacts.ensure()
    path = artifacts.run_dir / f"trajectory_seed-{int(seed):04d}.npz"
    payload = {
        "ee_positions": np.asarray(ee_positions, dtype=np.float64),
        "q_positions": np.asarray(q_positions, dtype=np.float64),
        "distances": np.asarray(distances, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
    }
    if target is not None:
        payload["target"] = np.asarray(target, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


# -------------------------------------------------------------------- load
def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load the JSON/CSV artifacts of a run directory."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    def _read_json(name: str) -> dict[str, Any] | None:
        path = run_dir / name
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON artifact {path}: {exc}") from exc

    episodes: list[dict[str, Any]] = []
    episodes_path = run_dir / "episodes.csv"
    if episodes_path.is_file():
        with episodes_path.open("r", encoding="utf-8") as handle:
            episodes = list(csv.DictReader(handle))

    trajectories = sorted(run_dir.glob("trajectory_*.npz"))
    return {
        "run_dir": run_dir,
        "metadata": _read_json("metadata.json"),
        "summary": _read_json("summary.json"),
        "episodes": episodes,
        "trajectories": trajectories,
    }


def discover_runs(root: str | Path) -> list[Path]:
    """Return run directories (those containing ``summary.json``) under root."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.glob("*/summary.json"))
