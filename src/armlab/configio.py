"""YAML configuration loading and resolution.

Configuration files are plain YAML mappings; they never instantiate arbitrary
Python objects (see implementation plan section 4.4). Every loader validates
its result by constructing the corresponding typed dataclass from
``armlab.config`` and rejects unknown keys, so typos fail loudly at startup.

Only the keys declared by the target dataclass are accepted. A small number of
documented metadata keys may accompany a config file (for example ``reward``
in environment files, ``scenario``, ``obstacles``); those are returned
separately instead of being silently injected into the dataclass.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from armlab.config import EnvParams, RewardParams, RLConfig, RobotParams

__all__ = [
    "BenchmarkConfig",
    "load_yaml",
    "dump_yaml",
    "load_robot_params",
    "load_env_config",
    "load_reward_params",
    "load_rl_config",
    "load_benchmark_config",
    "config_hash",
    "save_resolved_config",
]


# --------------------------------------------------------------------- I/O
def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path`` with a helpful error on failure."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {path} must contain a YAML mapping, got {type(data)!r}.")
    return dict(data)


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``data`` to ``path`` as YAML, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable(data), handle, sort_keys=False)
    return path


# ---------------------------------------------------------------- helpers
def _jsonable(value: Any) -> Any:
    """Recursively convert numpy/dataclass values to YAML/JSON-friendly types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _dataclass_fields(cls: type) -> set:
    return {f.name for f in fields(cls)}


def _build(cls: type, data: Mapping[str, Any], source: str):
    """Construct ``cls`` from ``data``, rejecting unknown keys."""
    allowed = _dataclass_fields(cls)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in {source}; allowed keys are {sorted(allowed)}."
        )
    return cls(**{k: data[k] for k in data})


# ------------------------------------------------------------ typed loaders
def load_robot_params(path: str | Path) -> RobotParams:
    """Load ``configs/robot/*.yaml`` into :class:`RobotParams`."""
    data = load_yaml(path)
    metadata_keys = {"name", "units", "legacy_unit_scale_m"}
    params = {k: v for k, v in data.items() if k not in metadata_keys}
    return _build(RobotParams, params, str(path))


def load_reward_params(data: Mapping[str, Any] | None, source: str = "<memory>") -> RewardParams:
    return _build(RewardParams, data or {}, source)


def load_env_config(path: str | Path) -> tuple[EnvParams, RewardParams, dict[str, Any]]:
    """Load an environment YAML file.

    Returns ``(env_params, reward_params, metadata)`` where ``metadata`` holds
    keys that are not part of :class:`EnvParams` (for example ``scenario`` or
    ``obstacles``).
    """
    data = load_yaml(path)
    meta_keys = {"scenario", "reward", "obstacles", "link_radius"}
    env_data = {k: v for k, v in data.items() if k not in meta_keys}
    env_params = _build(EnvParams, env_data, str(path))
    reward_params = load_reward_params(data.get("reward"), str(path))
    metadata = {k: _jsonable(v) for k, v in data.items() if k in meta_keys}
    return env_params, reward_params, metadata


def load_rl_config(path: str | Path) -> dict[str, Any]:
    """Load a training config file.

    Returns a dict with ``seed``, ``env_id``, training-loop parameters and the
    nested ``agent`` :class:`RLConfig` (when present).
    """
    data = load_yaml(path)
    if "agent" in data:
        data["agent"] = _build(RLConfig, data["agent"], f"{path}[agent]")
    return data


# -------------------------------------------------------------- benchmark
@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated benchmark-suite configuration (``configs/benchmark/*.yaml``)."""

    suite: str = "default"
    schema_version: str = "1.0"
    output_dir: str = "results"
    scenarios: tuple[str, ...] = ("static_reaching",)
    controllers: tuple[str, ...] = ("ik",)
    n_eval_episodes: int = 20
    base_eval_seed: int = 100_000
    smoke: bool = True
    render: bool = False
    video: bool = False
    require_physics: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "controllers", tuple(self.controllers))
        if not self.scenarios:
            raise ValueError("benchmark config must list at least one scenario.")
        if not self.controllers:
            raise ValueError("benchmark config must list at least one controller.")
        if self.n_eval_episodes < 1:
            raise ValueError("n_eval_episodes must be >= 1.")
        if self.base_eval_seed < 0:
            raise ValueError("base_eval_seed must be >= 0.")

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scenarios"] = list(self.scenarios)
        d["controllers"] = list(self.controllers)
        return d


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return _build(BenchmarkConfig, load_yaml(path), str(path))


# ----------------------------------------------------------- resolution
def config_hash(data: Mapping[str, Any]) -> str:
    """Stable SHA-256 of a resolved configuration mapping."""
    payload = json.dumps(_jsonable(data), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_resolved_config(run_dir: str | Path, sections: Mapping[str, Any]) -> Path:
    """Persist the exact resolved configuration next to a run's outputs."""
    return dump_yaml(_jsonable(sections), Path(run_dir) / "config.resolved.yaml")
