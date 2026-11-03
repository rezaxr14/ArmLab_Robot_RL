"""Validated configuration dataclasses for ArmLab.

All parameters that appear in tests, training scripts, or benchmarks must live
here (see implementation plan Phase A). Configuration errors raise
``ValueError`` explicitly rather than being silently clamped.

Units: meters, radians, seconds, kilograms. Angles are stored in radians.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

__all__ = [
    "RobotParams",
    "RewardParams",
    "EnvParams",
    "RLConfig",
    "PhysicsParams",
]


def _triple(name: str, value: tuple[float, float, float]) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values, got shape {arr.shape}.")
    return tuple(float(v) for v in arr)  # type: ignore[return-value]


def _check_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")


def _check_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}.")


@dataclass(frozen=True)
class RobotParams:
    """Physical parameters of the 3-link spatial arm.

    Joint convention (radians):
        theta1: waist rotation about the world +z axis.
        theta2: shoulder angle measured from the vertical (+z) axis, positive
            tilts the second link toward +x when theta1 == 0.
        theta3: elbow angle measured relative to the second link
            (0 = fully extended in a straight line).
    """

    link_lengths: tuple[float, float, float] = (5.0, 10.0, 8.0)
    joint_limits_low: tuple[float, float, float] = (-np.pi, -np.pi / 2, -np.pi)
    joint_limits_high: tuple[float, float, float] = (np.pi, np.pi / 2, np.pi)

    def __post_init__(self) -> None:
        object.__setattr__(self, "link_lengths", _triple("link_lengths", self.link_lengths))
        object.__setattr__(
            self, "joint_limits_low", _triple("joint_limits_low", self.joint_limits_low)
        )
        object.__setattr__(
            self, "joint_limits_high", _triple("joint_limits_high", self.joint_limits_high)
        )
        for length in self.link_lengths:
            _check_positive("link_lengths entries", length)
        low = np.asarray(self.joint_limits_low)
        high = np.asarray(self.joint_limits_high)
        if np.any(low >= high):
            raise ValueError(
                "Each joint limit low must be strictly less than high; "
                f"got low={self.joint_limits_low}, high={self.joint_limits_high}."
            )

    @property
    def max_reach(self) -> float:
        """Maximum distance from the shoulder joint to the end effector."""
        return float(self.link_lengths[1] + self.link_lengths[2])

    @property
    def base_height(self) -> float:
        return float(self.link_lengths[0])

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RewardParams:
    """Reward shaping parameters for the reaching environment.

    The per-step reward is the sum of explicit components, all surfaced in
    ``info["reward_components"]`` (implementation plan C4):

    - ``progress``: ``progress_scale * (prev_distance - distance)``.
    - ``distance``: ``-distance_penalty_scale * distance``.
    - ``action_cost``: ``-action_penalty * |action|``.
    - ``time_cost``: ``-time_penalty``.
    - ``success``: ``success_bonus`` when the target is reached (terminal).
    - ``limit_penalty``: ``-limit_penalty`` per step at a joint limit.
    """

    distance_penalty_scale: float = 0.1  # penalty = -scale * distance
    progress_scale: float = 1.0  # reward = scale * distance improvement
    success_bonus: float = 100.0
    success_threshold: float = 1.0  # length units; EE within this = success
    time_penalty: float = 0.0  # per-step penalty (>= 0)
    action_penalty: float = 0.0  # per-step penalty * |action|
    limit_penalty: float = 0.0  # per-step penalty at a joint limit
    success_terminates: bool = True

    def __post_init__(self) -> None:
        _check_non_negative("distance_penalty_scale", self.distance_penalty_scale)
        _check_non_negative("progress_scale", self.progress_scale)
        _check_non_negative("success_bonus", self.success_bonus)
        _check_positive("success_threshold", self.success_threshold)
        _check_non_negative("time_penalty", self.time_penalty)
        _check_non_negative("action_penalty", self.action_penalty)
        _check_non_negative("limit_penalty", self.limit_penalty)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EnvParams:
    """Kinematic reaching environment parameters.

    Action space: Discrete(6) unless ``control_mode == "continuous"`",
    in which case Box(-1, 1, (3,)) scales to ``continuous_action_scale``.
    Discrete action a moves joint ``a // 2`` by ``action_step_size`` rad in the
    direction ``+1`` if ``a % 2 == 0`` else ``-1``.
    """

    control_mode: str = "discrete"  # "discrete" | "continuous"
    action_step_size: float = np.deg2rad(5.0)
    continuous_action_scale: float = np.deg2rad(5.0)
    max_episode_steps: int = 250
    action_repeat: int = 1
    target_min_radius: float = 6.0  # sampling annulus in horizontal plane
    target_max_radius: float = 15.0
    target_min_height: float = 5.0
    target_max_height: float = 20.0
    observation_kind: str = "joint_ee_target"  # "joint_ee_target" | "relative"
    #: Fraction of the reachable annulus excluded at both radial boundaries when
    #: sampling the default (normal-control) scenario, so targets are not placed
    #: on the singular inner/outer workspace boundary. Boundary/singularity
    #: scenarios sample their own targets and ignore this.
    target_annulus_margin: float = 0.05

    def __post_init__(self) -> None:
        if self.control_mode not in ("discrete", "continuous"):
            raise ValueError(
                f"control_mode must be 'discrete' or 'continuous', got {self.control_mode!r}."
            )
        if self.observation_kind not in ("joint_ee_target", "relative"):
            raise ValueError(
                "observation_kind must be 'joint_ee_target' or 'relative', "
                f"got {self.observation_kind!r}."
            )
        _check_positive("action_step_size", self.action_step_size)
        _check_positive("continuous_action_scale", self.continuous_action_scale)
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1.")
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be >= 1.")
        _check_non_negative("target_min_radius", self.target_min_radius)
        if self.target_max_radius < self.target_min_radius:
            raise ValueError("target_max_radius must be >= target_min_radius.")
        _check_non_negative("target_min_height", self.target_min_height)
        if self.target_max_height < self.target_min_height:
            raise ValueError("target_max_height must be >= target_min_height.")
        if not 0.0 <= self.target_annulus_margin < 0.9:
            raise ValueError("target_annulus_margin must be in [0, 0.9).")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RLConfig:
    """Dueling Double DQN hyperparameters and schedules.

    The epsilon schedule is multiplicative per environment step:
        epsilon_t = max(epsilon_min, epsilon_start * epsilon_decay ** t)
    Target network updates are explicit and configurable:
        - mode "hard": copy local -> target every ``target_update_every``
          learn steps.
        - mode "soft": Polyak update with coefficient ``tau`` after every
          learn step.
    """

    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 128
    buffer_size: int = 100_000
    hidden_sizes: tuple[int, ...] = (256, 256)
    stream_hidden_size: int = 128
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    learn_every: int = 4
    gradient_clip: float = 10.0
    target_update_mode: str = "soft"  # "soft" | "hard"
    target_update_every: int = 500  # learn steps between hard updates
    tau: float = 1e-3  # Polyak coefficient for soft updates
    min_buffer_samples: int = 64

    def __post_init__(self) -> None:
        if self.target_update_mode not in ("soft", "hard"):
            raise ValueError(
                f"target_update_mode must be 'soft' or 'hard', got {self.target_update_mode!r}."
            )
        _check_positive("gamma", self.gamma)
        if self.gamma > 1.0:
            raise ValueError(f"gamma must be <= 1.0, got {self.gamma}.")
        _check_positive("lr", self.lr)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.buffer_size < self.batch_size:
            raise ValueError("buffer_size must be >= batch_size.")
        if not self.hidden_sizes or any(int(h) < 1 for h in self.hidden_sizes):
            raise ValueError("hidden_sizes must be a non-empty sequence of positive ints.")
        _check_positive("stream_hidden_size", float(self.stream_hidden_size))
        if not (0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0):
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError("epsilon_decay must be in (0, 1].")
        if self.learn_every < 1:
            raise ValueError("learn_every must be >= 1.")
        _check_non_negative("gradient_clip", self.gradient_clip)
        if self.target_update_every < 1:
            raise ValueError("target_update_every must be >= 1.")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1].")
        if self.min_buffer_samples < 1:
            raise ValueError("min_buffer_samples must be >= 1.")

    def epsilon_at(self, step: int) -> float:
        """Epsilon after ``step`` environment steps."""
        return max(self.epsilon_min, self.epsilon_start * (self.epsilon_decay**step))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RLConfig:
        data = dict(data)
        if "hidden_sizes" in data:
            data["hidden_sizes"] = tuple(data["hidden_sizes"])
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(frozen=True)
class PhysicsParams:
    """MuJoCo physics environment parameters (Phase E).

    ``sim_dt`` is the physics timestep; ``control_decimation`` sim steps per
    environment step (so env dt = sim_dt * control_decimation).
    """

    sim_dt: float = 0.002
    control_decimation: int = 10
    max_episode_steps: int = 500
    target_site_name: str = "target"
    ee_site_name: str = "ee"
    success_threshold: float = 0.2
    action_scale: float = 5.0  # rad/s per unit of [-1, 1] action
    contact_termination: bool = True
    # Body-pair contact excludes for the MJCF (body names). The base column
    # is separated from the floor via geom contype flags in the model
    # builder instead, because world geoms cannot appear in <exclude>.
    exclude_pairs: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self) -> None:
        _check_positive("sim_dt", self.sim_dt)
        if self.control_decimation < 1:
            raise ValueError("control_decimation must be >= 1.")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1.")
        _check_positive("success_threshold", self.success_threshold)
        _check_positive("action_scale", self.action_scale)

    @property
    def control_dt(self) -> float:
        return float(self.sim_dt * self.control_decimation)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["exclude_pairs"] = [list(p) for p in self.exclude_pairs]
        return d
