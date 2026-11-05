"""Typed configuration for ArmForge.

Everything the notebook (the "control panel") sets ends up in one :class:`RunConfig`,
which round-trips through YAML so a run directory is fully reproducible.

Unknown keys raise immediately: a typo in a config must never silently fall back to a
default and quietly change an experiment.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

EVAL_SEED_BASE = 1_000_000  # evaluation seeds never overlap training seeds (< 1e6)


# --------------------------------------------------------------------------------------
# Physics / scene
# --------------------------------------------------------------------------------------


@dataclass
class RandomizationConfig:
    """Per-episode domain randomisation (multiplicative scales around the nominal value)."""

    enabled: bool = False
    friction_scale: tuple[float, float] = (0.7, 1.3)  # table + object + pad sliding friction
    mass_scale: tuple[float, float] = (0.7, 1.4)  # manipulated object mass
    damping_scale: tuple[float, float] = (0.7, 1.5)  # arm joint damping
    kp_scale: tuple[float, float] = (0.85, 1.15)  # arm position-actuator stiffness
    action_noise: float = 0.0  # std of Gaussian noise added to the policy action


@dataclass
class PhysicsConfig:
    """Solver and contact parameters.

    Friction triples are MuJoCo ``(sliding, torsional, rolling)``. With the elliptic cone
    (required for ``condim`` 4/6) torsional friction resists spinning about the contact
    normal and rolling friction resists rolling; both matter for pushing and grasping a cube.
    """

    timestep: float = 0.002
    substeps: int = 20  # control period = timestep * substeps = 40 ms (25 Hz)
    gravity: float = 9.81
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    impratio: float = 10.0
    noslip_iterations: int = 0  # >0 reduces contact creep; off by default (cost, and cone interaction)
    # MuJoCo combines two geoms' friction with the per-component MAX, so the effective
    # table/object sliding friction below is max(0.5, 0.6) = 0.6.
    table_friction: tuple[float, float, float] = (0.5, 0.005, 0.0001)
    object_friction: tuple[float, float, float] = (0.6, 0.01, 0.0005)
    pad_friction: float = 1.0  # sliding friction on the Robotiq pad boxes (Menagerie: 0.6-0.7)
    object_mass: float = 0.08  # kg
    object_half_size: float = 0.02  # m  (a 40 mm cube; the 2F-85 opens to 85 mm)
    object_condim: int = 4
    # Contact softness. MuJoCo clamps the time constant to >= 2 * timestep (0.004 s here), so 0.004 sits exactly on the
    # stability edge; a small light cube squeezed between stiff pads then occasionally diverges. 0.01 is ~2.5x softer.
    solref: tuple[float, float] = (0.01, 1.0)  # contact stiffness (time constant, s) / damping ratio
    pad_solref: tuple[float, float] = (0.01, 1.0)  # gripper pads (Menagerie ships 0.004; the pads' priority makes theirs win)
    solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    arm_joint_damping: float = 2.0  # N m s/rad added on top of the actuator's own damping
    arm_frictionloss: float = 0.1  # N m  (Coulomb friction in the arm joints)
    arm_gravity_comp: bool = True  # the real UR controller compensates gravity; MuJoCo `gravcomp` on arm+gripper bodies
    # Menagerie's arm servos are heavily over-damped (kd/kp = 0.2 s: only ~18% of a commanded move happens per
    # 40 ms control step). Scaling kd makes the servo respond in ~50 ms, closer to a real UR joint controller.
    servo_damping_scale: float = 0.25
    # Gain applied to the joint step the IK asks for, to cancel the servo lag. 0 = derive it from the model
    # (1 / fraction-of-a-step-achieved); 1 = no compensation.
    servo_lead_gain: float = 0.0
    randomize: RandomizationConfig = field(default_factory=RandomizationConfig)


@dataclass
class SceneConfig:
    """Static scene content around the manipulated object."""

    tray: bool = False  # a walled tray used as the place target
    distractors: int = 0  # extra loose cubes that can be knocked around
    tray_center: tuple[float, float] = (0.0, 0.30)  # (forward, lateral) offset from workspace centre
    tray_half_extent: tuple[float, float] = (0.09, 0.09)
    tray_wall_height: float = 0.03
    camera_default: str = "front"
    workspace_center: tuple[float, float] = (0.50, 0.0)  # world (x, y) of the workspace middle


@dataclass
class TaskConfig:
    name: str = "reach"  # reach | push | pickplace
    horizon: int = 50  # control steps per episode (fixed length, truncation only)
    success_threshold: float = 0.03  # m, goal-distance tolerance
    reward: str = "sparse"  # sparse | dense   (goal term, relabel-able by HER)
    shaping: float = 0.0  # weight of the non-relabelled shaping terms (approach/grasp/lift)
    max_ee_step: float = 0.025  # m of end-effector motion per unit action per control step
    workspace_half_extent: tuple[float, float] = (0.17, 0.30)  # (forward, lateral) half sizes
    workspace_z: tuple[float, float] = (0.012, 0.32)  # reachable pinch-site height band
    push_height: float = 0.02  # push task: fixed pinch height above the table (m)
    goal_height_max: float = 0.18  # pickplace: maximum goal height above the table
    air_goal_prob: float = 0.0  # pickplace: chance a goal is in the air instead of on the table
    object_start_extent: float = 0.12  # half-width of the box objects spawn in
    min_goal_distance: float = 0.06  # goals are resampled until at least this far from the start


@dataclass
class EnvConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    seed: int = 0


# --------------------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------------------


@dataclass
class AlgoConfig:
    """SAC with an optional distributional (TQC) critic ensemble and hindsight relabelling."""

    critic: str = "tqc"  # tqc | twin   (twin == classic clipped double-Q SAC)
    hidden: tuple[int, ...] = (256, 256, 256)
    critic_ensemble: int = 5
    n_quantiles: int = 25
    top_quantiles_to_drop: int = 2  # per critic; the main TQC knob against over-estimation
    layer_norm: bool = True
    gamma: float = 0.98
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    target_entropy_scale: float = 1.0  # target entropy = -scale * action_dim
    batch_size: int = 1024
    updates_per_step: float = 0.5  # gradient steps per environment transition (UTD ratio)
    grad_clip: float = 10.0
    her_prob: float = 0.8  # probability a sampled transition is goal-relabelled
    clip_return: bool = True  # clamp bootstrap targets to the sparse-reward return range
    obs_clip: float = 5.0
    action_l2: float = 0.0  # penalty on |pre-tanh action|^2 in the actor loss


@dataclass
class CurriculumConfig:
    """Success-gated goal curriculum: goals start close and spread out as the policy improves."""

    enabled: bool = True
    start_scale: float = 0.35  # fraction of the full goal range at the start
    end_scale: float = 1.0
    promote_success: float = 0.7  # raise difficulty when the rolling success rate exceeds this
    demote_success: float = 0.2
    step: float = 0.1
    window_episodes: int = 200


@dataclass
class DemoConfig:
    """Scripted-expert demonstrations used to seed the replay buffer and warm-start the actor."""

    episodes: int = 0  # 0 disables
    bc_steps: int = 2000
    bc_batch_size: int = 512
    keep_only_successful: bool = True
    noise_std: float = 0.1


@dataclass
class TrainConfig:
    total_steps: int = 1_000_000  # environment transitions
    num_envs: int = 4
    start_steps: int = 5000  # uniform-random exploration before the policy acts
    warmup_updates_after: int = 2000  # transitions collected before gradient steps begin
    buffer_episodes: int = 40_000
    eval_every: int = 25_000
    eval_episodes: int = 20
    log_every: int = 5_000
    checkpoint_every: int = 100_000
    # Policy snapshots used to render "failing -> learning -> solved" media. Values <= 1.0 are
    # fractions of total_steps, values > 1 are absolute step counts. 0 snapshots the random init.
    milestones: tuple[float, ...] = (0.0, 0.02, 0.1, 0.35, 1.0)
    device: str = "auto"  # auto | cpu | cuda
    seed: int = 0
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    demos: DemoConfig = field(default_factory=DemoConfig)
    vec_backend: str = "subprocess"  # subprocess | sync
    save_buffer: bool = True  # also checkpoint the replay buffer (latest_buffer.pt) so a resumed run keeps its experience
    # Resuming a checkpoint that has no saved buffer: collect this many episodes with the loaded policy before any
    # gradient step, so the critics are not refit on a handful of fresh episodes.
    resume_refill_episodes: int = 400


@dataclass
class RunConfig:
    name: str = "run"
    out_dir: str = "runs"
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --------------------------------------------------------------------------------------
# (De)serialisation
# --------------------------------------------------------------------------------------


def _coerce(value: Any, hint: Any) -> Any:
    """Convert YAML/JSON values to the declared field type (lists -> tuples, ints -> floats)."""
    origin = getattr(hint, "__origin__", None)
    if dataclasses.is_dataclass(hint):
        return from_dict(hint, value)
    if origin is tuple:
        return tuple(value)
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def from_dict(cls: type, data: dict[str, Any] | None):
    """Build dataclass ``cls`` from a nested dict, rejecting unknown keys."""
    data = {} if data is None else dict(data)
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} key(s): {unknown}; valid keys: {sorted(names)}")
    kwargs = {k: _coerce(v, hints[k]) for k, v in data.items()}
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    """Dataclass -> plain nested dict (tuples become lists so YAML stays readable)."""
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [to_dict(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; ``override`` wins. Neither input is mutated."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(data: dict, overrides: dict[str, Any] | None) -> dict:
    """Apply ``{"train.total_steps": 1000, "env.physics.noslip_iterations": 0}`` style overrides."""
    out = copy.deepcopy(data)
    for dotted, value in (overrides or {}).items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Load a YAML run config (or the defaults when ``path`` is None) and apply overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    data = apply_overrides(data, overrides)
    return from_dict(RunConfig, data)


def save_config(cfg: RunConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
