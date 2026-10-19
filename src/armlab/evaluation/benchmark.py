"""Standardized evaluation entry point (implementation plan Phase G).

One function evaluates any supported controller or policy on a named scenario
with a fixed set of held-out seeds and writes the artifacts documented in
``armlab.evaluation.artifacts``. Training scenarios and evaluation seeds are
kept separate; ``smoke=True`` runs are labelled development results.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from armlab.config import EnvParams, RobotParams
from armlab.controllers import (
    CartesianDLSController,
    JointPDController,
    JointPositionController,
    RRTPlannerController,
)
from armlab.evaluation.artifacts import (
    create_run_dir,
    save_trajectory,
    write_episodes_csv,
    write_metrics_csv,
)
from armlab.evaluation.metrics import EpisodeRecord, aggregate, compute_metrics
from armlab.kinematics import RobotKinematics
from armlab.scenarios import list_scenarios, make_scenario

__all__ = [
    "CONTROLLERS",
    "evaluate_run",
    "run_benchmark",
    "list_controllers",
    "make_policy",
]

#: Controller names that can be evaluated through :func:`evaluate_run`.
CONTROLLERS = ("ik", "pd", "jacobian", "mpc", "rrt", "dqn", "sac")

#: Controllers that read a discrete action space.
DISCRETE_CONTROLLERS = ("ik", "pd", "jacobian", "mpc", "rrt", "dqn")

_DEFAULT_STEP_SIZE = float(np.deg2rad(5.0))


def list_controllers() -> list[str]:
    return list(CONTROLLERS)


def _resolve_attempt_seed(base_seed: int, attempt: int, max_attempts: int = 20) -> int:
    """Deterministically derive a fallback seed for a failed episode."""
    return base_seed + 1_000_000 + attempt * 7919


def make_policy(
    controller: str,
    kinematics: RobotKinematics,
    obstacles: Sequence = (),
    link_radius: float = 0.3,
    seed: int = 0,
    checkpoint: str | None = None,
    joint_limits: tuple | None = None,
) -> Callable[[np.ndarray], Any]:
    """Build a callable ``policy(obs) -> action`` for a controller name.

    Classical controllers are evaluated in the continuous-action environment
    so their true bounded joint commands are used (not quantized to the DQN's
    discrete increments). ``dqn`` and ``sac`` keep their native action spaces.
    """
    from armlab.config import EnvParams
    from armlab.evaluation.runner import controller_policy_continuous

    obstacles = list(obstacles)
    action_scale = EnvParams().continuous_action_scale
    if controller == "ik":
        inner = JointPositionController(kinematics, kp=0.6, max_dq=0.3)
        return controller_policy_continuous(inner, action_scale)
    if controller == "pd":
        inner = JointPDController(kinematics, kp=0.6, kd=0.1, dt=0.05, max_dq=0.3)
        return controller_policy_continuous(inner, action_scale)
    if controller == "jacobian":
        inner = CartesianDLSController(
            kinematics, kp=0.8, damping=0.1, max_dq=0.3, joint_limits=joint_limits
        )
        return controller_policy_continuous(inner, action_scale)
    if controller == "mpc":
        from armlab.controllers import ShootingMPCController

        inner = ShootingMPCController(
            kinematics,
            horizon=4,
            n_samples=24,
            action_magnitude=0.2,
            seed=seed,
            joint_limits=joint_limits,
        )
        return controller_policy_continuous(inner, action_scale)
    if controller == "rrt":
        inner = RRTPlannerController(
            kinematics,
            obstacles=obstacles,
            link_radius=link_radius,
            seed=seed,
            joint_limits=joint_limits,
        )
        return controller_policy_continuous(inner, action_scale)
    if controller == "dqn":
        return _make_dqn_policy(checkpoint)
    if controller == "sac":
        return _make_sac_policy(checkpoint)
    raise ValueError(f"Unknown controller {controller!r}. Available: {list(CONTROLLERS)}.")


def _make_dqn_policy(checkpoint: str | None) -> Callable[[np.ndarray], int]:
    if not checkpoint:
        raise ValueError("controller 'dqn' requires --checkpoint pointing to a .pt file.")
    agent = _load_agent(checkpoint)

    def policy(obs: np.ndarray) -> int:
        return int(agent.choose_action(np.asarray(obs, dtype=np.float32), greedy=True))

    return policy


def _load_agent(checkpoint: str):
    import torch

    from armlab.config import RLConfig
    from armlab.rl.agent import DQNAgent

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = RLConfig.from_dict(payload.get("config", {})) if payload.get("config") else RLConfig()
    agent = DQNAgent(
        state_dim=int(payload["state_dim"]),
        action_dim=int(payload["action_dim"]),
        config=cfg,
        device="cpu",
    )
    agent.load_checkpoint(checkpoint)
    agent.qnetwork_local.eval()
    return agent


def _make_sac_policy(checkpoint: str | None) -> Callable[[np.ndarray], np.ndarray]:
    if not checkpoint:
        raise ValueError("controller 'sac' requires --checkpoint pointing to a .zip model.")
    from armlab.rl.sac import load_sac

    model = load_sac(checkpoint)

    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(3)

    return policy


def _build_env(scenario: str, robot: RobotParams, control_mode: str):
    env_params = EnvParams(control_mode=control_mode)
    return make_scenario(scenario, seed=0, robot=robot, env_params=env_params)


def _episode_row(
    record: EpisodeRecord,
    metrics: Any,
    seed: int,
    index: int,
) -> dict[str, Any]:
    dists = record.distances
    min_error = float(np.min(dists)) if dists.size else float("inf")
    final_error = float(dists[-1]) if dists.size else float("inf")
    steps_to_success = int(metrics.time_to_reach) if np.isfinite(metrics.time_to_reach) else -1
    if record.terminated and metrics.success:
        reason = "success"
    elif record.truncated:
        reason = "time_limit"
    elif record.terminated:
        reason = "failure"
    else:
        reason = "ended"
    return {
        "episode": index,
        "seed": seed,
        "success": bool(metrics.success),
        "final_error": final_error,
        "min_error": min_error,
        "length": int(record.steps),
        "steps_to_success": steps_to_success,
        "total_return": float(metrics.total_reward),
        "collisions": int(metrics.collisions),
        "termination_reason": reason,
    }


def _rollout(
    env,
    policy: Callable[[np.ndarray], Any],
    seed: int,
    obstacles: Sequence,
    link_radius: float,
):
    """Physics-agnostic rollout returning ``(EpisodeRecord, metrics)``."""
    obs, info = env.reset(seed=seed)
    kin = getattr(env.unwrapped, "kin", None)
    ee_list = [np.asarray(info["ee_position"], dtype=float)]
    q_list = [np.asarray(info["q"], dtype=float)]
    dist_list = [float(info["distance"])]
    rewards: list[float] = []
    links_list = [kin.link_positions(info["q"])] if kin is not None else []

    terminated = truncated = False
    step_budget = env.unwrapped.env_params.max_episode_steps * 2 + 10
    steps = 0
    while not (terminated or truncated) and steps < step_budget:
        action = policy(np.asarray(obs, dtype=np.float32))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        ee_list.append(np.asarray(info["ee_position"], dtype=float))
        q_list.append(np.asarray(info["q"], dtype=float))
        dist_list.append(float(info["distance"]))
        if kin is not None:
            links_list.append(kin.link_positions(info["q"]))
        steps += 1

    record = EpisodeRecord(
        ee_positions=np.stack(ee_list),
        distances=np.array(dist_list),
        rewards=rewards,
        q_positions=np.stack(q_list),
        success_threshold=env.unwrapped.reward_params.success_threshold,
        link_positions=np.stack(links_list) if links_list else None,
        steps=len(rewards),
        terminated=terminated,
        truncated=truncated,
    )
    metrics = compute_metrics(record, obstacles=obstacles, link_radius=link_radius)
    return record, metrics


def evaluate_run(
    scenario: str,
    controller: str,
    seed: int = 0,
    n_episodes: int = 20,
    base_eval_seed: int = 100_000,
    output_dir: str | Path = "results",
    robot: RobotParams | None = None,
    checkpoint: str | None = None,
    smoke: bool = True,
    save_trajectories: bool = True,
    config_sections: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one controller on one scenario and write a run directory.

    ``seed`` is the *run* seed recorded in metadata; evaluation episodes use
    ``base_eval_seed + i`` so repeated runs use identical held-out seeds.
    """
    if scenario not in list_scenarios():
        raise ValueError(f"Unknown scenario {scenario!r}. Available: {list_scenarios()}.")
    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller {controller!r}. Available: {list(CONTROLLERS)}.")

    robot = robot or RobotParams()
    # DQN uses the discrete action variant; every other controller uses the
    # continuous variant so its bounded joint command is applied faithfully.
    control_mode = "discrete" if controller == "dqn" else "continuous"
    bundle = _build_env(scenario, robot, control_mode)

    # Reset seeded state of the scenario env; `make_scenario` envs are seeded
    # through their per-episode reset seeds, so nothing else is needed here.
    bundle.env.reset(seed=base_eval_seed)

    kinematics = RobotKinematics(robot)
    policy = make_policy(
        controller,
        kinematics,
        obstacles=bundle.obstacles,
        link_radius=bundle.metadata.get("link_radius", 0.3),
        seed=seed,
        checkpoint=checkpoint,
        joint_limits=(robot.joint_limits_low, robot.joint_limits_high),
    )

    artifacts = create_run_dir(output_dir, scenario, controller, seed)
    started = time.time()
    metrics_list = []
    episode_rows: list[dict[str, Any]] = []
    for i in range(n_episodes):
        ep_seed = base_eval_seed + i
        record, metrics = _rollout(
            bundle.env,
            policy,
            ep_seed,
            bundle.obstacles,
            bundle.metadata.get("link_radius", 0.3),
        )
        metrics_list.append(metrics)
        episode_rows.append(_episode_row(record, metrics, ep_seed, i))
        if save_trajectories:
            save_trajectory(
                artifacts,
                ep_seed,
                record.ee_positions,
                record.q_positions,
                record.distances,
                record.rewards,
                target=bundle.env.unwrapped.target.copy(),
            )
        inner = getattr(policy, "controller", None)
        if inner is not None and hasattr(inner, "reset"):
            inner.reset()

    summary = aggregate(metrics_list)
    summary.update(
        {
            "scenario": scenario,
            "controller": controller,
            "run_seed": int(seed),
            "base_eval_seed": int(base_eval_seed),
            "success_count": int(sum(1 for m in metrics_list if m.success)),
            "n_episodes": len(metrics_list),
            "smoke": bool(smoke),
            "aggregation": "per-episode; success_rate is the episode fraction",
            "wall_clock_seconds": float(time.time() - started),
            "control_mode": control_mode,
        }
    )

    write_episodes_csv(artifacts, episode_rows)
    write_metrics_csv(artifacts, summary)
    artifacts.write_summary(summary)
    artifacts.write_metadata(
        {
            "scenario": scenario,
            "controller": controller,
            "seed": int(seed),
            "n_episodes": n_episodes,
            "base_eval_seed": int(base_eval_seed),
            "smoke": bool(smoke),
            "checkpoint": checkpoint,
            "obstacles": [_obstacle_dict(o) for o in bundle.obstacles],
            "config_sections": _safe_sections(config_sections),
        }
    )
    bundle.env.close()
    return {
        "run_dir": str(artifacts.run_dir),
        "summary": summary,
        "episodes": episode_rows,
        "obstacles": bundle.obstacles,
    }


def _obstacle_dict(obstacle: Any) -> dict[str, Any]:
    center = getattr(obstacle, "center", None)
    if center is not None:
        return {
            "type": "sphere",
            "center": list(np.asarray(center, float)),
            "radius": float(obstacle.radius),
        }
    return {
        "type": "box",
        "low": list(np.asarray(obstacle.low, float)),
        "high": list(np.asarray(obstacle.high, float)),
    }


def _safe_sections(sections: dict[str, Any] | None) -> dict[str, Any]:
    if not sections:
        return {}
    out: dict[str, Any] = {}
    for key, value in sections.items():
        if hasattr(value, "as_dict"):
            out[key] = value.as_dict()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def run_benchmark(
    benchmark_config,
    output_dir: str | Path | None = None,
    robot: RobotParams | None = None,
    scenarios: Sequence[str] | None = None,
    controllers: Sequence[str] | None = None,
    checkpoints: dict[str, str] | None = None,
    n_episodes: int | None = None,
) -> dict[str, Any]:
    """Run the configured scenario x controller matrix.

    Returns a dict with the list of produced run directories and a summary.
    Each (scenario, controller) pair is one run with its own directory.
    """
    checkpoints = checkpoints or {}
    output_dir = Path(output_dir or benchmark_config.output_dir)
    scenario_list = list(scenarios or benchmark_config.scenarios)
    controller_list = list(controllers or benchmark_config.controllers)
    episodes = int(n_episodes if n_episodes is not None else benchmark_config.n_eval_episodes)

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scenario in scenario_list:
        for controller in controller_list:
            try:
                checkpoint = checkpoints.get(controller)
                if controller in ("dqn", "sac") and not checkpoint:
                    failures.append(
                        {
                            "scenario": scenario,
                            "controller": controller,
                            "reason": "missing checkpoint",
                        }
                    )
                    continue
                run = evaluate_run(
                    scenario=scenario,
                    controller=controller,
                    seed=benchmark_config.base_eval_seed,
                    n_episodes=episodes,
                    base_eval_seed=benchmark_config.base_eval_seed,
                    output_dir=output_dir,
                    robot=robot,
                    checkpoint=checkpoint,
                    smoke=benchmark_config.smoke,
                    config_sections={"benchmark": benchmark_config.as_dict()},
                )
                runs.append(run)
            except Exception as exc:  # keep going; report per-cell failures
                failures.append(
                    {"scenario": scenario, "controller": controller, "reason": str(exc)}
                )

    result = {
        "suite": benchmark_config.suite,
        "schema_version": benchmark_config.schema_version,
        "smoke": benchmark_config.smoke,
        "n_runs": len(runs),
        "runs": [r["run_dir"] for r in runs],
        "failures": failures,
        "results": [
            {
                "scenario": r["summary"]["scenario"],
                "controller": r["summary"]["controller"],
                "success_rate": r["summary"]["success_rate"],
                "n_episodes": r["summary"]["n_episodes"],
            }
            for r in runs
        ],
    }
    return result
