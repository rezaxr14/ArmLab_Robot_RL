"""Unified rollout runner for policies and classical controllers.

Any controller that exposes ``act(q, target) -> dq`` (see
armlab.controllers.base.Controller) can be evaluated on the same kinematic
environment and metrics as an RL policy, guaranteeing identical evaluation
semantics: same success definition, same episode limits, same metrics.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from armlab.controllers.base import discrete_argmax
from armlab.evaluation.metrics import (
    EpisodeMetrics,
    EpisodeRecord,
    aggregate,
    compute_metrics,
)
from armlab.planning.collision import Obstacle

__all__ = ["rollout_controller", "evaluate_controller", "PolicyFn"]


# A policy maps the observation to a discrete or continuous action.
PolicyFn = Callable[[np.ndarray], object]


def rollout_controller(
    env,
    policy: PolicyFn,
    seed: int,
    obstacles: Optional[List[Obstacle]] = None,
    link_radius: float = 0.3,
) -> Tuple[EpisodeRecord, EpisodeMetrics]:
    """Run one episode and record everything metrics need.

    ``policy`` receives the observation and returns an action compatible with
    the env's action space. For classical controllers, wrap them with
    ``controller_policy`` below.
    """
    env_params = env.unwrapped.env_params
    obs, info = env.reset(seed=seed)

    ee_list = [info["ee_position"]]
    q_list = [info["q"]]
    dist_list = [info["distance"]]
    rewards: List[float] = []
    links_list = [env.unwrapped.kin.link_positions(info["q"])]

    terminated = truncated = False
    while not (terminated or truncated):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        ee_list.append(info["ee_position"])
        q_list.append(info["q"])
        dist_list.append(info["distance"])
        links_list.append(env.unwrapped.kin.link_positions(info["q"]))

    record = EpisodeRecord(
        ee_positions=np.stack(ee_list),
        distances=np.array(dist_list),
        rewards=rewards,
        q_positions=np.stack(q_list),
        success_threshold=env.unwrapped.reward_params.success_threshold,
        link_positions=np.stack(links_list),
        steps=len(rewards),
        terminated=terminated,
        truncated=truncated,
    )
    metrics = compute_metrics(record, obstacles=obstacles, link_radius=link_radius)
    return record, metrics


def controller_policy(controller, step_size: float) -> PolicyFn:
    """Adapt a Controller (act(q, target) -> dq) to a policy(obs) -> action."""

    def policy(obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        # joint_ee_target observation: [q(3), ee(3), target(3)]
        q = obs[:3]
        target = obs[6:9]
        dq = controller.act(q, target)
        return discrete_argmax(np.asarray(dq, dtype=float), step_size)

    return policy


def evaluate_controller(
    env,
    policy: PolicyFn,
    n_episodes: int,
    base_seed: int = 0,
    obstacles: Optional[List[Obstacle]] = None,
    link_radius: float = 0.3,
) -> dict:
    """Evaluate a policy over ``n_episodes`` with distinct seeds.

    Seeds are ``base_seed + i`` so runs are exactly reproducible.
    Returns the aggregate report from armlab.evaluation.metrics.aggregate.
    """
    all_metrics: List[EpisodeMetrics] = []
    for i in range(n_episodes):
        _, metrics = rollout_controller(
            env, policy, seed=base_seed + i, obstacles=obstacles, link_radius=link_radius
        )
        all_metrics.append(metrics)
    return aggregate(all_metrics)
