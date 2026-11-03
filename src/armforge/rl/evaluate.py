"""Deterministic policy evaluation and rollout recording."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import EVAL_SEED_BASE, EnvConfig
from ..env.arm_env import ArmForgeEnv


def run_episode(env: ArmForgeEnv, policy, seed: int, goal_scale: float = 1.0, camera: str | None = None, render_every: int = 1, size=(480, 360)):
    """One episode. ``policy(obs_dict) -> action``. Returns ``(summary, frames)``."""
    obs = env.reset(seed=seed, goal_scale=goal_scale)
    frames: list[np.ndarray] = []
    if camera:
        frames.append(env.render(camera, *size))
    done, total, info, step = False, 0.0, {"is_success": False, "distance": float("nan")}, 0
    first_success = None
    best = float("inf")
    while not done:
        obs, reward, done, info = env.step(policy(obs))
        total += reward
        step += 1
        best = min(best, info["distance"])
        if info["is_success"] and first_success is None:
            first_success = step
        if camera and (step % render_every == 0 or done):
            frames.append(env.render(camera, *size))
    summary = {
        "seed": seed,
        "success": bool(info["is_success"]),
        "ever_success": first_success is not None,
        "final_distance": float(info["distance"]),
        "best_distance": float(best),
        "return": float(total),
        "first_success_step": first_success,
        "steps": step,
    }
    return summary, frames


def evaluate_policy(env_cfg: EnvConfig, policy_fn, episodes: int, seed0: int = EVAL_SEED_BASE, goal_scale: float = 1.0, env: ArmForgeEnv | None = None) -> dict:
    """Success rate and distance statistics over ``episodes`` fixed-seed episodes."""
    own = env is None
    env = env or ArmForgeEnv(replace(env_cfg, seed=seed0))
    try:
        rows = [run_episode(env, policy_fn, seed0 + i, goal_scale)[0] for i in range(episodes)]
    finally:
        if own:
            env.close()
    return {
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "ever_success_rate": float(np.mean([r["ever_success"] for r in rows])),
        "mean_final_distance": float(np.mean([r["final_distance"] for r in rows])),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "episodes": episodes,
        "rows": rows,
    }


def agent_policy(agent, deterministic: bool = True):
    return lambda obs: agent.act(obs["observation"], obs["desired_goal"], deterministic=deterministic)


def random_policy(env: ArmForgeEnv, seed: int = 0):
    rng = np.random.default_rng(seed)
    return lambda obs: env.action_space.sample(rng)
