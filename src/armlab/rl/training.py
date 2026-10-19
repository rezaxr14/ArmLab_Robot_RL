"""Training loop for the Dueling Double DQN agent.

Produces reproducible runs: everything is seeded (python ``random``, numpy,
torch, env, agent), and per-episode metrics are written to a JSON artifact.
Checkpoints are saved for the best and the final policy.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

__all__ = ["TrainResult", "train_dqn", "set_global_seed"]

try:
    import torch

    from armlab.rl.agent import DQNAgent
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "armlab.rl.training requires torch. Install with: pip install -e '.[rl]'"
    ) from exc


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainResult:
    """Summary of a training run."""

    episode_scores: List[float] = field(default_factory=list)
    evaluation_scores: List[float] = field(default_factory=list)
    best_score: float = -np.inf
    best_episode: int = -1
    total_env_steps: int = 0
    elapsed_seconds: float = 0.0
    checkpoint_path: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "episode_scores": [float(s) for s in self.episode_scores],
            "evaluation_scores": [float(s) for s in self.evaluation_scores],
            "best_score": float(self.best_score),
            "best_episode": int(self.best_episode),
            "total_env_steps": int(self.total_env_steps),
            "elapsed_seconds": float(self.elapsed_seconds),
            "checkpoint_path": self.checkpoint_path,
        }


def train_dqn(
    agent: "DQNAgent",
    env,
    n_episodes: int = 200,
    max_steps_per_episode: int = 250,
    eval_every: int = 0,
    eval_episodes: int = 5,
    eval_env=None,
    eval_seed: int = 10_000,
    checkpoint_path: Optional[str] = None,
    log_fn: Callable[[str], None] = print,
) -> TrainResult:
    """Train ``agent`` on ``env`` for ``n_episodes``.

    Args:
        agent: configured DQNAgent.
        env: a Gymnasium env (reset must accept seed=).
        n_episodes: number of training episodes.
        max_steps_per_episode: per-episode step cap.
        eval_every: evaluate every N episodes (0 disables evaluation).
        eval_episodes: episodes per evaluation.
        eval_env: env for evaluation (defaults to env).
        eval_seed: seed for evaluation episodes (fixed across evaluations).
        checkpoint_path: if set, best and final checkpoints are written here
            (best gets ``.best`` suffix, final uses the given path).
        log_fn: progress logger (set to ``lambda _: None`` to silence).
    """
    result = TrainResult()
    scores_window: deque = deque(maxlen=100)
    start = time.time()

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=episode)
        score = 0.0
        for _ in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, terminated, truncated)
            state = next_state
            score += float(reward)
            if terminated or truncated:
                break

        result.episode_scores.append(score)
        result.total_env_steps = agent.env_steps
        scores_window.append(score)
        if score > result.best_score:
            result.best_score = score
            result.best_episode = episode
            if checkpoint_path:
                agent.save_checkpoint(Path(checkpoint_path).with_suffix(".best"))

        if episode % 10 == 0 or episode == n_episodes:
            log_fn(
                f"episode {episode}/{n_episodes} "
                f"score={score:.2f} avg100={np.mean(scores_window):.2f} "
                f"eps={agent.epsilon:.3f}"
            )

        if eval_every and episode % eval_every == 0:
            target_env = eval_env or env
            eval_scores = _evaluate(agent, target_env, eval_episodes, eval_seed)
            result.evaluation_scores.append(float(np.mean(eval_scores)))
            log_fn(f"  eval@{episode}: mean={np.mean(eval_scores):.2f}")

    result.elapsed_seconds = time.time() - start
    if checkpoint_path:
        agent.save_checkpoint(checkpoint_path)
        result.checkpoint_path = str(checkpoint_path)
    return result


def _evaluate(agent: "DQNAgent", env, episodes: int, seed: int) -> List[float]:
    """Greedy rollout of the agent's current policy."""
    scores = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        score = 0.0
        for _ in range(env.unwrapped.env_params.max_episode_steps):
            action = agent.choose_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            if terminated or truncated:
                break
        scores.append(score)
    return scores
