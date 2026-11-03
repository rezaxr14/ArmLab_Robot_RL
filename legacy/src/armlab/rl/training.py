"""Reusable training loop for the Dueling Double DQN agent (plan Phase D4).

Everything is seeded (python ``random``, numpy, torch, environment resets and
the agent's replay sampler). Per-episode metrics are recorded and written as
JSON + CSV, and best/latest checkpoints are saved so training can resume.
Training never opens a GUI, and environments are closed in ``finally`` blocks.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from armlab.evaluation.artifacts import collect_metadata

__all__ = ["TrainResult", "train_dqn", "set_global_seed"]

try:
    import torch

    from armlab.rl.agent import DQNAgent
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "armlab.rl.training requires torch. Install with: pip install -e '.[rl]'"
    ) from exc


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainResult:
    """Summary of a training run."""

    episode_scores: list[float] = field(default_factory=list)
    episode_success: list[bool] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_final_distance: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    eval_success_rate: list[float] = field(default_factory=list)
    eval_mean_return: list[float] = field(default_factory=list)
    evaluation_scores: list[float] = field(default_factory=list)
    best_score: float = -np.inf
    best_success_rate: float = -np.inf
    best_episode: int = -1
    total_env_steps: int = 0
    elapsed_seconds: float = 0.0
    checkpoint_path: str | None = None
    best_checkpoint_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_scores": [float(s) for s in self.episode_scores],
            "episode_success": [bool(s) for s in self.episode_success],
            "episode_lengths": [int(n) for n in self.episode_lengths],
            "episode_final_distance": [float(d) for d in self.episode_final_distance],
            "eval_steps": [int(s) for s in self.eval_steps],
            "eval_success_rate": [float(s) for s in self.eval_success_rate],
            "eval_mean_return": [float(s) for s in self.eval_mean_return],
            "evaluation_scores": [float(s) for s in self.evaluation_scores],
            "best_score": float(self.best_score),
            "best_success_rate": float(self.best_success_rate),
            "best_episode": int(self.best_episode),
            "total_env_steps": int(self.total_env_steps),
            "elapsed_seconds": float(self.elapsed_seconds),
            "checkpoint_path": self.checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
        }


def _eval_greedy(agent: DQNAgent, env, episodes: int, seed: int) -> dict[str, float]:
    """Greedy evaluation returning measured success/length/distance metrics."""
    returns: list[float] = []
    successes = 0
    lengths: list[int] = []
    final_distances: list[float] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        steps = 0
        done = False
        info: dict[str, Any] = info
        while not done:
            action = agent.choose_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            done = bool(terminated or truncated)
        success = bool(info.get("success", False))
        successes += int(success)
        returns.append(total)
        lengths.append(steps)
        final_distances.append(float(info.get("distance_to_target", info.get("distance", np.nan))))
    return {
        "success_rate": successes / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_final_distance": float(np.mean(final_distances)),
    }


def train_dqn(
    agent: DQNAgent,
    env,
    n_episodes: int = 200,
    max_steps_per_episode: int = 250,
    eval_every: int = 0,
    eval_episodes: int = 5,
    eval_env=None,
    eval_seed: int = 10_000,
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
    max_steps: int | None = None,
    seed: int = 0,
    log_fn: Callable[[str], None] = print,
) -> TrainResult:
    """Train ``agent`` on ``env`` and record reproducible metrics.

    Args:
        agent: configured DQNAgent.
        env: a Gymnasium env (``reset`` accepts a seed).
        n_episodes: maximum number of training episodes.
        max_steps_per_episode: per-episode step cap.
        eval_every: evaluate every N episodes (0 disables evaluation).
        eval_episodes: episodes per evaluation.
        eval_env: separate env for evaluation (defaults to ``env``); a separate
            env is recommended so training state is untouched.
        eval_seed: base seed for fixed evaluation episodes.
        checkpoint_path: if set, latest and best checkpoints are written here.
        output_dir: if set, metrics JSON/CSV and a training curve are written.
        max_steps: optional global environment-step budget (smoke runs).
        seed: base training seed; episode ``e`` resets with ``seed + e``.
        log_fn: progress logger (pass ``lambda _: None`` to silence).
    """
    set_global_seed(seed)
    result = TrainResult()
    scores_window: deque = deque(maxlen=100)
    start = time.time()
    episode = 0

    try:
        for episode in range(1, n_episodes + 1):
            state, _ = env.reset(seed=seed + episode)
            score = 0.0
            steps = 0
            final_info: dict[str, Any] = {}
            for _ in range(max_steps_per_episode):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, final_info = env.step(action)
                agent.step(state, action, reward, next_state, terminated, truncated)
                state = next_state
                score += float(reward)
                steps += 1
                if terminated or truncated:
                    break

            success = bool(final_info.get("success", False))
            final_dist = float(
                final_info.get("distance_to_target", final_info.get("distance", np.nan))
            )
            result.episode_scores.append(score)
            result.episode_success.append(success)
            result.episode_lengths.append(steps)
            result.episode_final_distance.append(final_dist)
            result.total_env_steps = agent.env_steps
            scores_window.append(score)

            if episode % 10 == 0 or episode == n_episodes:
                log_fn(
                    f"episode {episode}/{n_episodes} score={score:.2f} "
                    f"avg100={np.mean(scores_window):.2f} eps={agent.epsilon:.3f} "
                    f"steps={agent.env_steps}"
                )

            if eval_every and episode % eval_every == 0:
                target_env = eval_env or env
                eval_metrics = _eval_greedy(agent, target_env, eval_episodes, eval_seed)
                result.eval_steps.append(agent.env_steps)
                result.eval_success_rate.append(eval_metrics["success_rate"])
                result.eval_mean_return.append(eval_metrics["mean_return"])
                result.evaluation_scores.append(eval_metrics["mean_return"])
                # Best = highest success rate, then lowest final distance.
                improved = eval_metrics["success_rate"] > result.best_success_rate or (
                    eval_metrics["success_rate"] == result.best_success_rate
                    and eval_metrics["mean_final_distance"] < result.best_score
                )
                result.best_success_rate = max(
                    result.best_success_rate, eval_metrics["success_rate"]
                )
                result.best_score = (
                    eval_metrics["mean_final_distance"] if improved else result.best_score
                )
                if improved:
                    result.best_episode = episode
                    if checkpoint_path:
                        best_path = Path(checkpoint_path).with_suffix(".best")
                        agent.save_checkpoint(best_path)
                        result.best_checkpoint_path = str(best_path)
                log_fn(
                    f"  eval@{episode}: success={eval_metrics['success_rate']:.2f} "
                    f"return={eval_metrics['mean_return']:.2f}"
                )

            if max_steps is not None and agent.env_steps >= max_steps:
                log_fn(f"reached max_steps={max_steps} at episode {episode}.")
                break
    finally:
        if checkpoint_path:
            agent.save_checkpoint(checkpoint_path)
            result.checkpoint_path = str(checkpoint_path)
        result.elapsed_seconds = time.time() - start
        if output_dir:
            _write_training_artifacts(Path(output_dir), result, agent, seed)

    if not np.isfinite(result.best_score):
        result.best_score = float(max(result.episode_scores)) if result.episode_scores else -np.inf
    if not np.isfinite(result.best_success_rate):
        result.best_success_rate = (
            float(np.mean(result.episode_success)) if result.episode_success else 0.0
        )

    return result


def _write_training_artifacts(
    output_dir: Path, result: TrainResult, agent: DQNAgent, seed: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "env_steps": int(agent.env_steps),
        "learn_steps": int(agent.learn_steps),
        "epsilon": float(agent.epsilon),
        "config": agent.config.as_dict(),
        "metrics": result.as_dict(),
        "metadata": collect_metadata(),
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    episode_path = output_dir / "episodes.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "return", "success", "length", "final_distance"])
        for i, (ret, succ, length, dist) in enumerate(
            zip(
                result.episode_scores,
                result.episode_success,
                result.episode_lengths,
                result.episode_final_distance,
                strict=False,
            ),
            start=1,
        ):
            writer.writerow([i, ret, int(succ), length, dist])
    eval_path = output_dir / "evaluations.csv"
    with eval_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["env_steps", "success_rate", "mean_return"])
        for steps, sr, mr in zip(
            result.eval_steps,
            result.eval_success_rate,
            result.eval_mean_return,
            strict=False,
        ):
            writer.writerow([steps, sr, mr])
    try:
        from armlab.evaluation.plots import plot_training_curves

        plot_training_curves(output_dir, output_path=output_dir / "training_curve.png")
    except Exception:  # plotting must never break training
        pass
