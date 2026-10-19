"""Soft Actor-Critic baseline via Stable-Baselines3 (implementation plan F4).

The from-scratch Dueling Double DQN remains the repository's "algorithm
implementation" project; SAC is intentionally delegated to Stable-Baselines3,
a well-tested reference implementation, so the continuous-control baseline
does not add a second large debugging surface.

Stable-Baselines3 and torch are optional: importing this module is safe, and
the public functions raise a clear installation error when the ``rl`` extra is
missing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SAC_AVAILABLE", "train_sac", "load_sac", "evaluate_sac"]

try:  # pragma: no cover - exercised only when the optional extra is present
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    SAC_AVAILABLE = True
except Exception:  # pragma: no cover
    SAC = None
    gym = None
    BaseCallback = object
    SAC_AVAILABLE = False


def _require_sac() -> None:
    if not SAC_AVAILABLE:
        raise ImportError(
            "SAC requires stable-baselines3 and torch. Install with: pip install -e '.[rl]'"
        )


def _make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


class _MetricsCallback(BaseCallback):  # pragma: no cover - needs SB3
    """Record rolling episode reward and timesteps at a configurable interval."""

    def __init__(self, log_every: int = 1000):
        super().__init__()
        self.log_every = max(1, int(log_every))
        self.history: list[dict[str, float]] = []
        self._episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(float(info["episode"]["r"]))
        if self.num_timesteps % self.log_every == 0:
            recent = self._episode_rewards[-50:]
            self.history.append(
                {
                    "timesteps": int(self.num_timesteps),
                    "mean_reward": float(np.mean(recent)) if recent else float("nan"),
                }
            )
        return True


def train_sac(
    env_id: str,
    output_dir: str | Path = "results",
    seed: int = 0,
    total_timesteps: int = 50_000,
    sac_params: dict[str, Any] | None = None,
    eval_episodes: int = 5,
    eval_seed: int = 100_000,
    deterministic_eval: bool = True,
    log_every: int = 1000,
    log_fn=print,
) -> dict[str, Any]:
    """Train and save an SAC policy; return paths and measured metrics.

    A short ``total_timesteps`` is a smoke run, not evidence of convergence.
    """
    _require_sac()
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be >= 1.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = dict(sac_params or {})

    env = _make_env(env_id, seed)
    callback = _MetricsCallback(log_every=log_every)
    started = time.time()
    model = SAC(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": params.pop("net_arch", [256, 256])},
        **params,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.time() - started

    model_path = output_dir / f"sac_{env_id.replace('/', '-')}_seed{seed}.zip"
    model.save(str(model_path.with_suffix("")))

    eval_metrics = evaluate_sac(
        model,
        env,
        n_episodes=eval_episodes,
        base_seed=eval_seed,
        deterministic=deterministic_eval,
    )
    payload = {
        "algorithm": "SAC (stable-baselines3)",
        "env_id": env_id,
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "elapsed_seconds": float(elapsed),
        "model_path": str(model_path),
        "sac_params": _jsonable(params),
        "training_history": callback.history,
        "eval": eval_metrics,
        "deterministic_eval": bool(deterministic_eval),
        "smoke": bool(total_timesteps < 20_000),
    }
    (output_dir / f"sac_metrics_seed{seed}.json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    env.close()
    log_fn(
        f"SAC trained {total_timesteps} steps in {elapsed:.1f}s; "
        f"eval success_rate={eval_metrics['success_rate']:.2f} "
        f"(smoke={payload['smoke']})"
    )
    return payload


def load_sac(path: str | Path):
    """Load a saved SAC model (accepts a path with or without ``.zip``)."""
    _require_sac()
    path = Path(path)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    if not path.is_file():
        raise FileNotFoundError(f"SAC model not found: {path}")
    return SAC.load(str(path))


def evaluate_sac(
    model,
    env,
    n_episodes: int = 5,
    base_seed: int = 100_000,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Greedy/analytic evaluation of a SAC policy on fixed held-out seeds."""
    _require_sac()
    successes = 0
    returns: list[float] = []
    final_errors: list[float] = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=base_seed + i)
        done = False
        total = 0.0
        final_dist = float(info["distance"])
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            final_dist = float(info["distance"])
            done = bool(terminated or truncated)
        threshold = env.unwrapped.reward_params.success_threshold
        successes += int(final_dist <= threshold)
        returns.append(total)
        final_errors.append(final_dist)
    return {
        "n_episodes": int(n_episodes),
        "success_rate": successes / n_episodes,
        "success_count": successes,
        "mean_return": float(np.mean(returns)),
        "final_error_mean": float(np.mean(final_errors)),
        "final_error_median": float(np.median(final_errors)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
