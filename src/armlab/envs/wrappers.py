"""Gymnasium wrappers used by the benchmark.

- ``ActionRepeat``: repeat each action N times, summing rewards and
  collapsing termination flags (the last step decides terminated/truncated).
- ``StackObservations``: concatenate the last k observations along a new
  leading axis.

Note: ``KinematicReachingEnv`` already supports an internal ``action_repeat``
parameter; the wrapper exists for composing with other envs uniformly.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

__all__ = ["ActionRepeat", "StackObservations"]


class ActionRepeat(gym.Wrapper):
    """Repeat an action ``n`` times per ``step`` call."""

    def __init__(self, env: gym.Env, n: int) -> None:
        super().__init__(env)
        if n < 1:
            raise ValueError(f"ActionRepeat n must be >= 1, got {n}.")
        self.n = int(n)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        obs, info = None, {}
        for _ in range(self.n):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class StackObservations(gym.ObservationWrapper):
    """Stack the last ``k`` observations into shape (k, *obs_shape)."""

    def __init__(self, env: gym.Env, k: int) -> None:
        if k < 1:
            raise ValueError(f"StackObservations k must be >= 1, got {k}.")
        super().__init__(env)
        k = int(k)
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.stack([low] * k), high=np.stack([high] * k), dtype=np.float32
        )
        self.k = k
        self._history: deque[np.ndarray] = deque(maxlen=k)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._history.clear()
        for _ in range(self.k):
            self._history.append(np.asarray(obs, dtype=np.float32))
        return self.observation(obs), info

    def observation(self, obs):
        self._history.append(np.asarray(obs, dtype=np.float32))
        return np.stack(list(self._history)).astype(np.float32)
