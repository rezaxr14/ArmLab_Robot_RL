"""Replay buffer for off-policy RL (numpy storage, torch sampling).

Design (implementation plan Phase D2 fixes):
- Storage is a fixed-capacity circular buffer of numpy arrays.
- Sampling uses a seeded ``numpy.random.Generator`` for reproducibility.
- Tensors are moved to the device at sample time, not at insertion time, so
  the buffer stays device-agnostic.
- ``terminated`` is stored separately from ``truncated`` so bootstrapping is
  correct under the Gymnasium API (only ``terminated`` masks the bootstrap).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["ReplayBuffer"]

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "armlab.rl.replay requires torch. Install with: pip install -e '.[rl]'"
    ) from exc


class ReplayBuffer:
    """Fixed-capacity uniform replay buffer with seeded sampling."""

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        seed: int = 0,
        device: "torch.device | str" = "cpu",
    ) -> None:
        if capacity < 1 or batch_size < 1:
            raise ValueError("capacity and batch_size must be >= 1.")
        if batch_size > capacity:
            raise ValueError("batch_size must be <= capacity.")
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._terminated = None
        self._truncated = None
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if self._states is None:
            shape = state.shape
            zeros = (self.capacity,)
            self._states = np.zeros(zeros + shape, dtype=np.float32)
            self._next_states = np.zeros(zeros + shape, dtype=np.float32)
            self._actions = np.zeros(self.capacity, dtype=np.int64)
            self._rewards = np.zeros(self.capacity, dtype=np.float32)
            self._terminated = np.zeros(self.capacity, dtype=np.float32)
            self._truncated = np.zeros(self.capacity, dtype=np.float32)

        i = self._idx
        self._states[i] = state
        self._next_states[i] = next_state
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._terminated[i] = float(bool(terminated))
        self._truncated[i] = float(bool(truncated))
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self) -> Tuple["torch.Tensor", ...]:
        """Sample a batch; returns (states, actions, rewards, next_states,
        terminated, truncated) as tensors on ``self.device``."""
        if self._size < self.batch_size:
            raise ValueError(
                f"Not enough samples: {self._size} < batch_size {self.batch_size}."
            )
        idx = self.rng.integers(0, self._size, size=self.batch_size)
        to = lambda arr: torch.as_tensor(arr[idx], device=self.device)  # noqa: E731
        return (
            to(self._states).float(),
            to(self._actions).long(),
            to(self._rewards).float(),
            to(self._next_states).float(),
            to(self._terminated).float(),
            to(self._truncated).float(),
        )
