"""Dueling Q-network architecture (PyTorch).

Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))

Requires the optional ``rl`` extra (torch); importing this module raises
ImportError with installation guidance when torch is unavailable.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["DuelingQNetwork"]

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - environment without torch
    raise ImportError(
        "armlab.rl.networks requires torch. Install with: pip install -e '.[rl]'"
    ) from exc


class DuelingQNetwork(nn.Module):
    """Shared feature trunk with separate value and advantage streams.

    Args:
        state_dim: observation dimension.
        action_dim: number of discrete actions.
        hidden_sizes: widths of the shared MLP (>= 1 layers).
        stream_hidden_size: width of each stream's hidden layer.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        stream_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if state_dim < 1 or action_dim < 1:
            raise ValueError("state_dim and action_dim must be >= 1.")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty.")

        layers = []
        prev = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        self.feature = nn.Sequential(*layers)
        self.value_stream = nn.Sequential(
            nn.Linear(prev, stream_hidden_size), nn.ReLU(),
            nn.Linear(stream_hidden_size, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev, stream_hidden_size), nn.ReLU(),
            nn.Linear(stream_hidden_size, action_dim),
        )

    def forward(self, state: "torch.Tensor") -> "torch.Tensor":
        features = self.feature(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))
