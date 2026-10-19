"""Corrected Dueling Double DQN agent.

Fixes relative to the legacy implementation (plan Phase D2):
1. Epsilon decay is an explicit per-environment-step schedule owned by the
   agent (``epsilon_at``), testable in isolation.
2. Target network updates are explicit: soft (Polyak, tau) or hard (every
   ``target_update_every`` learn steps), selected by config.
3. The target network is initialized as an exact copy of the local network.
4. The replay buffer is seeded; tensors are moved to the device at sample
   time; no dead experience filtering.
5. ``save_checkpoint``/``load_checkpoint`` persist epsilon, step counts, and
   config so resume/eval are reproducible.
6. Gradient clipping (``gradient_clip``) and Huber loss stabilize updates.
7. ``choose_action`` accepts arbitrary leading batch dims for state arrays
   and returns a plain Python int for single states.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Union

import numpy as np

from armlab.config import RLConfig

__all__ = ["DQNAgent"]

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from armlab.rl.networks import DuelingQNetwork
    from armlab.rl.replay import ReplayBuffer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "armlab.rl.agent requires torch. Install with: pip install -e '.[rl]'"
    ) from exc


class DQNAgent:
    """Dueling Double DQN agent with explicit schedules and checkpointing."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[RLConfig] = None,
        device: Union[str, "torch.device"] = "cpu",
        seed: int = 0,
    ) -> None:
        if state_dim < 1 or action_dim < 1:
            raise ValueError("state_dim and action_dim must be >= 1.")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.config = config or RLConfig()
        self.device = torch.device(device)

        cfg = self.config
        self.qnetwork_local = DuelingQNetwork(
            state_dim, action_dim, cfg.hidden_sizes, cfg.stream_hidden_size
        ).to(self.device)
        # Fix 3: exact copy at init.
        self.qnetwork_target = DuelingQNetwork(
            state_dim, action_dim, cfg.hidden_sizes, cfg.stream_hidden_size
        ).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber

        self.memory = ReplayBuffer(
            cfg.buffer_size, cfg.batch_size, seed=seed, device=self.device
        )
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Fix 1: epsilon is an explicit function of env steps.
        self.env_steps = 0
        self.learn_steps = 0

    # ----------------------------------------------------------- epsilon
    @property
    def epsilon(self) -> float:
        return self.config.epsilon_at(self.env_steps)

    def decay_epsilon(self) -> float:
        """Advance the schedule by one environment step; returns new epsilon."""
        self.env_steps += 1
        return self.epsilon

    # -------------------------------------------------------------- act
    def choose_action(self, state: np.ndarray, greedy: bool = False) -> Union[int, np.ndarray]:
        """Epsilon-greedy action selection.

        A single state of shape (state_dim,) returns a Python int; a batch of
        shape (B, state_dim) returns an int64 array of actions. ``greedy=True``
        (or epsilon == 0) disables exploration entirely.
        """
        arr = np.asarray(state, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape (state_dim,) or (B, state_dim); got {arr.shape}."
            )

        if not greedy and self.rng.random() < self.epsilon:
            actions = self.rng.integers(0, self.action_dim, size=arr.shape[0])
        else:
            with torch.no_grad():
                t = torch.as_tensor(arr, device=self.device)
                self.qnetwork_local.eval()
                q = self.qnetwork_local(t)
                self.qnetwork_local.train()
                actions = q.argmax(dim=1).cpu().numpy()

        if single:
            return int(actions[0])
        return actions.astype(np.int64)

    # ------------------------------------------------------------ learn
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store one transition and learn every ``learn_every`` steps."""
        self.memory.add(state, action, reward, next_state, terminated, truncated)
        self.decay_epsilon()  # advances env_steps by one
        if self.env_steps % self.config.learn_every == 0 and len(self.memory) >= max(
            self.config.min_buffer_samples, self.config.batch_size
        ):
            loss = self.learn()
            self.learn_steps += 1
            self._maybe_update_target()

    def learn(self) -> float:
        """One Double-DQN gradient step; returns the loss value."""
        states, actions, rewards, next_states, terminated, _truncated = self.memory.sample()

        with torch.no_grad():
            # Double DQN: local net selects, target net evaluates.
            best_actions = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
            q_next = self.qnetwork_target(next_states).gather(1, best_actions)
            # (B,) -> (B, 1) to avoid broadcasting against q_next (B, 1).
            rewards_col = rewards.unsqueeze(1)
            terminated_col = terminated.unsqueeze(1)
            # Only terminated masks the bootstrap (Gymnasium semantics).
            q_targets = rewards_col + self.config.gamma * q_next * (1.0 - terminated_col)

        q_expected = self.qnetwork_local(states).gather(1, actions.long().unsqueeze(1))
        loss = self.loss_fn(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        grad_clip = self.config.gradient_clip
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def _maybe_update_target(self) -> None:
        cfg = self.config
        if cfg.target_update_mode == "soft":
            self.soft_update()
        elif self.learn_steps % cfg.target_update_every == 0:
            self.hard_update()

    def soft_update(self) -> None:
        """Polyak: theta_target = tau*theta_local + (1-tau)*theta_target."""
        tau = self.config.tau
        with torch.no_grad():
            for tp, lp in zip(
                self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
            ):
                tp.data.mul_(1.0 - tau).add_(lp.data, alpha=tau)

    def hard_update(self) -> None:
        """Copy local weights exactly onto the target network."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    # ------------------------------------------------------- checkpoints
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model, optimizer, and schedules to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "qnetwork_local": self.qnetwork_local.state_dict(),
                "qnetwork_target": self.qnetwork_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "env_steps": self.env_steps,
                "learn_steps": self.learn_steps,
                "config": self.config.as_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Restore a checkpoint saved by ``save_checkpoint``."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("state_dim") != self.state_dim or payload.get("action_dim") != self.action_dim:
            raise ValueError(
                "Checkpoint dims do not match agent: "
                f"ckpt=({payload.get('state_dim')}, {payload.get('action_dim')}), "
                f"agent=({self.state_dim}, {self.action_dim})."
            )
        self.qnetwork_local.load_state_dict(payload["qnetwork_local"])
        self.qnetwork_target.load_state_dict(payload["qnetwork_target"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.env_steps = int(payload.get("env_steps", 0))
        self.learn_steps = int(payload.get("learn_steps", 0))

    def set_greedy(self) -> None:
        """Convenience: force epsilon to its minimum (greedy policy)."""
        # epsilon is derived from env_steps; jumping far ahead reaches min.
        self.env_steps = max(
            self.env_steps,
            int(np.ceil(np.log(self.config.epsilon_min / self.config.epsilon_start) /
                        np.log(self.config.epsilon_decay))) if self.config.epsilon_decay < 1 else 0,
        )
