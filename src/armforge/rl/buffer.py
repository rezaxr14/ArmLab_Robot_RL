"""Episode replay buffer with on-the-fly hindsight relabelling (the *future* strategy).

Whole fixed-length episodes are stored as dense tensors on the training device::

    obs [cap, T+1, obs_dim]   ag [cap, T+1, goal_dim]   goal [cap, goal_dim]
    act [cap, T,   act_dim]   shaping [cap, T]

A sample is ``(episode, t)``. With probability ``her_prob`` its goal is replaced by the achieved
goal at a uniformly random *future* step ``t' in [t+1, T]`` of the same episode, and the reward is
recomputed for that goal - so a failed episode still teaches how to reach where it actually got.
Because episodes are fixed length (time-limit truncation only), bootstrapping is never cut.
"""

from __future__ import annotations

import torch

from ..env.reward import goal_distance, goal_reward


class EpisodeBuffer:
    def __init__(self, capacity: int, horizon: int, obs_dim: int, goal_dim: int, act_dim: int, device, threshold: float, reward_mode: str):
        self.capacity, self.horizon, self.device = capacity, horizon, device
        self.threshold, self.reward_mode = threshold, reward_mode
        f = dict(dtype=torch.float32, device=device)
        self.obs = torch.zeros(capacity, horizon + 1, obs_dim, **f)
        self.ag = torch.zeros(capacity, horizon + 1, goal_dim, **f)
        self.goal = torch.zeros(capacity, goal_dim, **f)
        self.act = torch.zeros(capacity, horizon, act_dim, **f)
        self.shaping = torch.zeros(capacity, horizon, **f)
        self.size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self.size

    @property
    def transitions(self) -> int:
        return self.size * self.horizon

    def add(self, obs, ag, goal, act, shaping) -> None:
        """Add ``n`` episodes (numpy or tensors, leading dimension ``n``)."""
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)  # noqa: E731
        obs, ag, goal, act, shaping = map(t, (obs, ag, goal, act, shaping))
        n = obs.shape[0]
        idx = (self._ptr + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx], self.ag[idx], self.goal[idx], self.act[idx], self.shaping[idx] = obs, ag, goal, act, shaping
        self._ptr = (self._ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, her_prob: float):
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        dev = self.device
        ep = torch.randint(0, self.size, (batch_size,), device=dev)
        t = torch.randint(0, self.horizon, (batch_size,), device=dev)
        # future index in [t+1, T]
        span = self.horizon - t  # number of candidate future steps, >= 1
        offset = torch.minimum((torch.rand(batch_size, device=dev) * span.float()).long(), span - 1)
        future = t + 1 + offset
        relabel = torch.rand(batch_size, device=dev) < her_prob
        goal = torch.where(relabel.unsqueeze(-1), self.ag[ep, future], self.goal[ep])
        next_ag = self.ag[ep, t + 1]
        dist = goal_distance(next_ag, goal)
        reward = goal_reward(dist, self.threshold, self.reward_mode) + self.shaping[ep, t]
        return {
            "obs": self.obs[ep, t],
            "act": self.act[ep, t],
            "reward": reward,
            "next_obs": self.obs[ep, t + 1],
            "goal": goal,
            "relabelled": relabel,
        }

    def sample_actions(self, batch_size: int):
        """Plain ``(obs, goal, action)`` triples without relabelling (used for BC on demos)."""
        ep = torch.randint(0, self.size, (batch_size,), device=self.device)
        t = torch.randint(0, self.horizon, (batch_size,), device=self.device)
        return self.obs[ep, t], self.goal[ep], self.act[ep, t]
