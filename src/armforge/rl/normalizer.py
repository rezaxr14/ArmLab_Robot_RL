"""Running mean/std normaliser (Welford/Chan parallel update) living on the training device."""

from __future__ import annotations

import torch


class RunningNorm:
    def __init__(self, dim: int, device, clip: float = 5.0, eps: float = 1e-4):
        self.dim = dim
        self.device = device
        self.clip = clip
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = torch.tensor(eps, device=device)

    @torch.no_grad()
    def update(self, x) -> None:
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(-1, self.dim)
        n = torch.tensor(float(x.shape[0]), device=self.device)
        if n == 0:
            return
        b_mean, b_var = x.mean(0), x.var(0, unbiased=False)
        delta = b_mean - self.mean
        total = self.count + n
        new_mean = self.mean + delta * n / total
        m2 = self.var * self.count + b_var * n + delta.pow(2) * self.count * n / total
        self.mean, self.var, self.count = new_mean, m2 / total, total

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var + 1e-8).clamp_min(1e-2)  # floor: constant features stay finite
        return ((x - self.mean) / std).clamp(-self.clip, self.clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count.cpu()}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"].to(self.device)
        self.var = state["var"].to(self.device)
        self.count = state["count"].to(self.device)
