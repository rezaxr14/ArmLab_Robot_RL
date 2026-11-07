"""Actor and batched critic ensemble.

The critic ensemble evaluates all ``E`` networks in one ``baddbmm`` per layer instead of a Python
loop, which matters on a GPU where five tiny MLPs would otherwise be launch-bound.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class EnsembleLinear(nn.Module):
    """``E`` independent linear layers: input ``[E, B, in]`` -> output ``[E, B, out]``."""

    def __init__(self, ensemble: int, in_features: int, out_features: int):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(ensemble, in_features, out_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(ensemble, 1, out_features).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, x, self.weight)


class QuantileCritic(nn.Module):
    """``E`` critics, each predicting ``N`` return quantiles for a (state-goal, action) pair."""

    def __init__(self, in_dim: int, act_dim: int, hidden: tuple[int, ...], ensemble: int, n_quantiles: int, layer_norm: bool):
        super().__init__()
        self.ensemble, self.n_quantiles = ensemble, n_quantiles
        dims = (in_dim + act_dim, *hidden)
        self.layers = nn.ModuleList([EnsembleLinear(ensemble, a, b) for a, b in zip(dims[:-1], dims[1:], strict=False)])
        self.norms = nn.ModuleList([nn.LayerNorm(b) if layer_norm else nn.Identity() for b in dims[1:]])
        self.head = EnsembleLinear(ensemble, dims[-1], n_quantiles)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns quantiles ``[B, E, N]``."""
        x = torch.cat([state, action], dim=-1).unsqueeze(0).expand(self.ensemble, -1, -1)
        for layer, norm in zip(self.layers, self.norms, strict=False):
            x = F.relu(norm(layer(x)))
        return self.head(x).permute(1, 0, 2)


class TanhGaussianActor(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden: tuple[int, ...], layer_norm: bool):
        super().__init__()
        dims = (in_dim, *hidden)
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=False):
            blocks.append(nn.Linear(a, b))
            if layer_norm:
                blocks.append(nn.LayerNorm(b))
            blocks.append(nn.ReLU())
        self.body = nn.Sequential(*blocks)
        self.mu = nn.Linear(dims[-1], act_dim)
        self.log_std = nn.Linear(dims[-1], act_dim)

    def forward(self, state: torch.Tensor, deterministic: bool = False, with_logp: bool = True):
        h = self.body(state)
        mu = self.mu(h)
        if deterministic:
            return torch.tanh(mu), None, mu
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        action = torch.tanh(u)
        logp = None
        if with_logp:
            normal_logp = -0.5 * ((u - mu) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)
            # log|d tanh(u)/du| = 2 (log 2 - u - softplus(-2u)), numerically stable form
            correction = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
            logp = (normal_logp - correction).sum(-1)
        return action, logp, mu
