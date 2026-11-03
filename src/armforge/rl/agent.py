"""SAC with an optional distributional TQC critic (Truncated Quantile Critics), goal-conditioned.

* ``critic="tqc"``: ``E`` critics x ``N`` quantiles. The next-state quantiles of all critics are
  pooled, sorted, and the largest ``drop`` per critic are discarded before forming the target, which
  controls value over-estimation without the pessimism of a hard ``min``.  (Kuznetsov et al., 2020)
* ``critic="twin"``: classic clipped double-Q SAC, expressed in the same code path as two critics
  with one quantile each and a single dropped value (so the target is ``min(Q1, Q2)``).

Inputs are ``[normalised observation, normalised goal]``. Hindsight relabelling happens in the buffer.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch

from ..config import AlgoConfig
from .networks import QuantileCritic, TanhGaussianActor
from .normalizer import RunningNorm


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return torch.device(name)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        cfg: AlgoConfig,
        device: torch.device,
        return_range: tuple[float, float] | None = None,
    ):
        self.cfg, self.device = cfg, device
        self.obs_dim, self.goal_dim, self.act_dim = obs_dim, goal_dim, act_dim
        self.return_range = return_range if cfg.clip_return else None
        in_dim = obs_dim + goal_dim

        if cfg.critic == "tqc":
            self.n_critics, self.n_quantiles = cfg.critic_ensemble, cfg.n_quantiles
            self.drop_total = cfg.top_quantiles_to_drop * cfg.critic_ensemble
        elif cfg.critic == "twin":
            self.n_critics, self.n_quantiles, self.drop_total = 2, 1, 1
        else:
            raise ValueError(f"unknown critic type {cfg.critic!r} (use 'tqc' or 'twin')")
        if self.drop_total >= self.n_critics * self.n_quantiles:
            raise ValueError("top_quantiles_to_drop removes every quantile")

        self.actor = TanhGaussianActor(in_dim, act_dim, cfg.hidden, cfg.layer_norm).to(device)
        self.critic = QuantileCritic(in_dim, act_dim, cfg.hidden, self.n_critics, self.n_quantiles, cfg.layer_norm).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=device, requires_grad=True)
        self.target_entropy = -cfg.target_entropy_scale * act_dim

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.obs_norm = RunningNorm(obs_dim, device, cfg.obs_clip)
        self.goal_norm = RunningNorm(goal_dim, device, cfg.obs_clip)
        self.tau_levels = ((torch.arange(self.n_quantiles, device=device, dtype=torch.float32) + 0.5) / self.n_quantiles).view(1, 1, -1, 1)
        self.updates = 0

    # ------------------------------------------------------------------ acting
    def _state(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.obs_norm(obs), self.goal_norm(goal)], dim=-1)

    @torch.no_grad()
    def act(self, obs, goal, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        single = obs.ndim == 1
        if single:
            obs, goal = obs.unsqueeze(0), goal.unsqueeze(0)
        action, _, _ = self.actor(self._state(obs, goal), deterministic=deterministic, with_logp=False)
        action = action.cpu().numpy()
        return action[0] if single else action

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ learning
    def _quantile_huber(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """current ``[B,E,N]``, target ``[B,K]`` -> scalar loss (summed over critics)."""
        delta = target[:, None, None, :] - current[:, :, :, None]  # [B,E,N,K]
        abs_delta = delta.abs()
        huber = torch.where(abs_delta <= 1.0, 0.5 * delta.pow(2), abs_delta - 0.5)
        weight = (self.tau_levels - (delta.detach() < 0).float()).abs()
        return (weight * huber).mean(dim=(0, 2, 3)).sum()

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        state = self._state(batch["obs"], batch["goal"])
        next_state = self._state(batch["next_obs"], batch["goal"])
        action, reward = batch["act"], batch["reward"]

        # ---- critic ----
        with torch.no_grad():
            next_action, next_logp, _ = self.actor(next_state)
            z_next = self.critic_target(next_state, next_action).reshape(state.shape[0], -1)
            z_next = torch.sort(z_next, dim=1).values[:, : z_next.shape[1] - self.drop_total]
            target = reward.unsqueeze(1) + cfg.gamma * (z_next - self.alpha.detach() * next_logp.unsqueeze(1))
            if self.return_range is not None:
                target = target.clamp(*self.return_range)
        current = self.critic(state, action)
        critic_loss = self._quantile_huber(current, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()

        # ---- actor + temperature (critic frozen for this pass) ----
        self.critic.requires_grad_(False)
        new_action, logp, mu = self.actor(state)
        q = self.critic(state, new_action).mean(dim=(1, 2))
        actor_loss = (self.alpha.detach() * logp - q).mean()
        if cfg.action_l2 > 0:
            actor_loss = actor_loss + cfg.action_l2 * mu.pow(2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()
        self.critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---- target network ----
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.lerp_(p, cfg.tau)
        self.updates += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": q.mean().item(),
            "entropy": -logp.mean().item(),
            "reward_mean": reward.mean().item(),
        }

    def bc_update(self, obs: torch.Tensor, goal: torch.Tensor, expert_action: torch.Tensor) -> float:
        """Behaviour-cloning step on the actor's deterministic output (demonstration warm start)."""
        state = self._state(obs, goal)
        pred, _, _ = self.actor(state, deterministic=True)
        loss = (pred - expert_action).pow(2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()
        return float(loss.item())

    # ------------------------------------------------------------------ persistence
    def policy_state(self) -> dict:
        """Everything needed to *run* the policy (no optimisers, no critics)."""
        return {
            "actor": self.actor.state_dict(),
            "obs_norm": self.obs_norm.state_dict(),
            "goal_norm": self.goal_norm.state_dict(),
            "dims": (self.obs_dim, self.goal_dim, self.act_dim),
        }

    def state_dict(self) -> dict:
        return {
            **self.policy_state(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "updates": self.updates,
        }

    def load_policy_state(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.obs_norm.load_state_dict(state["obs_norm"])
        self.goal_norm.load_state_dict(state["goal_norm"])

    def load_state_dict(self, state: dict) -> None:
        self.load_policy_state(state)
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])
        self.updates = state["updates"]

    def save(self, path: str | Path, extra: dict | None = None, policy_only: bool = False) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = self.policy_state() if policy_only else self.state_dict()
        payload["algo_cfg"] = self.cfg.__dict__
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)
