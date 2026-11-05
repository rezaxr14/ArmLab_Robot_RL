"""Training loop: parallel collection, HER replay, success-gated curriculum, evaluation, snapshots.

A run directory is self-describing and resumable::

    runs/<name>/config.yaml        # exact RunConfig used
    runs/<name>/metrics.csv        # one row per log interval (train + eval + losses)
    runs/<name>/latest.pt          # full state (agent + optimisers + counters) for resuming
    runs/<name>/best.pt            # policy with the best evaluation success so far
    runs/<name>/milestones/*.pt    # policy snapshots used to render failing -> learning -> solved
    runs/<name>/summary.json       # written at the end
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..config import RunConfig, save_config
from ..env.arm_env import ArmForgeEnv
from ..env.tasks import make_task
from ..env.vec import make_vec_env
from .agent import SACAgent, resolve_device
from .buffer import EpisodeBuffer
from .evaluate import EVAL_SEED_BASE, agent_policy, evaluate_policy
from .expert import rollout_expert

METRIC_FIELDS = [
    "steps", "episodes", "time_s", "sps", "train_success", "goal_scale", "eval_success",
    "eval_ever_success", "eval_final_distance", "eval_return", "critic_loss", "actor_loss",
    "alpha", "q_mean", "entropy", "buffer_episodes", "updates",
]  # fmt: skip


def milestone_steps(milestones: tuple[float, ...], total_steps: int) -> list[int]:
    """Fractions (<= 1.0) of ``total_steps`` or absolute counts (> 1) -> sorted unique step numbers."""
    out = {int(round(m * total_steps)) if m <= 1.0 else int(m) for m in milestones}
    return sorted(min(s, total_steps) for s in out)


def milestone_name(step: int) -> str:
    return f"step_{step:09d}.pt"


class Trainer:
    def __init__(self, cfg: RunConfig, resume: bool = False, menagerie_root: str | None = None, log=print):
        self.cfg, self.log, self.menagerie_root = cfg, log, menagerie_root
        self.run_dir = Path(cfg.out_dir) / cfg.name
        (self.run_dir / "milestones").mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.run_dir / "config.yaml")
        tc = cfg.train
        self.device = resolve_device(tc.device)
        torch.manual_seed(tc.seed)
        np.random.seed(tc.seed)
        self.rng = np.random.default_rng(tc.seed)

        self.vec = make_vec_env(cfg.env, tc.num_envs, tc.vec_backend, menagerie_root, base_seed=tc.seed * 7919)
        self.eval_env = ArmForgeEnv(replace(cfg.env, seed=EVAL_SEED_BASE), menagerie_root)
        e = self.eval_env
        self.obs_dim, self.goal_dim, self.act_dim, self.horizon = e.obs_dim, e.goal_dim, e.action_dim, e.horizon

        t = cfg.env.task
        task = make_task(t.name, t.push_height)
        lo = -(1.0 + task.max_extra_penalty(t.shaping)) / (1.0 - cfg.algo.gamma)
        return_range = (lo, 0.0) if t.reward == "sparse" else None
        self.agent = SACAgent(self.obs_dim, self.goal_dim, self.act_dim, cfg.algo, self.device, return_range)
        self.buffer = EpisodeBuffer(
            tc.buffer_episodes, self.horizon, self.obs_dim, self.goal_dim, self.act_dim, self.device, t.success_threshold, t.reward
        )

        cur = tc.curriculum
        self.goal_scale = cur.start_scale if cur.enabled else 1.0
        self.steps = 0
        self.episodes = 0
        self.unstable_episodes = 0
        self.demo_steps = 0
        self.saved_milestones: set[int] = set()
        self.best_eval = (-1.0, float("inf"))
        self.resumed = False
        self.min_buffer_episodes = 0  # no gradient steps until the buffer holds this many episodes (resume without a saved buffer)
        if resume and (self.run_dir / "latest.pt").exists():
            self._load_latest()

        self.train_success = deque(maxlen=cur.window_episodes)
        self._since_scale_change = 0
        self._csv_path = self.run_dir / "metrics.csv"
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as fh:
                csv.DictWriter(fh, METRIC_FIELDS).writeheader()

    # ------------------------------------------------------------------ persistence
    def _save_latest(self) -> None:
        state = {
            "agent": self.agent.state_dict(),
            "steps": self.steps,
            "episodes": self.episodes,
            "goal_scale": self.goal_scale,
            "milestones": sorted(self.saved_milestones),
            "best_eval": list(self.best_eval),
        }
        tmp = self.run_dir / "latest.pt.tmp"
        torch.save(state, tmp)
        tmp.replace(self.run_dir / "latest.pt")
        if self.cfg.train.save_buffer and self.buffer.size > 0:
            btmp = self.run_dir / "latest_buffer.pt.tmp"
            torch.save(self.buffer.state_dict(), btmp)
            btmp.replace(self.run_dir / "latest_buffer.pt")

    def _load_latest(self) -> None:
        state = torch.load(self.run_dir / "latest.pt", map_location=self.device, weights_only=False)  # our own file
        self.agent.load_state_dict(state["agent"])
        self.steps, self.episodes = state["steps"], state["episodes"]
        self.goal_scale = state["goal_scale"]
        self.saved_milestones = set(state["milestones"])
        self.best_eval = tuple(state["best_eval"])
        self.resumed = True
        buffer_file = self.run_dir / "latest_buffer.pt"
        if buffer_file.exists():
            self.buffer.load_state_dict(torch.load(buffer_file, map_location="cpu", weights_only=False))
            self.log(f"[resume] step {self.steps}, goal_scale {self.goal_scale:.2f}, replay buffer restored ({self.buffer.size} episodes)")
        else:
            self.min_buffer_episodes = self.cfg.train.resume_refill_episodes
            self.log(
                f"[resume] step {self.steps}, goal_scale {self.goal_scale:.2f}; no saved replay buffer - collecting "
                f"{self.min_buffer_episodes} episodes with the loaded policy before training resumes"
            )

    def _snapshot(self, step: int) -> None:
        if step in self.saved_milestones:
            return
        self.agent.save(self.run_dir / "milestones" / milestone_name(step), extra={"step": step}, policy_only=True)
        self.saved_milestones.add(step)
        self.log(f"[milestone] saved policy snapshot at step {step}")

    # ------------------------------------------------------------------ demonstrations
    def _seed_with_demos(self) -> None:
        d = self.cfg.train.demos
        if d.episodes <= 0 or self.resumed:
            return
        self.log(f"[demos] collecting {d.episodes} scripted-expert episodes (noise {d.noise_std})")
        rng = np.random.default_rng(self.cfg.train.seed + 17)
        kept = succ = 0
        for i in range(d.episodes):
            ep = rollout_expert(self.eval_env, seed=500_000 + i, noise_std=d.noise_std, goal_scale=self.goal_scale, rng=rng)
            succ += int(ep["success"])
            if d.keep_only_successful and not ep["success"]:
                continue
            self._add_episode(ep["obs"], ep["ag"], ep["goal"], ep["act"], ep["shaping"])
            kept += 1
            self.demo_steps += self.horizon
        self.log(f"[demos] expert success {succ}/{d.episodes}; stored {kept} episodes")
        if kept == 0:
            self.log("[demos] no successful demonstrations - skipping behaviour cloning")
            return
        for k in range(d.bc_steps):
            o, g, a = self.buffer.sample_actions(d.bc_batch_size)
            loss = self.agent.bc_update(o, g, a)
            if (k + 1) % max(1, d.bc_steps // 4) == 0:
                self.log(f"[bc] {k + 1}/{d.bc_steps} loss {loss:.4f}")

    def _add_episode(self, obs, ag, goal, act, shaping) -> None:
        self.buffer.add(obs[None], ag[None], np.asarray(goal)[None], act[None], shaping[None])
        self.agent.obs_norm.update(obs)
        self.agent.goal_norm.update(np.concatenate([ag, np.asarray(goal)[None]], axis=0))

    # ------------------------------------------------------------------ curriculum
    def _maybe_adjust_curriculum(self) -> None:
        cur = self.cfg.train.curriculum
        if not cur.enabled or len(self.train_success) < cur.window_episodes // 2:
            return
        if self._since_scale_change < cur.window_episodes // 2:
            return
        rate = float(np.mean(self.train_success))
        new = self.goal_scale
        if rate >= cur.promote_success:
            new = min(cur.end_scale, self.goal_scale + cur.step)
        elif rate <= cur.demote_success:
            new = max(cur.start_scale, self.goal_scale - cur.step)
        if new != self.goal_scale:
            self.log(f"[curriculum] train success {rate:.2f} -> goal_scale {self.goal_scale:.2f} -> {new:.2f}")
            self.goal_scale = new
            self.vec.set_goal_scale(new)
            self.train_success.clear()
            self._since_scale_change = 0

    # ------------------------------------------------------------------ main loop
    def _evaluate(self) -> dict:
        tc = self.cfg.train
        res = evaluate_policy(self.cfg.env, agent_policy(self.agent), tc.eval_episodes, seed0=EVAL_SEED_BASE, goal_scale=1.0, env=self.eval_env)
        key = (res["success_rate"], -res["mean_final_distance"])
        if key > (self.best_eval[0], -self.best_eval[1]):
            self.best_eval = (res["success_rate"], res["mean_final_distance"])
            self.agent.save(self.run_dir / "best.pt", extra={"step": self.steps, **{k: v for k, v in res.items() if k != "rows"}}, policy_only=True)
        return res

    def _log_row(self, t0: float, step0: int, last_losses: dict, evaluation: dict | None) -> None:
        elapsed = time.time() - t0
        row = {
            "steps": self.steps,
            "episodes": self.episodes,
            "time_s": round(elapsed, 1),
            "sps": round((self.steps - step0) / max(elapsed, 1e-9), 1),
            "train_success": round(float(np.mean(self.train_success)), 4) if self.train_success else "",
            "goal_scale": round(self.goal_scale, 3),
            "buffer_episodes": len(self.buffer),
            "updates": self.agent.updates,
            **{k: round(v, 5) for k, v in last_losses.items() if k in METRIC_FIELDS},
        }
        if evaluation is not None:
            row.update(
                eval_success=evaluation["success_rate"],
                eval_ever_success=evaluation["ever_success_rate"],
                eval_final_distance=round(evaluation["mean_final_distance"], 5),
                eval_return=round(evaluation["mean_return"], 3),
            )
        with open(self._csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, METRIC_FIELDS).writerow(row)
        msg = f"step {self.steps:>9,d} | eps {self.episodes:>6,d} | sps {row['sps']:>6} | train_succ {row['train_success']} | scale {row['goal_scale']}"
        if evaluation is not None:
            msg += f" | EVAL succ {evaluation['success_rate']:.2f} dist {evaluation['mean_final_distance']:.3f}"
        if "critic_loss" in last_losses:
            msg += f" | critic {last_losses['critic_loss']:.3f} alpha {last_losses['alpha']:.3f}"
        self.log(msg)

    def train(self) -> dict:
        cfg, tc, ac = self.cfg, self.cfg.train, self.cfg.algo
        marks = milestone_steps(tc.milestones, tc.total_steps)
        if 0 in marks and not self.resumed:
            self._snapshot(0)  # the untrained, randomly initialised policy
        self._seed_with_demos()

        n = tc.num_envs
        obs = self.vec.reset(seeds=[int(self.rng.integers(0, 900_000)) for _ in range(n)], goal_scale=self.goal_scale)
        ep = [self._new_episode(obs, i) for i in range(n)]
        credit, losses, evaluation = 0.0, {}, None
        next_eval = (self.steps // tc.eval_every + 1) * tc.eval_every
        next_log = (self.steps // tc.log_every + 1) * tc.log_every
        next_ckpt = (self.steps // tc.checkpoint_every + 1) * tc.checkpoint_every
        t0, step0 = time.time(), self.steps
        random_until = 0 if self.resumed else tc.start_steps

        while self.steps < tc.total_steps:
            if self.steps < random_until:
                actions = self.rng.uniform(-1, 1, (n, self.act_dim)).astype(np.float32)
            else:
                actions = self.agent.act(obs["observation"], obs["desired_goal"])
            next_obs, _, dones, infos, terminal = self.vec.step(actions)

            for i in range(n):
                ep[i]["obs"].append(terminal["observation"][i])
                ep[i]["ag"].append(terminal["achieved_goal"][i])
                ep[i]["act"].append(actions[i])
                ep[i]["shp"].append(infos[i]["shaping"])
                ep[i]["bad"] = ep[i].get("bad", False) or bool(infos[i].get("unstable", False))
                if dones[i] and ep[i]["bad"]:
                    self.unstable_episodes += 1  # a physics blow-up: never let it into the buffer or the normalisers
                    if self.unstable_episodes in (1, 10, 100) or self.unstable_episodes % 1000 == 0:
                        self.log(f"[warn] {self.unstable_episodes} episode(s) discarded because the simulation diverged")
                    ep[i] = self._new_episode(next_obs, i)
                elif dones[i]:
                    e = ep[i]
                    self._add_episode(np.stack(e["obs"]), np.stack(e["ag"]), e["goal"], np.stack(e["act"]).astype(np.float32), np.array(e["shp"], np.float32))
                    self.episodes += 1
                    self._since_scale_change += 1
                    self.train_success.append(float(infos[i]["is_success"]))
                    ep[i] = self._new_episode(next_obs, i)
            obs = next_obs
            self.steps += n

            if self.steps >= tc.warmup_updates_after and self.buffer.transitions >= ac.batch_size and self.buffer.size >= self.min_buffer_episodes:
                credit += n * ac.updates_per_step
                while credit >= 1.0:
                    losses = self.agent.update(self.buffer.sample(ac.batch_size, ac.her_prob))
                    credit -= 1.0
            self._maybe_adjust_curriculum()

            for m in marks:
                if m not in self.saved_milestones and m > 0 and self.steps >= m and m < tc.total_steps:
                    self._snapshot(m)
            if self.steps >= next_eval:
                evaluation = self._evaluate()
                next_eval += tc.eval_every
                self._log_row(t0, step0, losses, evaluation)  # eval rows are always logged
                evaluation = None
                next_log = (self.steps // tc.log_every + 1) * tc.log_every
            elif self.steps >= next_log:
                self._log_row(t0, step0, losses, None)
                next_log += tc.log_every
            if self.steps >= next_ckpt:
                self._save_latest()
                next_ckpt += tc.checkpoint_every

        final_eval = self._evaluate()
        self._snapshot(tc.total_steps)
        self.agent.save(self.run_dir / "final.pt", extra={"step": self.steps}, policy_only=True)
        self._log_row(t0, step0, losses, final_eval)
        self._save_latest()
        summary = {
            "name": cfg.name,
            "task": cfg.env.task.name,
            "total_steps": self.steps,
            "episodes": self.episodes,
            "unstable_episodes_discarded": self.unstable_episodes,
            "demo_steps": self.demo_steps,
            "wall_time_s": round(time.time() - t0, 1),
            "device": str(self.device),
            "final_eval": {k: v for k, v in final_eval.items() if k != "rows"},
            "best_eval_success": self.best_eval[0],
            "milestones": sorted(self.saved_milestones),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    @staticmethod
    def _new_episode(obs: dict, i: int) -> dict:
        return {
            "obs": [obs["observation"][i]],
            "ag": [obs["achieved_goal"][i]],
            "goal": obs["desired_goal"][i],
            "act": [],
            "shp": [],
        }

    def close(self) -> None:
        self.vec.close()
        self.eval_env.close()
