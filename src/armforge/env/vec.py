"""Vectorised environments: a synchronous reference and a subprocess backend.

Both expose the same interface and *auto-reset*: when an episode ends the returned observation
is already the first observation of the next episode, and the finished episode's last
observation/info are available as ``terminal_obs`` / ``infos[i]``.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace

import numpy as np

from ..config import EnvConfig


def _stack(obs_list):
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def _make(cfg: EnvConfig, menagerie_root):
    from .arm_env import ArmForgeEnv  # imported lazily so workers import mujoco themselves

    return ArmForgeEnv(cfg, menagerie_root)


class SyncVecEnv:
    def __init__(self, cfgs: list[EnvConfig], menagerie_root=None):
        self.envs = [_make(c, menagerie_root) for c in cfgs]
        self.num_envs = len(self.envs)
        e0 = self.envs[0]
        self.obs_dim, self.action_dim, self.goal_dim, self.horizon = e0.obs_dim, e0.action_dim, e0.goal_dim, e0.horizon

    def reset(self, seeds=None, goal_scale=None):
        seeds = seeds if seeds is not None else [None] * self.num_envs
        return _stack([e.reset(seed=s, goal_scale=goal_scale) for e, s in zip(self.envs, seeds, strict=False)])

    def set_goal_scale(self, scale: float):
        for e in self.envs:
            e.set_goal_scale(scale)

    def step(self, actions):
        obs, rews, dones, infos, terminal = [], [], [], [], []
        for e, a in zip(self.envs, actions, strict=False):
            o, r, d, info = e.step(a)
            terminal.append(o)
            if d:
                o = e.reset()
            obs.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        return _stack(obs), np.array(rews, np.float32), np.array(dones), infos, _stack(terminal)

    def close(self):
        for e in self.envs:
            e.close()


def _worker(remote, cfg, menagerie_root):
    env = _make(cfg, menagerie_root)
    try:
        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                seed, scale = payload
                remote.send(env.reset(seed=seed, goal_scale=scale))
            elif cmd == "step":
                o, r, d, info = env.step(payload)
                terminal = o
                if d:
                    o = env.reset()
                remote.send((o, r, d, info, terminal))
            elif cmd == "scale":
                env.set_goal_scale(payload)
                remote.send(None)
            elif cmd == "meta":
                remote.send((env.obs_dim, env.action_dim, env.goal_dim, env.horizon))
            elif cmd == "close":
                env.close()
                remote.send(None)
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        remote.close()


class SubprocVecEnv:
    def __init__(self, cfgs: list[EnvConfig], menagerie_root=None):
        # "spawn" is required: forking after MuJoCo/CUDA initialisation is unsafe.
        ctx = mp.get_context("spawn")
        self.num_envs = len(cfgs)
        self.remotes, self.procs = [], []
        for cfg in cfgs:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, cfg, menagerie_root), daemon=True)
            proc.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(proc)
        self.remotes[0].send(("meta", None))
        self.obs_dim, self.action_dim, self.goal_dim, self.horizon = self.remotes[0].recv()

    def reset(self, seeds=None, goal_scale=None):
        seeds = seeds if seeds is not None else [None] * self.num_envs
        for r, s in zip(self.remotes, seeds, strict=False):
            r.send(("reset", (s, goal_scale)))
        return _stack([r.recv() for r in self.remotes])

    def set_goal_scale(self, scale: float):
        for r in self.remotes:
            r.send(("scale", scale))
        for r in self.remotes:
            r.recv()

    def step(self, actions):
        for r, a in zip(self.remotes, actions, strict=False):
            r.send(("step", a))
        results = [r.recv() for r in self.remotes]
        obs, rews, dones, infos, terminal = zip(*results, strict=False)
        return _stack(obs), np.array(rews, np.float32), np.array(dones), list(infos), _stack(terminal)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
                r.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)


def make_vec_env(cfg: EnvConfig, num_envs: int, backend: str = "subprocess", menagerie_root=None, base_seed: int = 0):
    """``num_envs`` copies of the env, each with its own seed stream."""
    cfgs = [replace(cfg, seed=base_seed + 1000 * i) for i in range(num_envs)]
    if backend == "sync" or num_envs == 1:
        return SyncVecEnv(cfgs, menagerie_root)
    if backend == "subprocess":
        return SubprocVecEnv(cfgs, menagerie_root)
    raise ValueError(f"unknown vec backend {backend!r}")
