"""Scripted experts using privileged simulator state.

They exist for two reasons: (1) to seed the replay buffer with successful demonstrations, and (2)
as an upper-bound reference in the README ("what a hand-written controller achieves"). They act
through the same action interface as the learned policy.
"""

from __future__ import annotations

import numpy as np


def _p_action(error: np.ndarray, max_step: float, gain: float = 1.0) -> np.ndarray:
    return np.clip(gain * error / max_step, -1.0, 1.0)


class ReachExpert:
    def __init__(self, env):
        self.env = env

    def reset(self) -> None:
        pass

    def act(self) -> np.ndarray:
        e = self.env
        return _p_action(e.goal - e.ee_pos, e.cfg.task.max_ee_step)


class PushExpert:
    """Get behind the cube on the far side from the goal (going around it), then push straight."""

    def __init__(self, env, standoff: float = 0.065):
        self.env, self.standoff = env, standoff

    def reset(self) -> None:
        pass

    def act(self) -> np.ndarray:
        e = self.env
        step = e.cfg.task.max_ee_step
        obj, ee, goal = e.object_pos("cube")[:2], e.ee_pos[:2], e.goal[:2]
        to_goal = goal - obj
        dist = np.linalg.norm(to_goal)
        if dist < 1e-6:
            return np.zeros(2)
        u = to_goal / dist
        n = np.array([-u[1], u[0]])
        rel = ee - obj
        along, lateral = float(rel @ u), float(rel @ n)
        behind = obj - self.standoff * u
        if along < -0.03 and abs(lateral) < 0.025:
            # lined up behind the cube: push, correcting sideways drift
            return _p_action(u * 0.03 - n * lateral * 0.6, step)
        if along > -0.03:
            # on the wrong side or beside it: swing wide around the cube first
            side = 1.0 if lateral >= 0 else -1.0
            waypoint = obj + n * side * 0.10 - u * 0.02
            if along < -0.08:
                waypoint = behind
            return _p_action(waypoint - ee, step)
        return _p_action(behind - ee, step)


class PickPlaceExpert:
    """Hover -> descend -> close -> lift -> carry -> lower -> open -> retreat."""

    GRASP_Z = 0.015
    HOVER_Z = 0.12

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self) -> None:
        self.phase = "hover"
        self.wait = 0
        self.phase_log: list[tuple[int, str]] = [(0, "hover")]
        self._t = 0

    def _go(self, phase: str) -> None:
        self.phase, self.wait = phase, 0
        self.phase_log.append((self._t, phase))

    def act(self) -> np.ndarray:
        self._t += 1
        e = self.env
        step = e.cfg.task.max_ee_step
        ee, obj, goal = e.ee_pos, e.object_pos("cube"), e.goal
        close_g, open_g = 1.0, -1.0
        in_air = goal[2] > e.layout.object_half_size + 0.03  # an air goal must be *held*, not released

        if self.phase == "hover":
            target = np.array([obj[0], obj[1], self.HOVER_Z])
            if np.linalg.norm(target[:2] - ee[:2]) < 0.008 and abs(ee[2] - self.HOVER_Z) < 0.01:
                self._go("descend")
            return np.append(_p_action(target - ee, step), open_g)
        if self.phase == "descend":
            target = np.array([obj[0], obj[1], self.GRASP_Z])
            if abs(ee[2] - self.GRASP_Z) < 0.006:
                self._go("close")
            return np.append(_p_action(target - ee, step), open_g)
        if self.phase == "close":
            self.wait += 1
            if self.wait >= 5 and e.pad_contacts().all():
                self._go("carry")
            elif self.wait > 12:
                self._go("hover")  # missed the cube: retry
            return np.append(np.zeros(3), close_g)
        if self.phase == "carry":
            # servo the *cube* (not the hand) to a point above the goal, then to the goal
            above = goal + np.array([0.0, 0.0, max(0.05, self.HOVER_Z - goal[2])]) if np.linalg.norm(obj[:2] - goal[:2]) > 0.02 else goal
            err = above - obj
            lost = e.gripper_opening() > 0.9 or np.linalg.norm(obj - ee) > 0.06  # closed on nothing / cube fell away
            self.wait = self.wait + 1 if lost else 0
            if self.wait >= 2:
                self._go("hover")  # dropped it: open, go back above the cube and regrasp
                return np.append(np.zeros(3), open_g)
            if np.linalg.norm(obj - goal) < 0.008:
                self._go("hold" if in_air else "release")
            lift_first = obj[2] < 0.05 and np.linalg.norm(obj[:2] - goal[:2]) > 0.02
            if lift_first:
                err = np.array([0.0, 0.0, 0.06])
            return np.append(_p_action(err, step), close_g)
        if self.phase == "hold":
            # keep the cube at the in-air goal with the gripper closed (success is judged at the last step)
            return np.append(_p_action(goal - obj, step), close_g)
        if self.phase == "release":
            self.wait += 1
            if self.wait > 4:
                self._go("retreat")
            return np.append(np.zeros(3), open_g)
        # retreat: straight up, gripper open
        return np.append(_p_action(np.array([0.0, 0.0, 0.08]), step), open_g)


def make_expert(env):
    return {"reach": ReachExpert, "push": PushExpert, "pickplace": PickPlaceExpert}[env.cfg.task.name](env)


def rollout_expert(env, seed: int, noise_std: float = 0.0, goal_scale: float = 1.0, rng: np.random.Generator | None = None):
    """Run the expert for one episode; returns arrays shaped like :class:`EpisodeBuffer` rows."""
    rng = rng or np.random.default_rng(seed)
    obs = env.reset(seed=seed, goal_scale=goal_scale)
    expert = make_expert(env)
    expert.reset()
    O, AG, A, S, trace = [obs["observation"]], [obs["achieved_goal"]], [], [], []
    goal, info, done = obs["desired_goal"], {}, False
    while not done:
        a = np.clip(expert.act() + (rng.normal(0, noise_std, env.action_dim) if noise_std > 0 else 0.0), -1, 1)
        obs, _, done, info = env.step(a)
        O.append(obs["observation"])
        AG.append(obs["achieved_goal"])
        A.append(a)
        S.append(info["shaping"])
        trace.append(
            {
                "t": len(A),
                "phase": getattr(expert, "phase", ""),
                "ee": env.ee_pos.round(4).tolist(),
                "ag": obs["achieved_goal"].round(4).tolist(),
                "distance": round(info["distance"], 4),
                "grip": round(env.gripper_opening(), 3),
                "pads": env.pad_contacts().astype(int).tolist(),
                "ncon": info["ncon"],
                "unstable": bool(info.get("unstable", False)),
            }
        )
    return {
        "obs": np.stack(O),
        "ag": np.stack(AG),
        "goal": goal,
        "act": np.stack(A).astype(np.float32),
        "shaping": np.array(S, np.float32),
        "success": bool(info["is_success"]),
        "phases": list(getattr(expert, "phase_log", [])),
        "trace": trace,
        "goal_height": float(goal[2]),
        "seed": seed,
    }


def summarize_episode(ep: dict) -> str:
    """One line: outcome, goal type, the phase timeline and where it ended - to see *why* an episode failed."""
    final = ep["trace"][-1]
    phases = " ".join(f"{name}@{t}" for t, name in ep["phases"]) or "-"
    kind = "air goal" if ep["goal_height"] > 0.05 else "table goal"
    blown = next((x for x in ep["trace"] if x["unstable"]), None)
    if blown is not None:
        phases += f" | SIMULATION DIVERGED at t={blown['t']} in phase '{blown['phase']}'"
    return (
        f"{'ok  ' if ep['success'] else 'FAIL'} seed {ep['seed']} | {kind} | phases: {phases} | "
        f"final dist {final['distance'] * 100:.1f} cm, grip {final['grip']:.2f}, pads {final['pads']}"
    )
