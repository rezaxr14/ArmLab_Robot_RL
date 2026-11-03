"""Task definitions: what is manipulated, how goals are sampled, extra observations, shaping.

Three tasks share one robot and one physics model:

* ``reach``      - move the Robotiq pinch point to a 3D goal (gripper closed).
* ``push``       - push a free 40 mm cube across the table to a goal with a *closed* gripper;
                   the outcome is decided by sliding/torsional friction and contact geometry.
* ``pickplace``  - grasp the cube with the 2F-85, lift it over the tray wall (optional) and put
                   it at a goal on the table, in the tray, or in the air.

Goals are 3D positions of the *achieved* quantity (end effector for reach, cube for the rest),
so one relabelling rule (hindsight experience replay) serves all tasks.
"""

from __future__ import annotations

import numpy as np

from ..mathutil import yaw_to_quat

# observation field names produced by the object block, in order
OBJECT_FIELDS = ("obj_pos", "obj_rel", "obj_quat", "obj_vel", "obj_angvel", "contacts")


class Task:
    name = "base"
    action_dim = 3
    uses_object = False
    grip_closed_at_start = True
    lock_z: float | None = None

    # ---- sampling ---------------------------------------------------------------------
    def sample_ee_start(self, env) -> np.ndarray:
        cx, cy = env.layout.workspace_center
        jitter = env.rng.uniform(-0.04, 0.04, size=2)
        return np.array([cx + jitter[0], cy + jitter[1], env.start_height])

    def reset_scene(self, env, goal_scale: float) -> None:
        raise NotImplementedError

    # ---- per-step -----------------------------------------------------------------------
    def command(self, env, action: np.ndarray) -> tuple[np.ndarray, float | None]:
        """Return ``(cartesian_delta_m, grip_delta_or_None)`` for a policy action."""
        raise NotImplementedError

    def achieved_goal(self, env) -> np.ndarray:
        raise NotImplementedError

    def extra_obs(self, env) -> list[tuple[str, np.ndarray]]:
        return []

    def extra_reward(self, env) -> float:
        return 0.0

    def max_extra_penalty(self, shaping: float) -> float:
        return 0.0


def _clip_xy_into_workspace(env, xy: np.ndarray, margin: float = 0.02) -> np.ndarray:
    cx, cy = env.layout.workspace_center
    hx, hy = env.cfg.task.workspace_half_extent
    return np.array(
        [
            np.clip(xy[0], cx - hx + margin, cx + hx - margin),
            np.clip(xy[1], cy - hy + margin, cy + hy - margin),
        ]
    )


class ReachTask(Task):
    name = "reach"
    action_dim = 3
    uses_object = False
    grip_closed_at_start = True

    def reset_scene(self, env, goal_scale: float) -> None:
        t = env.cfg.task
        cx, cy = env.layout.workspace_center
        hx, hy = t.workspace_half_extent
        start = env.ee_start
        # no objects are used in this task: park them at the table edge, out of the hand's way
        h = env.layout.object_half_size
        for k, name in enumerate(("cube", *env.layout.distractor_names)):
            env.set_free_body(name, [cx + 0.40, cy - 0.50 + 0.07 * k, h + 5e-4], [1, 0, 0, 0])
        for _ in range(20):
            full = np.array(
                [
                    env.rng.uniform(cx - hx + 0.02, cx + hx - 0.02),
                    env.rng.uniform(cy - hy + 0.02, cy + hy - 0.02),
                    env.rng.uniform(0.04, min(0.28, t.workspace_z[1] - 0.02)),
                ]
            )
            goal = start + goal_scale * (full - start)
            if np.linalg.norm(goal - start) >= t.min_goal_distance:
                break
        env.goal = goal
        env.place_goal_marker(goal)

    def command(self, env, action):
        return action[:3] * env.cfg.task.max_ee_step, None

    def achieved_goal(self, env):
        return env.ee_pos.copy()


class _ObjectTask(Task):
    uses_object = True

    # ---- object helpers -----------------------------------------------------------------
    def _spawn_xy(self, env, avoid: list[np.ndarray], min_sep: float) -> np.ndarray:
        cx, cy = env.layout.workspace_center
        ext = env.cfg.task.object_start_extent
        for _ in range(50):
            xy = np.array([cx + env.rng.uniform(-ext, ext), cy + env.rng.uniform(-ext, ext)])
            if all(np.linalg.norm(xy - a) >= min_sep for a in avoid):
                return xy
        return xy

    def _place_objects(self, env) -> np.ndarray:
        h = env.layout.object_half_size
        # when the pusher starts low (push task) keep the cube well clear of it
        avoid = [env.ee_start[:2]] if self.lock_z is not None else []
        obj_xy = self._spawn_xy(env, avoid, 0.10)
        yaw = env.rng.uniform(-0.35, 0.35) if env.cfg.task.name == "pickplace" else env.rng.uniform(-np.pi, np.pi)
        env.set_free_body("cube", [obj_xy[0], obj_xy[1], h + 5e-4], yaw_to_quat(yaw))
        placed = [obj_xy]
        for name in env.layout.distractor_names:
            xy = self._spawn_xy(env, placed, 0.075)
            env.set_free_body(name, [xy[0], xy[1], h + 5e-4], yaw_to_quat(env.rng.uniform(-np.pi, np.pi)))
            placed.append(xy)
        return obj_xy

    def _interpolate_goal(self, env, start: np.ndarray, full_goal: np.ndarray, scale: float) -> np.ndarray:
        """Curriculum: pull the goal towards the object's start position by ``1 - scale``."""
        t = env.cfg.task
        goal = start + scale * (full_goal - start)
        if np.linalg.norm(goal[:2] - start[:2]) < t.min_goal_distance and abs(goal[2] - start[2]) < 0.03:
            direction = full_goal[:2] - start[:2]
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            goal = goal.copy()
            goal[:2] = start[:2] + direction * t.min_goal_distance
        return goal

    # ---- observation block --------------------------------------------------------------
    def extra_obs(self, env):
        pos = env.object_pos("cube")
        quat = env.object_quat("cube")
        if quat[0] < 0:  # q and -q are the same rotation: canonicalise the sign
            quat = -quat
        out = [
            ("obj_pos", pos),
            ("obj_rel", pos - env.ee_pos),
            ("obj_quat", quat),
            ("obj_vel", env.object_linvel("cube")),
            ("obj_angvel", env.object_angvel("cube")),
            ("contacts", env.pad_contacts().astype(np.float64)),
        ]
        for name in env.layout.distractor_names:
            out.append((name, env.object_pos(name)))
        return out

    def achieved_goal(self, env):
        return env.object_pos("cube")

    def extra_reward(self, env) -> float:
        shaping = env.cfg.task.shaping
        if shaping <= 0:
            return 0.0
        d_ee_obj = float(np.linalg.norm(env.object_pos("cube") - env.ee_pos))
        return -shaping * min(d_ee_obj, 0.5)

    def max_extra_penalty(self, shaping: float) -> float:
        return shaping * 0.5


class PushTask(_ObjectTask):
    name = "push"
    action_dim = 2
    grip_closed_at_start = True

    def __init__(self, lock_z: float = 0.02):
        self.lock_z = float(lock_z)

    def sample_ee_start(self, env):
        cx, cy = env.layout.workspace_center
        hx, hy = env.cfg.task.workspace_half_extent
        pos = np.array([cx - 0.5 * hx + env.rng.uniform(-0.03, 0.03), cy + env.rng.uniform(-0.6 * hy, 0.6 * hy), self.lock_z])
        return pos

    def reset_scene(self, env, goal_scale: float) -> None:
        t = env.cfg.task
        h = env.layout.object_half_size
        obj_xy = self._place_objects(env)
        # keep the start of the pusher clear of the cube (it is placed by IK before the cube)
        start = np.array([obj_xy[0], obj_xy[1], h])
        for _ in range(30):
            ang = env.rng.uniform(-np.pi, np.pi)
            radius = env.rng.uniform(t.min_goal_distance, 0.26)
            xy = _clip_xy_into_workspace(env, obj_xy + radius * np.array([np.cos(ang), np.sin(ang)]))
            full = np.array([xy[0], xy[1], h])
            goal = self._interpolate_goal(env, start, full, goal_scale)
            if np.linalg.norm(goal[:2] - obj_xy) >= t.min_goal_distance:
                break
        env.goal = goal
        env.place_goal_marker(goal)

    def command(self, env, action):
        d = np.array([action[0], action[1], 0.0]) * env.cfg.task.max_ee_step
        return d, None


class PickPlaceTask(_ObjectTask):
    name = "pickplace"
    action_dim = 4
    grip_closed_at_start = False

    def reset_scene(self, env, goal_scale: float) -> None:
        t = env.cfg.task
        h = env.layout.object_half_size
        obj_xy = self._place_objects(env)
        start = np.array([obj_xy[0], obj_xy[1], h])
        layout = env.layout
        for _ in range(30):
            if layout.tray_center is not None:
                hx, hy = env.cfg.scene.tray_half_extent
                gx = layout.tray_center[0] + env.rng.uniform(-0.5, 0.5) * max(hx - 2 * h, 0.0)
                gy = layout.tray_center[1] + env.rng.uniform(-0.5, 0.5) * max(hy - 2 * h, 0.0)
                gz = layout.tray_floor_top + h + 1e-3
            else:
                ang = env.rng.uniform(-np.pi, np.pi)
                radius = env.rng.uniform(t.min_goal_distance, 0.26)
                gx, gy = _clip_xy_into_workspace(env, obj_xy + radius * np.array([np.cos(ang), np.sin(ang)]))
                gz = h
            if env.rng.random() < t.air_goal_prob:
                gz = env.rng.uniform(h + 0.05, max(t.goal_height_max, h + 0.06))
            full = np.array([gx, gy, gz])
            goal = self._interpolate_goal(env, start, full, goal_scale)
            if np.linalg.norm(goal - start) >= t.min_goal_distance:
                break
        env.goal = goal
        env.place_goal_marker(goal)

    def command(self, env, action):
        return action[:3] * env.cfg.task.max_ee_step, float(action[3])

    def extra_reward(self, env) -> float:
        shaping = env.cfg.task.shaping
        if shaping <= 0:
            return 0.0
        base = super().extra_reward(env)
        both_pads = float(env.pad_contacts().all())
        return base - 0.1 * shaping * (1.0 - both_pads)

    def max_extra_penalty(self, shaping: float) -> float:
        return shaping * 0.6


TASKS = {"reach": ReachTask, "push": PushTask, "pickplace": PickPlaceTask}


def make_task(name: str, push_height: float = 0.02) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return PushTask(push_height) if name == "push" else TASKS[name]()
