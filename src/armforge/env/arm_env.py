"""ArmForgeEnv: a native MuJoCo goal-conditioned tabletop environment (no Gymnasium).

Physics (all MuJoCo, nothing scripted):
  * UR5e with its real mesh/inertia/damping/force limits and per-joint PD position actuators,
    plus Coulomb joint friction (``frictionloss``) and viscous damping.
  * Robotiq 2F-85 with its 4-bar linkage, spring links, equality constraints, tendon-coupled
    actuator and silicone-pad friction.
  * Elliptic friction cones with torsional + rolling friction on the manipulated cubes and table.
  * Optional per-episode domain randomisation of friction, mass, joint damping and actuator gain.

Control: the policy commands a Cartesian displacement of the pinch point (and the gripper for
pick-and-place); differential IK converts it to joint targets for the arm's PD actuators, so
the controlled system includes actuator lag, force saturation, contact loads and gravity.

API (deliberately tiny):
    obs = env.reset(seed=None, goal_scale=1.0)           -> dict(observation, achieved_goal, desired_goal)
    obs, reward, done, info = env.step(action)
    env.compute_reward(achieved_goal, desired_goal)       -> vectorised, used for HER relabelling
"""

from __future__ import annotations

import numpy as np

import mujoco

from ..config import EnvConfig
from ..scene import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    GRIPPER_ACTUATOR,
    GRIPPER_DRIVER_JOINTS,
    GRIPPER_PREFIX,
    HOME_QPOS,
    TOOL_DOWN_ROT,
    TOOL_SITE,
    scene_layout,
    write_scene_xml,
)
from .controller import CartesianController
from .reward import goal_distance, goal_reward
from .spaces import Box
from .tasks import make_task

GRIPPER_CTRL_MAX = 255.0
GRIPPER_RATE = 0.35  # gripper command change per unit action per control step (0..1 range)
SETTLE_STEPS_CLOSED = 150  # physics steps to let the gripper close / objects settle after reset


class ArmForgeEnv:
    def __init__(self, cfg: EnvConfig, menagerie_root: str | None = None):
        self.cfg = cfg
        self.task = make_task(cfg.task.name, cfg.task.push_height)
        self.layout = scene_layout(cfg)
        self.rng = np.random.default_rng(cfg.seed)

        path = write_scene_xml(cfg, menagerie_root)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        self._lookup_ids()
        self._store_nominal_physics()

        phys = cfg.physics
        self.dt = self.model.opt.timestep * phys.substeps
        self.substeps = phys.substeps
        self.start_height = 0.20
        self.horizon = cfg.task.horizon
        self.goal_scale = 1.0
        self.goal = np.zeros(3)
        self.grip = 0.0 if not self.task.grip_closed_at_start else 1.0
        self.ee_start = np.zeros(3)
        self._prev_ee = np.zeros(3)
        self._ee_vel = np.zeros(3)
        self.t = 0
        self._renderer = None
        self._renderer_size = None

        self.controller = CartesianController(
            self.model,
            self.site_id,
            self.arm_qadr,
            self.arm_dadr,
            self.arm_jids,
            TOOL_DOWN_ROT,
        )
        self.servo_gain = self._servo_lead_gain()
        self.action_dim = self.task.action_dim
        self.action_space = Box.symmetric(self.action_dim)
        self.goal_dim = 3
        first = self.reset(seed=cfg.seed)
        self.obs_dim = first["observation"].shape[0]

    # ------------------------------------------------------------------ construction
    def _lookup_ids(self) -> None:
        m = self.model
        name2id = mujoco.mj_name2id
        obj = mujoco.mjtObj
        self.site_id = name2id(m, obj.mjOBJ_SITE, TOOL_SITE)
        self.arm_jids = np.array([name2id(m, obj.mjOBJ_JOINT, n) for n in ARM_JOINTS])
        self.arm_qadr = m.jnt_qposadr[self.arm_jids]
        self.arm_dadr = m.jnt_dofadr[self.arm_jids]
        self.arm_act = np.array([name2id(m, obj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS])
        self.grip_act = name2id(m, obj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
        drivers = [name2id(m, obj.mjOBJ_JOINT, n) for n in GRIPPER_DRIVER_JOINTS]
        self.grip_qadr = m.jnt_qposadr[drivers]
        self.goal_mocap = m.body_mocapid[name2id(m, obj.mjOBJ_BODY, "goal_marker")]
        self.table_geom = name2id(m, obj.mjOBJ_GEOM, "table_top")
        self._free = {}
        for name in ("cube", *self.layout.distractor_names):
            jid = name2id(m, obj.mjOBJ_JOINT, f"{name}_free")
            self._free[name] = (m.jnt_qposadr[jid], m.jnt_dofadr[jid])
        pad_names = {
            "right": (f"{GRIPPER_PREFIX}right_pad1", f"{GRIPPER_PREFIX}right_pad2"),
            "left": (f"{GRIPPER_PREFIX}left_pad1", f"{GRIPPER_PREFIX}left_pad2"),
        }
        self.pad_geoms = {
            side: np.array([name2id(m, obj.mjOBJ_GEOM, n) for n in names]) for side, names in pad_names.items()
        }
        self.cube_geom = name2id(m, obj.mjOBJ_GEOM, "cube_geom")
        self.cube_body = name2id(m, obj.mjOBJ_BODY, "cube")

    def _store_nominal_physics(self) -> None:
        m = self.model
        self._nom = {
            "geom_friction": m.geom_friction.copy(),
            "body_mass": m.body_mass.copy(),
            "body_inertia": m.body_inertia.copy(),
            "dof_damping": m.dof_damping.copy(),
            "gainprm": m.actuator_gainprm.copy(),
            "biasprm": m.actuator_biasprm.copy(),
        }
        # geoms whose sliding friction is randomised: table, the cubes, gripper pads
        ids = [self.table_geom, self.cube_geom]
        for name in self.layout.distractor_names:
            ids.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom"))
        for geoms in self.pad_geoms.values():
            ids.extend(int(g) for g in geoms)
        self._rand_geoms = np.array(ids)
        self._cube_bodies = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("cube", *self.layout.distractor_names)]
        )

    def _servo_lead_gain(self) -> float:
        """Gain that cancels the arm servo lag: 1 / (fraction of a commanded move achieved per control step).

        The position servo ``tau = kp (c - q) - kd qdot`` is heavily over-damped, so the joint follows its
        target with time constant kd/kp; over one control period ``dt`` it covers ``1 - exp(-dt kp/kd)``.
        """
        manual = self.cfg.physics.servo_lead_gain
        if manual > 0:
            return float(manual)
        kp = self.model.actuator_gainprm[self.arm_act, 0]
        kd = -self.model.actuator_biasprm[self.arm_act, 2]
        tau = float(np.mean(kd / kp))
        achieved = 1.0 - np.exp(-self.dt / max(tau, 1e-6))
        return float(np.clip(1.0 / achieved, 1.0, 6.0))

    # ------------------------------------------------------------------ randomisation
    def _randomize_physics(self) -> dict[str, float]:
        m, r = self.model, self.cfg.physics.randomize
        for key, arr in (
            ("geom_friction", m.geom_friction),
            ("body_mass", m.body_mass),
            ("body_inertia", m.body_inertia),
            ("dof_damping", m.dof_damping),
            ("gainprm", m.actuator_gainprm),
            ("biasprm", m.actuator_biasprm),
        ):
            arr[:] = self._nom[key]
        if not r.enabled:
            return {}
        fr = self.rng.uniform(*r.friction_scale)
        ms = self.rng.uniform(*r.mass_scale)
        dm = self.rng.uniform(*r.damping_scale)
        kp = self.rng.uniform(*r.kp_scale)
        m.geom_friction[self._rand_geoms, 0] = self._nom["geom_friction"][self._rand_geoms, 0] * fr
        m.body_mass[self._cube_bodies] = self._nom["body_mass"][self._cube_bodies] * ms
        m.body_inertia[self._cube_bodies] = self._nom["body_inertia"][self._cube_bodies] * ms
        m.dof_damping[self.arm_dadr] = self._nom["dof_damping"][self.arm_dadr] * dm
        m.actuator_gainprm[self.arm_act, 0] = self._nom["gainprm"][self.arm_act, 0] * kp
        m.actuator_biasprm[self.arm_act, 1] = self._nom["biasprm"][self.arm_act, 1] * kp
        return {"friction": fr, "mass": ms, "damping": dm, "kp": kp}

    # ------------------------------------------------------------------ state accessors
    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def object_pos(self, name: str) -> np.ndarray:
        adr = self._free[name][0]
        return self.data.qpos[adr : adr + 3].copy()

    def object_quat(self, name: str) -> np.ndarray:
        adr = self._free[name][0]
        return self.data.qpos[adr + 3 : adr + 7].copy()

    def object_linvel(self, name: str) -> np.ndarray:
        dadr = self._free[name][1]
        return self.data.qvel[dadr : dadr + 3].copy()

    def object_angvel(self, name: str) -> np.ndarray:
        dadr = self._free[name][1]
        return self.data.qvel[dadr + 3 : dadr + 6].copy()

    def set_free_body(self, name: str, pos, quat) -> None:
        adr, dadr = self._free[name]
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat
        self.data.qvel[dadr : dadr + 6] = 0.0

    def place_goal_marker(self, goal: np.ndarray) -> None:
        self.data.mocap_pos[self.goal_mocap] = goal

    def pad_contacts(self) -> np.ndarray:
        """``[right_pad_touches_cube, left_pad_touches_cube]`` from the live contact list."""
        n = self.data.ncon
        out = np.zeros(2, dtype=bool)
        if n == 0:
            return out
        g1, g2 = self.data.contact.geom1[:n], self.data.contact.geom2[:n]
        touching_cube = (g1 == self.cube_geom) | (g2 == self.cube_geom)
        for i, side in enumerate(("right", "left")):
            pads = self.pad_geoms[side]
            on_pad = np.isin(g1, pads) | np.isin(g2, pads)
            out[i] = bool(np.any(touching_cube & on_pad))
        return out

    def gripper_opening(self) -> float:
        """0 = open, 1 = fully closed on the driver-joint range (0 .. 0.8 rad)."""
        return float(np.clip(self.data.qpos[self.grip_qadr].mean() / 0.8, 0.0, 1.0))

    # ------------------------------------------------------------------ observation
    def _observation(self) -> dict[str, np.ndarray]:
        d = self.data
        ee = self.ee_pos
        parts = [
            ee,
            self._ee_vel,
            [self.gripper_opening()],
            d.qpos[self.arm_qadr],
            d.qvel[self.arm_dadr],
        ]
        for _, value in self.task.extra_obs(self):
            parts.append(value)
        obs = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts]).astype(np.float32)
        return {
            "observation": obs,
            "achieved_goal": self.task.achieved_goal(self).astype(np.float32),
            "desired_goal": self.goal.astype(np.float32),
        }

    # ------------------------------------------------------------------ reset
    def set_goal_scale(self, scale: float) -> None:
        self.goal_scale = float(np.clip(scale, 0.0, 1.0))

    def reset(self, seed: int | None = None, goal_scale: float | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if goal_scale is not None:
            self.set_goal_scale(goal_scale)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._randomize_physics()
        self.t = 0
        self.grip = 1.0 if self.task.grip_closed_at_start else 0.0

        # arm to the task start pose by IK on a scratch copy
        self.ee_start = self.task.sample_ee_start(self)
        q0 = np.array(HOME_QPOS)
        q, pos_err, rot_err = self.controller.solve_ik(self._scratch, self.ee_start, q0)
        if pos_err > 2e-3 or rot_err > 2e-2:
            raise RuntimeError(
                f"start pose unreachable: ee_start={self.ee_start.round(3)} pos_err={pos_err:.4f} rot_err={rot_err:.4f}"
            )
        d.qpos[self.arm_qadr] = q
        d.ctrl[self.arm_act] = q
        d.ctrl[self.grip_act] = GRIPPER_CTRL_MAX * self.grip
        mujoco.mj_forward(m, d)

        self.task.reset_scene(self, self.goal_scale)
        mujoco.mj_forward(m, d)
        for _ in range(SETTLE_STEPS_CLOSED if self.task.grip_closed_at_start else 40):
            mujoco.mj_step(m, d)
        self._sync_kinematics()
        self._prev_ee = self.ee_pos
        self._ee_vel = np.zeros(3)
        return self._observation()

    def _sync_kinematics(self) -> None:
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        if a.size != self.action_dim:
            raise ValueError(f"action has {a.size} entries, expected {self.action_dim}")
        noise = self.cfg.physics.randomize.action_noise
        if noise > 0 and self.cfg.physics.randomize.enabled:
            a = np.clip(a + self.rng.normal(0.0, noise, a.shape), -1.0, 1.0)

        delta, grip_cmd = self.task.command(self, a)
        if grip_cmd is not None:
            self.grip = float(np.clip(self.grip + GRIPPER_RATE * grip_cmd, 0.0, 1.0))
        cfg_t = self.cfg.task
        cx, cy = self.layout.workspace_center
        hx, hy = cfg_t.workspace_half_extent
        target = self.ee_pos + delta
        target[0] = np.clip(target[0], cx - hx, cx + hx)
        target[1] = np.clip(target[1], cy - hy, cy + hy)
        target[2] = np.clip(target[2], *cfg_t.workspace_z)
        if self.task.lock_z is not None:
            target[2] = self.task.lock_z

        self._sync_kinematics()
        self.data.ctrl[self.arm_act] = self.controller.joint_targets(self.data, target, self.servo_gain)
        self.data.ctrl[self.grip_act] = GRIPPER_CTRL_MAX * self.grip
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self._sync_kinematics()

        ee = self.ee_pos
        self._ee_vel = (ee - self._prev_ee) / self.dt
        self._prev_ee = ee
        self.t += 1

        obs = self._observation()
        dist = float(goal_distance(obs["achieved_goal"], obs["desired_goal"]))
        success = dist <= cfg_t.success_threshold
        reward = float(goal_reward(dist, cfg_t.success_threshold, cfg_t.reward)) + self.task.extra_reward(self)
        done = self.t >= self.horizon
        info = {
            "is_success": bool(success),
            "distance": dist,
            "shaping": self.task.extra_reward(self),
            "ncon": int(self.data.ncon),
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------ HER hook
    def compute_reward(self, achieved_goal, desired_goal):
        c = self.cfg.task
        return goal_reward(goal_distance(achieved_goal, desired_goal), c.success_threshold, c.reward)

    # ------------------------------------------------------------------ rendering
    def render(self, camera: str = "front", width: int = 640, height: int = 480) -> np.ndarray:
        size = (height, width)
        if self._renderer is None or self._renderer_size != size:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_size = size
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
