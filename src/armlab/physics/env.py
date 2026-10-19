"""MuJoCo physics environment for the 3-link arm (Phase E).

Gymnasium-style API with real simulation integration:
- Observations: [qpos(3), qvel(3), ee_pos(3), target(3)] -> shape (12,).
- Actions: Box(-1, 1, (3,)) mapped to bounded joint velocity setpoints
  (physics action_scale rad/s) held for ``control_decimation`` sim steps.
  No qpos teleportation occurs during ``step``.
- Contacts: arm geoms collide with obstacles/floor; the target is a mocap
  body with collisions disabled; the base column is excluded from floor
  contact via MJCF excludes.
- terminated: success (within success_threshold) or, optionally, collision.
- truncated: episode step limit.

MuJoCo is imported lazily so the base armlab package works without it;
instantiating this env without MuJoCo raises a clear installation message.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from armlab.config import PhysicsParams, RewardParams, RobotParams
from armlab.physics.model import build_arm_xml
from armlab.targets import sample_uniform_reachable

__all__ = ["PhysicsReachingEnv", "MUJOCO_AVAILABLE"]

try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:  # pragma: no cover
    mujoco = None
    MUJOCO_AVAILABLE = False

# Base class for the physics env; gymnasium is always installed.
gymnasium_env_base = gym.Env


class _EnvShim:
    """Minimal stand-in giving sample_uniform_reachable its EnvParams."""

    target_min_radius = 6.0
    target_max_radius = 15.0
    target_min_height = 5.0
    target_max_height = 20.0


class PhysicsReachingEnv(gymnasium_env_base):
    """MuJoCo-backed reaching environment (dynamics + contacts).

    Inherits gym.Env (gymnasium is a base dependency); only MuJoCo itself is
    optional, guarded by the __new__ check below.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __new__(cls, *args, **kwargs):
        if not MUJOCO_AVAILABLE:
            raise ImportError(
                "PhysicsReachingEnv requires MuJoCo. Install with: "
                "pip install -e '.[physics]' (or pip install mujoco)."
            )
        return super().__new__(cls)

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        physics_params: PhysicsParams | None = None,
        reward_params: RewardParams | None = None,
        obstacles: list | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.robot_params = robot_params or RobotParams()
        self.physics_params = physics_params or PhysicsParams()
        self.reward_params = reward_params or RewardParams()
        self.render_mode = render_mode

        xml_specs = [tuple(spec) for spec in (obstacles or [])]
        xml = build_arm_xml(self.robot_params, self.physics_params, xml_specs)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.physics_params.ee_site_name.encode(),
        )
        self._target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.physics_params.target_site_name.encode(),
        )
        self._obstacle_names = [spec[0] for spec in xml_specs]

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        reach = self.robot_params.base_height + self.robot_params.max_reach
        high = np.float32(reach + 1.0)
        obs_high = np.concatenate([np.full(6, np.float32(np.pi * 2)), np.full(6, high)]).astype(
            np.float32
        )
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.np_random = np.random.default_rng(0)
        self.elapsed_steps = 0
        self._renderer = None

    # ----------------------------------------------------------- helpers
    def _ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def _target_position(self) -> np.ndarray:
        return self.data.mocap_pos[0].copy()

    def _set_target(self, pos: np.ndarray) -> None:
        self.data.mocap_pos[0] = np.asarray(pos, dtype=float)

    def _collision_occurred(self) -> tuple[bool, int, list]:
        """Return (collided, n_contacts, geom_name_pairs).

        Counts contacts between the two arm link geoms and anything else
        except the base column. The MJCF excludes handle base/floor, so the
        remaining reportable contacts are arm-vs-obstacle, arm-vs-floor, and
        (already excluded by MuJoCo's parent-child filter) never arm-arm.
        """
        pairs = []
        arm_names = {"link2_geom", "link3_geom"}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            names = {g1, g2}
            if (
                (names & arm_names)
                and "base_geom" not in names
                and "floor" not in names - arm_names
            ):
                pairs.append((g1, g2))
        return bool(pairs), len(pairs), pairs

    def _collision_with_floor(self) -> bool:
        arm_names = {"link2_geom", "link3_geom"}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            names = {g1, g2}
            if (names & arm_names) and "floor" in names:
                return True
        return False

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        # Seeded initial configuration near home with small random offsets.
        q0 = self.np_random.uniform(-0.1, 0.1, size=3)
        self.data.qpos[:] = q0
        self.data.qvel[:] = 0.0

        if options and "target" in options:
            target = np.asarray(options["target"], dtype=float).reshape(3)
        else:
            target = sample_uniform_reachable(self.robot_params, _EnvShim(), self.np_random)
        self._set_target(target)

        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps = 0
        return self._get_obs(), self._info()

    def step(self, action):
        pp = self.physics_params
        action = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
        self.data.ctrl[:] = action * pp.action_scale
        for _ in range(pp.control_decimation):
            mujoco.mj_step(self.model, self.data)
        self.elapsed_steps += 1

        ee = self._ee_position()
        target = self._target_position()
        dist = float(np.linalg.norm(ee - target))
        success = bool(dist < pp.success_threshold)
        collided, n_contacts, pairs = self._collision_occurred()
        floor_hit = self._collision_with_floor()
        terminated = success or (pp.contact_termination and (collided or floor_hit))
        truncated = bool(self.elapsed_steps >= pp.max_episode_steps)

        rp = self.reward_params
        reward = -rp.distance_penalty_scale * dist - rp.time_penalty
        if success:
            reward += rp.success_bonus

        info = self._info()
        info.update(
            {
                "collided": bool(collided or floor_hit),
                "obstacle_contacts": collided,
                "floor_contact": floor_hit,
                "n_contacts": n_contacts,
                "contact_pairs": pairs,
            }
        )
        if self.render_mode is not None:
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[:3].copy()
        qvel = self.data.qvel[:3].copy()
        return np.concatenate([qpos, qvel, self._ee_position(), self._target_position()]).astype(
            np.float32
        )

    def _info(self) -> dict[str, Any]:
        ee = self._ee_position()
        target = self._target_position()
        return {
            "distance": float(np.linalg.norm(ee - target)),
            "ee_position": ee,
            "target_position": target,
            "q": self.data.qpos[:3].copy(),
        }

    # ---------------------------------------------------------- rendering
    def render(self):
        if self.render_mode is None:
            return None
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        except Exception:
            # Headless environments may lack GL; UI failure must not break
            # evaluation (implementation plan E5).
            return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
