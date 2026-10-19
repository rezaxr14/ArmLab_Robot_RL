"""Phase E acceptance tests for the MuJoCo physics environment.

Tests skip only when MuJoCo is absent (optional dependency, per plan).
"""

import numpy as np
import pytest

from armlab.config import PhysicsParams, RobotParams

mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed (optional extra)")

from armlab.physics import PhysicsReachingEnv, register_physics_envs
from armlab.physics.model import build_arm_xml


class TestModel:
    def test_xml_compiles(self):
        xml = build_arm_xml()
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.nq == 3
        assert model.nu == 3

    def test_joint_limits_in_model(self):
        robot = RobotParams(joint_limits_low=(-1.0, -0.5, -2.0),
                            joint_limits_high=(1.0, 0.5, 2.0))
        model = mujoco.MjModel.from_xml_string(build_arm_xml(robot))
        np.testing.assert_allclose(model.jnt_range[0], [-1.0, 1.0])
        np.testing.assert_allclose(model.jnt_range[1], [-0.5, 0.5])
        np.testing.assert_allclose(model.jnt_range[2], [-2.0, 2.0])

    def test_actuator_ctrlrange_bounded(self):
        model = mujoco.MjModel.from_xml_string(build_arm_xml())
        for i in range(model.nu):
            assert model.actuator_ctrllimited[i]
            np.testing.assert_allclose(model.actuator_ctrlrange[i], [-5.0, 5.0])

    def test_obstacle_xml(self):
        xml = build_arm_xml(obstacles=[("obs1", "sphere", (10, 0, 12), (1.5,))])
        model = mujoco.MjModel.from_xml_string(xml)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, b"obs1")
        assert gid >= 0


class TestPhysicsEnv:
    def test_simulation_advances(self):
        env = PhysicsReachingEnv()
        env.reset(seed=0)
        for _ in range(10):
            env.step(np.zeros(3))
        assert env.data.time > 0

    def test_states_finite_on_neutral_rollout(self):
        env = PhysicsReachingEnv()
        obs, _ = env.reset(seed=1)
        assert np.all(np.isfinite(obs))
        for _ in range(30):
            obs, *_ = env.step(np.zeros(3))
            assert np.all(np.isfinite(obs))

    def test_kinematics_matches_mujoco_site(self):
        env = PhysicsReachingEnv()
        env.reset(seed=2)
        from armlab.kinematics import RobotKinematics

        kin = RobotKinematics(RobotParams())
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(20):
            q = rng.uniform([-np.pi, -np.pi / 2, -np.pi], [np.pi, np.pi / 2, np.pi])
            env.data.qpos[:] = q
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            worst = max(worst, np.linalg.norm(kin.end_effector(q) - env._ee_position()))
        assert worst < 1e-6

    def test_actuator_bounds_enforced(self):
        env = PhysicsReachingEnv()
        env.reset(seed=0)
        env.step(np.array([100.0, -100.0, 50.0]))  # clipped to [-1, 1] then * scale
        assert np.all(np.abs(env.data.ctrl) <= 5.0 + 1e-9)

    def test_reset_deterministic(self):
        env = PhysicsReachingEnv()
        o1, i1 = env.reset(seed=7)
        o2, i2 = env.reset(seed=7)
        np.testing.assert_allclose(o1, o2, atol=1e-12)
        np.testing.assert_allclose(i1["ee_position"], i2["ee_position"], atol=1e-12)

    def test_headless_short_episode(self):
        env = PhysicsReachingEnv(render_mode=None)
        env.reset(seed=3)
        done = False
        steps = 0
        while not done and steps < 20:
            _, _, term, trunc, info = env.step(np.zeros(3))
            done = term or trunc
            steps += 1
        assert np.isfinite(info["distance"])

    def test_clear_configuration_no_collision(self):
        env = PhysicsReachingEnv()
        env.reset(seed=0)
        # Home-ish pose is far from floor/obstacles.
        for _ in range(10):
            _, _, term, trunc, info = env.step(np.zeros(3))
            assert not info["collided"]

    def test_colliding_configuration_detected(self):
        # Arm pointing straight down drives links into the floor.
        obstacles = [("obs_low", "sphere", (7.0, 0.0, 1.0), (1.0,))]
        env = PhysicsReachingEnv(obstacles=obstacles)
        env.reset(seed=0, options={"target": np.array([7.0, 0.0, 1.0])})
        env.data.qpos[:] = [0.0, np.pi * 0.8, 0.0]  # folded down toward floor
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        collided, n, pairs = env._collision_occurred()
        floor_hit = env._collision_with_floor()
        assert collided or floor_hit, f"expected contact, got pairs={pairs}"

    def test_contact_termination(self):
        env = PhysicsReachingEnv(
            physics_params=PhysicsParams(contact_termination=True, max_episode_steps=400)
        )
        env.reset(seed=0)
        terminated = False
        info = {}
        # Phase 1: swing shoulder to horizontal; forearm stays extended at
        # z = l1 (no floor contact yet).
        for _ in range(100):
            _, _, terminated, trunc, info = env.step(np.array([0.0, 1.0, 0.0]))
            if terminated:
                break
        # Phase 2: with the shoulder held horizontal, fold the elbow; the
        # forearm sweeps below the floor plane and contact must terminate.
        while not terminated:
            for _ in range(100):
                _, _, terminated, trunc, info = env.step(np.array([0.0, 0.0, 1.0]))
                if terminated:
                    break
            # Reset elbow direction if it ran to the limit without contact.
            if not terminated:
                for _ in range(100):
                    _, _, terminated, trunc, info = env.step(np.array([0.0, 0.0, -1.0]))
                    if terminated:
                        break
            if env.elapsed_steps >= env.physics_params.max_episode_steps:
                break
        assert terminated and info.get("collided"), (
            f"expected floor contact termination, info={info}"
        )

    def test_no_teleportation_during_step(self):
        env = PhysicsReachingEnv()
        env.reset(seed=0)
        q_before = env.data.qpos[:3].copy()
        env.step(np.array([0.2, 0.0, 0.0]))
        # Joints moved continuously: bounded by ctrl scale * dt * decimation.
        max_delta = 5.0 * env.physics_params.control_dt + 1e-6
        assert np.all(np.abs(env.data.qpos[:3] - q_before) <= max_delta)

    def test_registration(self):
        register_physics_envs()
        import gymnasium as gym

        env = gym.make("ArmLabPhysics-v0")
        obs, _ = env.reset(seed=1)
        assert obs.shape == (12,)
        env.close()
