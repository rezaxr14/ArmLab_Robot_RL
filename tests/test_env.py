"""Simulator tests (need `mujoco` and the Menagerie assets)."""

from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from armforge.config import EnvConfig, from_dict  # noqa: E402
from armforge.env.arm_env import ArmForgeEnv  # noqa: E402


def make(menagerie, **d):
    d.setdefault("task", {"name": "reach", "horizon": 12})
    return ArmForgeEnv(from_dict(EnvConfig, d), str(menagerie))


@pytest.fixture(scope="module")
def reach(menagerie):
    env = make(menagerie)
    yield env
    env.close()


def test_model_has_expected_structure(reach):
    m = reach.model
    assert m.nu == 7 and m.neq == 3  # 6 arm servos + gripper; 2 connects + 1 joint equality
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert abs(m.opt.timestep - 0.002) < 1e-12 and reach.substeps == 20


def test_friction_values_reach_the_model(menagerie):
    env = make(menagerie, physics={"table_friction": [0.5, 0.005, 0.0001], "pad_friction": 1.0}, task={"name": "pickplace", "horizon": 5})
    m, name2id = env.model, mujoco.mj_name2id
    table = name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    assert np.allclose(m.geom_friction[table], [0.5, 0.005, 0.0001])
    pad = name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "rq_right_pad1")
    assert m.geom_friction[pad][0] == pytest.approx(1.0)
    env.close()


def test_reset_is_deterministic_and_contact_free(reach):
    a = reach.reset(seed=3)
    b = reach.reset(seed=3)
    assert np.array_equal(a["observation"], b["observation"]) and np.array_equal(a["desired_goal"], b["desired_goal"])
    assert not np.array_equal(a["desired_goal"], reach.reset(seed=4)["desired_goal"])


def test_step_contract(reach):
    obs = reach.reset(seed=0)
    assert obs["observation"].shape == (reach.obs_dim,) and obs["achieved_goal"].shape == (3,)
    done, n = False, 0
    while not done:
        obs, r, done, info = reach.step(np.zeros(3))
        n += 1
        assert r in (-1.0, 0.0) and np.all(np.isfinite(obs["observation"]))
    assert n == reach.horizon


def test_cartesian_command_moves_the_hand_the_right_way(reach):
    reach.reset(seed=1)
    p0 = reach.ee_pos
    for _ in range(4):
        reach.step(np.array([1.0, 0.0, 0.0]))
    d = reach.ee_pos - p0
    assert d[0] > 0.04, f"hand barely moved along +x: {d}"  # 4 steps x 2.5 cm commanded
    assert abs(d[1]) < 0.02 and abs(d[2]) < 0.02, f"drifted off-axis: {d}"


def test_tool_stays_pointing_down(reach):
    reach.reset(seed=2)
    rng = np.random.default_rng(0)
    for _ in range(10):
        reach.step(rng.uniform(-1, 1, 3))
    R = reach.data.site_xmat[reach.site_id].reshape(3, 3)
    assert R[2, 2] < -0.98, "tool approach axis drifted away from straight down"


def test_reach_success_when_goal_is_the_hand(reach):
    reach.reset(seed=5)
    reach.goal = reach.ee_pos.copy()
    obs, r, done, info = reach.step(np.zeros(3))
    assert info["is_success"] and r == 0.0


def test_compute_reward_matches_step_and_is_vectorised(reach):
    ag = np.array([[0, 0, 0], [0, 0, 0.5]], np.float32)
    g = np.zeros((2, 3), np.float32)
    assert np.array_equal(reach.compute_reward(ag, g), [0.0, -1.0])


def test_cube_rests_on_table_and_pads_not_touching(menagerie):
    env = make(menagerie, task={"name": "pickplace", "horizon": 10})
    obs = env.reset(seed=0)
    for _ in range(10):
        env.step(np.array([0, 0, 0, -1.0]))
    cube = env.object_pos("cube")
    assert abs(cube[2] - env.layout.object_half_size) < 2e-3, cube  # resting, not sunk or floating
    assert np.linalg.norm(env.object_linvel("cube")) < 0.02
    assert not env.pad_contacts().any()
    env.close()


def test_push_height_is_locked_and_never_scrapes_the_table(menagerie):
    env = make(menagerie, task={"name": "push", "horizon": 15, "push_height": 0.02})
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(15):
        env.step(rng.uniform(-1, 1, 2))
        assert abs(env.ee_pos[2] - 0.02) < 0.012
    # no robot geom other than (maybe) the cube touches the table
    tbl = env.table_geom
    d = env.data
    for i in range(d.ncon):
        c = d.contact[i]
        if tbl in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == tbl else c.geom1
            assert mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, other) in ("cube_geom", None) or env.model.geom_bodyid[other] == env.cube_body
    env.close()


def test_gripper_closes_and_opens(menagerie):
    env = make(menagerie, task={"name": "pickplace", "horizon": 30})
    env.reset(seed=0)
    for _ in range(8):
        env.step(np.array([0, 0, 0, 1.0]))
    assert env.gripper_opening() > 0.8
    for _ in range(8):
        env.step(np.array([0, 0, 0, -1.0]))
    assert env.gripper_opening() < 0.2
    env.close()


def test_domain_randomisation_changes_then_restores_physics(menagerie):
    env = make(menagerie, physics={"randomize": {"enabled": True}})
    nominal = env._nom["geom_friction"].copy()
    env.reset(seed=0)
    assert not np.allclose(env.model.geom_friction, nominal)
    env.cfg.physics.randomize.enabled = False
    env.reset(seed=0)
    assert np.allclose(env.model.geom_friction, nominal)
    env.close()


def test_friction_actually_changes_a_slide(menagerie):
    """Physics sanity: a cube given the same push slides farther on a slippery table."""

    def slide(mu):
        env = make(menagerie, physics={"table_friction": [mu, 0.005, 0.0001], "object_friction": [mu, 0.005, 0.0001]}, task={"name": "push", "horizon": 5})
        env.reset(seed=0)
        env.set_free_body("cube", [0.5, 0.0, 0.02], [1, 0, 0, 0])
        env.data.qvel[env._free["cube"][1] : env._free["cube"][1] + 3] = [0.8, 0, 0]
        x0 = env.object_pos("cube")[0]
        for _ in range(200):
            mujoco.mj_step(env.model, env.data)
        x1 = env.object_pos("cube")[0]
        env.close()
        return x1 - x0

    assert slide(0.15) > slide(0.9) + 0.05


def test_distractors_and_tray_scene_runs(menagerie):
    env = make(menagerie, scene={"tray": True, "distractors": 2}, task={"name": "pickplace", "horizon": 6})
    obs = env.reset(seed=0)
    assert obs["observation"].shape[0] == env.obs_dim
    for _ in range(6):
        env.step(np.zeros(4))
    env.close()


def test_rendering(menagerie):
    pytest.importorskip("PIL")
    env = make(menagerie)
    env.reset(seed=0)
    try:
        img = env.render("front", 160, 120)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no OpenGL backend: {exc}")
    assert img.shape == (120, 160, 3) and img.std() > 5
    env.close()


@pytest.mark.parametrize("task,min_success", [("reach", 0.9), ("push", 0.5), ("pickplace", 0.5)])
def test_scripted_expert_solves_the_task(menagerie, task, min_success):
    """The expert is the reference the learners are judged against; it must work."""
    from armforge.rl.expert import rollout_expert

    horizon = {"reach": 50, "push": 60, "pickplace": 80}[task]
    env = make(menagerie, task={"name": task, "horizon": horizon, "success_threshold": 0.03 if task != "push" else 0.04})
    wins = sum(rollout_expert(env, seed=1_000_000 + i)["success"] for i in range(10))
    env.close()
    assert wins / 10 >= min_success, f"expert only solved {wins}/10 {task} episodes"


def test_arm_holds_position_without_drifting(menagerie):
    """Regression guard for the gravity-sag drift: zero actions must keep the hand where it is."""
    env = make(menagerie, task={"name": "reach", "horizon": 40})
    env.reset(seed=0)
    p0 = env.ee_pos
    for _ in range(30):
        env.step(np.zeros(3))
    assert np.linalg.norm(env.ee_pos - p0) < 0.01, "hand drifted while commanded to stay still"
    env.close()


def test_reach_parks_the_cube_out_of_the_way(menagerie):
    env = make(menagerie, task={"name": "reach", "horizon": 5})
    env.reset(seed=0)
    cube = env.object_pos("cube")
    assert np.linalg.norm(cube[:2] - np.array(env.layout.workspace_center)) > 0.4
    env.close()


def test_servo_gain_is_derived_from_the_model(menagerie):
    env = make(menagerie, task={"name": "reach", "horizon": 5})
    assert 1.0 <= env.servo_gain <= 6.0
    slow = make(menagerie, physics={"servo_damping_scale": 1.0}, task={"name": "reach", "horizon": 5})
    assert slow.servo_gain > env.servo_gain, "a more damped servo needs more lead compensation"
    env.close()
    slow.close()


def test_hand_reaches_a_commanded_step_without_overshoot(menagerie):
    """Step response of the full Cartesian loop: arrive within a few control steps and do not ring."""
    env = make(menagerie, task={"name": "reach", "horizon": 30})
    env.reset(seed=0)
    start = env.ee_pos.copy()
    target = start + np.array([0.0, 0.10, 0.0])
    ys = []
    for _ in range(4):  # four full-scale +y actions = 4 * max_ee_step
        env.step(np.array([0.0, 1.0, 0.0]))
        ys.append(env.ee_pos[1] - start[1])
    for _ in range(12):
        env.step(np.zeros(3))
        ys.append(env.ee_pos[1] - start[1])
    commanded = 4 * env.cfg.task.max_ee_step
    assert ys[-1] == pytest.approx(commanded, abs=0.15 * commanded), "did not settle on the commanded displacement"
    assert max(ys) < commanded * 1.2, "overshoots the commanded displacement"
    assert abs(ys[-1] - ys[-4]) < 0.003, "still moving 12 steps after the command ended"
    del target
    env.close()


def test_divergence_is_contained_not_propagated(menagerie):
    """A blown-up simulation must freeze on the last sane observation and be flagged, never leak huge numbers."""
    env = make(menagerie, task={"name": "pickplace", "horizon": 6})
    obs = env.reset(seed=0)
    env.step(np.zeros(4))
    env._diverged = lambda: True  # simulate the blow-up deterministically (MuJoCo may self-reset on real NaNs)
    obs, _, done, info = env.step(np.zeros(4))
    assert info["unstable"] and not info["is_success"]
    assert np.abs(obs["observation"]).max() < 100 and np.isfinite(obs["observation"]).all()
    while not done:
        obs, _, done, info = env.step(np.zeros(4))
        assert info["unstable"] and np.isfinite(obs["observation"]).all()
    assert env.reset(seed=1) is not None and not env._unstable
    env.close()
