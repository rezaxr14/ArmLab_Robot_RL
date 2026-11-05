import numpy as np
import pytest

from armforge.fkcheck import FkReport, compare_poses, cpu_fk_fn
from armforge.kinematics import XmlKinematics

ARM = """
<mujoco><worldbody>
  <body name="base" pos="0.1 0.2 0.3" quat="0.7071068 0 0 0.7071068">
    <body name="l1" pos="0 0 0.2"><joint name="j1" axis="0 0 1"/>
      <body name="l2" pos="0.4 0 0"><joint name="j2" axis="0 1 0"/>
        <site name="tool" pos="0.3 0 0" quat="0 1 0 0"/>
      </body></body></body></worldbody></mujoco>"""


def _qs(n=25, dof=2, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-np.pi, np.pi, dof) for _ in range(n)]


def test_identical_implementations_agree_exactly():
    kin = XmlKinematics(ARM)
    fk = cpu_fk_fn(kin, ["j1", "j2"], ["tool"])
    rep = compare_poses("self", _qs(), fk, fk, ["tool"])
    assert rep.samples == 25 and rep.max_pos_err == 0.0 and rep.max_rot_err < 1e-12
    assert rep.passed()


def test_a_one_millimetre_offset_is_detected_and_measured():
    kin = XmlKinematics(ARM)
    fk = cpu_fk_fn(kin, ["j1", "j2"], ["tool"])

    def shifted(q):
        pos, R = fk(q)["tool"]
        return {"tool": (pos + np.array([0.0, 0.0, 1e-3]), R)}

    rep = compare_poses("shifted", _qs(), fk, shifted, ["tool"])
    assert rep.max_pos_err == pytest.approx(1e-3, rel=1e-6)
    assert not rep.passed()


def test_a_rotation_error_is_detected_and_measured():
    kin = XmlKinematics(ARM)
    fk = cpu_fk_fn(kin, ["j1", "j2"], ["tool"])
    from armforge.mathutil import axis_angle_to_mat

    def tilted(q):
        pos, R = fk(q)["tool"]
        return {"tool": (pos, axis_angle_to_mat([0, 0, 1], 0.01) @ R)}

    rep = compare_poses("tilted", _qs(), fk, tilted, ["tool"])
    assert rep.max_rot_err == pytest.approx(0.01, rel=1e-3)
    assert not rep.passed()


def test_cpu_fk_matches_a_hand_computed_pose():
    kin = XmlKinematics(ARM)
    pos, _ = cpu_fk_fn(kin, ["j1", "j2"], ["tool"])(np.zeros(2))["tool"]
    # base rotated +90 deg about z: local +x becomes world +y; offsets (0.4 + 0.3) along local x, 0.2 up
    assert np.allclose(pos, [0.1, 0.2 + 0.7, 0.3 + 0.2], atol=1e-12)


def test_empty_report_never_passes():
    assert not FkReport("none").passed()


def test_table_lists_every_site():
    kin = XmlKinematics(ARM)
    fk = cpu_fk_fn(kin, ["j1", "j2"], ["tool"])
    assert "tool" in compare_poses("t", _qs(3), fk, fk, ["tool"]).table()


def test_real_robot_cpu_fk_matches_mujoco(menagerie):
    pytest.importorskip("mujoco")
    from armforge.config import EnvConfig, from_dict
    from armforge.fkcheck import benchmark, compare_random, compare_trajectory

    cfg = from_dict(EnvConfig, {"task": {"name": "reach", "horizon": 12}})
    rep = compare_random(cfg, samples=200, menagerie_root=menagerie)
    assert rep.passed(1e-7, 1e-7), rep.table()
    traj, gap = compare_trajectory(cfg, episodes=1, menagerie_root=menagerie)
    assert traj.passed(1e-7, 1e-7), traj.table()
    assert gap < 1e-4, "environment ee_pos disagrees with the independent CPU FK"
    assert benchmark(cfg, calls=50, menagerie_root=menagerie)["mujoco"] > 0
