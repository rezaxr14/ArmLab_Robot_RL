import numpy as np
import pytest

from armforge.kinematics import XmlKinematics
from armforge.mathutil import axis_angle_to_mat, mat_to_quat, quat_to_mat, rotation_error

PLANAR = """
<mujoco><worldbody>
  <body name="b0" pos="0 0 0"><joint name="j1" axis="0 0 1"/>
    <body name="b1" pos="1 0 0"><joint name="j2" axis="0 0 1"/>
      <site name="tip" pos="1 0 0"/>
    </body></body></worldbody></mujoco>"""

CLASSED = """
<mujoco><default><default class="a"><joint axis="0 1 0"/></default></default>
<worldbody><body name="b0" childclass="a"><joint name="j"/><site name="s" pos="1 0 0"/></body></worldbody></mujoco>"""


def test_planar_two_link_matches_closed_form():
    kin = XmlKinematics(PLANAR)
    for q1, q2 in [(0, 0), (0.3, -0.7), (1.2, 0.4)]:
        pos, _ = kin.site_pose("tip", {"j1": q1, "j2": q2})
        expect = [np.cos(q1) + np.cos(q1 + q2), np.sin(q1) + np.sin(q1 + q2), 0]
        assert np.allclose(pos, expect, atol=1e-12)


def test_childclass_axis_inheritance():
    kin = XmlKinematics(CLASSED)
    pos, _ = kin.site_pose("s", {"j": np.pi / 2})  # rotation about +y takes +x to -z
    assert np.allclose(pos, [0, 0, -1], atol=1e-12)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = axis_angle_to_mat(rng.normal(size=3), rng.uniform(0, np.pi))
        assert np.allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-10)
    assert np.allclose(rotation_error(np.eye(3), axis_angle_to_mat([0, 0, 1], 0.1)), [0, 0, 0.0998], atol=1e-3)


def test_ik_solves_planar_target():
    kin = XmlKinematics(PLANAR)
    q, pe, _ = kin.solve_ik("tip", ["j1", "j2"], [1.0, 1.0, 0.0], np.eye(3), [0.3, 0.3], iterations=200)
    assert pe < 1e-3


# ---- the real composed UR5e + Robotiq (needs the Menagerie files) --------------------
@pytest.fixture(scope="module")
def robot(menagerie):
    from armforge.scene import compose_robot

    root = compose_robot(menagerie)
    return root, XmlKinematics(root)


def test_home_pose_tool_points_down(robot):
    from armforge.scene import ARM_JOINTS, HOME_QPOS, TOOL_SITE

    _, kin = robot
    pos, R = kin.site_pose(TOOL_SITE, dict(zip(ARM_JOINTS, HOME_QPOS, strict=False)))
    assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-3), "gripper approach axis should point down at home"
    assert np.allclose(pos, [0.492, 0.134, 0.3322], atol=2e-3)


def test_workspace_is_reachable_tool_down(robot):
    import itertools

    from armforge.scene import ARM_JOINTS, HOME_QPOS, TOOL_DOWN_ROT, TOOL_SITE

    _, kin = robot
    lim = {j: (-np.pi, np.pi) for j in ARM_JOINTS}
    q, pe, re_ = kin.solve_ik(TOOL_SITE, list(ARM_JOINTS), [0.5, 0, 0.25], TOOL_DOWN_ROT, np.array(HOME_QPOS), limits=lim)
    assert pe < 1e-4 and re_ < 1e-3
    for x, y, z in itertools.product([0.33, 0.5, 0.67], [-0.30, 0, 0.30], [0.012, 0.15, 0.32]):
        _, pe, re_ = kin.solve_ik(TOOL_SITE, list(ARM_JOINTS), [x, y, z], TOOL_DOWN_ROT, q, iterations=600, limits=lim)
        assert pe < 1e-3 and re_ < 1e-2, (x, y, z, pe, re_)
