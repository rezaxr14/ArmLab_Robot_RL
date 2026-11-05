import xml.etree.ElementTree as ET
from collections import Counter

import pytest

from armforge.config import EnvConfig, from_dict
from armforge.scene import _prefix_gripper, build_scene_tree, build_scene_xml, scene_layout

VARIANTS = {
    "reach": {"task": {"name": "reach"}},
    "push": {"task": {"name": "push"}},
    "pickplace": {"task": {"name": "pickplace"}},
    "clutter": {"task": {"name": "pickplace"}, "scene": {"tray": True, "distractors": 2}},
}


def test_prefixing_renames_names_refs_and_classes():
    root = ET.fromstring(
        """<mujoco><default><default class="c"><geom/></default></default>
        <asset><mesh name="m" file="a.stl"/><material name="mat"/></asset>
        <worldbody><body name="b" childclass="c"><geom mesh="m" material="mat" class="c"/>
        <joint name="j"/></body></worldbody>
        <equality><joint joint1="j" joint2="j"/></equality></mujoco>"""
    )
    _prefix_gripper(root, "p_")
    assert root.find(".//body").get("name") == "p_b" and root.find(".//body").get("childclass") == "p_c"
    g = root.find(".//geom[@mesh]")
    assert (g.get("mesh"), g.get("material"), g.get("class")) == ("p_m", "p_mat", "p_c")
    assert root.find(".//equality/joint").get("joint1") == "p_j"


@pytest.mark.parametrize("variant", VARIANTS)
def test_composed_scene_is_self_consistent(menagerie, variant):
    cfg = from_dict(EnvConfig, VARIANTS[variant])
    root = build_scene_tree(cfg, menagerie)
    ET.fromstring(build_scene_xml(cfg, menagerie))  # well-formed
    for tag, key in [("body", "name"), ("joint", "name"), ("geom", "name"), ("site", "name"), ("camera", "name"), ("actuator", "name"), ("default", "class")]:
        names = [e.get(key) for e in root.iter(tag) if e.get(key)]
        assert not [n for n, c in Counter(names).items() if c > 1], f"duplicate {tag} names"
    names = {e.get("name") for e in root.iter() if e.get("name")}
    for e in root.iter():
        for attr in ("mesh", "material", "texture", "joint", "joint1", "joint2", "body1", "body2", "tendon", "site"):
            if e.get(attr):
                assert e.get(attr) in names, f"dangling {attr}={e.get(attr)} on <{e.tag}>"
    classes = {e.get("class") for e in root.iter("default") if e.get("class")}
    used = {e.get(a) for e in root.iter() for a in ("class", "childclass") if e.get(a) and e.tag != "default"}
    assert used <= classes
    import os

    assert all(os.path.isfile(m.get("file")) for m in root.find("asset").iter("mesh") if m.get("file"))
    assert len(list(root.iter("freejoint"))) == 1 + cfg.scene.distractors


def test_layout_and_actuators(menagerie):
    cfg = from_dict(EnvConfig, VARIANTS["clutter"])
    layout = scene_layout(cfg)
    assert layout.tray_center is not None and len(layout.distractor_names) == 2
    root = build_scene_tree(cfg, menagerie)
    acts = [a.get("name") for a in root.find("actuator")]
    assert acts[:6] == ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"] and "rq_fingers_actuator" in acts


def test_friction_and_damping_are_written(menagerie):
    cfg = from_dict(EnvConfig, {"physics": {"pad_friction": 0.77, "arm_joint_damping": 3.5, "table_friction": [0.45, 0.004, 0.0002]}})
    root = build_scene_tree(cfg, menagerie)
    assert root.find(".//default[@class='ur5e']/joint").get("damping") == "3.5"
    assert root.find(".//default[@class='rq_pad_box1']/geom").get("friction").split()[0] == "0.77"
    assert root.find(".//geom[@name='table_top']").get("friction") == "0.45 0.004 0.0002"
    opt = root.find("option")
    assert opt.get("cone") == "elliptic"


def test_gravity_compensation_is_applied_to_robot_only(menagerie):
    on = build_scene_tree(from_dict(EnvConfig, {}), menagerie)
    off = build_scene_tree(from_dict(EnvConfig, {"physics": {"arm_gravity_comp": False}}), menagerie)
    robot = [b for b in on.find(".//body[@name='base']").iter("body")]
    assert len(robot) > 15 and all(b.get("gravcomp") == "1" for b in robot)
    assert on.find(".//body[@name='cube']").get("gravcomp") is None
    assert all(b.get("gravcomp") is None for b in off.iter("body"))


def test_servo_damping_scale_scales_kd_only(menagerie):
    base = build_scene_tree(from_dict(EnvConfig, {"physics": {"servo_damping_scale": 1.0}}), menagerie)
    soft = build_scene_tree(from_dict(EnvConfig, {"physics": {"servo_damping_scale": 0.25}}), menagerie)
    for cls in ("ur5e", "size1"):
        a = base.find(f".//default[@class='{cls}']/general").get("biasprm").split()
        b = soft.find(f".//default[@class='{cls}']/general").get("biasprm").split()
        assert a[:2] == b[:2], "kp terms must not change"
        assert float(b[2]) == pytest.approx(0.25 * float(a[2]))
