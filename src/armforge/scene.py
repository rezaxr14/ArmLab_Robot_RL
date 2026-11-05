"""Scene generation: UR5e + Robotiq 2F-85 composed from Menagerie, plus the tabletop world.

The two Menagerie models are merged *textually* (ElementTree) instead of with ``MjSpec.attach``:

* the result is plain MJCF you can open in the MuJoCo viewer or diff,
* the merge is pure Python, so it is unit-tested without the simulator, and
* every name/class clash is resolved explicitly (the gripper's names, materials, meshes and
  default classes get an ``rq_`` prefix).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .assets import cache_root, ensure_menagerie
from .config import EnvConfig
from .mathutil import fmt, look_at_xyaxes

GRIPPER_PREFIX = "rq_"
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
GRIPPER_ACTUATOR = GRIPPER_PREFIX + "fingers_actuator"
GRIPPER_DRIVER_JOINTS = (GRIPPER_PREFIX + "right_driver_joint", GRIPPER_PREFIX + "left_driver_joint")
TOOL_SITE = GRIPPER_PREFIX + "pinch"
HOME_QPOS = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)  # Menagerie "home" keyframe
ARM_BASE_QUAT = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # +90 deg about z: reach along +x
# Tool frame pointing straight down with the finger-closing axis along world y:
# site x = world x, site y = -world y, site z (approach) = -world z.
TOOL_DOWN_ROT = np.diag([1.0, -1.0, -1.0])

# Attributes whose value is the *name* of another element (only renamed if it names a gripper item).
_REF_ATTRS = (
    "mesh", "material", "texture", "body1", "body2", "joint", "joint1", "joint2",
    "tendon", "site", "geom1", "geom2", "body", "target", "objname",
)  # fmt: skip


# --------------------------------------------------------------------------------------
# Robot composition
# --------------------------------------------------------------------------------------


def _absolute_meshes(root: ET.Element, model_dir: Path) -> None:
    """Point every mesh at an absolute file and give it an explicit name (default: file stem)."""
    meshdir = ""
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")
    for mesh in root.iter("mesh"):
        file = mesh.get("file")
        if file is None:
            continue
        mesh.set("name", mesh.get("name", Path(file).stem))
        mesh.set("file", str((model_dir / meshdir / file).resolve()))


def _prefix_gripper(root: ET.Element, prefix: str) -> None:
    """Rename every named item, material, mesh and default class of the gripper tree."""
    names: set[str] = set()
    classes: set[str] = set()
    for el in root.iter():
        if el.tag == "default":
            if el.get("class"):
                classes.add(el.get("class"))
        elif el.get("name"):
            names.add(el.get("name"))
    for el in root.iter():
        if el.tag != "default" and el.get("name") in names:
            el.set("name", prefix + el.get("name"))
        for attr in _REF_ATTRS:
            if el.get(attr) in names:
                el.set(attr, prefix + el.get(attr))
        for attr in ("class", "childclass"):
            if el.get(attr) in classes:
                el.set(attr, prefix + el.get(attr))


def _section(root: ET.Element, tag: str) -> ET.Element:
    node = root.find(tag)
    if node is None:
        node = ET.SubElement(root, tag)
    return node


def compose_robot(menagerie_root: str | Path | None = None) -> ET.Element:
    """Return a ``<mujoco>`` tree: UR5e with a Robotiq 2F-85 mounted on its flange.

    Dropped on purpose: both sources' ``<compiler>``, ``<option>`` and ``<keyframe>`` (the
    keyframe's qpos length no longer matches once the gripper joints are added).
    """
    root_dir = ensure_menagerie(menagerie_root)
    arm_dir = root_dir / "universal_robots_ur5e"
    grip_dir = root_dir / "robotiq_2f85"
    arm = ET.parse(arm_dir / "ur5e.xml").getroot()
    grip = ET.parse(grip_dir / "2f85.xml").getroot()

    _absolute_meshes(arm, arm_dir)
    _absolute_meshes(grip, grip_dir)
    _prefix_gripper(grip, GRIPPER_PREFIX)

    for tag in ("compiler", "option", "keyframe"):
        for el in arm.findall(tag):
            arm.remove(el)

    # defaults + assets
    arm_default = arm.find("default")
    grip_default = grip.find("default")
    if arm_default is None or grip_default is None:
        raise ValueError("expected a top-level <default> in both Menagerie models")
    for child in list(grip_default):
        arm_default.append(child)
    arm_asset = _section(arm, "asset")
    for child in list(grip.find("asset")):
        arm_asset.append(child)

    # mount the gripper at the UR5e flange (attachment_site frame == gripper frame)
    wrist = arm.find(".//body[@name='wrist_3_link']")
    site = None if wrist is None else wrist.find("site[@name='attachment_site']")
    if wrist is None or site is None:
        raise ValueError("UR5e model has no wrist_3_link/attachment_site")
    mount = ET.SubElement(
        wrist,
        "body",
        {
            "name": GRIPPER_PREFIX + "mount",
            "pos": site.get("pos", "0 0 0"),
            "quat": site.get("quat", "1 0 0 0"),
        },
    )
    for body in grip.find("worldbody").findall("body"):
        mount.append(body)

    # constraints, tendons, contact exclusions, actuators
    for tag in ("tendon", "equality", "contact", "actuator"):
        src = grip.find(tag)
        if src is not None:
            dst = _section(arm, tag)
            for child in list(src):
                dst.append(child)

    # Menagerie's base faces world +y at the home pose; turn it so the arm reaches along +x.
    base = arm.find(".//body[@name='base']")
    if base is None:
        raise ValueError("UR5e model has no 'base' body")
    base.set("quat", fmt(ARM_BASE_QUAT))

    arm.set("model", "armforge")
    return arm


# --------------------------------------------------------------------------------------
# Tabletop world
# --------------------------------------------------------------------------------------

TABLE_TOP_Z = 0.0  # the arm base sits on the table surface, which is the world z = 0 plane
TRAY_WALL_HALF_THICKNESS = 0.004
TRAY_FLOOR_HALF_THICKNESS = 0.004

CAMERAS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = {
    # name: (position, look-at target, up vector)
    "front": ((1.20, -0.90, 0.62), (0.50, 0.02, 0.08), (0.0, 0.0, 1.0)),
    "side": ((0.50, -1.15, 0.42), (0.50, 0.00, 0.10), (0.0, 0.0, 1.0)),
    "top": ((0.50, 0.00, 1.25), (0.50, 0.00, 0.00), (1.0, 0.0, 0.0)),
    "close": ((0.92, -0.50, 0.34), (0.50, 0.00, 0.06), (0.0, 0.0, 1.0)),
    "hero": ((1.45, -1.05, 0.85), (0.38, 0.06, 0.14), (0.0, 0.0, 1.0)),
}


@dataclass(frozen=True)
class SceneLayout:
    """World-frame facts about the generated scene that the environment needs."""

    workspace_center: tuple[float, float]
    object_half_size: float
    tray_center: tuple[float, float] | None
    tray_floor_top: float | None
    distractor_names: tuple[str, ...]


def scene_layout(cfg: EnvConfig) -> SceneLayout:
    s = cfg.scene
    cx, cy = s.workspace_center
    tray_center = (cx + s.tray_center[0], cy + s.tray_center[1]) if s.tray else None
    floor_top = 2 * TRAY_FLOOR_HALF_THICKNESS if s.tray else None
    names = tuple(f"distractor_{i}" for i in range(s.distractors))
    return SceneLayout((cx, cy), cfg.physics.object_half_size, tray_center, floor_top, names)


def _add(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def _cube(world: ET.Element, name: str, rgba: str, cfg: EnvConfig, pos) -> None:
    p = cfg.physics
    h = p.object_half_size
    body = _add(world, "body", name=name, pos=fmt(pos))
    _add(body, "freejoint", name=f"{name}_free")
    _add(
        body,
        "geom",
        name=f"{name}_geom",
        type="box",
        size=fmt([h, h, h]),
        mass=fmt(p.object_mass),
        friction=fmt(p.object_friction),
        condim=p.object_condim,
        solref=fmt(p.solref),
        solimp=fmt(p.solimp),
        rgba=rgba,
    )


def build_scene_tree(cfg: EnvConfig, menagerie_root: str | Path | None = None) -> ET.Element:
    """Full MJCF tree for ``cfg``: composed robot + table + objects + cameras + physics options."""
    root = compose_robot(menagerie_root)
    p, s, t = cfg.physics, cfg.scene, cfg.task
    layout = scene_layout(cfg)
    cx, cy = layout.workspace_center
    h = p.object_half_size

    # ---- global settings (inserted first so they precede everything that uses them) ----
    settings: list[ET.Element] = []
    compiler = ET.Element("compiler", {"angle": "radian", "autolimits": "true"})
    option = ET.Element(
        "option",
        {
            "timestep": fmt(p.timestep),
            "gravity": fmt([0, 0, -p.gravity]),
            "integrator": p.integrator,
            "cone": p.cone,
            "impratio": fmt(p.impratio),
            "noslip_iterations": str(p.noslip_iterations),
        },
    )
    visual = ET.Element("visual")
    _add(visual, "global", offwidth=1920, offheight=1080)
    _add(visual, "quality", shadowsize=4096)
    _add(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.5 0.5 0.5", specular="0 0 0")
    _add(visual, "rgba", haze="0.15 0.25 0.35 1")
    settings += [compiler, option, visual]
    for i, el in enumerate(settings):
        root.insert(i, el)

    # ---- gravity compensation (the UR controller does this on hardware; the cube is NOT compensated) ----
    base_body = root.find(".//body[@name='base']")
    if base_body is None:
        raise ValueError("composed robot has no 'base' body")
    if p.arm_gravity_comp:
        for body in base_body.iter("body"):
            body.set("gravcomp", "1")

    # ---- servo damping: scale the kd term of the arm position actuators (biasprm = "0 -kp -kd") ----
    for cls in ("ur5e", "size1"):
        general = root.find(f".//default[@class='{cls}']/general")
        if general is not None and general.get("biasprm"):
            parts = general.get("biasprm").split()
            parts[2] = f"{float(parts[2]) * p.servo_damping_scale:g}"
            general.set("biasprm", " ".join(parts))

    # ---- arm joint friction/damping and gripper pad friction ----
    ur5e_default = root.find(".//default[@class='ur5e']")
    joint_default = None if ur5e_default is None else ur5e_default.find("joint")
    if joint_default is None:
        raise ValueError("UR5e default class has no <joint> defaults")
    joint_default.set("damping", fmt(p.arm_joint_damping))
    joint_default.set("frictionloss", fmt(p.arm_frictionloss))
    for cls in (GRIPPER_PREFIX + "pad_box1", GRIPPER_PREFIX + "pad_box2"):
        geom = root.find(f".//default[@class='{cls}']/geom")
        if geom is None:
            raise ValueError(f"gripper default class {cls} has no <geom>")
        geom.set("friction", fmt([p.pad_friction, 0.005, 0.0001]))
        geom.set("condim", "4")

    # ---- assets ----
    asset = _section(root, "asset")
    _add(asset, "texture", type="skybox", builtin="gradient", rgb1="0.35 0.55 0.75", rgb2="0.02 0.03 0.05", width=512, height=3072)
    _add(asset, "texture", type="2d", name="floor_tex", builtin="checker", mark="edge", rgb1="0.20 0.30 0.40", rgb2="0.10 0.20 0.30", markrgb="0.8 0.8 0.8", width=300, height=300)
    _add(asset, "material", name="floor_mat", texture="floor_tex", texuniform="true", texrepeat="6 6", reflectance="0.15")
    _add(asset, "material", name="table_mat", rgba="0.62 0.47 0.33 1", specular="0.15", shininess="0.1")
    _add(asset, "material", name="leg_mat", rgba="0.25 0.25 0.28 1")

    # ---- world ----
    world = root.find("worldbody")
    _add(world, "light", name="key", pos="1.2 -1.0 2.2", dir="-0.35 0.4 -1", diffuse="0.85 0.85 0.85", specular="0.2 0.2 0.2", castshadow="true")
    _add(world, "light", name="fill", pos="-0.6 1.0 1.6", dir="0.35 -0.5 -1", diffuse="0.35 0.35 0.40", specular="0 0 0", castshadow="false")
    _add(world, "geom", name="floor", type="plane", size="0 0 0.05", pos="0 0 -0.75", material="floor_mat")
    _add(
        world, "geom", name="table_top", type="box", pos="0.45 0 -0.02", size="0.55 0.65 0.02",
        material="table_mat", friction=fmt(p.table_friction), condim=p.object_condim,
        solref=fmt(p.solref), solimp=fmt(p.solimp),
    )  # fmt: skip
    for i, (lx, ly) in enumerate([(-0.05, -0.60), (-0.05, 0.60), (0.95, -0.60), (0.95, 0.60)]):
        _add(world, "geom", name=f"table_leg_{i}", type="cylinder", size="0.025 0.365", pos=fmt([lx, ly, -0.405]), material="leg_mat", contype=0, conaffinity=0)

    if s.tray:
        tx, ty = layout.tray_center
        hx, hy = s.tray_half_extent
        tw, wh = TRAY_WALL_HALF_THICKNESS, s.tray_wall_height
        tray_kw = dict(rgba="0.22 0.32 0.72 1", friction=fmt(p.table_friction), condim=p.object_condim, solref=fmt(p.solref), solimp=fmt(p.solimp))
        _add(world, "geom", name="tray_floor", type="box", pos=fmt([tx, ty, TRAY_FLOOR_HALF_THICKNESS]), size=fmt([hx, hy, TRAY_FLOOR_HALF_THICKNESS]), **tray_kw)
        wall_z = wh / 2
        _add(world, "geom", name="tray_wall_n", type="box", pos=fmt([tx + hx + tw, ty, wall_z]), size=fmt([tw, hy + 2 * tw, wh / 2]), **tray_kw)
        _add(world, "geom", name="tray_wall_s", type="box", pos=fmt([tx - hx - tw, ty, wall_z]), size=fmt([tw, hy + 2 * tw, wh / 2]), **tray_kw)
        _add(world, "geom", name="tray_wall_e", type="box", pos=fmt([tx, ty + hy + tw, wall_z]), size=fmt([hx, tw, wh / 2]), **tray_kw)
        _add(world, "geom", name="tray_wall_w", type="box", pos=fmt([tx, ty - hy - tw, wall_z]), size=fmt([hx, tw, wh / 2]), **tray_kw)

    _cube(world, "cube", "0.86 0.16 0.14 1", cfg, [cx, cy, h])
    palette = ["0.20 0.45 0.85 1", "0.95 0.75 0.15 1", "0.30 0.70 0.35 1", "0.70 0.35 0.80 1"]
    for i, name in enumerate(layout.distractor_names):
        _cube(world, name, palette[i % len(palette)], cfg, [cx - 0.12, cy + 0.12 * (i + 1), h])

    marker = _add(world, "body", name="goal_marker", mocap="true", pos=fmt([cx, cy, h]))
    if t.name == "reach":
        _add(marker, "geom", name="goal_geom", type="sphere", size="0.014", rgba="0.15 0.95 0.35 0.55", contype=0, conaffinity=0, group=1)
    else:
        _add(marker, "geom", name="goal_geom", type="box", size=fmt([h, h, h]), rgba="0.15 0.95 0.35 0.35", contype=0, conaffinity=0, group=1)

    for name, (pos, target, up) in CAMERAS.items():
        _add(world, "camera", name=name, pos=fmt(pos), xyaxes=fmt(look_at_xyaxes(pos, target, up)), fovy=45)

    return root


def build_scene_xml(cfg: EnvConfig, menagerie_root: str | Path | None = None) -> str:
    root = build_scene_tree(cfg, menagerie_root)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def write_scene_xml(cfg: EnvConfig, menagerie_root: str | Path | None = None) -> Path:
    """Write the scene XML into the cache (content-addressed) and return its path."""
    import hashlib

    xml = build_scene_xml(cfg, menagerie_root)
    digest = hashlib.sha1(xml.encode()).hexdigest()[:12]
    out = cache_root() / "scenes" / f"armforge_{digest}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text(xml, encoding="utf-8")
    return out
