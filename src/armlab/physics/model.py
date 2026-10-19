"""MuJoCo model builder for the 3-link arm (Phase E).

The MJCF is generated programmatically from ``RobotParams`` so the physics
model always matches the analytical kinematics:

- waist: hinge about world +z at the origin.
- shoulder: hinge about +y at height l1; theta2 measured from vertical
  (positive tilts toward +x at theta1=0).
- elbow: hinge about +y relative to link 2.

Velocity actuators provide bounded control commands. The end-effector site
("ee") and target site ("target") allow direct pose comparison with the
analytical forward kinematics. The target body is a mocap body with all
collision flags disabled so it never affects dynamics.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from armlab.config import PhysicsParams, RobotParams

__all__ = ["build_arm_xml", "OBSTACLE_XML"]

# NOTE: mujoco itself is imported lazily by the env, not here; this module
# only produces the XML string so it stays testable without MuJoCo installed.


def _obstacle_xml(name: str, kind: str, pos: Sequence[float], size: Sequence[float]) -> str:
    # Obstacles use contype/conaffinity 3: they collide with arm links
    # (1 & 3 != 0) but not with the base column (4 & 3 == 0).
    pos_str = " ".join(f"{v:.4f}" for v in pos)
    if kind == "sphere":
        return (
            f'<body name="{name}" pos="{pos_str}">'
            f'<geom name="{name}" type="sphere" size="{size[0]:.4f}" '
            f'rgba="0.8 0.2 0.2 0.6" contype="3" conaffinity="3"/>'
            f"</body>"
        )
    if kind == "box":
        sz = " ".join(f"{v:.4f}" for v in size)
        return (
            f'<body name="{name}" pos="{pos_str}">'
            f'<geom name="{name}" type="box" size="{sz}" '
            f'rgba="0.2 0.6 0.2 0.6" contype="3" conaffinity="3"/>'
            f"</body>"
        )
    raise ValueError(f"Unknown obstacle kind {kind!r}; use 'sphere' or 'box'.")


OBSTACLE_XML = _obstacle_xml  # alias for callers


def build_arm_xml(
    robot: Optional[RobotParams] = None,
    physics: Optional[PhysicsParams] = None,
    obstacles: Optional[List[Tuple[str, str, Sequence[float], Sequence[float]]]] = None,
    include_floor: bool = True,
) -> str:
    """Build the arm MJCF as a string.

    Args:
        robot: arm geometry; defaults to RobotParams().
        physics: physics parameters (only dt is used here); defaults.
        obstacles: list of (name, kind, pos, size) tuples; kind is "sphere"
            (size = [radius]) or "box" (size = [hx, hy, hz]).
        include_floor: add a ground plane.

    Returns:
        Complete MJCF XML document.
    """
    robot = robot or RobotParams()
    physics = physics or PhysicsParams()
    l1, l2, l3 = robot.link_lengths
    lo = robot.joint_limits_low
    hi = robot.joint_limits_high

    exclusions = "".join(
        f'<exclude body1="{b1}" body2="{b2}"/>' for b1, b2 in physics.exclude_pairs
    )
    obstacle_bodies = "".join(_obstacle_xml(*spec) for spec in (obstacles or []))
    floor_xml = (
        '<geom name="floor" type="plane" size="30 30 0.1" rgba="0.8 0.8 0.8 1" '
        'contype="1" conaffinity="1"/>'
        if include_floor
        else ""
    )
    # Base column uses contype/conaffinity 4 so it never collides with the
    # floor (contype 1) or obstacles (contype 2): resting contact of the
    # intended fixed-base geometry must not count as a collision (Phase E4).
    base_geom_extra = ' contype="4" conaffinity="4"' if include_floor else ""

    return f"""
<mujoco model="armlab_3link">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{physics.sim_dt}" gravity="0 0 -9.81"/>

  <default>
    <geom friction="0.9 0.1 0.1" margin="0.001"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <velocity ctrllimited="true" kv="50"/>
  </default>

  <asset>
    <material name="arm" rgba="0.2 0.4 0.8 1"/>
    <material name="target" rgba="0.9 0.1 0.1 0.5"/>
  </asset>

  <worldbody>
    {floor_xml}
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.5 {l1 / 2:.4f}"
            pos="0 0 {l1 / 2:.4f}" material="arm"{base_geom_extra}/>
      <body name="link1" pos="0 0 0">
        <joint name="waist" type="hinge" axis="0 0 1" pos="0 0 0" damping="0.5"
               range="{lo[0]:.4f} {hi[0]:.4f}" limited="true"/>
        <inertial pos="0 0 0.5" mass="1.0" diaginertia="0.2 0.2 0.2"/>
        <body name="link2" pos="0 0 {l1:.4f}">
          <joint name="shoulder" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.5"
                 range="{lo[1]:.4f} {hi[1]:.4f}" limited="true"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 {l2:.4f}"
                size="0.25" material="arm" mass="2.0"/>
          <body name="link3" pos="0 0 {l2:.4f}">
            <joint name="elbow" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.5"
                   range="{lo[2]:.4f} {hi[2]:.4f}" limited="true"/>
            <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0 {l3:.4f}"
                  size="0.25" material="arm" mass="1.5"/>
            <site name="ee" pos="0 0 {l3:.4f}" size="0.2"/>
          </body>
        </body>
      </body>
    </body>
    {obstacle_bodies}
    <body name="target_body" mocap="true" pos="10 0 12">
      <geom name="target_geom" type="sphere" size="0.4" material="target"
            contype="0" conaffinity="0"/>
      <site name="target" pos="0 0 0" size="0.1"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="waist_vel" joint="waist" ctrlrange="-5 5"/>
    <velocity name="shoulder_vel" joint="shoulder" ctrlrange="-5 5"/>
    <velocity name="elbow_vel" joint="elbow" ctrlrange="-5 5"/>
  </actuator>

  <contact>
    {exclusions}
  </contact>
</mujoco>
"""
