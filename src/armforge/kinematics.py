"""Pure-numpy forward kinematics straight from an MJCF tree.

This module exists for two reasons:

1. **Independent oracle.** The test-suite compares MuJoCo's own site positions with this FK, so
   a mistake in the composed UR5e+Robotiq model (a wrong attachment frame, a mis-resolved joint
   axis) is caught by something that shares no code with MuJoCo.
2. **Offline design.** Workspace placement, tool orientation and the start pose can be checked
   without the simulator.

Supported subset (everything the composed model uses): nested ``body`` with ``pos``/``quat``,
hinge and slide ``joint`` with ``axis`` (including values inherited from ``<default>`` classes
and ``childclass``), and ``site`` with ``pos``/``quat``. Free joints are ignored (a free body is
placed by its own qpos, which this walker does not model).
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np

from .mathutil import axis_angle_to_mat, quat_to_mat


def _vec(text: str | None, default, n: int) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    out = np.array([float(v) for v in text.split()], dtype=float)
    if out.size != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return out


class _Defaults:
    """Resolves attributes through nested ``<default class=...>`` inheritance."""

    def __init__(self, root: ET.Element):
        self.classes: dict[str | None, tuple[str | None, dict[str, dict[str, str]]]] = {}
        for top in root.findall("default"):
            self._walk(top, None, None)

    def _walk(self, node: ET.Element, name: str | None, parent: str | None) -> None:
        attrs = {child.tag: dict(child.attrib) for child in node if child.tag != "default"}
        self.classes[name] = (parent, attrs)
        for child in node.findall("default"):
            self._walk(child, child.get("class"), name)

    def get(self, tag: str, attr: str, cls: str | None):
        seen = set()
        while cls in self.classes and cls not in seen:
            seen.add(cls)
            parent, attrs = self.classes[cls]
            if attr in attrs.get(tag, {}):
                return attrs[tag][attr]
            cls = parent
        # fall back to the unnamed top-level default
        _, attrs = self.classes.get(None, (None, {}))
        return attrs.get(tag, {}).get(attr)


class XmlKinematics:
    """Forward kinematics of the serial/tree structure described by an MJCF file."""

    def __init__(self, root: ET.Element | str):
        if isinstance(root, str):
            root = ET.fromstring(root) if root.lstrip().startswith("<") else ET.parse(root).getroot()
        self.root = root
        self.defaults = _Defaults(root)
        self.joints: list[str] = []  # hinge/slide joints in tree order
        self._bodies: list[dict] = []
        world = root.find("worldbody")
        if world is None:
            raise ValueError("MJCF has no <worldbody>")
        for body in world.findall("body"):
            self._parse_body(body, parent=None, childclass=None)

    # -- parsing ---------------------------------------------------------------------
    def _parse_body(self, body: ET.Element, parent: int | None, childclass: str | None) -> None:
        childclass = body.get("childclass", childclass)
        idx = len(self._bodies)
        record = {
            "name": body.get("name", f"body{idx}"),
            "parent": parent,
            "pos": _vec(body.get("pos"), [0, 0, 0], 3),
            "quat": _vec(body.get("quat"), [1, 0, 0, 0], 4),
            "joints": [],
            "sites": {},
        }
        for unsupported in ("euler", "axisangle", "xyaxes", "zaxis"):
            if body.get(unsupported) is not None:
                raise ValueError(f"body {record['name']}: '{unsupported}' orientation not supported")
        for joint in body.findall("joint"):
            cls = joint.get("class", childclass)
            jtype = joint.get("type") or self.defaults.get("joint", "type", cls) or "hinge"
            if jtype not in ("hinge", "slide"):
                continue
            axis = joint.get("axis") or self.defaults.get("joint", "axis", cls) or "0 0 1"
            jpos = joint.get("pos") or self.defaults.get("joint", "pos", cls) or "0 0 0"
            name = joint.get("name", f"joint{len(self.joints)}")
            self.joints.append(name)
            record["joints"].append(
                {"name": name, "type": jtype, "axis": _vec(axis, [0, 0, 1], 3), "pos": _vec(jpos, [0, 0, 0], 3)}
            )
        for site in body.findall("site"):
            sname = site.get("name")
            if sname:
                record["sites"][sname] = (_vec(site.get("pos"), [0, 0, 0], 3), _vec(site.get("quat"), [1, 0, 0, 0], 4))
        self._bodies.append(record)
        for child in body.findall("body"):
            self._parse_body(child, idx, childclass)

    # -- queries ---------------------------------------------------------------------
    @property
    def site_names(self) -> list[str]:
        return [s for b in self._bodies for s in b["sites"]]

    def _transforms(self, qpos: dict[str, float] | None) -> list[np.ndarray]:
        qpos = qpos or {}
        transforms: list[np.ndarray] = []
        for b in self._bodies:
            T = np.eye(4)
            T[:3, :3] = quat_to_mat(b["quat"])
            T[:3, 3] = b["pos"]
            parent_T = transforms[b["parent"]] if b["parent"] is not None else np.eye(4)
            T = parent_T @ T
            for j in b["joints"]:
                value = float(qpos.get(j["name"], 0.0))
                J = np.eye(4)
                if j["type"] == "hinge":
                    R = axis_angle_to_mat(j["axis"], value)
                    J[:3, :3] = R
                    J[:3, 3] = j["pos"] - R @ j["pos"]  # rotate about the joint anchor
                else:
                    axis = j["axis"] / np.linalg.norm(j["axis"])
                    J[:3, 3] = axis * value
                T = T @ J
            transforms.append(T)
        return transforms

    def forward(self, qpos: dict[str, float] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World pose ``(position, rotation_matrix)`` of every site for joint values ``qpos``."""
        transforms = self._transforms(qpos)
        sites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for b, T in zip(self._bodies, transforms, strict=False):
            for name, (pos, quat) in b["sites"].items():
                S = np.eye(4)
                S[:3, :3] = quat_to_mat(quat)
                S[:3, 3] = pos
                W = T @ S
                sites[name] = (W[:3, 3].copy(), W[:3, :3].copy())
        return sites

    def body_pose(self, name: str, qpos: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """World pose ``(position, rotation_matrix)`` of a named body."""
        transforms = self._transforms(qpos)
        for b, T in zip(self._bodies, transforms, strict=False):
            if b["name"] == name:
                return T[:3, 3].copy(), T[:3, :3].copy()
        raise KeyError(f"no body named {name!r}")

    def site_pose(self, site: str, qpos: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.forward(qpos)[site]

    def solve_ik(
        self,
        site: str,
        joint_names: list[str],
        target_pos,
        target_rot,
        q0,
        iterations: int = 400,
        damping: float = 1e-2,
        step: float = 0.8,
        limits: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Damped-least-squares 6D IK using finite-difference Jacobians.

        Returns ``(q, position_error_m, rotation_error_rad)``. Intended for offline checks.
        """
        from .mathutil import rotation_error

        q = np.asarray(q0, dtype=float).copy()
        target_pos = np.asarray(target_pos, float)
        for _ in range(iterations):
            pos, R = self.site_pose(site, dict(zip(joint_names, q, strict=False)))
            err = np.concatenate([target_pos - pos, rotation_error(R, target_rot)])
            if np.linalg.norm(err[:3]) < 1e-5 and np.linalg.norm(err[3:]) < 1e-4:
                break
            J = np.zeros((6, len(q)))
            eps = 1e-6
            for i in range(len(q)):
                dq = q.copy()
                dq[i] += eps
                p2, R2 = self.site_pose(site, dict(zip(joint_names, dq, strict=False)))
                J[:3, i] = (p2 - pos) / eps
                # world-frame angular velocity from the rotation difference
                dR = R2 @ R.T
                J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * eps)
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            q = q + step * dq
            if limits:
                for i, name in enumerate(joint_names):
                    lo, hi = limits.get(name, (-np.inf, np.inf))
                    q[i] = float(np.clip(q[i], lo, hi))
        pos, R = self.site_pose(site, dict(zip(joint_names, q, strict=False)))
        return q, float(np.linalg.norm(target_pos - pos)), float(np.linalg.norm(rotation_error(R, target_rot)))
