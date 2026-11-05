"""Forward-kinematics cross-check: a pure-numpy CPU walk of the MJCF tree vs MuJoCo's own FK.

Two independent implementations of the same question - "where is the tool, given the six joint
angles?" - are compared:

* **CPU / numpy** - :class:`armforge.kinematics.XmlKinematics` reads the generated MJCF and chains
  the body/joint transforms itself. It shares no code with MuJoCo.
* **MuJoCo** - ``mj_kinematics`` on the compiled model, which is what the simulator and the
  controller actually use.

If the composed UR5e + Robotiq model has a wrong attachment frame, a mis-resolved joint axis or a
wrong base placement, the two disagree. Agreement to ~1e-9 m is expected (both are exact rigid-body
chains); anything above 1e-6 m means one of them is wrong.

Three comparisons are provided:

``compare_random``      random joint configurations within the joint limits
``compare_trajectory``  the joint states an actual (expert) episode visits, replayed through the CPU FK
``benchmark``           CPU numpy FK vs MuJoCo FK call time (numpy is a checker, not a fast path)

The numeric core (:func:`compare_poses`) takes plain callables, so it is unit-tested without MuJoCo.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig
from .kinematics import XmlKinematics
from .mathutil import rotation_error

PoseFn = Callable[[np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]


@dataclass
class FkReport:
    """Position error (metres) and orientation error (radians) between two FK implementations."""

    label: str
    samples: int = 0
    sites: tuple[str, ...] = ()
    pos_err: dict[str, np.ndarray] = field(default_factory=dict)
    rot_err: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def max_pos_err(self) -> float:
        return float(max((v.max() for v in self.pos_err.values()), default=0.0))

    @property
    def max_rot_err(self) -> float:
        return float(max((v.max() for v in self.rot_err.values()), default=0.0))

    def passed(self, pos_tol: float = 1e-6, rot_tol: float = 1e-6) -> bool:
        return self.samples > 0 and self.max_pos_err <= pos_tol and self.max_rot_err <= rot_tol

    def table(self) -> str:
        rows = [f"{self.label}: {self.samples} configurations", f"  {'site':<22}{'pos max (mm)':>14}{'pos mean (mm)':>15}{'rot max (deg)':>15}"]
        for s in self.sites:
            p, r = self.pos_err[s], self.rot_err[s]
            rows.append(f"  {s:<22}{p.max() * 1e3:>14.6f}{p.mean() * 1e3:>15.6f}{np.degrees(r.max()):>15.6f}")
        return "\n".join(rows)


def compare_poses(label: str, qs: Sequence[np.ndarray], cpu_fk: PoseFn, ref_fk: PoseFn, sites: Sequence[str]) -> FkReport:
    """Run both FK functions on every joint vector in ``qs`` and collect per-site errors."""
    pos = {s: [] for s in sites}
    rot = {s: [] for s in sites}
    for q in qs:
        a, b = cpu_fk(q), ref_fk(q)
        for s in sites:
            pos[s].append(np.linalg.norm(a[s][0] - b[s][0]))
            rot[s].append(np.linalg.norm(rotation_error(a[s][1], b[s][1])))
    return FkReport(label, len(qs), tuple(sites), {s: np.array(v) for s, v in pos.items()}, {s: np.array(v) for s, v in rot.items()})


# --------------------------------------------------------------------------------------
# bindings to the two implementations
# --------------------------------------------------------------------------------------


def cpu_fk_fn(kin: XmlKinematics, joint_names: Sequence[str], sites: Sequence[str]) -> PoseFn:
    def fk(q: np.ndarray):
        poses = kin.forward(dict(zip(joint_names, np.asarray(q, float))))
        return {s: poses[s] for s in sites}

    return fk


def mujoco_fk_fn(model, data, joint_names: Sequence[str], sites: Sequence[str]) -> PoseFn:
    import mujoco

    qadr = [model.joint(j).qposadr[0] for j in joint_names]
    ids = {s: model.site(s).id for s in sites}

    def fk(q: np.ndarray):
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        return {s: (data.site_xpos[i].copy(), data.site_xmat[i].reshape(3, 3).copy()) for s, i in ids.items()}

    return fk


def build_checker(cfg: EnvConfig, menagerie_root=None):
    """Compile the scene for ``cfg`` and return ``(model, data, kin, joint_names, sites, q_lo, q_hi)``."""
    import mujoco

    from .scene import ARM_JOINTS, TOOL_SITE, build_scene_tree, write_scene_xml

    kin = XmlKinematics(build_scene_tree(cfg, menagerie_root))
    model = mujoco.MjModel.from_xml_path(str(write_scene_xml(cfg, menagerie_root)))  # exactly what the env compiles
    data = mujoco.MjData(model)
    joints = tuple(ARM_JOINTS)
    wanted = ("attachment_site", TOOL_SITE)
    sites = tuple(s for s in wanted if s in kin.site_names)
    if TOOL_SITE not in sites:
        raise RuntimeError(f"tool site {TOOL_SITE!r} not found by the CPU walker; sites: {kin.site_names}")
    rng = np.array([model.joint(j).range for j in joints])
    return model, data, kin, joints, sites, rng[:, 0], rng[:, 1]


# --------------------------------------------------------------------------------------
# the three comparisons
# --------------------------------------------------------------------------------------


def compare_random(cfg: EnvConfig, samples: int = 1000, seed: int = 0, menagerie_root=None) -> FkReport:
    """Random joint vectors within the joint limits (clipped to +/-pi)."""
    model, data, kin, joints, sites, lo, hi = build_checker(cfg, menagerie_root)
    rng = np.random.default_rng(seed)
    qs = [rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi)) for _ in range(samples)]
    return compare_poses("random joint configurations", qs, cpu_fk_fn(kin, joints, sites), mujoco_fk_fn(model, data, joints, sites), sites)


def compare_trajectory(cfg: EnvConfig, episodes: int = 3, seed: int = 1_000_000, menagerie_root=None) -> tuple[FkReport, float]:
    """Replay the joint states of scripted-expert episodes through the CPU FK.

    Returns ``(report, env_vs_cpu_ee_mm)``: the numpy-vs-MuJoCo report for every visited joint vector, and
    the largest disagreement (mm) between the environment's own ``ee_pos`` and the CPU FK of that state.
    """
    from .env.arm_env import ArmForgeEnv
    from .rl.expert import make_expert
    from .scene import ARM_JOINTS, TOOL_SITE, build_scene_tree

    env = ArmForgeEnv(cfg, str(menagerie_root) if menagerie_root else None)
    try:
        kin = XmlKinematics(build_scene_tree(cfg, menagerie_root))
        sites = (TOOL_SITE,)
        cpu = cpu_fk_fn(kin, ARM_JOINTS, sites)
        qs, env_ee = [], []
        for e in range(episodes):
            env.reset(seed=seed + e)
            expert = make_expert(env)
            expert.reset()
            done = False
            while not done:
                _, _, done, info = env.step(expert.act())
                if info.get("unstable"):
                    break
                qs.append(env.data.qpos[env.arm_qadr].copy())
                env_ee.append(env.ee_pos.copy())
        # reference: MuJoCo evaluated on the same joint vectors (not the live state, so it is a pure FK comparison)
        ref = mujoco_fk_fn(env.model, env._scratch, ARM_JOINTS, sites)
        report = compare_poses(f"{episodes} expert episode(s), every visited state", qs, cpu, ref, sites)
        gap = max((np.linalg.norm(cpu(q)[TOOL_SITE][0] - p) for q, p in zip(qs, env_ee)), default=0.0) * 1e3
        return report, float(gap)
    finally:
        env.close()


def benchmark(cfg: EnvConfig, calls: int = 2000, menagerie_root=None) -> dict[str, float]:
    """Microseconds per FK call (tool + flange site) for the CPU numpy walker and for MuJoCo."""
    model, data, kin, joints, sites, lo, hi = build_checker(cfg, menagerie_root)
    rng = np.random.default_rng(0)
    qs = [rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi)) for _ in range(calls)]
    out = {}
    for name, fn in (("cpu_numpy", cpu_fk_fn(kin, joints, sites)), ("mujoco", mujoco_fk_fn(model, data, joints, sites))):
        t0 = time.perf_counter()
        for q in qs:
            fn(q)
        out[name] = (time.perf_counter() - t0) / calls * 1e6
    return out
