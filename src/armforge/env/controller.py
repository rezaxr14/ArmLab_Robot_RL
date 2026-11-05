"""Cartesian end-effector control for the UR5e through damped-least-squares differential IK.

The policy outputs a small Cartesian displacement of the Robotiq *pinch* point. Each control
step this module turns it into joint position targets for the UR5e's built-in position
actuators (kp 2000/500, forcerange 150/28 Nm), while holding the tool pointing straight down.
Because the actuators are real PD loops with force limits, contacts, payload and friction all
act on the commanded motion - the arm is not teleported.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..mathutil import rotation_error


class CartesianController:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_id: int,
        arm_qpos_adr: np.ndarray,
        arm_dof_adr: np.ndarray,
        joint_ids: np.ndarray,
        target_rot: np.ndarray,
        damping: float = 0.05,
        rot_gain: float = 1.0,
        max_dq: float = 0.12,
    ):
        self.model = model
        self.site_id = site_id
        self.qadr = np.asarray(arm_qpos_adr)
        self.dadr = np.asarray(arm_dof_adr)
        self.target_rot = np.asarray(target_rot, dtype=float)
        self.damping = float(damping)
        self.rot_gain = float(rot_gain)
        self.max_dq = float(max_dq)
        rng = model.jnt_range[np.asarray(joint_ids)]
        self.q_lo, self.q_hi = rng[:, 0].copy(), rng[:, 1].copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # -- core DLS ---------------------------------------------------------------------
    def _jacobian(self, data: mujoco.MjData) -> np.ndarray:
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.site_id)
        return np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])

    def _dls(self, J: np.ndarray, err: np.ndarray) -> np.ndarray:
        return J.T @ np.linalg.solve(J @ J.T + (self.damping**2) * np.eye(6), err)

    def _errors(self, data: mujoco.MjData, target_pos: np.ndarray) -> np.ndarray:
        pos = data.site_xpos[self.site_id]
        R = data.site_xmat[self.site_id].reshape(3, 3)
        return np.concatenate([target_pos - pos, self.rot_gain * rotation_error(R, self.target_rot)])

    # -- per-control-step target --------------------------------------------------------
    def joint_targets(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        gain: float = 1.0,
        max_lead: float = 0.15,
    ) -> np.ndarray:
        """Joint position targets that move the pinch point towards ``target_pos`` this step.

        The target is built on the *measured* joints (no integrated command: an integrator around the
        servo lag oscillates). ``gain`` > 1 compensates the servo lag so that the achieved motion matches
        the requested one; the target may lead the measured joints by at most ``max_lead`` rad, which
        also bounds the force a blocked arm can apply.

        ``data`` must be kinematics-current (``mj_kinematics`` + ``mj_comPos``).
        """
        J = self._jacobian(data)
        dq = self._dls(J, self._errors(data, target_pos))
        peak = np.max(np.abs(dq))
        if peak > self.max_dq:
            dq = dq * (self.max_dq / peak)
        q_meas = data.qpos[self.qadr]
        q = np.clip(q_meas + gain * dq, q_meas - max_lead, q_meas + max_lead)
        return np.clip(q, self.q_lo, self.q_hi)

    # -- offline IK (resets, start pose) ------------------------------------------------
    def solve_ik(
        self,
        scratch: mujoco.MjData,
        target_pos: np.ndarray,
        q_init: np.ndarray,
        iterations: int = 300,
        pos_tol: float = 5e-5,
        rot_tol: float = 5e-4,
        step: float = 0.8,
    ) -> tuple[np.ndarray, float, float]:
        """Iterative 6D IK on a scratch ``MjData``; returns ``(q, pos_err_m, rot_err_rad)``."""
        q = np.asarray(q_init, dtype=float).copy()
        for _ in range(iterations):
            scratch.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, scratch)
            err = self._errors(scratch, np.asarray(target_pos, dtype=float))
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                break
            dq = self._dls(self._jacobian(scratch), err)
            peak = np.max(np.abs(dq))
            if peak > 0.3:
                dq = dq * (0.3 / peak)
            q = np.clip(q + step * dq, self.q_lo, self.q_hi)
        scratch.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, scratch)
        err = self._errors(scratch, np.asarray(target_pos, dtype=float))
        return q, float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]) / max(self.rot_gain, 1e-9))
