from __future__ import annotations

import mujoco
import numpy as np

from ..params import MusculoskeletalParams

FINGER_JOINTS = {"mcp": [f"mcp{f}_flexion_r" for f in "2345"], "pm": [f"pm{f}_flexion_r" for f in "2345"],
                 "md": [f"md{f}_flexion_r" for f in "2345"], "mcp_abd": [f"mcp{f}_abduction_r" for f in "2345"]}
GRID_DEG = np.arange(-90.0, 120.1, 1.0)     # where the curves are tabulated (beyond: the end values' slopes)


class PassiveFingers:
    """Passive torques of the finger joints (fingers 2-5). Flexion (MCP, PIP, DIP): measured moment-angle curves,
    steep towards the ends of the range, nearly flat around rest; a curve is the measured total and the model's
    muscles already give part of it passively, so the joint adds the rest (the joint tissues: capsule, ligaments,
    skin), tabulated at the reference posture. MCP abduction: the linear stiffness of MusculoskeletalParams (no
    measured curve). A constant per joint (set_rest) makes the relaxed posture an equilibrium, whatever the muscles'
    passive pull there. The muscle solver applies it every control tick as an external torque, so its inverse
    dynamics and the simulation both see it (the damping is the model's own, MusculoskeletalParams.passive_damping)."""

    def __init__(self, model, params: MusculoskeletalParams) -> None:
        self.model, self.p, self.scale = model, params, 1.0
        m = model.m
        self.kinds = [kind for kind, joints in FINGER_JOINTS.items() for _ in joints]
        names = [name for joints in FINGER_JOINTS.values() for name in joints]
        self.dofs = [int(m.jnt_dofadr[m.joint(n).id]) for n in names]
        self.qpos = [int(m.jnt_qposadr[m.joint(n).id]) for n in names]
        self.table = np.zeros((len(names), len(GRID_DEG)))
        self.offset = np.zeros(len(names))

    def tabulate(self) -> None:
        """The joint tissues' share for the model's muscles as they are now (strength included)."""
        ref = self.model.default_qpos
        self.table = np.array([self._joint(kind, ref, q, v) for kind, q, v in zip(self.kinds, self.qpos, self.dofs)])

    def joint_torques(self, qpos: np.ndarray) -> np.ndarray:
        """(nv,) the finger joints' passive torques at this joint state, zero elsewhere."""
        out = np.zeros(self.model.m.nv)
        out[self.dofs] = self.torque(qpos)
        return out

    def torque(self, qpos: np.ndarray) -> np.ndarray:
        """(joints,) passive torque (Nm, + = flexion) of the joint tissues here, rest offsets included."""
        deg = np.degrees(qpos[self.qpos])
        x = np.clip(deg, GRID_DEG[0], GRID_DEG[-1])
        tau = np.array([np.interp(v, GRID_DEG, t) for v, t in zip(x, self.table)])
        slope = (self.table[:, -1] - self.table[:, -2], self.table[:, 1] - self.table[:, 0])   # Nm per grid step
        tau += np.where(deg > GRID_DEG[-1], slope[0] * (deg - GRID_DEG[-1]), 0.0)
        tau += np.where(deg < GRID_DEG[0], slope[1] * (deg - GRID_DEG[0]), 0.0)
        return self.scale * tau + self.offset

    def set_rest(self, needed: np.ndarray, q_rest: np.ndarray, scale: float) -> None:
        """Offsets so that at q_rest the joints give needed (nv,) torque (gravity and passive muscles balanced).
        scale: the subject's passive stiffness."""
        self.scale, self.offset[:] = scale, 0.0
        self.offset = needed[self.dofs] - self.torque(q_rest)

    def _joint(self, kind: str, ref: np.ndarray, qpos: int, dof: int) -> np.ndarray:
        """The joint's own passive torque over GRID_DEG."""
        if kind == "mcp_abd":
            return -self.p.passive_stiffness["mcp_abd"] * np.radians(GRID_DEG)
        return self._measured(kind, GRID_DEG) - self._muscles(self.model, ref, qpos, dof)

    def _measured(self, kind: str, deg: np.ndarray) -> np.ndarray:
        """The measured total passive torque (Nm, + = flexion) at these angles."""
        p = self.p
        if kind == "mcp":
            a, b, c, dd, e, f = p.passive_mcp
            return 0.01 * (a * (np.exp(-b * (deg - e)) - 1) - c * (np.exp(dd * (deg - f)) - 1))      # N cm -> Nm
        coef, valid = ((p.passive_pip_cubic, p.passive_pip_valid_deg) if kind == "pm" else
                       (p.passive_dip_cubic, p.passive_dip_valid_deg))
        return np.polynomial.Polynomial(coef)(np.radians(np.clip(deg, *valid)))       # constant beyond

    @staticmethod
    def _muscles(model, ref: np.ndarray, qpos: int, dof: int) -> np.ndarray:
        """Passive muscle torque (Nm, + = flexion) at this joint over GRID_DEG, the rest of the hand at ref."""
        d = mujoco.MjData(model.m)
        out = []
        for deg in GRID_DEG:
            d.qpos[:] = ref
            d.qpos[qpos] = np.radians(deg)
            mujoco.mj_forward(model.m, d)
            out.append(d.qfrc_actuator[dof])           # no activation: the muscles' passive force only
        return np.array(out)
