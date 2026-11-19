from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..inputs.subject import Subject
from ..params import MusculoskeletalParams

MODEL_XML = Path(__file__).resolve().parent.parent / "models" / "myoforearm_r.xml"

# Hand and wrist joints, in the order HandMotion stores them.
HAND_JOINTS = [
    "pro_sup_r", "deviation_r", "flexion_r",
    "cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r",
    "mcp2_flexion_r", "mcp2_abduction_r", "pm2_flexion_r", "md2_flexion_r",
    "mcp3_flexion_r", "mcp3_abduction_r", "pm3_flexion_r", "md3_flexion_r",
    "mcp4_flexion_r", "mcp4_abduction_r", "pm4_flexion_r", "md4_flexion_r",
    "mcp5_flexion_r", "mcp5_abduction_r", "pm5_flexion_r", "md5_flexion_r",
]
ELBOW_JOINT = "elbow_flexion_r"
# Intrinsic hand muscles: absent after a transradial amputation and invisible to forearm electrodes.
INTRINSIC_PREFIXES = ("OP_", "RI", "LU_RB", "UI_UB")
R_MODEL_REF = 0.032          # m, outermost muscle line-of-action radius in the model's proximal forearm
GRAVITY = 9.81               # m/s^2


class ForearmModel:
    """The musculoskeletal model. `m`, `d` are the MuJoCo model and data; the rest is anatomy looked up by name."""

    def __init__(self, xml: str | Path = MODEL_XML, params: MusculoskeletalParams | None = None) -> None:
        self.params = params or MusculoskeletalParams()
        self.m = m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(m)
        mujoco.mj_forward(m, self.d)
        self.default_qpos = self.d.qpos.copy()
        self.joint_names = [m.joint(i).name for i in range(m.njnt)]
        self.muscle_names = [m.actuator(i).name for i in range(m.nu)]
        self.extrinsic = np.array([not n.startswith(INTRINSIC_PREFIXES) for n in self.muscle_names])
        self.qmin, self.qmax = m.jnt_range[:, 0].copy(), m.jnt_range[:, 1].copy()
        self.hand_qpos = np.array([m.jnt_qposadr[m.joint(n).id] for n in HAND_JOINTS])
        self.elbow_qpos = int(m.jnt_qposadr[m.joint(ELBOW_JOINT).id])
        # Tendon slack length per muscle, from MuJoCo's muscle length normalisation.
        lr, rg = m.actuator_lengthrange, m.actuator_gainprm[:, :2]
        self.tendon_slack = lr[:, 0] - rg[:, 0] * (lr[:, 1] - lr[:, 0]) / (rg[:, 1] - rg[:, 0])
        self.path_sites = []                 # per muscle: site ids along the path, origin -> insertion
        for t in m.actuator_trnid[:, 0]:
            adr, num = m.tendon_adr[t], m.tendon_num[t]
            self.path_sites.append(np.array([m.wrap_objid[k] for k in range(adr, adr + num)
                                             if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_SITE]))
        # The same paths padded to one length (the last site repeated) for vectorised belly points.
        n_sites = max(len(p) for p in self.path_sites)
        self.path_padded = np.array([np.concatenate([p, np.repeat(p[-1:], n_sites - len(p))]) for p in self.path_sites])
        elbow, wrist = self._axis_points(self.d)
        self.forearm_length = float(np.linalg.norm(wrist - elbow))       # m

    @property
    def nu(self) -> int:
        return self.m.nu

    # -- poses -------------------------------------------------------------------------------------------------

    def full_q(self, hand: np.ndarray, elbow: np.ndarray | float) -> np.ndarray:
        """(T, nq) joint vectors from (T, 23) hand angles and (T,) elbow angles."""
        q = np.zeros(hand.shape[:-1] + (self.m.nq,))
        q[..., self.hand_qpos] = hand
        q[..., self.elbow_qpos] = elbow
        return q

    def set_pose(self, qpos: np.ndarray) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)

    # -- geometry in the forearm frame -------------------------------------------------------------------------

    def _axis_points(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Elbow (ulna origin) and wrist (lunate) centres in world coordinates."""
        return d.xpos[self.m.body("ulna_r").id].copy(), d.xpos[self.m.body("lunate_r").id].copy()

    def forearm_frame(self, d: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(origin, axes) of the forearm frame: origin at the elbow, z towards the wrist, x towards the radial border,
        attached to the ulna (it does not rotate in pronation). Rows of `axes` are x, y, z."""
        d = self.d if d is None else d
        elbow, wrist = self._axis_points(d)
        z = wrist - elbow
        z /= np.linalg.norm(z)
        R_ulna = d.xmat[self.m.body("ulna_r").id].reshape(3, 3)
        x = R_ulna[:, 0] - (R_ulna[:, 0] @ z) * z
        x /= np.linalg.norm(x)
        return elbow, np.stack([x, np.cross(z, x), z])

    def belly_points(self, n_per_muscle: int, d: mujoco.MjData | None = None) -> np.ndarray:
        """(nu, n, 3) points along each muscle belly in the forearm frame: the proximal part of the tendon path
        whose arc length is the fibre length."""
        d = self.d if d is None else d
        origin, axes = self.forearm_frame(d)
        p = (d.site_xpos[self.path_padded] - origin) @ axes.T                   # (nu, sites, 3)
        seg = np.linalg.norm(np.diff(p, axis=1), axis=2)
        s = np.concatenate([np.zeros((len(p), 1)), np.cumsum(seg, axis=1)], axis=1)
        total = s[:, -1]
        fibre = np.clip(total - np.maximum(self.tendon_slack, 0.0), 0.2 * total, total)
        target = (np.arange(n_per_muscle) + 0.5) / n_per_muscle * fibre[:, None]   # (nu, n) arc length
        k = np.clip((s[:, None, :] <= target[..., None]).sum(axis=2) - 1, 0, s.shape[1] - 2)
        s0 = np.take_along_axis(s, k, axis=1)
        frac = np.divide(target - s0, np.take_along_axis(seg, k, axis=1), out=np.zeros_like(target),
                         where=np.take_along_axis(seg, k, axis=1) > 0)
        rows = np.arange(len(p))[:, None]
        return p[rows, k] + frac[..., None] * (p[rows, k + 1] - p[rows, k])

    def scale_strength(self, factors: np.ndarray) -> None:
        """Maximal isometric force of every muscle times these factors (active and passive force alike)."""
        self.m.actuator_gainprm[:, 2] *= factors
        self.m.actuator_biasprm[:, 2] *= factors

    def radius_extent(self, d: mujoco.MjData | None = None) -> tuple[float, float]:
        """Distance from the elbow (m) of the radius head and of the styloid tip, in the forearm frame."""
        d = self.d if d is None else d
        g = self.m.geom("radius_r").id
        mesh = self.m.geom_dataid[g]
        adr = self.m.mesh_vertadr[mesh]
        v = self.m.mesh_vert[adr:adr + self.m.mesh_vertnum[mesh]]
        origin, axes = self.forearm_frame(d)
        z = ((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T) - origin) @ axes[2]
        return float(z.min()), float(z.max())

    def moment_arms(self, qpos: np.ndarray) -> np.ndarray:
        """(nu, nv) muscle moment arms (m) at joint state qpos."""
        self.set_pose(qpos)
        R = np.zeros((self.m.nu, self.m.nv))
        mujoco.mju_sparse2dense(R, self.d.actuator_moment, self.d.moment_rownnz, self.d.moment_rowadr,
                                self.d.moment_colind)
        return R

    def fibre_lengths(self, d: mujoco.MjData | None = None) -> np.ndarray:
        """Current fibre lengths (m): path length minus tendon slack (rigid tendon)."""
        return (self.d if d is None else d).actuator_length - self.tendon_slack

    def fibre_lengths_at(self, qpos: np.ndarray) -> np.ndarray:
        """Fibre lengths (m) at joint state qpos (moves the model there)."""
        self.set_pose(qpos)
        return np.maximum(self.fibre_lengths(), 1e-3)

    # -- loads besides the muscles -----------------------------------------------------------------------------

    def set_environment(self, subject: Subject, q_start: np.ndarray, q_rest: np.ndarray) -> None:
        """The physics of a run besides the muscles: gravity (perpendicular to the forearm at q_start when it lies on
        an armrest) and passive joint springs that hold the hand at the relaxed posture q_rest without muscle drive."""
        self.set_pose(q_start)
        gravity = np.asarray(subject.gravity, float)
        gravity = gravity / np.linalg.norm(gravity)
        if subject.armrest:
            # Forearm on a horizontal support: gravity perpendicular to the forearm.
            axis = np.subtract(*self._axis_points(self.d)[::-1])
            axis /= np.linalg.norm(axis)
            gravity -= (gravity @ axis) * axis
            gravity /= np.linalg.norm(gravity)
        self.m.opt.gravity[:] = GRAVITY * gravity
        self._set_passive_springs(q_rest, subject.passive_stiffness)

    def _set_passive_springs(self, q_rest: np.ndarray, scale: float) -> None:
        """Joint springs whose equilibrium is q_rest: a relaxed hand rests there without muscle drive (each spring's
        reference is offset to balance gravity and passive muscle forces at q_rest). Leaves d at q_rest."""
        m, d = self.m, self.d
        k_all = np.zeros(m.njnt)
        for name in self.joint_names:
            k_all[m.joint(name).id] = _joint_stiffness(name, self.params.passive_stiffness) * scale
        m.jnt_stiffness[:] = k_all
        m.qpos_spring[:] = q_rest
        self.set_pose(q_rest)
        static = d.qfrc_bias - d.qfrc_actuator           # torque the springs must supply (actuator = passive muscle)
        for jid in np.flatnonzero(k_all):
            a, v = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
            m.qpos_spring[a] = q_rest[a] + np.clip(static[v] / k_all[jid], -self.params.spring_offset_max,
                                                    self.params.spring_offset_max)

    def support(self, subject: Subject) -> np.ndarray:
        """(nv,) fraction of each joint's static torque carried by the armrest (or a resting hand)."""
        p = self.params
        arm, wrist = p.armrest_support if subject.armrest else p.held_support
        support = np.zeros(self.m.nv)
        for name, frac in [(ELBOW_JOINT, arm), ("pro_sup_r", p.pronation_support * arm), ("deviation_r", wrist),
                           ("flexion_r", wrist)]:
            support[self.m.jnt_dofadr[self.m.joint(name).id]] = frac
        return support

    def grip_torque(self, grip: np.ndarray, closure: np.ndarray) -> np.ndarray:
        """(T, nv) torques opposing finger flexion when squeezing an object. grip: (T,) load level in [0, 1];
        closure: (T, 4) closure of fingers 2-5 in [0, 1] (only a closing finger meets the object)."""
        tau = np.zeros((len(grip), self.m.nv))
        for k, finger in enumerate("2345"):
            for joint, torque in self.params.grip_torque.items():
                dof = self.m.jnt_dofadr[self.m.joint(f"{joint}{finger}_flexion_r").id]
                tau[:, dof] = -torque * grip * np.clip(closure[:, k] / self.params.grip_contact_closure, 0, 1)
        return tau


def _joint_stiffness(name: str, stiffness: dict[str, float]) -> float:
    """A joint's passive stiffness (Nm/rad); finger joints by prefix (MusculoskeletalParams.passive_stiffness)."""
    if name.startswith("mcp"):
        return stiffness["mcp_abd" if "abduction" in name else "mcp"]
    if name[:2] in ("pm", "md"):
        return stiffness[name[:2]]
    return stiffness.get(name, 0.0)
