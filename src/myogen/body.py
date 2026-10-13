"""ForearmModel: the MuJoCo forearm (24 joints, 48 muscles), its anatomy by name, and the loads besides the muscles.

The forearm frame used everywhere (electrodes, EMG sources, viewer): origin at the elbow, z along the forearm
towards the wrist, attached to the ulna (it does not rotate in pronation). Angles around the forearm:
0 = radial border, +90 deg dorsal (extensors), -90 deg palmar (flexors), +-180 deg ulnar border.
"""
from pathlib import Path

import mujoco
import numpy as np

MODEL_XML = Path(__file__).resolve().parent / "models" / "myoforearm_r.xml"

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

# Passive joint stiffness (Nm/rad) around the relaxed posture: capsules, ligaments, resting muscle tone.
PASSIVE_STIFFNESS = {"pro_sup_r": 0.1, "deviation_r": 0.25, "flexion_r": 0.2, "cmc_flexion_r": 0.03,
                     "cmc_abduction_r": 0.03, "mp_flexion_r": 0.02, "ip_flexion_r": 0.01, "mcp": 0.04,
                     "mcp_abd": 0.02, "pm": 0.02, "md": 0.01}
# Torque opposing finger flexion at full grip load (Nm, ~5-10 N fingertip force).
GRIP_TORQUE = {"mcp": 0.35, "pm": 0.18, "md": 0.08}


class ForearmModel:
    """The musculoskeletal model. `m`, `d` are the MuJoCo model and data; the rest is anatomy looked up by name."""

    def __init__(self, xml=MODEL_XML):
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
        elbow, wrist = self._axis_points(self.d)
        self.forearm_length = float(np.linalg.norm(wrist - elbow))       # m

    @property
    def nu(self):
        return self.m.nu

    def default_data(self):
        """A fresh MjData at the model's default pose (where the BioMime territories are fitted)."""
        d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, d)
        return d

    # -- poses -------------------------------------------------------------------------------------------------

    def full_q(self, hand, elbow):
        """(T, nq) joint vectors from (T, 23) hand angles and (T,) elbow angles."""
        q = np.zeros(hand.shape[:-1] + (self.m.nq,))
        q[..., self.hand_qpos] = hand
        q[..., self.elbow_qpos] = elbow
        return q

    def set_pose(self, qpos):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)

    # -- geometry in the forearm frame -------------------------------------------------------------------------

    def _axis_points(self, d):
        """Elbow (ulna origin) and wrist (lunate) centres in world coordinates."""
        return d.xpos[self.m.body("ulna_r").id].copy(), d.xpos[self.m.body("lunate_r").id].copy()

    def forearm_frame(self, d=None):
        """(origin, axes) of the forearm frame (see the module docstring); rows of `axes` are x, y, z."""
        d = self.d if d is None else d
        elbow, wrist = self._axis_points(d)
        z = wrist - elbow
        z /= np.linalg.norm(z)
        R_ulna = d.xmat[self.m.body("ulna_r").id].reshape(3, 3)
        x = R_ulna[:, 0] - (R_ulna[:, 0] @ z) * z
        x /= np.linalg.norm(x)
        return elbow, np.stack([x, np.cross(z, x), z])

    def belly_points(self, n_per_muscle, d=None):
        """(nu, n, 3) points along each muscle belly in the forearm frame: the proximal part of the tendon path
        whose arc length is the fibre length."""
        d = self.d if d is None else d
        origin, axes = self.forearm_frame(d)
        out = np.zeros((self.m.nu, n_per_muscle, 3))
        u = (np.arange(n_per_muscle) + 0.5) / n_per_muscle
        for i, sites in enumerate(self.path_sites):
            p = (d.site_xpos[sites] - origin) @ axes.T
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
            fibre = np.clip(s[-1] - max(self.tendon_slack[i], 0.0), 0.2 * s[-1], s[-1])
            out[i] = np.stack([np.interp(u * fibre, s, p[:, k]) for k in range(3)], axis=1)
        return out

    def moment_arms(self, qpos):
        """(nu, nv) muscle moment arms (m) at joint state qpos."""
        self.set_pose(qpos)
        R = np.zeros((self.m.nu, self.m.nv))
        mujoco.mju_sparse2dense(R, self.d.actuator_moment, self.d.moment_rownnz, self.d.moment_rowadr,
                                self.d.moment_colind)
        return R

    def fibre_lengths(self, d=None):
        """Current fibre lengths (m): path length minus tendon slack (rigid tendon)."""
        return (self.d if d is None else d).actuator_length - self.tendon_slack

    # -- loads besides the muscles -----------------------------------------------------------------------------

    def prepare(self, subject, q_start, q_rest):
        """Set gravity and the passive joint springs for a run. Returns the fibre lengths at the relaxed posture."""
        self.set_pose(q_start)
        gravity = np.asarray(subject.gravity, float)
        gravity = gravity / np.linalg.norm(gravity)
        if subject.armrest:
            # Forearm on a horizontal support: gravity perpendicular to the forearm.
            axis = np.subtract(*self._axis_points(self.d)[::-1])
            axis /= np.linalg.norm(axis)
            gravity -= (gravity @ axis) * axis
            gravity /= np.linalg.norm(gravity)
        self.m.opt.gravity[:] = 9.81 * gravity
        self._set_passive_springs(q_rest, subject.passive_stiffness)
        return np.maximum(self.fibre_lengths(), 1e-3)

    def _set_passive_springs(self, q_rest, scale):
        """Joint springs whose equilibrium is q_rest: a relaxed hand rests there without muscle drive (each spring's
        reference is offset to balance gravity and passive muscle forces at q_rest). Leaves d at q_rest."""
        m, d = self.m, self.d
        k_all = np.zeros(m.njnt)
        for name in self.joint_names:
            k_all[m.joint(name).id] = _joint_stiffness(name) * scale
        m.jnt_stiffness[:] = k_all
        m.qpos_spring[:] = q_rest
        self.set_pose(q_rest)
        static = d.qfrc_bias - d.qfrc_actuator           # torque the springs must supply (actuator = passive muscle)
        for jid in np.flatnonzero(k_all):
            a, v = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
            m.qpos_spring[a] = q_rest[a] + np.clip(static[v] / k_all[jid], -0.6, 0.6)

    def support(self, subject):
        """(nv,) fraction of each joint's static torque carried by the armrest (or a resting hand)."""
        arm, wrist = (0.925, 0.6) if subject.armrest else (0.15, 0.0)
        support = np.zeros(self.m.nv)
        for name, frac in [(ELBOW_JOINT, arm), ("pro_sup_r", 0.7 * arm), ("deviation_r", wrist), ("flexion_r", wrist)]:
            support[self.m.jnt_dofadr[self.m.joint(name).id]] = frac
        return support

    def grip_torque(self, grip, closure):
        """(T, nv) torques opposing finger flexion when squeezing an object. grip: (T,) load level in [0, 1];
        closure: (T, 4) closure of fingers 2-5 in [0, 1] (only a closing finger meets the object)."""
        tau = np.zeros((len(grip), self.m.nv))
        for k, finger in enumerate("2345"):
            for joint, torque in GRIP_TORQUE.items():
                dof = self.m.jnt_dofadr[self.m.joint(f"{joint}{finger}_flexion_r").id]
                tau[:, dof] = -torque * grip * np.clip(closure[:, k] / 0.6, 0, 1)
        return tau


def _joint_stiffness(name):
    if name.startswith("mcp"):
        return PASSIVE_STIFFNESS["mcp_abd" if "abduction" in name else "mcp"]
    if name[:2] in ("pm", "md"):
        return PASSIVE_STIFFNESS[name[:2]]
    return PASSIVE_STIFFNESS.get(name, 0.0)
