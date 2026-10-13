"""The musculoskeletal forearm model: loading, anatomy bookkeeping and geometry in the forearm frame.

Joints and muscles are addressed by name, so the rest of the pipeline works with either model:
"myoforearm" (packaged, 24 DoF incl. elbow, 48 muscles) or MyoSim's "myohand" (23 DoF, 39 muscles).
"""
import functools
from dataclasses import dataclass

import mujoco
import myo_sim
import numpy as np

from ..config import PACKAGE

MODEL_XML = PACKAGE / "models" / "myoforearm_r.xml"

# Hand/wrist joints driven by the posture latent, in JointMap output order.
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


def load(name="myoforearm"):
    """(model, data) after a forward pass."""
    if name == "myoforearm":
        if not MODEL_XML.exists():
            raise FileNotFoundError(f"{MODEL_XML} missing: run `myogen build-model` first")
        model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        data = mujoco.MjData(model)
    elif name == "myohand":
        model, data = myo_sim.load("myohand_r")
    else:
        raise ValueError(f"unknown model {name!r}: myoforearm | myohand")
    mujoco.mj_forward(model, data)
    return model, data


@functools.cache
def shared_model():
    """One (model, data, anatomy) per process, reused by every trial it simulates."""
    m, d = load()
    return m, d, Anatomy.from_model(m, d)


@dataclass
class Anatomy:
    joint_names: list
    muscle_names: list
    extrinsic: np.ndarray       # (nu,) bool: muscle belly in the forearm
    qmin: np.ndarray
    qmax: np.ndarray
    tendon_ids: np.ndarray      # tendon driven by each muscle
    tendon_slack: np.ndarray    # (nu,) tendon slack length (m), from MuJoCo's muscle length normalisation
    path_sites: list            # per muscle: site ids along the path, origin -> insertion
    forearm_length: float       # m, elbow to wrist
    hand_qpos: np.ndarray       # qpos index of each HAND_JOINTS entry
    elbow_qpos: int             # qpos index of the elbow, -1 if the model has none

    @classmethod
    def from_model(cls, m, d):
        muscles = [m.actuator(i).name for i in range(m.nu)]
        tendon_ids = m.actuator_trnid[:, 0].copy()
        lr, rg = m.actuator_lengthrange, m.actuator_gainprm[:, :2]
        L0 = (lr[:, 1] - lr[:, 0]) / (rg[:, 1] - rg[:, 0])
        path_sites = []
        for t in tendon_ids:
            adr, num = m.tendon_adr[t], m.tendon_num[t]
            path_sites.append(np.array([m.wrap_objid[k] for k in range(adr, adr + num)
                                        if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_SITE]))
        elbow, wrist = forearm_axis_points(m, d)
        elbow_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ELBOW_JOINT)
        return cls(
            joint_names=[m.joint(i).name for i in range(m.njnt)],
            muscle_names=muscles,
            extrinsic=np.array([not n.startswith(INTRINSIC_PREFIXES) for n in muscles]),
            qmin=m.jnt_range[:, 0].copy(),
            qmax=m.jnt_range[:, 1].copy(),
            tendon_ids=tendon_ids,
            tendon_slack=lr[:, 0] - rg[:, 0] * L0,
            path_sites=path_sites,
            forearm_length=float(np.linalg.norm(wrist - elbow)),
            hand_qpos=np.array([m.jnt_qposadr[m.joint(n).id] for n in HAND_JOINTS]),
            elbow_qpos=int(m.jnt_qposadr[elbow_id]) if elbow_id >= 0 else -1,
        )

    def hand_limits(self):
        return self.qmin[self.hand_qpos], self.qmax[self.hand_qpos]

    def full_q(self, q_hand, elbow=None):
        """(..., nq) joint vectors from (..., 23) hand joints and the elbow angle(s)."""
        q = np.zeros(q_hand.shape[:-1] + (len(self.qmin),))
        q[..., self.hand_qpos] = q_hand
        if self.elbow_qpos >= 0 and elbow is not None:
            q[..., self.elbow_qpos] = elbow
        return q


def set_pose(m, d, qpos):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)


def forearm_axis_points(m, d):
    """Elbow (ulna origin) and wrist (lunate) centres in world coordinates."""
    return d.xpos[m.body("ulna_r").id].copy(), d.xpos[m.body("lunate_r").id].copy()


def forearm_frame(m, d):
    """(origin, axes): origin at the elbow, z along the forearm, attached to the ulna (rows are the axes).

    The ulna does not rotate in pronation, so an armband is roughly fixed in this frame while the radius-side
    muscles move underneath it.
    """
    elbow, wrist = forearm_axis_points(m, d)
    z = wrist - elbow
    z /= np.linalg.norm(z)
    R_ulna = d.xmat[m.body("ulna_r").id].reshape(3, 3)
    x = R_ulna[:, 0] - (R_ulna[:, 0] @ z) * z
    x /= np.linalg.norm(x)
    return elbow, np.stack([x, np.cross(z, x), z])


def belly_points(m, d, anat, n_per_muscle, rng=None, spread=0.0):
    """(nu, n_per_muscle, 3) points along each muscle belly in the forearm frame (z = distance from the elbow).

    The belly is the proximal part of the tendon path whose arc length is the fibre length.
    """
    origin, axes = forearm_frame(m, d)
    out = np.zeros((m.nu, n_per_muscle, 3))
    u = (np.arange(n_per_muscle) + 0.5) / n_per_muscle
    for i, sites in enumerate(anat.path_sites):
        p = (d.site_xpos[sites] - origin) @ axes.T
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        fibre = np.clip(s[-1] - max(anat.tendon_slack[i], 0.0), 0.2 * s[-1], s[-1])
        out[i] = np.stack([np.interp(u * fibre, s, p[:, k]) for k in range(3)], axis=1)
    if spread > 0 and rng is not None:
        out[..., :2] += rng.normal(0.0, spread, size=out[..., :2].shape)    # muscle width
    return out


def fibre_lengths(m, d, anat):
    """Current fibre lengths (m): path length minus tendon slack (rigid tendon)."""
    return d.actuator_length - anat.tendon_slack


def dense_moments(m, d, out=None):
    """Muscle moment arms as a dense (nu, nv) matrix."""
    if out is None:
        out = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(out, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
    return out


def muscle_gain_bias(m, d):
    """Per-muscle (gain, bias) with actuator_force = gain * act + bias at the current state."""
    gain, bias = np.empty(m.nu), np.empty(m.nu)
    for a in range(m.nu):
        L, V = d.actuator_length[a], d.actuator_velocity[a]
        gain[a] = mujoco.mju_muscleGain(L, V, m.actuator_lengthrange[a], m.actuator_acc0[a], m.actuator_gainprm[a, :9])
        bias[a] = mujoco.mju_muscleBias(L, m.actuator_lengthrange[a], m.actuator_acc0[a], m.actuator_biasprm[a, :9])
    return gain, bias
