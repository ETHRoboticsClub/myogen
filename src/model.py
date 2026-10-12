"""Model loading and anatomical bookkeeping (muscle groups, forearm frame, belly geometry).

Two models are supported, addressed by joint/muscle *names* so the rest of the pipeline is model-agnostic:
  * "myoforearm" (default): models/myoforearm_r.xml, built from MyoArm by src/build_model.py
                            (24 DoF incl. elbow, 48 muscles incl. SUP/BRD/ANC/BIC/BRA/TRI)
  * "myohand":              MyoSim's MyoHand (23 DoF, 39 muscles)
"""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import myo_sim
import numpy as np

MODEL_XML = Path(__file__).resolve().parent.parent / "models" / "myoforearm_r.xml"
DEFAULT_MODEL = "myoforearm"

# Hand/wrist joints driven by the posture latent (see motion.JointMap), in JointMap output order.
HAND_JOINTS = [
    "pro_sup_r", "deviation_r", "flexion_r",
    "cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r",
    "mcp2_flexion_r", "mcp2_abduction_r", "pm2_flexion_r", "md2_flexion_r",
    "mcp3_flexion_r", "mcp3_abduction_r", "pm3_flexion_r", "md3_flexion_r",
    "mcp4_flexion_r", "mcp4_abduction_r", "pm4_flexion_r", "md4_flexion_r",
    "mcp5_flexion_r", "mcp5_abduction_r", "pm5_flexion_r", "md5_flexion_r",
]
ELBOW_JOINT = "elbow_flexion_r"

# Intrinsic hand muscles: their bellies live in the hand, so they are absent after a transradial amputation
# and invisible to forearm electrodes.
INTRINSIC_PREFIXES = ("OP_", "RI", "LU_RB", "UI_UB")


def load(name=DEFAULT_MODEL):
    if name == "myoforearm":
        if not MODEL_XML.exists():
            raise FileNotFoundError(f"{MODEL_XML} missing: run `python -m src.build_model` first")
        model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        data = mujoco.MjData(model)
    elif name == "myohand":
        model, data = myo_sim.load("myohand_r")
    else:
        raise ValueError(name)
    mujoco.mj_forward(model, data)
    return model, data


@dataclass
class Anatomy:
    joint_names: list
    muscle_names: list
    extrinsic: np.ndarray       # bool mask (nu,)
    qmin: np.ndarray
    qmax: np.ndarray
    tendon_ids: np.ndarray      # tendon driven by each muscle
    tendon_slack: np.ndarray    # LT per muscle (m), from MuJoCo's muscle length normalisation
    path_sites: list            # per muscle: array of site ids along the path (origin -> insertion)
    forearm_length: float
    hand_qpos: np.ndarray       # qpos index of each HAND_JOINTS entry
    elbow_qpos: int             # qpos index of the elbow (-1 if the model has none)

    @classmethod
    def from_model(cls, m, d):
        muscles = [m.actuator(i).name for i in range(m.nu)]
        extrinsic = np.array([not n.startswith(INTRINSIC_PREFIXES) for n in muscles])
        tendon_ids = m.actuator_trnid[:, 0].copy()
        lr, rg = m.actuator_lengthrange, m.actuator_gainprm[:, :2]
        L0 = (lr[:, 1] - lr[:, 0]) / (rg[:, 1] - rg[:, 0])
        slack = lr[:, 0] - rg[:, 0] * L0
        path_sites = []
        for t in tendon_ids:
            adr, num = m.tendon_adr[t], m.tendon_num[t]
            path_sites.append(np.array([m.wrap_objid[k] for k in range(adr, adr + num)
                                        if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_SITE]))
        elbow, wrist = forearm_axis_points(m, d)
        return cls(
            joint_names=[m.joint(i).name for i in range(m.njnt)],
            muscle_names=muscles,
            extrinsic=extrinsic,
            qmin=m.jnt_range[:, 0].copy(),
            qmax=m.jnt_range[:, 1].copy(),
            tendon_ids=tendon_ids,
            tendon_slack=slack,
            path_sites=path_sites,
            forearm_length=float(np.linalg.norm(wrist - elbow)),
            hand_qpos=np.array([m.jnt_qposadr[m.joint(n).id] for n in HAND_JOINTS]),
            elbow_qpos=_qpos_or(m, ELBOW_JOINT),
        )

    def hand_limits(self):
        return self.qmin[self.hand_qpos], self.qmax[self.hand_qpos]

    def full_q(self, q_hand, elbow=None):
        """Assemble (T, nq) model joint vectors from (T, 23) hand joints (+ elbow angle(s))."""
        q = np.zeros(q_hand.shape[:-1] + (len(self.qmin),))
        q[..., self.hand_qpos] = q_hand
        if self.elbow_qpos >= 0 and elbow is not None:
            q[..., self.elbow_qpos] = elbow
        return q


def _qpos_or(m, name, default=-1):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(m.jnt_qposadr[jid]) if jid >= 0 else default


def forearm_axis_points(m, d):
    """Elbow (ulna origin) and wrist (lunate) centres in world coordinates."""
    return d.xpos[m.body("ulna_r").id].copy(), d.xpos[m.body("lunate_r").id].copy()


def forearm_frame(m, d):
    """Frame attached to the ulna: origin at the elbow, z along the forearm, x = anatomical reference.

    The ulna does not rotate during pronation, so a proximal armband is roughly fixed in this frame
    while the radius-side muscles move underneath it.
    """
    elbow, wrist = forearm_axis_points(m, d)
    z = wrist - elbow
    z /= np.linalg.norm(z)
    R_ulna = d.xmat[m.body("ulna_r").id].reshape(3, 3)
    x = R_ulna[:, 0] - (R_ulna[:, 0] @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return elbow, np.stack([x, y, z])  # rows are the frame axes


def belly_points(m, d, anat, n_per_muscle, rng=None, spread=0.0):
    """Sample source points along each muscle belly, in the forearm frame.

    The belly is taken as the proximal part of the tendon path with arc length = fibre length
    (current path length minus tendon slack). Returns (nu, n_per_muscle, 3) in metres,
    coordinates (x, y, z) with z = axial distance from the elbow.
    """
    origin, axes = forearm_frame(m, d)
    out = np.zeros((m.nu, n_per_muscle, 3))
    u = (np.arange(n_per_muscle) + 0.5) / n_per_muscle
    for i, sites in enumerate(anat.path_sites):
        p = (d.site_xpos[sites] - origin) @ axes.T
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        fibre = np.clip(s[-1] - max(anat.tendon_slack[i], 0.0), 0.2 * s[-1], s[-1])
        si = u * fibre
        out[i] = np.stack([np.interp(si, s, p[:, k]) for k in range(3)], axis=1)
    if spread > 0 and rng is not None:
        # Muscles have width: jitter source points transversally (fixed per subject, passed via rng seed).
        out[..., :2] += rng.normal(0.0, spread, size=out[..., :2].shape)
    return out


def dense_moments(m, d, out=None):
    """Muscle moment arms as a dense (nu, nv) matrix (MuJoCo stores them sparsely)."""
    if out is None:
        out = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(out, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
    return out


def muscle_gain_bias(m, d):
    """Per-muscle (gain, bias) so that actuator_force = gain * act + bias at the current state."""
    gain = np.empty(m.nu)
    bias = np.empty(m.nu)
    for a in range(m.nu):
        L, V = d.actuator_length[a], d.actuator_velocity[a]
        gain[a] = mujoco.mju_muscleGain(L, V, m.actuator_lengthrange[a], m.actuator_acc0[a], m.actuator_gainprm[a, :9])
        bias[a] = mujoco.mju_muscleBias(L, m.actuator_lengthrange[a], m.actuator_acc0[a], m.actuator_biasprm[a, :9])
    return gain, bias
