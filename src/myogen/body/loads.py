"""Loads besides the muscles: passive joint springs, gravity, arm support and grip load."""
import mujoco
import numpy as np

# Passive joint stiffness (Nm/rad) around the relaxed posture: capsules, ligaments, resting muscle tone.
PASSIVE_STIFFNESS = {"pro_sup_r": 0.1, "deviation_r": 0.25, "flexion_r": 0.2, "cmc_flexion_r": 0.03,
                     "cmc_abduction_r": 0.03, "mp_flexion_r": 0.02, "ip_flexion_r": 0.01, "mcp": 0.04,
                     "mcp_abd": 0.02, "pm": 0.02, "md": 0.01}
# Torque opposing finger flexion at full grip load (Nm, ~5-10 N fingertip force).
GRIP_TORQUE = {"mcp": 0.35, "pm": 0.18, "md": 0.08}


def joint_stiffness(name):
    if name.startswith("mcp"):
        return PASSIVE_STIFFNESS["mcp_abd" if "abduction" in name else "mcp"]
    if name[:2] in ("pm", "md"):
        return PASSIVE_STIFFNESS[name[:2]]
    return PASSIVE_STIFFNESS.get(name, 0.0)


def set_passive_springs(m, d, anat, q_rest, scale):
    """Joint springs whose equilibrium is q_rest: a relaxed hand rests there with no muscle drive.

    Each spring's reference is offset so that at q_rest it balances gravity and the passive muscle forces.
    Leaves d at q_rest.
    """
    k_all = np.zeros(m.njnt)
    for name in anat.joint_names:
        k_all[m.joint(name).id] = joint_stiffness(name) * scale
    m.jnt_stiffness[:] = k_all
    m.qpos_spring[:] = q_rest
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q_rest
    mujoco.mj_forward(m, d)
    static = d.qfrc_bias - d.qfrc_actuator           # torque the springs must supply (actuator = passive muscle)
    for jid in np.flatnonzero(k_all):
        a, v = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        m.qpos_spring[a] = q_rest[a] + np.clip(static[v] / k_all[jid], -0.6, 0.6)


def set_gravity(m, d, session):
    """Gravity for this session. On an armrest it is perpendicular to the forearm at the current pose; the
    projected direction is written back to session.gravity_dir so it is stored with the trial."""
    if session.arm_support > 0.5:
        axis = d.xpos[m.body("lunate_r").id] - d.xpos[m.body("ulna_r").id]
        axis /= np.linalg.norm(axis)
        g = session.gravity_dir - (session.gravity_dir @ axis) * axis
        session.gravity_dir = g / np.linalg.norm(g)
    m.opt.gravity[:] = 9.81 * session.gravity_dir


def support_fractions(m, session):
    """(nv,) fraction of each joint's static torque carried by the armrest / a resting hand."""
    support = np.zeros(m.nv)
    for name, frac in [("elbow_flexion_r", session.arm_support), ("pro_sup_r", 0.7 * session.arm_support),
                       ("deviation_r", session.wrist_support), ("flexion_r", session.wrist_support)]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            support[m.jnt_dofadr[jid]] = frac
    return support


def grip_load_torque(m, latent, load_level):
    """(T, nv) joint torques opposing finger flexion, as when squeezing an object. load_level: (T,) in [0, 1]."""
    tau = np.zeros((len(latent), m.nv))
    for k, finger in enumerate("2345"):
        closure = latent[:, 1 + k]
        for joint, torque in GRIP_TORQUE.items():
            dof = m.jnt_dofadr[m.joint(f"{joint}{finger}_flexion_r").id]
            tau[:, dof] = -torque * load_level * np.clip(closure / 0.6, 0, 1)
    return tau
