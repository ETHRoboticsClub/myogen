import mujoco
import numpy as np
from common import REFS, Forearm, Report
from scipy.optimize import linprog

r = REFS["finger_force"]
f = Forearm()
m, d = f.model.m, f.model.d
q = f.q.copy()
for k in "2345":
    for joint, deg in r["posture_deg"].items():
        q[f.qpos(f"{joint}{k}_flexion_r")] = np.radians(deg)
f.model.set_environment(f.subject, q, f.q)          # gravity, and the passive fingers relaxed at the usual rest


def torques(act: np.ndarray) -> np.ndarray:
    d.qpos[:], d.qvel[:], d.act[:] = q, 0, act
    mujoco.mj_forward(m, d)
    return d.qfrc_actuator.copy()


passive = torques(np.zeros(m.nu)) + f.model.passive_fingers.joint_torques(q) - d.qfrc_bias
active = np.array([torques(a) for a in np.eye(m.nu)]) - torques(np.zeros(m.nu))    # (nu, nv) at full activation

report = Report("Maximal single-finger pressing force (N) vs Shinohara et al. 2003, mean +- 2 SD")
for tip, (mean, sd) in r["fingers"].items():
    k = r["tip_sites"][tip]
    dofs = [f.dof(f"{j}{k}_{a}_r") for j, a in (("mcp", "flexion"), ("mcp", "abduction"), ("pm", "flexion"),
                                                 ("md", "flexion"))]
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jac, None, m.site(f"{tip}tip_r").id)
    flat = q.copy()
    flat[f.qpos(f"mcp{k}_flexion_r")] = 0
    d.qpos[:] = flat
    mujoco.mj_forward(m, d)
    jac0 = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jac0, None, m.site(f"{tip}tip_r").id)
    palmar = jac0[:, f.dof(f"mcp{k}_flexion_r")] / np.linalg.norm(jac0[:, f.dof(f"mcp{k}_flexion_r")])
    # Unknowns: activations (nu), the normal force F and two tangential forces (friction pyramid). The finger's joints
    # balance: muscles + passive = J^T (force on the sensor); the sensor measures F.
    tangent = np.linalg.svd(palmar[None])[2][1:]                        # (2, 3) orthogonal to palmar
    directions = np.vstack([palmar, tangent])                           # (3, 3): F, t1, t2
    a_eq = np.column_stack([active[:, dofs].T, -(jac[:, dofs].T @ directions.T)])
    mu = r["friction"]
    a_ub = np.zeros((4, m.nu + 3))                                       # |t_i| <= mu F
    for row, (i, s) in enumerate([(1, 1), (1, -1), (2, 1), (2, -1)]):
        a_ub[row, m.nu + i], a_ub[row, m.nu] = s, -mu
    res = linprog(np.r_[np.zeros(m.nu), -1.0, 0, 0], A_eq=a_eq, b_eq=-passive[dofs], A_ub=a_ub, b_ub=np.zeros(4),
                  bounds=[(0, 1)] * m.nu + [(0, None), (None, None), (None, None)], method="highs")
    force = res.x[m.nu] if res.success else 0.0
    report.check(f"{tip}: force", force, mean - 2 * sd, mean + 2 * sd)
    if res.success:
        used = np.argsort(-res.x[:m.nu])[:4]
        print(f"  {tip}: " + ", ".join(f"{f.model.muscle_names[i]} {res.x[i]:.2f}" for i in used))
report.show()
