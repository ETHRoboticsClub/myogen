import mujoco
import numpy as np
from common import REFS, Forearm, Report

f = Forearm()
m, d = f.model.m, f.model.d


def torque(act: np.ndarray) -> np.ndarray:
    """(nv,) joint torques of the muscles at these activations, at the reference posture (force-length included)."""
    d.qpos[:], d.qvel[:], d.act[:] = f.q, 0, act
    mujoco.mj_forward(m, d)
    return d.qfrc_actuator.copy()


active = np.array([torque(a) for a in np.eye(m.nu)]) - torque(np.zeros(m.nu))     # (nu, nv), passive removed

report = Report("Maximal isometric wrist moments (Nm) vs Delp et al. 1996, mean +- 2 SD")
total = {}
for name, r in REFS["strength_nm"].items():
    along = r["sign"] * active[:, f.dof(r["joint"])]
    total[name] = along[along > 0].sum()          # every muscle that helps fully active, the others relaxed
    report.check(name, total[name], r["mean"] - 2 * r["sd"], r["mean"] + 2 * r["sd"])
    top = np.argsort(-along)[:3]
    print(f"  {name}: {', '.join(f'{f.model.muscle_names[i]} {along[i]:.1f}' for i in top)}")
report.check("flexion / extension", total["flexion"] / total["extension"], 1.2, 2.3)   # Delp: 12.2 / 7.1 = 1.7
report.show()
