import mujoco
import numpy as np
from common import REFS, Forearm, Report

f = Forearm()
m, d = f.model.m, f.model.d


def arms_cm(q: np.ndarray) -> np.ndarray:
    """(nu, nv) moment arms at q; muscle force pulls (negative in MuJoCo): + = the muscle turns the joint its + way."""
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    R = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(R, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
    return -R * 100


def signed(joint: str) -> list[tuple[str, int]]:
    """The joint's muscles from REFS, each with the sign of its anatomical action."""
    groups = REFS["actions"][joint]
    return [(n, 1) for n in groups["positive"]] + [(n, -1) for n in groups["negative"]]


report = Report("Muscle actions (moment arm, cm, signed by the action anatomy gives it)")
at_rest = arms_cm(f.q)
for joint in REFS["actions"]:
    lo, hi = REFS["moment_arm_cm"].get(joint, REFS["moment_arm_cm"]["default"])
    for name, sign in signed(joint):
        report.check(f"{name} on {joint}", sign * at_rest[f.muscle(name), f.dof(joint)], lo, hi)

# Over the joint's whole range, every muscle keeps pulling its anatomical way (its path wraps correctly).
for joint in REFS["actions"]:
    weakest = np.inf
    for angle in np.linspace(*m.jnt_range[m.joint(joint).id], 15):
        q = f.q.copy()
        q[f.qpos(joint)] = angle
        arms = arms_cm(q)
        weakest = min(weakest, *(sign * arms[f.muscle(n), f.dof(joint)] for n, sign in signed(joint)))
    report.check(f"{joint}: weakest action over its range (cm)", weakest, REFS["min_moment_arm_cm"], np.inf)

for joint, (need_lo, need_hi) in REFS["functional_rom_deg"].items():
    lim = np.degrees(m.jnt_range[m.joint(joint).id])
    report.check(f"{joint} lower limit covers {need_lo} deg", lim[0], -360, need_lo)
    report.check(f"{joint} upper limit covers {need_hi} deg", lim[1], need_hi, 360)
report.show()
