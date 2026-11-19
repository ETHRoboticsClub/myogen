import mujoco
import numpy as np
from common import REFS, Forearm, Report

f = Forearm()
m, d = f.model.m, f.model.d
d.qpos[:] = f.q
mujoco.mj_forward(m, d)
R = np.zeros((m.nu, m.nv))
mujoco.mju_sparse2dense(R, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
arm_cm = -R * 100       # muscle force pulls (negative in MuJoCo): + = the muscle turns the joint its + way

report = Report("Muscle actions (moment arm, cm, signed by the action anatomy gives it)")
for joint, groups in REFS["actions"].items():
    lo, hi = REFS["moment_arm_cm"].get(joint, REFS["moment_arm_cm"]["default"])
    for sign, names in ((1, groups["positive"]), (-1, groups["negative"])):
        for name in names:
            report.check(f"{name} on {joint}", sign * arm_cm[f.muscle(name), f.dof(joint)], lo, hi)

for joint, (need_lo, need_hi) in REFS["functional_rom_deg"].items():
    lim = np.degrees(m.jnt_range[m.joint(joint).id])
    report.check(f"{joint} lower limit covers {need_lo} deg", lim[0], -360, need_lo)
    report.check(f"{joint} upper limit covers {need_hi} deg", lim[1], need_hi, 360)
report.show()
