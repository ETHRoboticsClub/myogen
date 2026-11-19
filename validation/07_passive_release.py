import mujoco
import numpy as np
from common import REFS, Forearm, Report

r = REFS["release"]
f = Forearm()
m, d = f.model.m, f.model.d
k = f.qpos(r["joint"])


def release(displacement: float) -> np.ndarray:
    """(T,) the joint's angle after letting it go this far from rest, every muscle relaxed (passive forces only)."""
    f.model.set_environment(f.subject, f.q, f.q)                 # springs hold the hand at rest
    q = f.q.copy()
    q[k] += displacement
    f.model.set_pose(q)
    d.act[:], d.ctrl[:] = 0, 0
    out = []
    for _ in range(int(r["seconds"] / m.opt.timestep)):
        mujoco.mj_step(m, d)
        out.append(d.qpos[k])
    return np.array(out)


def damping_ratio(x: np.ndarray) -> float:
    """zeta of the mass-spring-damper fitted to the motion: x'' = -w^2 x - 2 zeta w x' + c (Xu et al.'s model M1)."""
    dt = m.opt.timestep
    v, a = np.gradient(x, dt), np.gradient(np.gradient(x, dt), dt)
    (k_i, b_i, _), *_ = np.linalg.lstsq(np.column_stack([-x, -v, np.ones_like(x)]), a, rcond=None)
    return float(b_i / (2 * np.sqrt(max(k_i, 1e-9))))


report = Report(f"Passive release of {r['joint']}: let go relaxed, back to rest (Xu et al. 2011)")
for deg in r["displacements_deg"]:
    x = release(np.radians(deg)) - f.q[k]
    back = np.flatnonzero(np.abs(x) < 0.1 * abs(np.radians(deg)))
    t_back = back[0] * m.opt.timestep if len(back) else np.inf
    report.check(f"{deg:+} deg: time back within 10 % (s)", t_back, 0, r["max_return_s"])
    report.check(f"{deg:+} deg: damping ratio", damping_ratio(x[: int(0.2 / m.opt.timestep)]), *r["damping_ratio"])
    print(f"  {deg:+} deg: angle from rest every 50 ms (deg) "
          f"{np.degrees(x[:: int(0.05 / m.opt.timestep)]).round(1)}")
report.show()
