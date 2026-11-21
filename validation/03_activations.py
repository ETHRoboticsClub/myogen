import numpy as np
from common import REFS, Forearm, Report

from myogen.simulation.kinematics import JointTrajectory
from myogen.simulation.muscle_solver import MuscleSolver

r = REFS["activation"]
f = Forearm()
m = f.model.m


def hold(q: np.ndarray, load: dict[str, float]) -> tuple[np.ndarray, float, float]:
    """Hold posture q against external joint torques {joint: Nm}: (settled activations, max reserve Nm,
    max tracking error deg)."""
    T = r["ticks"]
    q_ref, torque = np.tile(q, (T, 1)), np.zeros((T, m.nv))
    for joint, nm in load.items():
        torque[:, f.dof(joint)] = nm
    held = JointTrajectory(q_ref, q, q_ref[:, f.model.hand_qpos], warmup=0)     # relaxed at q: springs hold it there
    a = MuscleSolver(f.model, f.subject).solve(held, torque)
    s = slice(-r["settle"], None)
    return (a.activation[s].mean(0), float(np.abs(a.reserve[s]).max()),
            float(np.degrees(np.abs(a.q[s] - q_ref[s]).max())))


report = Report("Muscle solver: isometric loads and relaxed postures")
for name, joints in r["rest_postures"].items():
    q = f.q.copy()
    for joint, deg in joints.items():
        q[f.qpos(joint)] = np.radians(deg)
    act, _, _ = hold(q, {})
    top = int(np.argmax(act))
    report.check(f"relaxed '{name}': max activation ({f.model.muscle_names[top]})", act[top], 0,
                 r["max_rest_activation"])

wrist = {"flexion": ("flexion_r", 1), "extension": ("flexion_r", -1), "radial": ("deviation_r", 1),
         "ulnar": ("deviation_r", -1)}
for task, (joint, sign) in wrist.items():
    act, reserve, err = hold(f.q, {joint: -sign * r["load_nm"]})       # the muscles must answer sign * load
    groups = REFS["actions"][joint]
    agonists, antagonists = (groups["positive"], groups["negative"])[::sign]
    ag, an = (sum(act[f.muscle(n)] for n in g) for g in (agonists, antagonists))
    report.check(f"{task}: agonist share of activation", ag / (ag + an + 1e-12), r["min_agonist_share"], 1)
    report.check(f"{task}: reserve torque (Nm)", reserve, 0, r["max_reserve_nm"])
    report.check(f"{task}: tracking error (deg)", err, 0, r["max_tracking_deg"])
    print(f"  {task}: " + ", ".join(f"{f.model.muscle_names[i]} {act[i]:.2f}" for i in np.argsort(-act)[:5]))

# Preferred direction: the load direction (0 = flexion, 90 = radial) each wrist muscle answers most.
angles = np.radians(np.arange(0, 360, 45))
acts = np.array([hold(f.q, {"flexion_r": -r["load_nm"] * np.cos(a), "deviation_r": -r["load_nm"] * np.sin(a)})[0]
                 for a in angles])
for name, (lo, hi) in r["preferred_direction_deg"].items():
    pd = np.degrees(np.angle(np.sum(acts[:, f.muscle(name)] * np.exp(1j * angles))))
    report.check(f"{name} preferred direction (deg)", lo + (pd - lo) % 360, lo, hi)
report.show()
