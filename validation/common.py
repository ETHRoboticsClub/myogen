import sys
from pathlib import Path

import numpy as np
import yaml

import myogen as mg
from myogen.inputs.motion import REST_POSTURE
from myogen.simulation.muscle_solver import MuscleActivity
from myogen.simulation.musculoskeletal import HAND_JOINTS, ForearmModel

REFS = yaml.safe_load((Path(__file__).parent / "references.yaml").read_text())


class Report:
    """Measured against reference, one row per check. show() exits 1 if any check failed."""

    def __init__(self, title: str) -> None:
        self.title, self.rows = title, []

    def check(self, name: str, value: float, lo: float, hi: float) -> bool:
        ok = bool(lo <= value <= hi)
        self.rows.append((name, value, lo, hi, ok))
        return ok

    def show(self) -> None:
        print(f"\n== {self.title}")
        for name, value, lo, hi, ok in self.rows:
            print(f"  {'ok  ' if ok else 'FAIL'} {name:42s} {value:9.3f}   [{lo:g}, {hi:g}]")
        failed = sum(not r[-1] for r in self.rows)
        print(f"  {len(self.rows) - failed}/{len(self.rows)} passed")
        sys.exit(1 if failed else 0)


class Forearm:
    """The model at the reference posture (relaxed hand, elbow from REFS), with names as in references.yaml."""

    def __init__(self, subject: mg.Subject | None = None) -> None:
        self.subject = subject or mg.Subject()
        self.model = ForearmModel()
        self.model.scale_strength(self.subject.muscle_forces(self.model.nu))
        hand = np.array([[REST_POSTURE[j] for j in HAND_JOINTS]])
        self.q = self.model.full_q(hand, np.radians(REFS["posture"]["elbow_deg"]))[0]

    def muscle(self, name: str) -> int:
        return self.model.muscle_names.index(f"{name}_r")

    def dof(self, joint: str) -> int:
        m = self.model.m
        return int(m.jnt_dofadr[m.joint(joint).id])

    def qpos(self, joint: str) -> int:
        m = self.model.m
        return int(m.jnt_qposadr[m.joint(joint).id])

    def activity(self, drive: dict[str, float], ticks: int, fs: float = 100.0) -> MuscleActivity:
        """Constant drive {muscle: level} at the reference posture, as the EMG step takes it."""
        nu, nv = self.model.nu, self.model.m.nv
        d = np.zeros((ticks, nu))
        for name, level in drive.items():
            d[:, self.muscle(name)] = level
        q = np.tile(self.q, (ticks, 1))
        zv = np.zeros((ticks, nv))
        return MuscleActivity(fs, d, d, np.zeros_like(d), d, np.ones((ticks, nu)), q, q, zv, zv, np.zeros(ticks))
