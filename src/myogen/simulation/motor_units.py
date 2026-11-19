from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import signal

from ..inputs.subject import Subject
from ..params import MotorUnitParams
from .geometry import ForearmGeometry
from .musculoskeletal import ForearmModel


@dataclass
class Spikes:
    """Every motor-unit discharge of a run."""
    times: np.ndarray       # (S,) s
    units: np.ndarray       # (S,) unit of each spike


class MotorUnitPool:
    """The motor units of the muscles and when they fire, from each muscle's drive (Fuglevand et al. 1993):
    recruitment by size (thresholds spread exponentially, most units recruited at low drive), rate coding,
    irregular intervals and a slow drive shared by a muscle's units."""

    def __init__(self, model: ForearmModel, subject: Subject, muscles: np.ndarray,
                 params: MotorUnitParams | None = None) -> None:
        """muscles: the muscles that have units."""
        self.p = p = params or MotorUnitParams()
        force = model.m.actuator_gainprm[:, 2]
        size = subject.per_muscle("muscle_size", model.nu, 1.0)[muscles]       # force already holds the strength
        counts = np.clip(np.round(p.units_per_newton * force[muscles] * size), p.min_units, p.max_units).astype(int)
        self.muscle = np.repeat(muscles, counts)              # (U,) muscle of each unit
        rank = np.concatenate([np.arange(n) / max(n - 1, 1) for n in counts])   # 0 smallest .. 1 largest
        self.threshold = p.recruitment_max * np.exp(np.log(p.recruitment_range) * (rank - 1))
        self.amplitude = np.exp(np.log(p.amplitude_range) * rank)          # later units are larger

    @property
    def n_units(self) -> int:
        return len(self.muscle)

    def rate(self, drive: np.ndarray) -> np.ndarray:
        """(..., U) firing rate (Hz) of every unit at this drive (zero below its threshold)."""
        p = self.p
        return np.where(drive >= self.threshold,
                        np.minimum(p.min_rate_hz + p.rate_gain_hz * (drive - self.threshold), p.peak_rate_hz), 0.0)

    def full_drive_power(self, n_muscles: int) -> np.ndarray:
        """(n_muscles,) sum over each muscle's units of rate x amplitude^2 at full drive."""
        rate = np.minimum(self.p.min_rate_hz + self.p.rate_gain_hz * (1 - self.threshold), self.p.peak_rate_hz)
        return np.bincount(self.muscle, rate * self.amplitude ** 2, minlength=n_muscles)

    def spikes(self, drive: np.ndarray, fs: float, rng: np.random.Generator) -> Spikes:
        """Every spike from drive (T, nu) in [0, 1] at fs."""
        p, n = self.p, self.n_units
        T = len(drive)
        common = _lowpass_noise(rng, T, drive.shape[1], p.common_drive_hz, fs)       # (T, nu), unit std
        unit_drive = drive[:, self.muscle] * (1 + p.common_drive * common[:, self.muscle])
        phase = rng.uniform(0, 1, n)            # recruited units start at a random point of their interval
        target = self._intervals(rng, n)
        times, units = [], []
        for t in range(T):
            d = unit_drive[t]
            active = d >= self.threshold
            step = self.rate(d) / fs
            phase = np.where(active, phase + step, rng.uniform(0, 1, n))   # derecruited: restarts anywhere
            fired = np.flatnonzero(active & (phase >= target))
            if len(fired):
                late = (phase[fired] - target[fired]) / step[fired]                     # fraction of the tick ago
                times.append((t + 1 - late) / fs)
                units.append(fired)
                phase[fired] -= target[fired]
                target[fired] = self._intervals(rng, len(fired))
        if not times:
            return Spikes(np.zeros(0), np.zeros(0, int))
        return Spikes(np.concatenate(times), np.concatenate(units))

    def _intervals(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Each interval in units of the mean interval: irregular firing (Gaussian, CV isi_cv)."""
        return np.clip(rng.normal(1.0, self.p.isi_cv, n), 0.3, 1.7)


class UnitPlacement:
    """Where each unit sits: a random point of its muscle's atlas volume (its territory), moved with the posture."""

    def __init__(self, geometry: ForearmGeometry, muscle: np.ndarray, seed: int, elbow_margin_m: float) -> None:
        """muscle: (U,) muscle of each unit; seed: the subject's."""
        self.geometry, self.elbow_margin = geometry, elbow_margin_m
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
        points = np.empty((len(muscle), 3))
        for m in np.unique(muscle):
            sel = muscle == m
            pts = geometry.anatomy.points[m]
            points[sel] = pts[rng.integers(0, len(pts), sel.sum())]
        self.atlas = geometry.anatomy.prepare(muscle, points)
        self.data = mujoco.MjData(geometry.model.m)        # scratch state for the posture

    def positions(self, qpos: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(positions (S, 3) in the forearm frame, in_limb (S,)) of these units at joint state qpos."""
        g = self.geometry
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(g.model.m, self.data)
        pos = g.inside(g.atlas_positions(self.atlas, g.bellies(self.data), units))
        in_limb = (pos[:, 2] < g.limb_length) & (pos[:, 2] > -self.elbow_margin)
        return pos, in_limb


def _lowpass_noise(rng: np.random.Generator, T: int, n: int, cutoff_hz: float, fs: float) -> np.ndarray:
    """(T, n) noise below cutoff_hz, unit standard deviation."""
    x = rng.standard_normal((T, n))
    if T > 15 and cutoff_hz < fs / 2:
        x = signal.sosfiltfilt(signal.butter(2, cutoff_hz, fs=fs, output="sos"), x, axis=0)
    return x / (x.std(0) + 1e-12)
