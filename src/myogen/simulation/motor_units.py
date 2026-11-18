from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from scipy import signal

from ..params import MotorUnitParams
from .volume_conductor import MUAPTable

WAVEFORMS = ("biphasic", "volume_conductor")
REFERENCE_DISTANCE_M = 0.02     # pole to end-plate distance at which unit amplitudes are normalised

if TYPE_CHECKING:
    from .emg import EMGModel


class MotorUnits:
    """The motor-unit pools of the muscles, for the EMG at the electrode poles: each unit fires a spike train set by
    the muscle's drive (recruitment by size, rate coding, irregular intervals, a drive shared by the muscle's units)
    and adds its action potential at every pole: an amplitude falling with distance (EMGModel's gain law) and a
    biphasic waveform that widens with distance (the tissue low-pass). Far units are slow and wide, so a bipolar pair
    cancels them, and few units firing at low drive leave a sparse, non-Gaussian signal.

    Fast by construction: spikes are sparse; each adds its gains into a few waveform-width bands at every pole, and
    each band is convolved once with its waveform."""

    def __init__(self, emg: EMGModel, params: MotorUnitParams | None = None) -> None:
        self.emg, self.p = emg, params or MotorUnitParams()
        p, model = self.p, emg.model
        rng = np.random.default_rng(np.random.SeedSequence([emg.sub.seed, 7]))
        force = model.m.actuator_gainprm[:, 2]
        muscles = np.flatnonzero(model.extrinsic)            # the hand's own muscles are not under a forearm band
        size = emg.sub.per_muscle("muscle_size", model.nu, 1.0)[muscles]       # force already holds the strength
        counts = np.clip(np.round(p.units_per_newton * force[muscles] * size), p.min_units, p.max_units).astype(int)
        self.muscle = np.repeat(muscles, counts)              # (U,) muscle of each unit
        rank = np.concatenate([np.arange(n) / max(n - 1, 1) for n in counts])   # 0 smallest .. 1 largest
        # Recruitment (Fuglevand 1993): thresholds spread exponentially, most units recruited at low drive.
        self.threshold = p.recruitment_max * np.exp(np.log(p.recruitment_range) * (rank - 1))
        self.amplitude = np.exp(np.log(p.amplitude_range) * rank)
        # Territory: a point along the belly (one of the EMG sources) and an offset across the muscle; in an atlas
        # muscle, a point of its volume instead.
        self.segment = rng.integers(0, emg.n_sources, len(self.muscle))
        radius = emg.sub.muscle_spread_mm / 1000 * np.sqrt(rng.uniform(0, 1, len(self.muscle)))
        angle = rng.uniform(0, 2 * np.pi, len(self.muscle))
        self.offset = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
        self.in_atlas = np.zeros(len(self.muscle), bool)
        self.atlas_index = np.full(len(self.muscle), -1)          # unit -> its row in self.atlas
        if emg.anatomy is not None:
            self.in_atlas = emg.anatomy.has[self.muscle]
            units = np.flatnonzero(self.in_atlas)
            points = np.empty((len(units), 3))
            for m in np.unique(self.muscle[units]):
                sel = self.muscle[units] == m
                pts = emg.anatomy.points[m]
                points[sel] = pts[rng.integers(0, len(pts), sel.sum())]
            self.atlas = emg.anatomy.prepare(self.muscle[units], points)
            self.atlas_index[units] = np.arange(len(units))
        if p.waveform not in WAVEFORMS:
            raise ValueError(f"waveform must be one of {' | '.join(WAVEFORMS)}, got {p.waveform!r}")
        self.table = None
        if p.waveform == "volume_conductor":
            # Waveforms from the layered cylinder of this subject's forearm at the electrodes' mean height.
            z = float(np.mean(emg.poles[..., 2]))
            self.table = MUAPTable(p.volume_conductor, float(emg.skin_radius(z)), emg.fat, emg.fs)
            self.kernels = list(self.table.basis)
        else:
            # One waveform per width band; a unit's waveform at a pole is interpolated between two bands (log width).
            self.bands = np.asarray(p.bands_ms) / 1000
            self.kernels = [_biphasic(width, emg.fs) for width in self.bands]
        self.scale = self._scale(emg.sub.muscle_amplitudes(model.extrinsic))

    @property
    def n_units(self) -> int:
        return len(self.muscle)

    def width_s(self, distance_m: np.ndarray) -> np.ndarray:
        """Waveform time scale at a pole this far from the unit: the propagating potential, low-passed by tissue."""
        return self.p.width_ms / 1000 + distance_m * 1000 * self.p.width_per_mm / 1000

    def _scale(self, muscle_amp: np.ndarray) -> np.ndarray:
        """(U,) amplitude of each unit so that a fully driven superficial muscle under a pole gives an RMS of
        muscle_amp, as the noise sources do."""
        p, emg = self.p, self.emg
        d0 = emg.fat + emg.params.superficial_depth_m
        if self.table is not None:     # the waveform there, the pole REFERENCE_DISTANCE_M from the end-plate
            w = self.table.weights_at(np.array(emg.params.superficial_depth_m), np.array(0.0),
                                      np.array(REFERENCE_DISTANCE_M))
            energy, attenuation = float(np.sum(w ** 2)) / emg.fs, 1.0          # orthonormal waveforms
        else:
            energy = float(np.sum(_biphasic(self.width_s(np.array(d0)), emg.fs) ** 2)) / emg.fs   # unit peak
            attenuation = float(np.exp(-d0 / emg.decay))
        rate = np.minimum(p.min_rate_hz + p.rate_gain_hz * (1 - self.threshold), p.peak_rate_hz)
        power = np.bincount(self.muscle, rate * self.amplitude ** 2, minlength=len(muscle_amp)) * energy
        per_muscle = np.divide(muscle_amp, np.sqrt(power) * attenuation, out=np.zeros(len(muscle_amp)),
                               where=power > 0)
        return self.amplitude * per_muscle[self.muscle]

    # -- spikes ------------------------------------------------------------------------------------------------

    def spikes(self, drive: np.ndarray, fs_ctrl: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(times in s, units) of every spike. drive: (T, nu) in [0, 1] at the control rate."""
        p, n = self.p, self.n_units
        T = len(drive)
        common = _lowpass_noise(rng, T, drive.shape[1], p.common_drive_hz, fs_ctrl)       # (T, nu), unit std
        unit_drive = drive[:, self.muscle] * (1 + p.common_drive * common[:, self.muscle])
        phase = rng.uniform(0, 1, n)            # recruited units start at a random point of their interval
        target = self._intervals(rng, n)
        times, units = [], []
        for t in range(T):
            d = unit_drive[t]
            active = d >= self.threshold
            rate = np.where(active, np.minimum(p.min_rate_hz + p.rate_gain_hz * (d - self.threshold), p.peak_rate_hz),
                            0.0)
            step = rate / fs_ctrl
            phase = np.where(active, phase + step, rng.uniform(0, 1, n))   # derecruited: restarts anywhere
            fired = np.flatnonzero(active & (phase >= target))
            if len(fired):
                late = (phase[fired] - target[fired]) / step[fired]                     # fraction of the tick ago
                times.append((t + 1 - late) / fs_ctrl)
                units.append(fired)
                phase[fired] -= target[fired]
                target[fired] = self._intervals(rng, len(fired))
        if not times:
            return np.zeros(0), np.zeros(0, int)
        return np.concatenate(times), np.concatenate(units)

    def _intervals(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Each interval in units of the mean interval: irregular firing (Gaussian, CV isi_cv)."""
        return np.clip(rng.normal(1.0, self.p.isi_cv, n), 0.3, 1.7)

    # -- signals -----------------------------------------------------------------------------------------------

    def pole_signals(self, drive: np.ndarray, qpos: np.ndarray, fs_ctrl: float, rng: np.random.Generator,
                     on_tick: Callable[[int, int], None] | None = None) -> np.ndarray:
        """(T * upsample, poles) signal at every pole."""
        emg = self.emg
        poles = emg.poles.reshape(-1, 3)
        upsample = int(round(emg.fs / fs_ctrl))
        n_samples, n_bands = len(drive) * upsample, len(self.kernels)
        times, units = self.spikes(np.clip(drive, 0, 1), fs_ctrl, rng)
        sample = np.minimum(np.round(times * emg.fs).astype(int), n_samples - 1)
        tick = np.minimum((times * fs_ctrl).astype(int), len(drive) - 1)
        impulses = np.zeros((n_samples, len(poles), n_bands), np.float32)
        order = np.argsort(tick, kind="stable")
        bounds = np.searchsorted(tick[order], np.arange(len(drive) + 1))
        for t in range(len(drive)):
            idx = order[bounds[t]:bounds[t + 1]]
            if not len(idx):
                pass
            elif self.table is not None:
                weights = self._at_poles(units[idx], qpos[t], poles)                       # (S, P, waveforms)
                at, which = np.unique(sample[idx], return_inverse=True)              # spikes sharing a sample add
                onehot = np.zeros((len(at), len(idx)), np.float32)
                onehot[which, np.arange(len(idx))] = 1
                impulses[at] += (onehot @ weights.reshape(len(idx), -1)).reshape((len(at),) + weights.shape[1:])
            else:
                gain, width = self._at_poles(units[idx], qpos[t], poles)                   # (S, P) each
                # Split each spike's gain between the two width bands around its width.
                pos = np.clip(np.interp(np.log(width), np.log(self.bands), np.arange(n_bands)), 0, n_bands - 1 - 1e-9)
                low, frac = pos.astype(int), (pos % 1).astype(np.float32)
                rows = np.broadcast_to(sample[idx][:, None], gain.shape)
                cols = np.broadcast_to(np.arange(len(poles)), gain.shape)
                np.add.at(impulses, (rows, cols, low), gain * (1 - frac))
                np.add.at(impulses, (rows, cols, low + 1), gain * frac)
            if on_tick is not None and (t + 1) % 200 == 0:
                on_tick(t + 1, len(drive))
        out = np.zeros((n_samples, len(poles)), np.float32)
        for b, kernel in enumerate(self.kernels):
            out += signal.oaconvolve(impulses[:, :, b], kernel[:, None], mode="full", axes=0)[
                len(kernel) // 2:len(kernel) // 2 + n_samples].astype(np.float32)
        if on_tick is not None:
            on_tick(len(drive), len(drive))
        return out

    def _at_poles(self, units: np.ndarray, qpos: np.ndarray, poles: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """At joint state qpos, these units at every pole: (S, P) amplitude and waveform width (biphasic), or
        (S, P, waveforms) waveform weights (volume_conductor)."""
        emg = self.emg
        emg.data.qpos[:] = qpos
        mujoco.mj_kinematics(emg.model.m, emg.data)
        atlas = self.in_atlas[units]
        pos = np.empty((len(units), 3))
        if not atlas.all():
            src = emg.sources(emg.data)                                          # (nu, K, 3)
            pos[~atlas] = src[self.muscle[units[~atlas]], self.segment[units[~atlas]]]
            pos[~atlas, :2] += self.offset[units[~atlas]]
        if atlas.any():
            pos[atlas] = emg.atlas_positions(self.atlas, emg.bellies(emg.data), self.atlas_index[units[atlas]])
        under_fat = emg.under_fat(pos[:, 2])
        r = np.linalg.norm(pos[:, :2], axis=1)
        pos[:, :2] *= (np.minimum(r, under_fat) / np.maximum(r, 1e-9))[:, None]
        in_limb = (pos[:, 2] < emg.limb_length) & (pos[:, 2] > -emg.params.elbow_margin_m)
        if self.table is not None:      # depth below the muscle surface, angle, distance along from the end-plate
            depth = under_fat - np.linalg.norm(pos[:, :2], axis=1)
            dtheta = np.arctan2(poles[None, :, 1], poles[None, :, 0]) - np.arctan2(pos[:, 1], pos[:, 0])[:, None]
            w = self.table.weights_at(depth[:, None], dtheta, poles[None, :, 2] - pos[:, 2, None])
            return (w * (self.scale[units] * in_limb)[:, None, None]).astype(np.float32)
        dist = np.linalg.norm(pos[:, None, :] - poles[None], axis=-1)
        gain = self.scale[units, None] * np.exp(-dist / emg.decay) * in_limb[:, None]
        return gain.astype(np.float32), self.width_s(dist)


def _biphasic(width_s: np.ndarray | float, fs: float) -> np.ndarray:
    """A propagating action potential seen at one pole: the first derivative of a Gaussian of this time scale,
    peak 1."""
    width = float(width_s)
    t = np.arange(-int(np.ceil(4 * width * fs)), int(np.ceil(4 * width * fs)) + 1) / fs
    w = -t / width * np.exp(-0.5 * (t / width) ** 2)
    return (w / np.abs(w).max()).astype(np.float32)


def _lowpass_noise(rng: np.random.Generator, T: int, n: int, cutoff_hz: float, fs: float) -> np.ndarray:
    """(T, n) noise below cutoff_hz, unit standard deviation."""
    x = rng.standard_normal((T, n))
    if T > 15 and cutoff_hz < fs / 2:
        x = signal.sosfiltfilt(signal.butter(2, cutoff_hz, fs=fs, output="sos"), x, axis=0)
    return x / (x.std(0) + 1e-12)
