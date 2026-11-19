from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import signal

from ..inputs.electrodes import Electrodes
from ..inputs.subject import Subject
from ..params import EMGParams
from .controller import MuscleActivity
from .geometry import ForearmGeometry
from .montage import Montage
from .motor_units import MotorUnitPool, Spikes, UnitPlacement
from .muap import muap_model
from .musculoskeletal import ForearmModel


class EMGModel:
    """Clean sEMG at the electrodes from the muscles' drive, block by block:
    drive -> MotorUnitPool (spikes) -> UnitPlacement (where each unit is at the posture) -> MUAP model (what each pole
    sees of it) -> superposition (pole signals) -> Montage (channels in mV)."""

    def __init__(self, model: ForearmModel, subject: Subject, electrodes: Electrodes,
                 params: EMGParams | None = None) -> None:
        self.params = p = params or EMGParams()
        self.fs = p.fs
        self.geometry = ForearmGeometry(model, subject)
        self.poles = self.geometry.poles(electrodes)                          # (C, 2 or 1 poles, xyz)
        muscles = np.flatnonzero(model.extrinsic & self.geometry.anatomy.has)  # the forearm muscles
        self.pool = MotorUnitPool(model, subject, muscles, p.motor_units)
        self.placement = UnitPlacement(self.geometry, self.pool.muscle, subject.seed, p.elbow_margin_m)
        self.muap = muap_model(self.geometry, p, self.poles)
        self.unit_gain = self._unit_gains(subject.muscle_amplitudes(model.extrinsic))
        self.montage = Montage(electrodes, p, subject.gain_mv, self.muap.propagates)

    def generate(self, activity: MuscleActivity, rng: np.random.Generator,
                 on_tick: Callable[[int, int], None] | None = None) -> np.ndarray:
        """(T * upsample, C) EMG in mV at self.fs, from the drive and joint states at the control rate.
        on_tick(k, T) is called every 200 ticks."""
        spikes = self.pool.spikes(np.clip(activity.drive, 0, 1), activity.fs, rng)
        return self.montage(self.superpose(spikes, activity.qpos, activity.fs, on_tick))

    def superpose(self, spikes: Spikes, qpos: np.ndarray, fs_ctrl: float,
                  on_tick: Callable[[int, int], None] | None = None) -> np.ndarray:
        """(T * upsample, poles) signal at every pole: every spike adds its unit's waveform (MUAP model weights at
        that tick's posture) into one impulse train per kernel; each train is convolved once with its kernel."""
        poles = self.poles.reshape(-1, 3)
        kernels = self.muap.kernels
        n_ticks, upsample = len(qpos), int(round(self.fs / fs_ctrl))
        n_samples = n_ticks * upsample
        sample = np.minimum(np.round(spikes.times * self.fs).astype(int), n_samples - 1)
        tick = np.minimum((spikes.times * fs_ctrl).astype(int), n_ticks - 1)
        impulses = np.zeros((n_samples, len(poles), len(kernels)), np.float32)
        order = np.argsort(tick, kind="stable")
        bounds = np.searchsorted(tick[order], np.arange(n_ticks + 1))
        for t in range(n_ticks):
            idx = order[bounds[t]:bounds[t + 1]]
            if len(idx):
                units = spikes.units[idx]
                pos, in_limb = self.placement.positions(qpos[t], units)
                w = self.muap.weights(pos, poles) * (self.unit_gain[units] * in_limb)[:, None, None]
                at, which = np.unique(sample[idx], return_inverse=True)              # spikes sharing a sample add
                onehot = np.zeros((len(at), len(idx)), np.float32)
                onehot[which, np.arange(len(idx))] = 1
                impulses[at] += (onehot @ w.reshape(len(idx), -1).astype(np.float32)).reshape((len(at),) + w.shape[1:])
            if on_tick is not None and (t + 1) % 200 == 0:
                on_tick(t + 1, n_ticks)
        out = np.zeros((n_samples, len(poles)), np.float32)
        for b, kernel in enumerate(kernels):
            out += signal.oaconvolve(impulses[:, :, b], kernel[:, None], mode="full", axes=0)[
                len(kernel) // 2:len(kernel) // 2 + n_samples].astype(np.float32)
        if on_tick is not None:
            on_tick(n_ticks, n_ticks)
        return out

    def _unit_gains(self, muscle_amp: np.ndarray) -> np.ndarray:
        """(U,) gain of each unit so that a fully driven superficial muscle under a pole gives an RMS of muscle_amp."""
        rms = np.sqrt(self.pool.full_drive_power(len(muscle_amp))) * self.muap.reference_rms(
            self.params.superficial_depth_m)
        per_muscle = np.divide(muscle_amp, rms, out=np.zeros(len(muscle_amp)), where=rms > 0)
        return self.pool.amplitude * per_muscle[self.pool.muscle]
