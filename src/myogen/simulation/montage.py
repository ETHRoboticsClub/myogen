from __future__ import annotations

import numpy as np

from ..inputs.electrodes import Electrodes
from ..params import EMGParams


class Montage:
    """Pole signals -> channels in mV: bipolar = pole 1 - pole 2 (delayed by the conduction time between them when
    the MUAP model does not propagate); monopolar = the electrode's pole against a distant reference, or against the
    average of the electrodes."""

    def __init__(self, electrodes: Electrodes, params: EMGParams, gain_mv: float, propagates: bool) -> None:
        """gain_mv: Subject.gain_mv; propagates: the MUAP model's waveforms already travel along the fibres."""
        self.monopolar, self.reference, self.fs = electrodes.monopolar, electrodes.reference, params.fs
        along_fibre = np.array([e.ied_mm / 1000 * np.cos(np.radians(e.orientation_deg))
                                for e in electrodes.electrodes])
        self.pole_delay = np.zeros_like(along_fibre) if propagates else along_fibre / params.conduction_velocity
        self.scale = (params.monopolar_scale if self.monopolar else params.bipolar_scale) * gain_mv

    def __call__(self, pole_emg: np.ndarray) -> np.ndarray:
        """(N, C) channels from (N, poles) pole signals (pole order: Electrodes, two per bipolar electrode)."""
        if not self.monopolar:
            emg = self._bipolar(pole_emg)
        elif self.reference == "average":
            emg = pole_emg - pole_emg.mean(axis=1, keepdims=True)
        else:
            emg = pole_emg
        return (emg * self.scale).astype(np.float32)

    def _bipolar(self, pole_emg: np.ndarray) -> np.ndarray:
        pole1, pole2 = pole_emg[:, 0::2], pole_emg[:, 1::2]
        t = np.arange(len(pole_emg))
        return np.stack([pole1[:, c] - np.interp(t - self.pole_delay[c] * self.fs, t, pole2[:, c], left=0.0)
                         for c in range(pole1.shape[1])], axis=1)
