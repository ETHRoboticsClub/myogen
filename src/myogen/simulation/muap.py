from __future__ import annotations

import numpy as np

from ..params import EMGParams
from .geometry import ForearmGeometry
from .volume_conductor import MUAPTable

WAVEFORMS = ("biphasic", "volume_conductor")
REFERENCE_DISTANCE_M = 0.02     # pole to end-plate distance at which unit amplitudes are normalised


class BiphasicMUAP:
    """A unit's action potential at a pole: amplitude exp(-distance / decay) and a biphasic waveform (first derivative
    of a Gaussian) that widens with distance (the tissue low-pass). Phenomenological: one decay length in every
    direction. The waveform is one of a few width bands, a unit's width interpolated between two (log width)."""

    propagates = False      # the waveform does not travel along the fibre: the bipolar montage adds the delay

    def __init__(self, geometry: ForearmGeometry, params: EMGParams) -> None:
        self.geometry, self.p, self.fs = geometry, params.motor_units, params.fs
        self.decay = params.decay_mm / 1000
        self.bands = np.asarray(self.p.bands_ms) / 1000
        self.kernels = [_biphasic(width, self.fs) for width in self.bands]

    def width_s(self, distance_m: np.ndarray) -> np.ndarray:
        """Waveform time scale at a pole this far from the unit: the propagating potential, low-passed by tissue."""
        return self.p.width_ms / 1000 + distance_m * 1000 * self.p.width_per_mm / 1000

    def weights(self, pos: np.ndarray, poles: np.ndarray) -> np.ndarray:
        """(S, P, kernels) weight of every kernel in the waveform of each unit (S, 3) at each pole (P, 3)."""
        dist = np.linalg.norm(pos[:, None, :] - poles[None], axis=-1)
        gain = np.exp(-dist / self.decay).astype(np.float32)
        n = len(self.bands)
        at = np.clip(np.interp(np.log(self.width_s(dist)), np.log(self.bands), np.arange(n)), 0, n - 1 - 1e-9)
        low, frac = at.astype(int), (at % 1).astype(np.float32)
        w = np.zeros(dist.shape + (n,), np.float32)
        np.put_along_axis(w, low[..., None], (gain * (1 - frac))[..., None], axis=-1)
        np.put_along_axis(w, low[..., None] + 1, (gain * frac)[..., None], axis=-1)
        return w

    def reference_rms(self, depth_m: float) -> float:
        """RMS per unit rate (1 Hz) of a unit this far below the fat, right under a pole."""
        d0 = self.geometry.fat + depth_m
        energy = float(np.sum(_biphasic(self.width_s(np.array(d0)), self.fs) ** 2)) / self.fs
        return np.sqrt(energy) * float(np.exp(-d0 / self.decay))


class LayeredMUAP:
    """A unit's action potential at a pole from a layered cylinder (bone, anisotropic muscle, fat, skin; Farina et al.
    2004) of this subject's forearm at the electrodes' mean height: the intracellular action potential travels from
    the end-plate to the tendons (docs/volume_conductor.md). Precomputed per geometry (MUAPTable)."""

    propagates = True       # each pole's waveform already travels along the fibre

    def __init__(self, geometry: ForearmGeometry, params: EMGParams, poles: np.ndarray) -> None:
        self.geometry = geometry
        self.fs = params.fs
        z = float(np.mean(poles[..., 2]))
        self.table = MUAPTable(params.motor_units.volume_conductor, float(geometry.skin_radius(z)), geometry.fat,
                               self.fs)
        self.kernels = list(self.table.basis)

    def weights(self, pos: np.ndarray, poles: np.ndarray) -> np.ndarray:
        """(S, P, kernels) weight of every kernel in the waveform of each unit (S, 3) at each pole (P, 3); the unit's
        end-plate is at its position."""
        depth = self.geometry.under_fat(pos[:, 2]) - np.linalg.norm(pos[:, :2], axis=1)
        dtheta = np.arctan2(poles[None, :, 1], poles[None, :, 0]) - np.arctan2(pos[:, 1], pos[:, 0])[:, None]
        return self.table.weights_at(depth[:, None], dtheta, poles[None, :, 2] - pos[:, 2, None])

    def reference_rms(self, depth_m: float) -> float:
        """RMS per unit rate (1 Hz) of a unit this far below the muscle surface, under a pole REFERENCE_DISTANCE_M
        from its end-plate."""
        w = self.table.weights_at(np.array(depth_m), np.array(0.0), np.array(REFERENCE_DISTANCE_M))
        return float(np.sqrt(np.sum(w ** 2) / self.fs))      # orthonormal waveforms


def muap_model(geometry: ForearmGeometry, params: EMGParams, poles: np.ndarray) -> BiphasicMUAP | LayeredMUAP:
    """The MUAP model params.motor_units.waveform names."""
    waveform = params.motor_units.waveform
    if waveform not in WAVEFORMS:
        raise ValueError(f"waveform must be one of {' | '.join(WAVEFORMS)}, got {waveform!r}")
    return BiphasicMUAP(geometry, params) if waveform == "biphasic" else LayeredMUAP(geometry, params, poles)


def _biphasic(width_s: np.ndarray | float, fs: float) -> np.ndarray:
    """A propagating action potential seen at one pole: the first derivative of a Gaussian of this time scale,
    peak 1."""
    width = float(width_s)
    t = np.arange(-int(np.ceil(4 * width * fs)), int(np.ceil(4 * width * fs)) + 1) / fs
    w = -t / width * np.exp(-0.5 * (t / width) ** 2)
    return (w / np.abs(w).max()).astype(np.float32)
