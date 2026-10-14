from __future__ import annotations

from typing import TYPE_CHECKING, overload

import mujoco
import numpy as np
from scipy import signal

from ..params import EMGParams
from .musculoskeletal import R_MODEL_REF

if TYPE_CHECKING:
    from ..inputs.electrodes import Electrodes
    from ..inputs.subject import Subject
    from .muscle_activity import MuscleActivity
    from .musculoskeletal import ForearmModel


class EMGModel:
    """Clean bipolar sEMG at the electrodes from muscle activity (forearm frame: z along the forearm from the
    elbow, metres). Each muscle belly is n_sources point sources emitting band-limited noise; each electrode pole
    sees them attenuated with distance; bipolar = pole 1 - pole 2 delayed by the conduction time between them."""

    def __init__(self, model: ForearmModel, subject: Subject, electrodes: Electrodes,
                 params: EMGParams | None = None) -> None:
        self.model, self.sub, self.params = model, subject, params or EMGParams()
        p = self.params
        self.n_sources, self.decay, self.fs = p.n_sources, p.decay_mm / 1000, p.fs
        self.data = mujoco.MjData(model.m)              # scratch state for the per-tick geometry
        self.fat = subject.fat_mm / 1000
        self.limb_length = subject.residual_length * model.forearm_length   # shorter for amputees

        # Fixed per-subject (x, y) offset of every source: muscles sit slightly differently in each person.
        self.source_offset = np.random.default_rng(subject.seed).normal(
            0.0, subject.muscle_spread_mm / 1000, size=(model.nu, self.n_sources, 2))
        self.poles = self._pole_positions(electrodes)                       # (C, 2 poles, xyz)

        # Delay (s) between the two poles: the along-fibre part of the pole spacing over the conduction velocity.
        along_fibre = np.array([e.ied_mm / 1000 * np.cos(np.radians(e.orientation_deg))
                                for e in electrodes.electrodes])
        self.pole_delay = along_fibre / p.conduction_velocity

        # Per-source amplitude, scaled so a fully driven superficial muscle under a pole gives ~1 (before gain_mv).
        norm = np.exp(-(self.fat + p.superficial_depth_m) / self.decay) * self.n_sources / 2
        self.source_amp = np.repeat(subject.muscle_amplitudes(model.extrinsic), self.n_sources) / norm

    def generate(self, activity: MuscleActivity, rng: np.random.Generator) -> np.ndarray:
        """(T * upsample, C) bipolar EMG in mV at self.fs, from the drive and joint angles at the control rate."""
        envelope = np.clip(activity.drive, 0, 1) ** self.sub.amp_exponent     # how strongly each muscle fires
        pole_emg = self._pole_signals(envelope, activity.qpos, activity.fs, rng)   # monopolar, every pole
        emg = self._bipolar(pole_emg)
        scale = self.sub.gain_mv * self.params.noise_to_mv * self.params.bipolar_scale
        return (emg * scale).astype(np.float32)

    # -- geometry ----------------------------------------------------------------------------------------------

    @overload
    def skin_radius(self, z: float) -> float: ...
    @overload
    def skin_radius(self, z: np.ndarray) -> np.ndarray: ...

    def skin_radius(self, z: float | np.ndarray) -> float | np.ndarray:
        """Forearm radius (m) at distance z from the elbow; it tapers towards the wrist."""
        frac_along = np.clip(z / self.model.forearm_length, 0, 1)
        return self.sub.forearm_radius_mm / 1000 * (1 - self.sub.taper * frac_along)

    def _pole_positions(self, electrodes: Electrodes) -> np.ndarray:
        """(C, 2, 3) both poles of every electrode as points on the skin."""
        poles = []
        for e in electrodes.electrodes:
            pole_ends = e.poles(self.skin_radius)   # [(angle, z), (angle, z)]
            if not all(0.0 <= z <= self.limb_length for _, z in pole_ends):
                raise ValueError(f"electrode {e.name} ({e.position_cm} cm) is off the forearm: positions must be "
                                 f"within 0-{self.limb_length * 100:.1f} cm of the elbow")
            poles += [[self.skin_radius(z) * np.cos(angle), self.skin_radius(z) * np.sin(angle), z]
                      for angle, z in pole_ends]
        return np.array(poles).reshape(len(electrodes), 2, 3)

    def sources(self, d: mujoco.MjData) -> np.ndarray:
        """(nu, K, 3) source positions in the forearm frame, mapped into this subject's cross-section."""
        model_pts = self.model.belly_points(self.n_sources, d)               # in the generic model's forearm
        z = model_pts[..., 2]
        under_fat = self.skin_radius(z) - self.fat                           # radius left for muscle
        # Rescale the model's cross-section to this subject's, then add the per-subject offsets.
        muscle_radius = under_fat * self.sub.radial_fill
        xy = model_pts[..., :2] / R_MODEL_REF * muscle_radius[..., None] + self.source_offset
        # Pull back any source the offset pushed into the fat layer.
        r = np.linalg.norm(xy, axis=-1)
        xy *= (np.minimum(r, under_fat) / np.maximum(r, 1e-9))[..., None]
        return np.concatenate([xy, z[..., None]], axis=-1)

    def gains(self, qpos: np.ndarray) -> np.ndarray:
        """(nu * K, 2 C) gain from every source to every pole at joint state qpos: exp(-distance / decay)."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model.m, self.data)
        src = self.sources(self.data).reshape(-1, 3)                         # (nu * K, 3)
        dist = np.linalg.norm(src[:, None, :] - self.poles.reshape(-1, 3)[None], axis=-1)   # (nu * K, 2 C)
        # Sources outside the (residual) limb are silent.
        in_limb = (src[:, 2] < self.limb_length) & (src[:, 2] > -self.params.elbow_margin_m)
        return (np.exp(-dist / self.decay) * (in_limb * self.source_amp)[:, None]).astype(np.float32)

    # -- signals -----------------------------------------------------------------------------------------------

    def _pole_signals(self, envelope: np.ndarray, qpos: np.ndarray, fs_ctrl: float, rng: np.random.Generator,
                      chunk_ticks: int = 200) -> np.ndarray:
        """(T * upsample, 2 C) monopolar signal at every pole: each source fires band-limited noise scaled by its
        muscle's envelope, and each pole sums the sources weighted by their gains at that tick's posture."""
        n_ticks, n_muscles = envelope.shape
        K, n_poles = self.n_sources, 2 * len(self.poles)
        n_src = n_muscles * K
        upsample = int(round(self.fs / fs_ctrl))          # EMG samples per control tick
        # Band-pass shaping white noise into the EMG spectrum; its state carries across chunks.
        f_lo, f_hi = self.sub.spectrum_hz
        bandpass = signal.butter(self.params.filter_order, [f_lo, min(f_hi, 0.45 * self.fs)], btype="bandpass",
                                 fs=self.fs, output="sos")
        filter_state = np.zeros((bandpass.shape[0], 2, n_src))
        tick_index = np.arange(n_ticks)
        pole_emg = np.empty((n_ticks * upsample, n_poles), np.float32)
        # Chunked so the (samples x sources) noise array stays small.
        for t0 in range(0, n_ticks, chunk_ticks):
            t1 = min(n_ticks, t0 + chunk_ticks)
            n = (t1 - t0) * upsample
            t_fine = t0 + np.arange(n) / upsample
            amp = np.stack([np.interp(t_fine, tick_index, envelope[:, i]) for i in range(n_muscles)], axis=1)
            noise, filter_state = signal.sosfilt(bandpass, rng.standard_normal((n, n_src)), axis=0, zi=filter_state)
            src = (noise.reshape(n, n_muscles, K) * amp[:, :, None]).reshape(t1 - t0, upsample, n_src)
            src = src.astype(np.float32)
            gain = np.stack([self.gains(q) for q in qpos[t0:t1]])                          # (ticks, n_src, 2C)
            pole_emg[t0 * upsample:t1 * upsample] = np.einsum("tus,tsp->tup", src, gain).reshape(n, n_poles)
        return pole_emg

    def _bipolar(self, pole_emg: np.ndarray) -> np.ndarray:
        """(N, C) pole 1 minus pole 2 delayed by the conduction time between them."""
        pole1, pole2 = pole_emg[:, 0::2], pole_emg[:, 1::2]
        t = np.arange(len(pole_emg))
        return np.stack([pole1[:, c] - np.interp(t - self.pole_delay[c] * self.fs, t, pole2[:, c], left=0.0)
                         for c in range(len(self.poles))], axis=1)
