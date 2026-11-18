from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, overload

import mujoco
import numpy as np
from scipy import signal

from ..params import EMGParams
from .anatomy import TRACK_POINTS, Anatomy, AtlasPoints
from .motor_units import MotorUnits
from .musculoskeletal import R_MODEL_REF

ENGINES = ("noise", "motor_units")
ANATOMIES = ("atlas", "lines")

if TYPE_CHECKING:
    from ..inputs.electrodes import Electrodes
    from ..inputs.subject import Subject
    from .muscle_activity import MuscleActivity
    from .musculoskeletal import ForearmModel


class EMGModel:
    """Clean sEMG at the electrodes from muscle activity (forearm frame: z along the forearm from the elbow,
    metres). The forearm muscles sit where the BodyParts3D atlas has them (anatomy "atlas", simulation/anatomy.py)
    or on the model's lines of action ("lines"). Each muscle is n_sources point sources emitting band-limited noise
    (engine "noise") or a motor-unit pool (MotorUnits); each electrode pole sees them
    attenuated with distance; bipolar = pole 1 - pole 2 delayed by the conduction time between them; monopolar = the
    electrode's single pole against a distant reference, or against the average of the electrodes."""

    def __init__(self, model: ForearmModel, subject: Subject, electrodes: Electrodes,
                 params: EMGParams | None = None) -> None:
        self.model, self.sub, self.params = model, subject, params or EMGParams()
        p = self.params
        self.n_sources, self.decay, self.fs = p.n_sources, p.decay_mm / 1000, p.fs
        self.data = mujoco.MjData(model.m)              # scratch state for the per-tick geometry
        self.fat = subject.fat_mm / 1000
        self.limb_length = subject.residual_length * model.forearm_length   # shorter for amputees
        self.monopolar, self.reference = electrodes.monopolar, electrodes.reference

        # Fixed per-subject (x, y) offset of every source: muscles sit slightly differently in each person.
        self.source_offset = np.random.default_rng(subject.seed).normal(
            0.0, subject.muscle_spread_mm / 1000, size=(model.nu, self.n_sources, 2))
        self.poles = self._pole_positions(electrodes)                       # (C, 2 or 1 poles, xyz)
        if p.anatomy not in ANATOMIES:
            raise ValueError(f"anatomy must be one of {' | '.join(ANATOMIES)}, got {p.anatomy!r}")
        self.anatomy = Anatomy(model, subject) if p.anatomy == "atlas" else None
        # The noise sources of an atlas muscle: its volume in n_sources slabs.
        if self.anatomy is not None:
            self.atlas_muscles = np.array(sorted(self.anatomy.points))
            self.atlas_sources = self.anatomy.prepare(
                np.repeat(self.atlas_muscles, self.n_sources),
                np.vstack([self.anatomy.centroids(i, self.n_sources) for i in self.atlas_muscles]))

        # Delay (s) between the two poles: the along-fibre part of the pole spacing over the conduction velocity.
        along_fibre = np.array([e.ied_mm / 1000 * np.cos(np.radians(e.orientation_deg))
                                for e in electrodes.electrodes])
        self.pole_delay = along_fibre / p.conduction_velocity
        if p.engine == "motor_units" and p.motor_units.waveform == "volume_conductor":
            self.pole_delay = np.zeros_like(along_fibre)     # each pole's waveform already propagates

        # Per-source amplitude, scaled so a fully driven superficial muscle under a pole gives ~1 (before gain_mv).
        norm = np.exp(-(self.fat + p.superficial_depth_m) / self.decay) * self.n_sources / 2
        self.source_amp = np.repeat(subject.muscle_amplitudes(model.extrinsic), self.n_sources) / norm
        if p.engine not in ENGINES:
            raise ValueError(f"engine must be one of {' | '.join(ENGINES)}, got {p.engine!r}")
        self.units = MotorUnits(self, p.motor_units) if p.engine == "motor_units" else None

    def generate(self, activity: MuscleActivity, rng: np.random.Generator,
                 on_tick: Callable[[int, int], None] | None = None) -> np.ndarray:
        """(T * upsample, C) EMG in mV at self.fs, from the drive and joint angles at the control rate.
        on_tick(k, T) is called after every chunk of ticks."""
        if self.units is not None:     # recruitment and rate coding make the amplitude follow the drive
            pole_emg = self.units.pole_signals(activity.drive, activity.qpos, activity.fs, rng, on_tick=on_tick)
        else:
            envelope = np.clip(activity.drive, 0, 1) ** self.sub.amp_exponent     # how strongly each muscle fires
            pole_emg = self._pole_signals(envelope, activity.qpos, activity.fs, rng, on_tick=on_tick)
        if self.monopolar:
            emg = pole_emg - pole_emg.mean(axis=1, keepdims=True) if self.reference == "average" else pole_emg
            scale = self.params.monopolar_scale
        else:
            emg, scale = self._bipolar(pole_emg), self.params.bipolar_scale
        scale *= self.sub.gain_mv * self.params.noise_to_mv
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
        """(C, 2, 3) both poles of every electrode as points on the skin; (C, 1, 3) its centre if monopolar."""
        poles = []
        for e in electrodes.electrodes:
            pole_ends = e.poles(self.skin_radius)   # [(angle, z), (angle, z)]
            if electrodes.monopolar:
                pole_ends = ((np.radians(e.angle_deg), e.position_cm / 100),)
            if not all(0.0 <= z <= self.limb_length for _, z in pole_ends):
                raise ValueError(f"electrode {e.name} ({e.position_cm} cm) is off the forearm: positions must be "
                                 f"within 0-{self.limb_length * 100:.1f} cm of the elbow")
            poles += [[self.skin_radius(z) * np.cos(angle), self.skin_radius(z) * np.sin(angle), z]
                      for angle, z in pole_ends]
        return np.array(poles).reshape(len(electrodes), -1, 3)

    def under_fat(self, z: np.ndarray) -> np.ndarray:
        """Radius (m) left for muscle at distance z from the elbow."""
        return self.skin_radius(z) - self.fat

    def xy_scale(self, z: np.ndarray) -> np.ndarray:
        """The generic model's cross-section -> this subject's, at distance z from the elbow."""
        return self.under_fat(z) * self.sub.radial_fill / R_MODEL_REF

    def bellies(self, d: mujoco.MjData) -> np.ndarray:
        """(nu, TRACK_POINTS, 3) the model's bellies now (they carry the posture to the atlas points)."""
        return self.model.belly_points(TRACK_POINTS, d)

    def atlas_positions(self, pts: AtlasPoints, belly: np.ndarray, sel: np.ndarray | slice = slice(None)) -> np.ndarray:
        """(N, 3) prepared atlas points (Anatomy.prepare) in this subject's forearm at the current pose."""
        return self.anatomy.positions(pts, belly, self.under_fat, self.xy_scale, sel)

    def muscle_points(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) points filling the muscles at this pose (an atlas muscle's volume, else its belly sources), and the
        muscle of each (for drawing)."""
        src = self.sources(d)
        lines = np.ones(self.model.nu, bool) if self.anatomy is None else ~self.anatomy.has
        pts, owner = [src[lines].reshape(-1, 3)], [np.repeat(np.flatnonzero(lines), self.n_sources)]
        if self.anatomy is not None:
            if not hasattr(self, "_volume"):
                muscles = sorted(self.anatomy.points)
                self._volume = self.anatomy.prepare(np.repeat(muscles, [len(self.anatomy.points[m]) for m in muscles]),
                                                    np.vstack([self.anatomy.points[m] for m in muscles]))
            pts.append(self.atlas_positions(self._volume, self.bellies(d)))
            owner.append(self._volume.muscle)
        return np.vstack(pts), np.concatenate(owner)

    def sources(self, d: mujoco.MjData) -> np.ndarray:
        """(nu, K, 3) source positions in the forearm frame, in this subject's cross-section."""
        model_pts = self.model.belly_points(self.n_sources, d)               # in the generic model's forearm
        z = model_pts[..., 2]
        # Rescale the model's cross-section to this subject's, then add the per-subject offsets.
        xy = model_pts[..., :2] * self.xy_scale(z)[..., None] + self.source_offset
        out = np.concatenate([xy, z[..., None]], axis=-1)
        if self.anatomy is not None:
            out[self.atlas_muscles] = self.atlas_positions(self.atlas_sources, self.bellies(d)).reshape(
                len(self.atlas_muscles), self.n_sources, 3)
            out[self.atlas_muscles, :, :2] += self.source_offset[self.atlas_muscles]
        # Pull back any source the offset pushed into the fat layer.
        under_fat = self.under_fat(out[..., 2])
        r = np.linalg.norm(out[..., :2], axis=-1)
        out[..., :2] *= (np.minimum(r, under_fat) / np.maximum(r, 1e-9))[..., None]
        return out

    def gains(self, qpos: np.ndarray) -> np.ndarray:
        """(nu * K, poles) gain from every source to every pole at joint state qpos: exp(-distance / decay)."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model.m, self.data)
        src = self.sources(self.data).reshape(-1, 3)                         # (nu * K, 3)
        dist = np.linalg.norm(src[:, None, :] - self.poles.reshape(-1, 3)[None], axis=-1)   # (nu * K, 2 C)
        # Sources outside the (residual) limb are silent.
        in_limb = (src[:, 2] < self.limb_length) & (src[:, 2] > -self.params.elbow_margin_m)
        return (np.exp(-dist / self.decay) * (in_limb * self.source_amp)[:, None]).astype(np.float32)

    # -- signals -----------------------------------------------------------------------------------------------

    def _pole_signals(self, envelope: np.ndarray, qpos: np.ndarray, fs_ctrl: float, rng: np.random.Generator,
                      chunk_ticks: int = 200, on_tick: Callable[[int, int], None] | None = None) -> np.ndarray:
        """(T * upsample, poles) monopolar signal at every pole: each source fires band-limited noise scaled by its
        muscle's envelope, and each pole sums the sources weighted by their gains at that tick's posture."""
        n_ticks, n_muscles = envelope.shape
        K, n_poles = self.n_sources, self.poles.shape[0] * self.poles.shape[1]
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
            if on_tick is not None:
                on_tick(t1, n_ticks)
        return pole_emg

    def _bipolar(self, pole_emg: np.ndarray) -> np.ndarray:
        """(N, C) pole 1 minus pole 2 delayed by the conduction time between them."""
        pole1, pole2 = pole_emg[:, 0::2], pole_emg[:, 1::2]
        t = np.arange(len(pole_emg))
        return np.stack([pole1[:, c] - np.interp(t - self.pole_delay[c] * self.fs, t, pole2[:, c], left=0.0)
                         for c in range(len(self.poles))], axis=1)
