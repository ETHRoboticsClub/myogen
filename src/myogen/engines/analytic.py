"""AnalyticEngine: an interference-pattern sEMG model, fast enough for hours of data on a CPU.

  * Each muscle belly is K point sources along its MuJoCo path, recomputed every control tick, so the bellies
    move under the electrodes with pronation and wrist posture. Fat pushes them away from the skin.
  * Each source emits band-limited noise, scaled by drive ** amp_exponent.
  * Each pole sees a source at distance r with gain exp(-r / decay). The activity propagates along the fibres
    (the forearm axis) at the conduction velocity, so the second pole sees it delayed by
    ied * cos(orientation) / velocity: bipolar = pole1(t) - pole2(t - delay). Inter-electrode distance and
    orientation therefore shape both amplitude and spectrum, as for real bipolar electrodes.
  * Amputees: no intrinsic muscles (Subject.muscle_amplitudes), no sources beyond the residual limb.
"""
import mujoco
import numpy as np
from scipy import signal

from . import CleanEMG, EMGEngine

R_MODEL_REF = 0.032      # m, outermost muscle line-of-action radius in the model's proximal forearm
# Calibration against the point-electrode model this replaces (same spatial selectivity at decay 12 mm):
# the bipolar derivation is 1.65x stronger, and it lowers the spectrum, compensated by Subject.spectrum_hz.
BIPOLAR_SCALE = 0.6


class AnalyticEngine(EMGEngine):
    name = "analytic"

    def __init__(self, n_sources=6, decay_mm=12.0, conduction_velocity=4.0, fs=2000.0):
        self.K, self.decay = n_sources, decay_mm / 1000
        self.velocity, self.fs = conduction_velocity, fs

    def prepare(self, model, subject, electrodes):
        self.model, self.sub = model, subject
        self.data = mujoco.MjData(model.m)          # scratch state for the per-tick geometry
        self.fat = subject.fat_mm / 1000
        self.length = subject.residual_length * model.forearm_length
        self.jitter = np.random.default_rng(subject.seed).normal(
            0.0, subject.muscle_spread_mm / 1000, size=(model.nu, self.K, 2))
        poles = []
        for e in electrodes.electrodes:
            ends = e.poles(self.skin_radius)
            if not all(0.0 <= z <= self.length for _, z in ends):
                raise ValueError(f"electrode {e.name} ({e.position_cm} cm) is off the forearm: positions must be "
                                 f"within 0-{self.length * 100:.1f} cm of the elbow")
            poles += [[self.skin_radius(z) * np.cos(a), self.skin_radius(z) * np.sin(a), z] for a, z in ends]
        self.poles = np.array(poles).reshape(len(electrodes), 2, 3)
        self.delay = np.array([e.ied_mm / 1000 * np.cos(np.radians(e.orientation_deg)) for e in electrodes.electrodes]
                              ) / self.velocity
        # A fully driven superficial muscle under a pole gives ~1 (before gain_mv).
        norm = np.exp(-(self.fat + 0.004) / self.decay) * self.K / 2
        self.source_amp = np.repeat(subject.muscle_amplitudes(model.extrinsic), self.K) / norm

    def skin_radius(self, z):
        return self.sub.forearm_radius_mm / 1000 * (1 - self.sub.taper * np.clip(z / self.model.forearm_length, 0, 1))

    def sources(self, d):
        """(nu, K, 3) source positions in the forearm frame, mapped into this subject's cross-section."""
        p = self.model.belly_points(self.K, d)
        z = p[..., 2]
        r_avail = (self.skin_radius(z) - self.fat) * self.sub.radial_fill
        xy = p[..., :2] / R_MODEL_REF * r_avail[..., None] + self.jitter
        r = np.linalg.norm(xy, axis=-1)
        xy *= (np.minimum(r, self.skin_radius(z) - self.fat) / np.maximum(r, 1e-9))[..., None]
        return np.concatenate([xy, z[..., None]], axis=-1)

    def gains(self, qpos):
        """(nu * K, 2 C) gain from every source to every pole at joint state qpos."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model.m, self.data)
        src = self.sources(self.data).reshape(-1, 3)
        dist = np.linalg.norm(src[:, None, :] - self.poles.reshape(-1, 3)[None], axis=-1)
        alive = (src[:, 2] < self.length) & (src[:, 2] > -0.02)
        return (np.exp(-dist / self.decay) * (alive * self.source_amp)[:, None]).astype(np.float32)

    def generate(self, activity, rng, chunk_ticks=200):
        T, nu = activity.drive.shape
        K, C = self.K, len(self.poles)
        up = int(round(self.fs / activity.fs))
        lo, hi = self.sub.spectrum_hz
        sos = signal.butter(2, [lo, min(hi, 0.45 * self.fs)], btype="bandpass", fs=self.fs, output="sos")
        zi = np.zeros((sos.shape[0], 2, nu * K))
        amp_ctrl = np.clip(activity.drive, 0, 1) ** self.sub.amp_exponent
        out = np.empty((T * up, 2 * C), np.float32)
        t_ctrl = np.arange(T)
        for t0 in range(0, T, chunk_ticks):
            t1 = min(T, t0 + chunk_ticks)
            n = (t1 - t0) * up
            tt = t0 + np.arange(n) / up
            amp = np.stack([np.interp(tt, t_ctrl, amp_ctrl[:, i]) for i in range(nu)], axis=1)
            src, zi = signal.sosfilt(sos, rng.standard_normal((n, nu * K)), axis=0, zi=zi)
            src = (src.reshape(n, nu, K) * amp[:, :, None]).reshape(t1 - t0, up, nu * K).astype(np.float32)
            G = np.stack([self.gains(q) for q in activity.qpos[t0:t1]])
            out[t0 * up:t1 * up] = np.einsum("trj,tjc->trc", src, G).reshape(n, 2 * C)
        pole1, pole2 = out[:, 0::2], out[:, 1::2]
        t = np.arange(len(out))
        emg = np.stack([pole1[:, c] - np.interp(t - self.delay[c] * self.fs, t, pole2[:, c], left=0.0)
                        for c in range(C)], axis=1)
        emg *= self.sub.gain_mv * 3.0 * BIPOLAR_SCALE        # unit-variance source noise -> mV
        return CleanEMG(emg.astype(np.float32), self.fs)
