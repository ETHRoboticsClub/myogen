"""Analytic sEMG engine: an interference-pattern model, fast enough for hours of data on a CPU.

  * Each muscle belly is K point sources along its MuJoCo path, recomputed every control tick, so bellies move
    under the electrodes with pronation and wrist posture.
  * Each source emits band-limited noise with a muscle-specific spectrum, scaled by excitation ** exponent.
  * Volume conductor: a source at distance r from an electrode contributes exp(-r / decay); fat pushes muscles
    away from the skin. Cross-talk comes out of the geometry.
  * Amputees: intrinsic muscles removed (muscle_amp), sources beyond the residual limb cut.
"""
import numpy as np
from scipy import signal

from ..body.model import belly_points
from .acquisition import acquire

R_MODEL_REF = 0.032   # m, outermost muscle line-of-action radius in the model's proximal forearm


class EMGGeometry:
    """Electrode positions and per-tick source -> electrode gain matrices for one subject / session."""

    def __init__(self, m, d, anat, subject, session, n_src=6):
        self.m, self.anat, self.sub, self.K = m, anat, subject, n_src
        rng = np.random.default_rng(subject.seed)
        self.jitter = rng.normal(0.0, subject.muscle_spread, size=(m.nu, n_src, 2))
        self.electrodes = self._place_electrodes(m, d, anat, subject, session, rng)     # (C, 3)
        self.norm = None

    def _place_electrodes(self, m, d, anat, subject, session, rng):
        lay, L = subject.layout, anat.forearm_length
        angles, levels = [], []
        for r in range(lay.rings):
            stagger = (np.pi / lay.per_ring) * r
            for k in range(lay.per_ring):
                angles.append(lay.rotation + session.rotation_shift + stagger + 2 * np.pi * k / lay.per_ring)
                levels.append((lay.level + session.level_shift) * L + r * lay.ring_gap)
        if lay.targets:
            # Targeted electrodes (e.g. DB1's FDS / EDC spots): above the muscle belly at target_level.
            P = belly_points(m, d, anat, 24)
            for name in lay.targets:
                pts = P[anat.muscle_names.index(name)]
                z_t = (lay.target_level + session.level_shift) * L
                p = pts[np.argmin(np.abs(pts[:, 2] - z_t))]
                angles.append(np.arctan2(p[1], p[0]) + session.rotation_shift * 0.3)
                levels.append(z_t)
        angles = np.array(angles) + rng.normal(0, lay.angle_jitter, len(angles))
        levels = np.clip(np.array(levels), 0.02, subject.residual_length * L - 0.01)
        rad = self.skin_radius(levels)
        return np.stack([rad * np.cos(angles), rad * np.sin(angles), levels], axis=1)

    def skin_radius(self, z):
        return self.sub.forearm_radius * (1 - self.sub.taper * np.clip(z / self.anat.forearm_length, 0, 1))

    def sources(self, d):
        """(nu, K, 3) source positions in the forearm frame, mapped into this subject's cross-section."""
        sub = self.sub
        p = belly_points(self.m, d, self.anat, self.K)
        z = p[..., 2]
        r_avail = (self.skin_radius(z) - sub.fat) * sub.radial_fill
        xy = p[..., :2] / R_MODEL_REF * r_avail[..., None] + self.jitter
        r = np.linalg.norm(xy, axis=-1)
        xy *= (np.minimum(r, self.skin_radius(z) - sub.fat) / np.maximum(r, 1e-9))[..., None]
        return np.concatenate([xy, z[..., None]], axis=-1)

    def gains(self, d):
        """(nu * K, C) gain matrix at the current state."""
        sub = self.sub
        src = self.sources(d).reshape(-1, 3)
        dist = np.linalg.norm(src[:, None, :] - self.electrodes[None], axis=-1)
        G = np.exp(-dist / sub.decay)
        alive = (src[:, 2] < sub.residual_length * self.anat.forearm_length) & (src[:, 2] > -0.02)
        G *= alive[:, None]
        if self.norm is None:
            # A fully excited superficial muscle under an electrode gives ~1 (before gain_mv).
            self.norm = np.exp(-(sub.fat + 0.004) / sub.decay) * self.K / 2
        G *= (np.repeat(sub.muscle_amp, self.K) / self.norm)[:, None]
        return G.astype(np.float32)


def synthesize(u, G, qd, subject, session, fs_ctrl, fs=2000, fs_out=2000, rng=None, chunk_ticks=200):
    """sEMG from excitations u (T, nu) and gain matrices G (T, nu * K, C) at fs_ctrl; qd (T, nv) for movement
    artifacts. Returns the device output (see acquisition.acquire)."""
    rng = rng or np.random.default_rng()
    T, nu = u.shape
    MK, C = G.shape[1:]
    K = MK // nu
    up = int(round(fs / fs_ctrl))
    sos = [signal.butter(2, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
           for lo, hi in zip(subject.spectrum_lo, subject.spectrum_hi)]
    zi = [np.zeros((s.shape[0], 2, K)) for s in sos]
    # Neural drive is smooth (< ~5 Hz): the controller's fast corrections are a simulation artifact.
    u_s = signal.sosfiltfilt(signal.butter(2, 4.0, fs=fs_ctrl, output="sos"), u, axis=0) if T > 15 else u
    amp_ctrl = np.clip(u_s, 0, 1) ** subject.amp_exponent
    out = np.empty((T * up, C), np.float32)
    t_ctrl = np.arange(T)
    for t0 in range(0, T, chunk_ticks):
        t1 = min(T, t0 + chunk_ticks)
        n = (t1 - t0) * up
        tt = t0 + np.arange(n) / up
        amp = np.stack([np.interp(tt, t_ctrl, amp_ctrl[:, i]) for i in range(nu)], axis=1)
        src = np.empty((n, nu, K))
        for i in range(nu):
            src[:, i, :], zi[i] = signal.sosfilt(sos[i], rng.standard_normal((n, K)), axis=0, zi=zi[i])
        src *= amp[:, :, None]
        src = src.reshape(t1 - t0, up, MK).astype(np.float32)
        out[t0 * up:t1 * up] = np.einsum("trj,tjc->trc", src, G[t0:t1]).reshape(n, C)
    out *= subject.gain_mv * 3.0          # unit-variance source noise -> mV scale
    return acquire(out, qd, subject, session, fs_ctrl, fs=fs, fs_out=fs_out, rng=rng)
