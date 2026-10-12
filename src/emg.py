"""Surface EMG forward model: muscle excitations + musculoskeletal geometry -> multichannel forearm sEMG.

Model (interference-pattern level, fast enough to generate hours of data on a CPU):

  * Each muscle belly is represented by K point sources sampled along its path in the forearm, taken from the
    MuJoCo tendon geometry at every control tick. Bellies therefore move under the electrodes with pronation and
    wrist posture (the "limb position effect" comes for free). Muscle width is a fixed transverse jitter.
  * Each source emits band-limited Gaussian noise (a stand-in for the summed motor-unit action potential train)
    with a muscle-specific spectrum. Its amplitude follows the neural excitation u(t)^beta: EMG reflects the
    drive, which precedes force by the activation dynamics.
  * Volume conductor: a source at distance d from a bipolar electrode on the skin contributes with gain
    exp(-d / decay). Subcutaneous fat pushes muscles away from the skin. Cross-talk and inter-channel
    correlation come out of the geometry, they are not added by hand.
  * Acquisition: per-channel gain, amplifier/skin noise, mains interference, movement artifacts, an optional
    bad contact, a 20-450 Hz amplifier band-pass and resampling to the target device rate.
  * Amputee mode: intrinsic muscles removed, sources distal to the residual limb cut, atrophy in muscle_amp.

Motor-unit level synthesis (MUAP templates + recruitment/rate coding, e.g. BioMime/NeuroMotion) can replace
the source model later without changing the interface: sources_to_channels() stays the same.
"""
import numpy as np
from scipy import signal

from .model import belly_points
from .subject import DEVICE_FS, Session, Subject

R_MODEL_REF = 0.032   # m, outermost muscle line-of-action radius in MyoHand's proximal forearm


class EMGGeometry:
    """Per-tick source->electrode gain matrices for one subject/session."""

    def __init__(self, m, d, anat, subject: Subject, session: Session, n_src=6):
        self.m, self.anat, self.sub, self.ses, self.K = m, anat, subject, session, n_src
        rng = np.random.default_rng(subject.seed)
        self.jitter = rng.normal(0.0, subject.muscle_spread, size=(m.nu, n_src, 2))
        lay = subject.layout
        L = anat.forearm_length
        per_ring = (lay.n_channels - len(lay.targets)) // lay.rings
        angles, levels = [], []
        for r in range(lay.rings):
            stagger = (np.pi / per_ring) * r
            for k in range(per_ring):
                angles.append(lay.rotation + session.rotation_shift + stagger + 2 * np.pi * k / per_ring)
                levels.append((lay.level + session.level_shift) * L + r * lay.ring_gap)
        # Targeted electrodes (e.g. DB1's FDS/EDC spots): above the muscle belly at target_level.
        if lay.targets:
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
        self.electrodes = np.stack([rad * np.cos(angles), rad * np.sin(angles), levels], axis=1)  # (C, 3)
        self.norm = None

    def skin_radius(self, z):
        L = self.anat.forearm_length
        return self.sub.forearm_radius * (1 - self.sub.taper * np.clip(z / L, 0, 1))

    def gains(self, d):
        """(nu*K, C) gain matrix at the current MuJoCo state."""
        sub, L = self.sub, self.anat.forearm_length
        p = belly_points(self.m, d, self.anat, self.K)                      # (nu, K, 3) forearm frame
        z = p[..., 2]
        # Map MyoHand lines of action to an anatomical cross-section of this subject.
        r_avail = (self.skin_radius(z) - sub.fat) * sub.radial_fill
        xy = p[..., :2] / R_MODEL_REF * r_avail[..., None] + self.jitter
        r = np.linalg.norm(xy, axis=-1)
        xy *= (np.minimum(r, self.skin_radius(z) - sub.fat) / np.maximum(r, 1e-9))[..., None]
        src = np.concatenate([xy, z[..., None]], axis=-1).reshape(-1, 3)
        dist = np.linalg.norm(src[:, None, :] - self.electrodes[None], axis=-1)  # (nu*K, C)
        G = np.exp(-dist / sub.decay)
        alive = (z.reshape(-1) < sub.residual_length * L) & (z.reshape(-1) > -0.02)
        G *= alive[:, None]
        if self.norm is None:
            # Normalise so a fully excited superficial muscle under an electrode is ~1 (before gain_mv).
            self.norm = np.exp(-(sub.fat + 0.004) / sub.decay) * self.K / 2
        G *= (np.repeat(sub.muscle_amp, self.K) / self.norm)[:, None]
        return G.astype(np.float32)


def _sos_bank(subject: Subject, fs):
    return [signal.butter(2, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
            for lo, hi in zip(subject.spectrum_lo, subject.spectrum_hi)]


def synthesize(u, G, qd, subject: Subject, session: Session, fs_ctrl, fs=2000, fs_out=2000,
               rng=None, chunk_ticks=200):
    """Render sEMG.

    u:  (T, nu) neural excitation at fs_ctrl
    G:  (T, nu*K, C) gain matrices at fs_ctrl
    qd: (T, nv) joint velocities (for movement artifacts)
    Returns emg (T * fs_out / fs_ctrl, C) float32 in mV.
    """
    rng = rng or np.random.default_rng()
    T, nu = u.shape
    MK, C = G.shape[1:]
    K = MK // nu
    up = int(round(fs / fs_ctrl))
    sos = _sos_bank(subject, fs)
    zi = [np.zeros((s.shape[0], 2, K)) for s in sos]
    # Neural drive is smooth (< ~5 Hz); the tracking controller's small fast corrections are an artifact of the
    # simulation, so the EMG amplitude follows a zero-phase low-passed drive (timing preserved).
    u_s = signal.sosfiltfilt(signal.butter(2, 4.0, fs=fs_ctrl, output="sos"), u, axis=0) if T > 15 else u
    amp_ctrl = np.clip(u_s, 0, 1) ** subject.amp_exponent
    out = np.empty((T * up, C), np.float32)
    t_ctrl = np.arange(T)
    for t0 in range(0, T, chunk_ticks):
        t1 = min(T, t0 + chunk_ticks)
        n = (t1 - t0) * up
        tt = t0 + np.arange(n) / up
        amp = np.stack([np.interp(tt, t_ctrl, amp_ctrl[:, i]) for i in range(nu)], axis=1)   # (n, nu)
        src = np.empty((n, nu, K))
        for i in range(nu):
            w = rng.standard_normal((n, K))
            src[:, i, :], zi[i] = signal.sosfilt(sos[i], w, axis=0, zi=zi[i])
        src *= amp[:, :, None]
        src = src.reshape(t1 - t0, up, MK).astype(np.float32)
        out[t0 * up:t1 * up] = np.einsum("trj,tjc->trc", src, G[t0:t1]).reshape(n, C)
    # Source noise is unit-variance before band-pass; restore a sensible amplitude scale.
    out *= subject.gain_mv * 3.0
    return acquire(out, qd, subject, session, fs_ctrl, fs=fs, fs_out=fs_out, rng=rng)


def acquire(out, qd, subject: Subject, session: Session, fs_ctrl, fs=2000, fs_out=2000, rng=None):
    """Recording chain shared by every source model (analytic, BioMime): electrode-skin gain, amplifier and
    skin noise, mains, movement artifacts, bad contact, 20-450 Hz amplifier, device front-end.

    out: (N, C) clean sEMG in mV at fs. qd: (T, nv) joint velocities at fs_ctrl (movement artifacts).
    """
    rng = rng or np.random.default_rng()
    N, C = out.shape
    out = (out * session.channel_gain[None]).astype(np.float32)
    t_ctrl = np.arange(len(qd))
    t = np.arange(N) / fs
    noise = subject.noise_uv * 1e-3 * session.noise_scale
    out += rng.normal(0, noise, out.shape).astype(np.float32)
    mains_amp = subject.mains_uv * 1e-3 * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t))
    for c in range(C):
        ph = rng.uniform(0, 2 * np.pi, 2)
        out[:, c] += (mains_amp * rng.uniform(0.5, 1.5) * (np.sin(2 * np.pi * subject.mains_hz * t + ph[0])
                      + 0.3 * np.sin(2 * np.pi * 3 * subject.mains_hz * t + ph[1]))).astype(np.float32)
    # Movement artifacts: low-frequency transients scaled by forearm/wrist speed.
    speed = np.abs(qd[:, :3]).sum(1)
    speed = np.interp(t * fs_ctrl, t_ctrl, speed)
    lp = signal.butter(2, 8, fs=fs, output="sos")
    art = signal.sosfilt(lp, rng.standard_normal((N, C)), axis=0) * 0.03
    out += (art * (session.motion_artifact * speed)[:, None]).astype(np.float32)
    if session.bad_channel >= 0:
        b = session.bad_channel
        out[:, b] = 0.1 * out[:, b] + rng.normal(0, 20 * noise, N) \
            + 30 * subject.mains_uv * 1e-3 * np.sin(2 * np.pi * subject.mains_hz * t)
    # Amplifier band-pass (causal, like hardware), then the device's own processing and sampling.
    hp = signal.butter(4, [20, min(450, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    out = signal.sosfilt(hp, out, axis=0).astype(np.float32)
    return device_output(out, fs, fs_out, subject)


def device_output(x, fs, fs_out, subject: Subject):
    """Emulate the recording hardware on band-passed sEMG x (mV at fs)."""
    dev, p = subject.layout.device, subject.device_params
    if dev == "otto_bock":
        # 13E200-style front end: ~90-450 Hz band (rejects mains and movement artifacts), rectify + smooth
        # (hardware envelope), sample at 100 Hz, output in volts (0..5 V range).
        x = signal.sosfilt(signal.butter(2, [90, min(450, 0.45 * fs)], btype="bandpass", fs=fs, output="sos"), x, axis=0)
        sos = signal.butter(2, p["smooth_hz"], fs=fs, output="sos")
        env = np.maximum(signal.sosfilt(sos, np.abs(x), axis=0), 0)
        step = int(fs // DEVICE_FS[dev])
        return np.clip(env[::step] * p["volts_per_mv"], 0, 5).astype(np.float32)
    if dev == "myo":
        step = int(fs // DEVICE_FS[dev])
        if p["alias"]:
            y = x[::step]          # no anti-aliasing: the 200 Hz Myo folds 100-450 Hz content back down
        else:
            y = signal.resample_poly(x, 1, step, axis=0)
        y = y * p["counts_per_mv"] + np.random.default_rng(subject.seed).normal(0, p.get("adc_noise", 0.0), y.shape)
        return np.clip(np.round(y), -128, 127).astype(np.float32)
    if fs_out != fs:
        g = np.gcd(int(fs_out), int(fs))
        x = signal.resample_poly(x, int(fs_out) // g, int(fs) // g, axis=0)
    return x.astype(np.float32)
