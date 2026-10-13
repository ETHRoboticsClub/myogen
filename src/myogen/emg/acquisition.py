"""The recording chain shared by both EMG engines: electrode gain, noise, mains, movement artifacts, a bad
contact, the amplifier band-pass and the device front-end.

Devices: "raw" (bipolar sEMG in mV at fs_out) or the NinaPro hardware, so synthetic data can be compared with
(and mixed into) the real datasets channel for channel:
  otto_bock (DB1)  hardware-rectified, smoothed envelope at 100 Hz (volts)
  myo (DB5)        two 8-channel armbands at 200 Hz, int8 counts
"""
import numpy as np
from scipy import signal

DEVICES = ("raw", "otto_bock", "myo")
DEVICE_FS = {"otto_bock": 100, "myo": 200}


def acquire(x, qd, subject, session, fs_ctrl, fs=2000, fs_out=2000, rng=None):
    """x: (N, C) clean sEMG in mV at fs; qd: (T, nv) joint velocities at fs_ctrl (movement artifacts).
    Returns the device output at fs_out (float32)."""
    rng = rng or np.random.default_rng()
    N, C = x.shape
    out = (x * session.channel_gain[None]).astype(np.float32)
    t = np.arange(N) / fs
    noise = subject.noise_uv * 1e-3 * session.noise_scale
    out += rng.normal(0, noise, out.shape).astype(np.float32)
    mains_amp = subject.mains_uv * 1e-3 * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t))
    for c in range(C):
        ph = rng.uniform(0, 2 * np.pi, 2)
        out[:, c] += (mains_amp * rng.uniform(0.5, 1.5) * (np.sin(2 * np.pi * subject.mains_hz * t + ph[0])
                      + 0.3 * np.sin(2 * np.pi * 3 * subject.mains_hz * t + ph[1]))).astype(np.float32)
    # Movement artifacts: low-frequency transients scaled by forearm / wrist speed.
    speed = np.interp(t * fs_ctrl, np.arange(len(qd)), np.abs(qd[:, :3]).sum(1))
    art = signal.sosfilt(signal.butter(2, 8, fs=fs, output="sos"), rng.standard_normal((N, C)), axis=0) * 0.03
    out += (art * (session.motion_artifact * speed)[:, None]).astype(np.float32)
    if session.bad_channel >= 0:
        b = session.bad_channel
        out[:, b] = 0.1 * out[:, b] + rng.normal(0, 20 * noise, N) \
            + 30 * subject.mains_uv * 1e-3 * np.sin(2 * np.pi * subject.mains_hz * t)
    # Amplifier band-pass (causal, like hardware).
    amp = signal.butter(4, [20, min(450, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    out = signal.sosfilt(amp, out, axis=0).astype(np.float32)
    return device_output(out, fs, fs_out, subject)


def device_output(x, fs, fs_out, subject):
    """Emulate the recording hardware on band-passed sEMG x (mV at fs)."""
    dev, p = subject.layout.device, subject.device_params
    if dev == "otto_bock":
        # 13E200-style: 90-450 Hz band, rectify + smooth, sample at 100 Hz, volts (0..5 V).
        x = signal.sosfilt(signal.butter(2, [90, min(450, 0.45 * fs)], btype="bandpass", fs=fs, output="sos"), x, axis=0)
        env = np.maximum(signal.sosfilt(signal.butter(2, p["smooth_hz"], fs=fs, output="sos"), np.abs(x), axis=0), 0)
        return np.clip(env[::int(fs // DEVICE_FS[dev])] * p["volts_per_mv"], 0, 5).astype(np.float32)
    if dev == "myo":
        step = int(fs // DEVICE_FS[dev])
        # Without anti-aliasing the 200 Hz Myo folds 100-450 Hz content back down.
        y = x[::step] if p["alias"] else signal.resample_poly(x, 1, step, axis=0)
        y = y * p["counts_per_mv"] + np.random.default_rng(subject.seed).normal(0, p.get("adc_noise", 0.0), y.shape)
        return np.clip(np.round(y), -128, 127).astype(np.float32)
    return resample(x, fs, fs_out).astype(np.float32)


def resample(x, fs, fs_out):
    if fs_out == fs:
        return x
    g = np.gcd(int(fs_out), int(fs))
    return signal.resample_poly(x, int(fs_out) // g, int(fs) // g, axis=0)
