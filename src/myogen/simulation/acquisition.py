from __future__ import annotations

import math

import numpy as np
from scipy import signal

from ..inputs.electrodes import DEVICE_FS, Electrodes
from ..params import AcquisitionParams


def record(clean: np.ndarray, fs: float, electrodes: Electrodes, joint_speed: np.ndarray, fs_motion: float,
           rng: np.random.Generator, params: AcquisitionParams | None = None) -> np.ndarray:
    """Device output (float32) from clean bipolar sEMG (N, C) in mV at fs. joint_speed: (T,) forearm / wrist
    speed (rad/s) at fs_motion, which scales the movement artifacts."""
    p = params or AcquisitionParams()
    x = (clean * channel_gains(electrodes)[None]).astype(np.float32)   # electrode contact and amplification
    x += skin_noise(x.shape, electrodes, rng)
    x += mains(x.shape, fs, electrodes, rng, p)
    x += movement_artifacts(x.shape, fs, joint_speed, fs_motion, electrodes, rng, p)
    x = amplifier(x, fs, electrodes, p)
    return device_output(x, fs, electrodes, rng)


def channel_gains(electrodes: Electrodes) -> np.ndarray:
    """(C,) overall gain times the per-channel contact quality."""
    C = len(electrodes)
    per_channel = np.ones(C) if electrodes.channel_gain is None else np.asarray(electrodes.channel_gain, float)
    return electrodes.gain * per_channel


def skin_noise(shape: tuple[int, int], electrodes: Electrodes, rng: np.random.Generator) -> np.ndarray:
    """(N, C) white skin + amplifier noise."""
    return rng.normal(0, electrodes.noise_uv * 1e-3, shape).astype(np.float32)


def mains(shape: tuple[int, int], fs: float, electrodes: Electrodes, rng: np.random.Generator,
          p: AcquisitionParams) -> np.ndarray:
    """(N, C) mains interference: slowly modulated fundamental + third harmonic, random phase and level per
    channel."""
    N, C = shape
    t = np.arange(N) / fs
    modulation = np.sin(2 * np.pi * rng.uniform(*p.mains_modulation_hz) * t)
    amp = electrodes.mains_uv * 1e-3 * (1 + p.mains_modulation * modulation)
    f = electrodes.mains_hz
    out = np.empty(shape, np.float32)
    for c in range(C):
        phase = rng.uniform(0, 2 * np.pi, 2)
        wave = np.sin(2 * np.pi * f * t + phase[0]) + p.mains_third_harmonic * np.sin(2 * np.pi * 3 * f * t + phase[1])
        out[:, c] = (amp * rng.uniform(*p.mains_channel_spread) * wave).astype(np.float32)
    return out


def movement_artifacts(shape: tuple[int, int], fs: float, joint_speed: np.ndarray, fs_motion: float,
                       electrodes: Electrodes, rng: np.random.Generator, p: AcquisitionParams) -> np.ndarray:
    """(N, C) low-pass noise scaled by the forearm / wrist speed."""
    t = np.arange(shape[0]) / fs
    speed = np.interp(t * fs_motion, np.arange(len(joint_speed)), joint_speed)
    lowpass = signal.butter(2, p.artifact_lowpass_hz, fs=fs, output="sos")
    noise = signal.sosfilt(lowpass, rng.standard_normal(shape), axis=0) * p.artifact_mv
    return (noise * (electrodes.motion_artifact * speed)[:, None]).astype(np.float32)


def amplifier(x: np.ndarray, fs: float, electrodes: Electrodes, p: AcquisitionParams) -> np.ndarray:
    """The amplifier's band-pass (Frontend.amplifier_hz)."""
    lo, hi = electrodes.frontend.amplifier_hz
    band = signal.butter(p.amplifier_order, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(band, x, axis=0).astype(np.float32)


def device_output(x: np.ndarray, fs: float, electrodes: Electrodes, rng: np.random.Generator) -> np.ndarray:
    """What the device delivers: raw (mV at Electrodes.fs), myo (int8 counts at 200 Hz) or otto_bock (rectified,
    smoothed envelope in V at 100 Hz)."""
    f = electrodes.frontend
    if electrodes.device == "otto_bock":
        band = signal.butter(2, [f.band_hz[0], min(f.band_hz[1], 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
        env = signal.sosfilt(signal.butter(2, f.smooth_hz, fs=fs, output="sos"),
                             np.abs(signal.sosfilt(band, x, axis=0)), axis=0)
        step = int(fs // DEVICE_FS["otto_bock"])
        return np.clip(np.maximum(env, 0)[::step] * f.volts_per_mv, 0, 5).astype(np.float32)
    if electrodes.device == "myo":
        step = int(fs // DEVICE_FS["myo"])
        y = (x[::step] if f.alias else signal.resample_poly(x, 1, step, axis=0)) * f.counts_per_mv
        return np.clip(np.round(y + rng.normal(0, f.adc_noise, y.shape)), -128, 127).astype(np.float32)
    return resample(x, fs, electrodes.fs).astype(np.float32)


def resample(x: np.ndarray, fs: float, fs_out: float) -> np.ndarray:
    if fs_out == fs:
        return x
    g = math.gcd(int(fs_out), int(fs))
    return signal.resample_poly(x, int(fs_out) // g, int(fs) // g, axis=0)
