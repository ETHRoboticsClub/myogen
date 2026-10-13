"""Electrodes: where the bipolar electrodes sit on the forearm and how the hardware records them.

Positions are anatomical and shared by both engines (forearm frame, see body.py):
    angle_deg        around the forearm: 0 = radial border, +90 dorsal (extensors), -90 palmar (flexors), +-180 ulnar
    position_cm      distance from the elbow along the forearm (the model's forearm is 25.8 cm)
    ied_mm           distance between the two poles
    orientation_deg  pole axis: 0 = along the forearm (the muscle fibres), 90 = around it

    Electrodes.ring(n=8, position_cm=7, rotation_deg=0)          # one armband
    Electrodes.myo() / Electrodes.otto_bock()                    # the NinaPro DB5 / DB1 hardware
    Electrodes([Electrode(angle_deg=90, position_cm=6)], device="raw")
    Electrodes.load("electrodes.yml") / .save(...)

Devices: raw (bipolar sEMG in mV at fs), myo (int8 counts at 200 Hz), otto_bock (rectified, smoothed envelope in V
at 100 Hz). Recording scales the clean EMG by `gain`, adds skin / amplifier noise, mains interference and movement
artifacts, then the amplifier band-pass and the device front-end (Frontend). The YAML file is the device profile:
`myogen.calibrate` fits gain, noise and the Otto Bock smoothing to recordings of a real device.
"""
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy import signal

DEVICES = ("raw", "myo", "otto_bock")
DEVICE_FS = {"myo": 200.0, "otto_bock": 100.0}


@dataclass
class Frontend:
    """The device's own processing; fields that do not apply to the device are ignored."""
    amplifier_hz: tuple = (20.0, 450.0)    # amplifier band-pass (all devices)
    counts_per_mv: float = 150.0           # myo: ADC gain
    adc_noise: float = 0.55                # myo: ADC noise, counts RMS
    alias: bool = False                    # myo: sample without anti-aliasing (folds 100-450 Hz content down)
    band_hz: tuple = (90.0, 450.0)         # otto_bock: electrode band-pass (rejects mains and artifacts)
    smooth_hz: float = 4.5                 # otto_bock: envelope smoothing (13E200: 3-6 Hz)
    volts_per_mv: float = 15.0             # otto_bock: output gain (set by the technician's knob)


@dataclass
class Electrode:
    angle_deg: float
    position_cm: float
    ied_mm: float = 20.0
    orientation_deg: float = 0.0
    name: str = ""

    def poles(self, skin_radius_m):
        """((angle_rad, z_m), (angle_rad, z_m)) of the two poles; skin_radius_m(z) converts arc length to angle."""
        z = self.position_cm / 100
        half = self.ied_mm / 2000
        dz = half * math.cos(math.radians(self.orientation_deg))
        darc = half * math.sin(math.radians(self.orientation_deg))
        centre = math.radians(self.angle_deg)
        return ((centre - darc / skin_radius_m(z - dz), z - dz), (centre + darc / skin_radius_m(z + dz), z + dz))


@dataclass
class Electrodes:
    electrodes: list
    device: str = "raw"
    fs: float = 2000.0              # output rate of the raw device (myo and otto_bock use their hardware rate)
    gain: float = 1.0               # overall amplitude: electrode contact and amplification
    noise_uv: float = 1.5           # skin + amplifier noise RMS (dry Myo electrodes: ~5)
    mains_hz: float = 50.0
    mains_uv: float = 2.0
    motion_artifact: float = 1.0    # scale of low-frequency movement artifacts
    channel_gain: list = field(default=None)   # (C,) per-channel gain (contact quality); None = 1
    frontend: Frontend = field(default_factory=Frontend)

    def __post_init__(self):
        self.electrodes = [e if isinstance(e, Electrode) else Electrode(**e) for e in self.electrodes]
        if isinstance(self.frontend, dict):
            self.frontend = Frontend(**{k: tuple(v) if isinstance(v, list) else v for k, v in self.frontend.items()})
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {' | '.join(DEVICES)}, got {self.device!r}")
        if not self.electrodes:
            raise ValueError("no electrodes")
        for i, e in enumerate(self.electrodes):
            e.name = e.name or f"ch{i}"

    def __len__(self):
        return len(self.electrodes)

    @property
    def names(self):
        return [e.name for e in self.electrodes]

    @property
    def fs_out(self):
        return DEVICE_FS.get(self.device, self.fs)

    # -- presets -----------------------------------------------------------------------------------------------

    @classmethod
    def ring(cls, n=8, position_cm=7.0, rotation_deg=0.0, rings=1, ring_gap_mm=35.0, ied_mm=20.0, **kwargs):
        """n equally spaced electrodes per ring; ring r is ring_gap_mm further from the elbow and rotated by half
        an electrode spacing per ring."""
        electrodes = [Electrode(angle_deg=_wrap(rotation_deg + 360 * (k + 0.5 * r) / n),
                                position_cm=position_cm + r * ring_gap_mm / 10, ied_mm=ied_mm, name=f"ring{r}_{k}")
                      for r in range(rings) for k in range(n)]
        return cls(electrodes, **kwargs)

    @classmethod
    def myo(cls, position_cm=3.1, rotation_deg=0.0):
        """NinaPro DB5: two Myo armbands (8 pods each), the second 3.5 cm distal and rotated 22.5 deg."""
        return cls.ring(8, position_cm, rotation_deg, rings=2, ring_gap_mm=35.0, ied_mm=15.0, device="myo",
                        noise_uv=5.0)

    @classmethod
    def otto_bock(cls, position_cm=2.6, rotation_deg=0.0):
        """NinaPro DB1: 8 Otto Bock electrodes around the proximal forearm + 2 over the FDS and EDC bellies
        (2.6 cm: as proximal as the BioMime grid allows, which starts at 1.55 cm)."""
        ring = cls.ring(8, position_cm, rotation_deg, device="otto_bock")
        targets = [Electrode(-120.0, 8.4, name="target_FDS"), Electrode(91.0, 8.4, name="target_EDC")]
        return cls(ring.electrodes + targets, device="otto_bock")

    @classmethod
    def preset(cls, name):
        presets = {"ring": cls.ring, "myo": cls.myo, "otto_bock": cls.otto_bock}
        if name not in presets:
            raise ValueError(f"unknown electrode preset {name!r}: {' | '.join(presets)}")
        return presets[name]()

    # -- files -------------------------------------------------------------------------------------------------

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return Path(path)

    @classmethod
    def load(cls, path):
        """An electrode YAML file, or a preset name (ring | myo | otto_bock)."""
        if not Path(path).exists() and str(path) in ("ring", "myo", "otto_bock"):
            return cls.preset(str(path))
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    # -- recording ---------------------------------------------------------------------------------------------

    def record(self, clean, fs, joint_speed, fs_motion, rng):
        """Device output (float32) from clean bipolar sEMG (N, C) in mV at fs. joint_speed: (T,) forearm / wrist
        speed (rad/s) at fs_motion, which scales the movement artifacts."""
        N, C = clean.shape
        gain = self.gain * (np.ones(C) if self.channel_gain is None else np.asarray(self.channel_gain, float))
        out = (clean * gain[None]).astype(np.float32)
        t = np.arange(N) / fs
        noise = self.noise_uv * 1e-3
        out += rng.normal(0, noise, out.shape).astype(np.float32)
        mains_amp = self.mains_uv * 1e-3 * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t))
        for c in range(C):
            ph = rng.uniform(0, 2 * np.pi, 2)
            out[:, c] += (mains_amp * rng.uniform(0.5, 1.5) * (np.sin(2 * np.pi * self.mains_hz * t + ph[0])
                          + 0.3 * np.sin(2 * np.pi * 3 * self.mains_hz * t + ph[1]))).astype(np.float32)
        speed = np.interp(t * fs_motion, np.arange(len(joint_speed)), joint_speed)
        art = signal.sosfilt(signal.butter(2, 8, fs=fs, output="sos"), rng.standard_normal((N, C)), axis=0) * 0.03
        out += (art * (self.motion_artifact * speed)[:, None]).astype(np.float32)
        lo, hi = self.frontend.amplifier_hz
        amp = signal.butter(4, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
        return self._device(signal.sosfilt(amp, out, axis=0).astype(np.float32), fs, rng)

    def _device(self, x, fs, rng):
        f = self.frontend
        if self.device == "otto_bock":
            # Band-pass, rectify + smooth, 100 Hz, 0..5 V.
            band = signal.butter(2, [f.band_hz[0], min(f.band_hz[1], 0.45 * fs)], btype="bandpass", fs=fs,
                                 output="sos")
            env = signal.sosfilt(signal.butter(2, f.smooth_hz, fs=fs, output="sos"),
                                 np.abs(signal.sosfilt(band, x, axis=0)), axis=0)
            step = int(fs // DEVICE_FS["otto_bock"])
            return np.clip(np.maximum(env, 0)[::step] * f.volts_per_mv, 0, 5).astype(np.float32)
        if self.device == "myo":
            step = int(fs // DEVICE_FS["myo"])
            y = (x[::step] if f.alias else signal.resample_poly(x, 1, step, axis=0)) * f.counts_per_mv
            return np.clip(np.round(y + rng.normal(0, f.adc_noise, y.shape)), -128, 127).astype(np.float32)
        return resample(x, fs, self.fs).astype(np.float32)


def resample(x, fs, fs_out):
    if fs_out == fs:
        return x
    g = math.gcd(int(fs_out), int(fs))
    return signal.resample_poly(x, int(fs_out) // g, int(fs) // g, axis=0)


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0
