from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEVICES = ("raw", "myo", "otto_bock")
DEVICE_FS = {"myo": 200.0, "otto_bock": 100.0}


@dataclass
class Frontend:
    """The device's own processing; fields that do not apply to the device are ignored."""
    amplifier_hz: tuple[float, float] = (20.0, 450.0)   # amplifier band-pass (all devices)
    counts_per_mv: float = 150.0                        # myo: ADC gain
    adc_noise: float = 0.55                             # myo: ADC noise, counts RMS
    alias: bool = False                                 # myo: no anti-aliasing (folds 100-450 Hz content down)
    band_hz: tuple[float, float] = (90.0, 450.0)        # otto_bock: electrode band-pass (rejects mains and artifacts)
    smooth_hz: float = 4.5                              # otto_bock: envelope smoothing (13E200: 3-6 Hz)
    volts_per_mv: float = 15.0                          # otto_bock: output gain (set by the technician's knob)


@dataclass
class Electrode:
    """One bipolar electrode, placed in anatomical terms."""
    angle_deg: float                # around the forearm: 0 radial border, +90 dorsal, -90 palmar, +-180 ulnar
    position_cm: float              # distance from the elbow along the forearm (the model's forearm is 25.8 cm)
    ied_mm: float = 20.0            # distance between the two poles
    orientation_deg: float = 0.0    # pole axis: 0 along the forearm (the muscle fibres), 90 around it
    name: str = ""

    def poles(self, skin_radius_m: Callable[[float], float]) -> tuple[tuple[float, float], tuple[float, float]]:
        """((angle_rad, z_m), (angle_rad, z_m)) of the two poles; skin_radius_m(z) converts arc length to angle."""
        z = self.position_cm / 100
        half = self.ied_mm / 2000
        dz = half * math.cos(math.radians(self.orientation_deg))
        darc = half * math.sin(math.radians(self.orientation_deg))
        centre = math.radians(self.angle_deg)
        return ((centre - darc / skin_radius_m(z - dz), z - dz), (centre + darc / skin_radius_m(z + dz), z + dz))


@dataclass
class Electrodes:
    """The electrodes and the device recording them: the device profile, saved as YAML."""
    electrodes: list[Electrode]
    device: str = "raw"             # raw | myo | otto_bock
    fs: float = 2000.0              # output rate of the raw device (myo and otto_bock use their hardware rate)
    gain: float = 1.0               # overall amplitude: electrode contact and amplification
    noise_uv: float = 1.5           # skin + amplifier noise RMS (dry Myo electrodes: ~5)
    mains_hz: float = 50.0
    mains_uv: float = 2.0
    motion_artifact: float = 1.0    # scale of low-frequency movement artifacts
    channel_gain: list[float] | None = None   # (C,) per-channel gain (contact quality); None = 1
    frontend: Frontend = field(default_factory=Frontend)

    def __post_init__(self) -> None:
        self.electrodes = [e if isinstance(e, Electrode) else Electrode(**e) for e in self.electrodes]
        if isinstance(self.frontend, dict):
            self.frontend = Frontend(**{k: tuple(v) if isinstance(v, list) else v for k, v in self.frontend.items()})
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {' | '.join(DEVICES)}, got {self.device!r}")
        if not self.electrodes:
            raise ValueError("no electrodes")
        for i, e in enumerate(self.electrodes):
            e.name = e.name or f"ch{i}"

    def __len__(self) -> int:
        return len(self.electrodes)

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.electrodes]

    @property
    def fs_out(self) -> float:
        return DEVICE_FS.get(self.device, self.fs)

    # -- presets -----------------------------------------------------------------------------------------------

    @classmethod
    def ring(cls, n: int = 8, position_cm: float = 7.0, rotation_deg: float = 0.0, rings: int = 1,
             ring_gap_mm: float = 35.0, ied_mm: float = 20.0, **kwargs: Any) -> Electrodes:
        """n equally spaced electrodes per ring; ring r is ring_gap_mm further from the elbow and rotated by half
        an electrode spacing per ring."""
        electrodes = [Electrode(angle_deg=_wrap(rotation_deg + 360 * (k + 0.5 * r) / n),
                                position_cm=position_cm + r * ring_gap_mm / 10, ied_mm=ied_mm, name=f"ring{r}_{k}")
                      for r in range(rings) for k in range(n)]
        return cls(electrodes, **kwargs)

    @classmethod
    def myo(cls, position_cm: float = 3.1, rotation_deg: float = 0.0) -> Electrodes:
        """NinaPro DB5: two Myo armbands (8 pods each), the second 3.5 cm distal and rotated 22.5 deg."""
        return cls.ring(8, position_cm, rotation_deg, rings=2, ring_gap_mm=35.0, ied_mm=15.0, device="myo",
                        noise_uv=5.0)

    @classmethod
    def otto_bock(cls, position_cm: float = 2.6, rotation_deg: float = 0.0) -> Electrodes:
        """NinaPro DB1: 8 Otto Bock electrodes around the proximal forearm + 2 over the FDS and EDC bellies."""
        ring = cls.ring(8, position_cm, rotation_deg, device="otto_bock")
        targets = [Electrode(-120.0, 8.4, name="target_FDS"), Electrode(91.0, 8.4, name="target_EDC")]
        return cls(ring.electrodes + targets, device="otto_bock")

    @classmethod
    def preset(cls, name: str) -> Electrodes:
        presets: dict[str, Callable[[], Electrodes]] = {"ring": cls.ring, "myo": cls.myo, "otto_bock": cls.otto_bock}
        if name not in presets:
            raise ValueError(f"unknown electrode preset {name!r}: {' | '.join(presets)}")
        return presets[name]()

    # -- files -------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Electrodes:
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> Electrodes:
        """An electrode YAML file, or a preset name (ring | myo | otto_bock)."""
        if not Path(path).exists() and str(path) in ("ring", "myo", "otto_bock"):
            return cls.preset(str(path))
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0
