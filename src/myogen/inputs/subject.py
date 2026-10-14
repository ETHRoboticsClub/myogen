from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class Subject:
    """The simulated person. The defaults are an average adult; Subject.random(seed) draws one from the
    population ranges in the comments (the middle of each range is the default)."""
    # Anatomy
    forearm_radius_mm: float = 41.5   # skin radius at the elbow; population [36, 47]
    taper: float = 0.375              # wrist radius = elbow radius * (1 - taper); [0.3, 0.45]
    fat_mm: float = 5.0               # subcutaneous fat; [2, 8]
    radial_fill: float = 0.885        # how far out the muscle bellies reach (fraction of radius minus fat)
    muscle_spread_mm: float = 4.0     # muscle width (transverse spread of the EMG sources); [2, 6]
    amputee: bool = False             # transradial: intrinsic hand muscles absent, extrinsic ones atrophied
    residual_length: float = 1.0      # fraction of the forearm left (amputees); [0.35, 0.8]
    # Posture
    elbow_angle_deg: float = 72.0     # used when the motion has no elbow angle; [40, 103]
    armrest: bool = True              # forearm resting on a support (as in NinaPro) or held up
    # World gravity direction (projected perpendicular to the forearm on an armrest).
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0)
    # Muscles and EMG
    gain_mv: float = 1.5              # EMG amplitude: mV RMS of a superficial muscle at full drive (lognormal)
    amp_exponent: float = 0.8         # EMG amplitude ~ drive ** exponent; [0.6, 1]
    spectrum_hz: tuple[float, float] = (60.0, 215.0)  # band of the EMG source noise
    passive_stiffness: float = 1.0    # scale of the passive joint stiffness (lognormal)
    muscle_amp: list[float] | None = None        # (nu,) per-muscle EMG amplitude factors (None = 1)
    recruitment_cost: list[float] | None = None  # (nu,) relative cost of recruiting each muscle (None = 1)
    # Behaviour: effort beyond the mechanical minimum, as co-activation of the muscles of the engaged joints
    cocontraction: float = 0.02       # baseline co-activation while the hand is engaged (with random episodes)
    hold_tone: float = 0.15           # co-activation while moving or holding a posture away from rest; [0.05, 0.25]
    move_cocontraction: float = 0.125  # extra co-activation proportional to movement speed; [0, 0.25]
    grip_load: float = 0.25           # probability of squeezing episodes while the hand is closed; [0, 0.5]
    seed: int = 0                     # randomness of the EMG noise and of the effort episodes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Subject:
        return cls(**{k: tuple(v) if k in ("gravity", "spectrum_hz") else v for k, v in data.items()})

    def muscle_amplitudes(self, extrinsic: np.ndarray) -> np.ndarray:
        """(nu,) EMG amplitude factor per muscle; amputees lose their intrinsic hand muscles."""
        amp = np.ones(len(extrinsic)) if self.muscle_amp is None else np.asarray(self.muscle_amp, float).copy()
        if self.amputee:
            amp[~np.asarray(extrinsic)] = 0.0
        return amp

    @classmethod
    def random(cls, seed: int, amputee: bool = False, n_muscles: int = 48) -> Subject:
        """A subject drawn from the population ranges in the field comments."""
        rng = np.random.default_rng(seed)
        amp = np.exp(rng.normal(0, 0.35, n_muscles))
        if amputee:
            amp *= np.exp(rng.normal(-0.3, 0.4, n_muscles))      # disuse atrophy
        lo = float(rng.uniform(40, 80))
        return cls(
            forearm_radius_mm=float(rng.uniform(36, 47)) * (0.92 if amputee else 1.0),
            taper=float(rng.uniform(0.3, 0.45)),
            fat_mm=float(rng.uniform(2, 8)),
            radial_fill=float(rng.uniform(0.8, 0.97)),
            muscle_spread_mm=float(rng.uniform(2, 6)),
            amputee=amputee,
            residual_length=float(rng.uniform(0.35, 0.8)) if amputee else 1.0,
            elbow_angle_deg=float(rng.uniform(40, 103)),
            armrest=bool(rng.random() < 0.5),
            gain_mv=float(np.exp(rng.normal(np.log(1.5), 0.4))),
            amp_exponent=float(rng.uniform(0.6, 1.0)),
            spectrum_hz=(lo, lo + float(rng.uniform(85, 225))),
            passive_stiffness=float(np.exp(rng.normal(0, 0.3))),
            muscle_amp=amp.tolist(),
            recruitment_cost=np.exp(rng.normal(0, 0.25, n_muscles)).tolist(),
            cocontraction=float(min(rng.exponential(0.02), 0.08)),
            hold_tone=float(rng.uniform(0.05, 0.25)),
            move_cocontraction=float(rng.uniform(0.0, 0.25)),
            grip_load=float(rng.uniform(0.0, 0.5)),
            seed=int(rng.integers(2**31)),
        )
