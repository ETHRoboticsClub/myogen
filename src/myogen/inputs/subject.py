from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

FIBRE_AMPLITUDE_EXPONENT = 0.5      # unit action potential ~ strength ** this
MUSCLE_RADIUS_MM = 36.5             # elbow radius under the fat at strength 1
# (strength, fat_mm) for Subject.preset
PRESETS = {"reference": (1.0, 5.0), "low_strength": (0.7, 5.0), "high_strength": (1.3, 5.0), "low_fat": (1.0, 2.0),
           "high_fat": (1.0, 9.0)}


@dataclass
class Subject:
    """The simulated person. The defaults are an average adult; Subject.random(seed) draws one from the
    population ranges in the comments (the middle of each range is the default)."""
    # Anatomy
    forearm_radius_mm: float = 41.5   # skin radius at the elbow; population [36, 47]
    taper: float = 0.375              # wrist radius = elbow radius * (1 - taper); [0.3, 0.45]
    fat_mm: float = 5.0               # subcutaneous fat; [2, 8]
    radial_fill: float = 0.885        # how far out the muscle bellies reach (fraction of radius minus fat)
    amputee: bool = False             # transradial: intrinsic hand muscles absent, extrinsic ones atrophied
    # Muscles relative to the model and the atlas (per-muscle lists: (nu,), None = unchanged)
    strength: float = 1.0             # maximal force of every muscle; [0.7, 1.3]
    muscle_strength: list[float] | None = None
    muscle_size: list[float] | None = None
    muscle_shift_deg: list[float] | None = None
    muscle_depth: list[float] | None = None
    section_rotation_deg: float = 0.0  # [-15, 15]
    residual_length: float = 1.0      # fraction of the forearm left (amputees); [0.35, 0.8]
    # Posture
    elbow_angle_deg: float = 72.0     # used when the motion has no elbow angle; [40, 103]
    armrest: bool = True              # forearm resting on a support (as in NinaPro) or held up
    # World gravity direction (projected perpendicular to the forearm on an armrest).
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0)
    # Muscles and EMG
    gain_mv: float = 1.5              # EMG amplitude: mV RMS of a superficial muscle at full drive (lognormal)
    passive_stiffness: float = 1.0    # scale of the passive joint stiffness (lognormal)
    muscle_amp: list[float] | None = None        # (nu,) per-muscle EMG amplitude factors (None = 1)
    recruitment_cost: list[float] | None = None  # (nu,) relative cost of recruiting each muscle (None = 1)
    # Behaviour: effort beyond the mechanical minimum, as co-activation of the muscles of the engaged joints
    cocontraction: float = 0.02       # baseline co-activation while the hand is engaged (with random episodes)
    hold_tone: float = 0.15           # co-activation while moving or holding a posture away from rest; [0.05, 0.25]
    move_cocontraction: float = 0.125  # extra co-activation proportional to movement speed; [0, 0.25]
    antagonist_ratio: float = 0.25    # co-activation of the antagonists relative to the agonists; [0.1, 0.4]
    grip_load: float = 0.0            # probability of squeezing episodes while the hand is closed (0: free movement)
    rest_tone: float = 0.02           # drive every muscle keeps at rest (real forearms are not silent); [0.01, 0.04]
    seed: int = 0                     # randomness of the EMG noise and of the effort episodes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Subject:
        known = {f.name for f in fields(cls)}          # fields of older versions are dropped
        return cls(**{k: tuple(v) if k == "gravity" else v for k, v in data.items() if k in known})

    def per_muscle(self, name: str, n: int, default: float) -> np.ndarray:
        """(n,) a per-muscle field, its default where None."""
        value = getattr(self, name)
        return np.full(n, default) if value is None else np.asarray(value, float)

    def muscle_forces(self, n: int) -> np.ndarray:
        """(n,) maximal force factor per muscle: strength times the per-muscle factors."""
        return self.strength * self.per_muscle("muscle_strength", n, 1.0)

    def muscle_amplitudes(self, extrinsic: np.ndarray) -> np.ndarray:
        """(nu,) EMG amplitude factor per muscle (muscle_amp, and the unit size of a stronger muscle: larger fibres,
        larger action potentials); amputees lose their intrinsic hand muscles."""
        amp = np.ones(len(extrinsic)) if self.muscle_amp is None else np.asarray(self.muscle_amp, float).copy()
        amp *= self.muscle_forces(len(extrinsic)) ** FIBRE_AMPLITUDE_EXPONENT
        if self.amputee:
            amp[~np.asarray(extrinsic)] = 0.0
        return amp

    @classmethod
    def build(cls, strength: float = 1.0, fat_mm: float = 5.0, **fields: Any) -> Subject:
        """A subject of this strength and fat; the forearm radius follows: fat plus the muscle radius, which grows
        with sqrt(strength) (force ~ cross-sectional area)."""
        return cls(strength=strength, fat_mm=fat_mm, forearm_radius_mm=fat_mm + MUSCLE_RADIUS_MM * np.sqrt(strength),
                   **fields)

    @classmethod
    def preset(cls, name: str, **fields: Any) -> Subject:
        """Subject.build with the (strength, fat_mm) of one of PRESETS."""
        if name not in PRESETS:
            raise ValueError(f"preset must be one of {' | '.join(PRESETS)}, got {name!r}")
        return cls.build(*PRESETS[name], **fields)

    @classmethod
    def random(cls, seed: int, amputee: bool = False, n_muscles: int = 48) -> Subject:
        """A subject drawn from the population ranges in the field comments."""
        rng = np.random.default_rng(seed)
        amp = np.exp(rng.normal(0, 0.35, n_muscles))
        if amputee:
            amp *= np.exp(rng.normal(-0.3, 0.4, n_muscles))      # disuse atrophy
        return cls(
            forearm_radius_mm=float(rng.uniform(36, 47)) * (0.92 if amputee else 1.0),
            taper=float(rng.uniform(0.3, 0.45)),
            fat_mm=float(rng.uniform(2, 8)),
            radial_fill=float(rng.uniform(0.8, 0.97)),
            amputee=amputee,
            residual_length=float(rng.uniform(0.35, 0.8)) if amputee else 1.0,
            elbow_angle_deg=float(rng.uniform(40, 103)),
            armrest=bool(rng.random() < 0.5),
            gain_mv=float(np.exp(rng.normal(np.log(1.5), 0.4))),
            passive_stiffness=float(np.exp(rng.normal(0, 0.3))),
            muscle_amp=amp.tolist(),
            recruitment_cost=np.exp(rng.normal(0, 0.25, n_muscles)).tolist(),
            cocontraction=float(min(rng.exponential(0.02), 0.08)),
            hold_tone=float(rng.uniform(0.05, 0.25)),
            move_cocontraction=float(rng.uniform(0.0, 0.25)),
            seed=int(rng.integers(2**31)),
            antagonist_ratio=float(rng.uniform(0.1, 0.4)),   # drawn after the others: their draws stay as they were
            rest_tone=float(rng.uniform(0.01, 0.04)),
            # drawn last, so the draws above stay as they were
            strength=float(rng.uniform(0.7, 1.3)),
            muscle_strength=np.exp(rng.normal(0, 0.15, n_muscles)).tolist(),
            muscle_size=np.exp(rng.normal(0, 0.15, n_muscles)).tolist(),
            muscle_shift_deg=rng.normal(0, 5.0, n_muscles).tolist(),
            muscle_depth=np.exp(rng.normal(0, 0.1, n_muscles)).tolist(),
            section_rotation_deg=float(rng.uniform(-15, 15)),
        )
