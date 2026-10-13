"""Virtual subjects and recording sessions: every between-user and between-session difference is an explicit,
stored random parameter (domain randomisation that can be ablated or matched to real data)."""
import math
from dataclasses import asdict, dataclass, field

import numpy as np

# Interpretable parameters a caller may fix (x) or range ([lo, hi]) instead of the built-in priors (SI units).
SUBJECT_OVERRIDES = ("residual_length", "fat", "forearm_radius", "gain_mv", "noise_uv")
SESSION_OVERRIDES = ("elbow_angle", "load_prob", "cocontraction_mean")


class _Params:
    """to_dict / from_dict for the parameter dataclasses (arrays <-> lists, as stored with every trial)."""

    def to_dict(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in params.items()})


@dataclass
class ElectrodeLayout:
    n_channels: int = 8              # total, including targeted electrodes
    rings: int = 1                   # rings after the first are staggered by half an electrode spacing
    level: float = 0.25              # ring position, fraction of forearm length from the elbow
    ring_gap: float = 0.035          # m between rings
    rotation: float = 0.0            # armband rotation (rad)
    angle_jitter: float = 0.0        # per-electrode placement irregularity (rad std)
    targets: tuple = ()              # extra electrodes over these muscles' bellies (after the rings)
    target_level: float = 0.3        # fraction of forearm length of the targeted electrodes
    device: str = "raw"

    @property
    def per_ring(self):
        return (self.n_channels - len(self.targets)) // self.rings

    def channel_names(self):
        return ([f"ring{r}_{k}" for r in range(self.rings) for k in range(self.per_ring)]
                + [f"target_{t}" for t in self.targets])


@dataclass
class Subject(_Params):
    seed: int
    amputee: bool
    # Anatomy
    forearm_radius: float            # m, skin radius at the elbow
    taper: float                     # radius at the wrist = forearm_radius * (1 - taper)
    fat: float                       # m, subcutaneous fat
    radial_fill: float               # how far out the muscle bellies reach (fraction of radius minus fat)
    muscle_spread: float             # m, transverse jitter of the source points (muscle width)
    residual_length: float           # fraction of the forearm left (1.0 = intact)
    decay: float                     # m, spatial decay length of bipolar EMG
    # Per muscle (nu,)
    muscle_amp: np.ndarray           # EMG amplitude factors (size, atrophy, fibre type)
    spectrum_lo: np.ndarray          # Hz
    spectrum_hi: np.ndarray          # Hz
    amp_exponent: float              # EMG amplitude ~ excitation ** exponent
    recruitment_cost: np.ndarray     # relative cost in the muscle-sharing optimisation
    passive_stiffness: float         # scale of the passive joint stiffness
    # Electrodes and noise
    layout: ElectrodeLayout = field(default_factory=ElectrodeLayout)
    gain_mv: float = 1.0             # mV RMS of a superficial muscle at full excitation
    noise_uv: float = 5.0            # amplifier + skin noise RMS
    mains_hz: float = 50.0
    mains_uv: float = 5.0
    device_params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        layout = params.pop("layout")
        subject = super().from_dict(params)
        subject.layout = ElectrodeLayout(**{**layout, "targets": tuple(layout["targets"])})
        return subject


@dataclass
class Session(_Params):
    """Per-recording changes around the subject: re-donning the armband, skin, posture, support, effort."""
    rotation_shift: float            # rad, armband rotation added to the layout's
    level_shift: float               # forearm-length fraction added to the layout's
    channel_gain: np.ndarray         # (C,)
    noise_scale: float
    bad_channel: int                 # -1 = none
    motion_artifact: float           # scale of low-frequency movement artifacts
    gravity_dir: np.ndarray          # unit vector, world frame
    cocontraction_mean: float
    load_prob: float                 # probability of grip-load episodes while the hand is closed
    elbow_angle: float               # rad, held (with slow drift) during the session
    arm_support: float               # fraction of the static elbow torque carried by an armrest (70 % on pro/sup)
    wrist_support: float             # same for the wrist (hand partly resting)
    cocontraction_move_gain: float   # extra co-contraction proportional to movement speed
    hold_tone: float                 # co-activation while holding a posture away from rest


def draw(rng, v, scale=1.0):
    """Config value -> sample: [lo, hi] is drawn uniformly, a number is fixed (and consumes no draw)."""
    if isinstance(v, (list, tuple)):
        return float(rng.uniform(v[0] * scale, v[1] * scale))
    return float(v) * scale


def channel_count(spec):
    return spec["per_ring"] * spec["rings"] + len(spec.get("targets", ()))


def sample_layout(rng, device, spec, residual, amputee):
    """spec: an electrodes.<device> config block (angles in degrees)."""
    level = spec["level"]
    if amputee and "level_amputee" in spec:
        # The ring must stay on the stump, at least amputee_margin above its end.
        lo, hi = spec["level_amputee"]
        level = [lo, max(hi, residual - spec["amputee_margin"])]
    deg = math.pi / 180
    targets = tuple(spec.get("targets", ()))
    return ElectrodeLayout(
        n_channels=channel_count(spec), rings=spec["rings"], level=draw(rng, level),
        ring_gap=draw(rng, spec["ring_gap"]), rotation=draw(rng, spec["rotation_deg"], deg),
        angle_jitter=draw(rng, spec["angle_jitter_deg"], deg), targets=targets,
        target_level=draw(rng, spec["target_level"]) if targets else 0.3, device=device)


def sample_device_params(rng, device, spec):
    if device == "otto_bock":
        # 13E200-style: rectification + low-pass smoothing; output gain set by the technician's knob.
        return {"smooth_hz": float(rng.uniform(3.0, 6.0)), "volts_per_mv": float(np.exp(rng.normal(np.log(15), 0.5))),
                "ied": draw(rng, spec["ied"])}
    if device == "myo":
        return {"counts_per_mv": float(np.exp(rng.normal(np.log(150), 0.4))), "alias": bool(rng.random() < 0.5),
                "adc_noise": float(rng.uniform(0.3, 0.8)),     # counts RMS
                "ied": draw(rng, spec["ied"])}
    return {"ied": draw(rng, spec["ied"])}


def sample_subject(rng, nu, extrinsic, electrodes, amputee=False, device="raw", overrides=None, orng=None):
    """electrodes: the electrodes.<device> config block. overrides: {name in SUBJECT_OVERRIDES: value},
    drawn with the separate generator `orng` so fixing one parameter does not change the others."""
    overrides = overrides or {}
    seed = int(rng.integers(2**31))
    amp = np.exp(rng.normal(0, 0.35, nu))
    residual = 1.0
    if amputee:
        residual = rng.uniform(0.35, 0.8)
        if overrides.get("residual_length") is not None:       # needed now: the electrode layout depends on it
            residual = draw(orng, overrides["residual_length"])
        amp = amp * np.exp(rng.normal(-0.3, 0.4, nu))            # disuse atrophy
        amp[~extrinsic] = 0.0                                      # intrinsic hand muscles are gone
    lo = rng.uniform(30, 70, nu)
    subject = Subject(
        seed=seed,
        amputee=amputee,
        forearm_radius=rng.uniform(0.036, 0.047) * (0.92 if amputee else 1.0),
        taper=rng.uniform(0.3, 0.45),
        fat=rng.uniform(0.002, 0.008),
        radial_fill=rng.uniform(0.8, 0.97),
        muscle_spread=rng.uniform(0.002, 0.006),
        residual_length=residual,
        # Small Otto Bock electrodes see more local signal.
        decay=rng.uniform(0.005, 0.010) if device == "otto_bock" else rng.uniform(0.008, 0.016),
        muscle_amp=amp,
        spectrum_lo=lo,
        spectrum_hi=lo + rng.uniform(60, 200, nu),
        amp_exponent=rng.uniform(0.6, 1.0),
        recruitment_cost=np.exp(rng.normal(0, 0.25, nu)),
        passive_stiffness=float(np.exp(rng.normal(0, 0.3))),
        layout=sample_layout(rng, device, electrodes, residual, amputee),
        gain_mv=float(np.exp(rng.normal(np.log(1.5), 0.4))),
        noise_uv=rng.uniform(3.0, 8.0) if device == "myo" else rng.uniform(0.7, 3.0),   # dry electrodes: noisier
        mains_hz=float(rng.choice([50.0, 60.0])),
        mains_uv=float(rng.exponential(2.0)),
        device_params=sample_device_params(rng, device, electrodes),
    )
    return _override(subject, {k: v for k, v in overrides.items() if k != "residual_length"}, orng)


def sample_session(rng, subject, support="random", overrides=None, orng=None):
    """support: supported (armrest, as in NinaPro) | free (arm held up) | random (50/50).
    overrides: {name in SESSION_OVERRIDES: value}, drawn with `orng` (see sample_subject)."""
    C = subject.layout.n_channels
    g = rng.normal(size=3)
    supported = support == "supported" or (support == "random" and rng.random() < 0.5)
    session = Session(
        rotation_shift=rng.normal(0, 0.15),
        level_shift=rng.normal(0, 0.03),
        channel_gain=np.exp(rng.normal(0, 0.15, C)),
        noise_scale=float(np.exp(rng.normal(0, 0.3))),
        bad_channel=int(rng.integers(C)) if rng.random() < 0.05 else -1,
        motion_artifact=float(rng.exponential(1.0)),
        gravity_dir=g / np.linalg.norm(g),
        cocontraction_mean=float(min(rng.exponential(0.02), 0.08)),
        load_prob=float(rng.uniform(0.0, 0.5)),
        elbow_angle=float(rng.uniform(0.7, 1.8)),
        arm_support=float(rng.uniform(0.85, 1.0) if supported else rng.uniform(0.0, 0.3)),
        wrist_support=float(rng.uniform(0.3, 0.9) if supported else 0.0),
        cocontraction_move_gain=float(rng.uniform(0.0, 0.25)),
        hold_tone=float(rng.uniform(0.05, 0.25)),
    )
    return _override(session, overrides, orng)


def _override(obj, overrides, orng):
    for k, v in (overrides or {}).items():
        if v is not None:
            setattr(obj, k, draw(orng, v))
    return obj
