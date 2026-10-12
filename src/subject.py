"""Virtual subjects and recording sessions: every source of between-user and between-session variability
is an explicit, logged random parameter (domain randomisation you can later ablate or match to real data)."""
from dataclasses import asdict, dataclass, field

import numpy as np


# Recording devices. "raw": bipolar sEMG at fs_out. The others emulate the NinaPro hardware so synthetic data
# can be compared with (and mixed into) the real datasets channel for channel:
#   otto_bock (DB1): 8 electrodes in a ring at the radio-humeral joint + 2 over the FDS and EDC activity spots,
#                    hardware-rectified and smoothed envelope sampled at 100 Hz (volts)
#   myo (DB5):       two 8-channel Myo armbands, the second distal and rotated 22.5 deg, 200 Hz, int8 counts
DEVICES = ("raw", "otto_bock", "myo")
DEVICE_FS = {"otto_bock": 100, "myo": 200}


@dataclass
class ElectrodeLayout:
    n_channels: int = 8              # total, including targeted electrodes
    rings: int = 1                   # 1 ring, or 2 rings staggered by half an electrode spacing
    level: float = 0.25              # ring position as fraction of forearm length from the elbow
    ring_gap: float = 0.035          # m between rings
    rotation: float = 0.0            # armband rotation (rad)
    angle_jitter: float = 0.0        # per-electrode placement irregularity (rad std)
    targets: tuple = ()              # extra electrodes placed over these muscles' bellies (after the rings)
    target_level: float = 0.3        # fraction of forearm length for the targeted electrodes
    device: str = "raw"


@dataclass
class Subject:
    seed: int
    amputee: bool
    # Anatomy
    forearm_radius: float            # m, skin radius at the elbow
    taper: float                     # radius at the wrist = radius * (1 - taper)
    fat: float                       # subcutaneous fat thickness (m)
    radial_fill: float               # how far out muscle bellies reach (fraction of radius minus fat)
    muscle_spread: float             # transverse jitter of source points (m), muscle width
    residual_length: float           # amputee: residual forearm fraction (1.0 = intact)
    # Volume conductor
    decay: float                     # spatial decay length of bipolar EMG (m)
    # Muscle-level
    muscle_amp: np.ndarray           # (nu,) EMG amplitude factors (size, atrophy, fibre type)
    spectrum_lo: np.ndarray          # (nu,) Hz
    spectrum_hi: np.ndarray          # (nu,) Hz
    amp_exponent: float              # EMG amplitude ~ excitation ** exponent
    recruitment_cost: np.ndarray     # (nu,) relative cost in the muscle-sharing optimisation
    passive_stiffness: float         # scale of passive joint stiffness (capsules, ligaments, resting tone)
    # Electrodes / noise
    layout: ElectrodeLayout = field(default_factory=ElectrodeLayout)
    gain_mv: float = 1.0             # EMG scale (mV at full excitation, superficial muscle)
    noise_uv: float = 5.0            # amplifier + skin noise RMS (uV)
    mains_hz: float = 50.0
    mains_uv: float = 5.0
    device_params: dict = field(default_factory=dict)

    def to_dict(self):
        out = asdict(self)
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
        return out


def sample_layout(rng, device, n_channels, rings, residual, amputee):
    if device == "otto_bock":
        level = rng.uniform(0.04, 0.14)
        return ElectrodeLayout(n_channels=10, rings=1, level=level, rotation=rng.uniform(-np.pi, np.pi),
                               angle_jitter=rng.uniform(0.0, 0.08), targets=("FDS3_r", "EDC3_r"),
                               target_level=rng.uniform(0.25, 0.4), device=device)
    if device == "myo":
        return ElectrodeLayout(n_channels=16, rings=2, level=rng.uniform(0.06, 0.18),
                               ring_gap=rng.uniform(0.03, 0.04), rotation=rng.uniform(-np.pi, np.pi),
                               angle_jitter=rng.uniform(0.0, 0.05), device=device)
    level = rng.uniform(0.15, 0.4) if not amputee else rng.uniform(0.12, max(0.15, residual - 0.12))
    return ElectrodeLayout(n_channels=n_channels, rings=rings, level=level, rotation=rng.uniform(-np.pi, np.pi),
                           angle_jitter=rng.uniform(0.0, 0.08), device=device)


def sample_device_params(rng, device):
    if device == "otto_bock":
        # 13E200-style: rectification + low-pass smoothing; output gain set by the technician's knob.
        return {"smooth_hz": float(rng.uniform(3.0, 6.0)), "volts_per_mv": float(np.exp(rng.normal(np.log(15), 0.5))),
                "ied": float(rng.uniform(0.015, 0.025))}
    if device == "myo":
        return {"counts_per_mv": float(np.exp(rng.normal(np.log(150), 0.4))), "alias": bool(rng.random() < 0.5),
                "adc_noise": float(rng.uniform(0.3, 0.8)),   # counts RMS
                "ied": float(rng.uniform(0.012, 0.018))}     # m between the bipolar contacts of a Myo pod
    return {"ied": float(rng.uniform(0.015, 0.025))}


def sample_subject(rng, nu, extrinsic, n_channels=8, rings=1, amputee=False, device="raw"):
    seed = int(rng.integers(2**31))
    amp = np.exp(rng.normal(0, 0.35, nu))
    residual = 1.0
    if amputee:
        residual = rng.uniform(0.35, 0.8)
        amp = amp * np.exp(rng.normal(-0.3, 0.4, nu))          # disuse atrophy, heterogeneous
        amp[~extrinsic] = 0.0                                    # intrinsic hand muscles are gone
    lo = rng.uniform(30, 70, nu)
    return Subject(
        seed=seed,
        amputee=amputee,
        forearm_radius=rng.uniform(0.036, 0.047) * (0.92 if amputee else 1.0),
        taper=rng.uniform(0.3, 0.45),
        fat=rng.uniform(0.002, 0.008),
        radial_fill=rng.uniform(0.8, 0.97),
        muscle_spread=rng.uniform(0.002, 0.006),
        residual_length=residual,
        # Pickup depth depends on electrode size / spacing: small Otto Bock electrodes see more local signal.
        decay=rng.uniform(0.005, 0.010) if device == "otto_bock" else rng.uniform(0.008, 0.016),
        muscle_amp=amp,
        spectrum_lo=lo,
        spectrum_hi=lo + rng.uniform(60, 200, nu),
        amp_exponent=rng.uniform(0.6, 1.0),
        recruitment_cost=np.exp(rng.normal(0, 0.25, nu)),
        passive_stiffness=float(np.exp(rng.normal(0, 0.3))),
        layout=sample_layout(rng, device, n_channels, rings, residual, amputee),
        # Superficial muscle at full drive ~0.5-3 mV RMS; amplifier+skin noise ~1-3 uV RMS: ~60 dB dynamic range.
        gain_mv=float(np.exp(rng.normal(np.log(1.5), 0.4))),
        noise_uv=rng.uniform(3.0, 8.0) if device == "myo" else rng.uniform(0.7, 3.0),   # dry electrodes: noisier
        mains_hz=float(rng.choice([50.0, 60.0])),
        mains_uv=float(rng.exponential(2.0)),
        device_params=sample_device_params(rng, device),
    )


@dataclass
class Session:
    """Per-recording perturbations around the subject: re-donning the armband, skin/sweat, fatigue."""
    rotation_shift: float
    level_shift: float
    channel_gain: np.ndarray
    noise_scale: float
    bad_channel: int                 # -1 = none
    motion_artifact: float           # scale of low-frequency movement artifacts
    gravity_dir: np.ndarray          # arm orientation (unit vector, world frame)
    cocontraction_mean: float
    load_prob: float
    elbow_angle: float               # rad, held (with slow drift) during the session; limb-position effect
    arm_support: float               # fraction of static torque (gravity + passive) on the elbow carried by a support
                                     # (armrest); 70 % of it on pro/sup (surface friction)
    wrist_support: float             # same for the wrist (hand partly resting)
    cocontraction_move_gain: float   # extra co-contraction proportional to movement speed (stiffening while moving)
    hold_tone: float                 # sustained co-activation while holding a posture away from rest

    def to_dict(self):
        out = asdict(self)
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
        return out


def sample_session(rng, subject: Subject, support="random"):
    """support: "supported" (forearm on an armrest, as in NinaPro), "free" (arm held up), or "random" (50/50)."""
    C = subject.layout.n_channels
    g = rng.normal(size=3)
    supported = support == "supported" or (support == "random" and rng.random() < 0.5)
    return Session(
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
