from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class MusculoskeletalParams:
    """Loads on the skeleton besides the muscles (simulation/musculoskeletal.py)."""
    # Passive joint stiffness (Nm/rad) around the relaxed posture: capsules, ligaments, resting muscle tone.
    # Finger joints are looked up by prefix (mcp, mcp_abd, pm, md).
    passive_stiffness: dict[str, float] = field(default_factory=lambda: {
        "pro_sup_r": 0.1, "deviation_r": 0.25, "flexion_r": 0.2, "cmc_flexion_r": 0.03, "cmc_abduction_r": 0.03,
        "mp_flexion_r": 0.02, "ip_flexion_r": 0.01, "mcp": 0.04, "mcp_abd": 0.02, "pm": 0.02, "md": 0.01})
    spring_offset_max: float = 0.6        # rad, max shift of a spring's reference to balance gravity at rest
    # Fraction of the static joint torque carried by the support: (elbow, wrist) on an armrest / arm held up.
    armrest_support: tuple[float, float] = (0.925, 0.6)
    held_support: tuple[float, float] = (0.15, 0.0)
    pronation_support: float = 0.7        # pronation support relative to the elbow's
    # Torque opposing finger flexion at full grip load (Nm, ~5-10 N fingertip force), per finger joint.
    grip_torque: dict[str, float] = field(default_factory=lambda: {"mcp": 0.35, "pm": 0.18, "md": 0.08})
    grip_contact_closure: float = 0.6     # finger closure at which the grip torque is fully felt


@dataclass(frozen=True)
class ControlParams:
    """The reference trajectory and how the muscles track it (simulation/muscle_activity.py)."""
    fs: float = 100.0                     # Hz, control rate and label rate
    warmup_s: float = 0.5                 # first posture held so the controller settles before recording
    min_rest_s: float = 1.0               # rest needed in a motion to take its median posture as the relaxed one
    joint_limit_tolerance: float = 0.05   # rad beyond a joint limit before a warning (angles are clipped)
    pd_bandwidth_hz: float = 8.0          # PD tracking bandwidth
    pd_damping: float = 1.0               # PD damping ratio
    effort_weight: float = 3e-3           # regularisation of activations (effort / co-contraction)
    smooth_weight: float = 1.0            # penalty on activation change between ticks (smooth neural drive)
    smooth_leak: float = 0.9              # pull towards leak * previous activation: unneeded drive decays ~100 ms
    horizon_ticks: int = 3                # ticks over which the activation dynamics are inverted
    acc_scale: float = 200.0              # rad/s^2 of acceleration error that costs as much as effort_weight
    reserve_cost: float = 1.0             # reserve torques (what the muscles cannot do): weight
    reserve_max_nm: float = 0.5           # ... and limit per joint
    # Thumb joints lack intrinsic muscles: cheaper, larger reserves stand in for them (without producing EMG).
    weak_joints: tuple[str, ...] = ("cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r")
    weak_reserve_cost: float = 0.1
    weak_reserve_max_nm: float = 3.0
    max_speed: float = 200.0              # rad/s: beyond this the simulation has diverged (reset to the reference)
    drive_lowpass_hz: float = 4.0         # neural drive smoothing (the controller's fast corrections are artifacts)


@dataclass(frozen=True)
class EffortParams:
    """Effort beyond the mechanical minimum: co-activation and grip (simulation/muscle_activity.py). Speeds and
    distances are in joint-range fractions; the subject sets the levels (Subject.cocontraction, hold_tone, ...)."""
    pip_closed_rad: float = 1.5           # PIP angle of a fully closed finger
    speed_lowpass_hz: float = 2.0         # smoothing of joint speed and engagement
    speed_deadzone: float = 0.15          # /s, below this a joint is not moving (sensor jitter)
    speed_max: float = 1.5                # /s, speed beyond which co-activation stops growing
    speed_engaged: float = 0.5            # /s, speed at which a joint counts as fully engaged
    return_effort: float = 1.0            # effort of moving back towards the relaxed posture, relative to moving away
    finger_coupling: float = 0.0          # share of a finger's engagement reaching its neighbour (^2 the next one)
    rest_effort: float = 1.0              # effort left where the motion is labelled rest (HandMotion.rest)
    rest_excitation: float = 1.0          # share of the controller's excitation in the EMG drive at labelled rest
    lead_s: float = 0.0                   # s, the effort precedes the movement (electromechanical delay)
    muscle_variability: float = 0.0       # log-sd of each muscle's own slow effort fluctuations (0: none)
    muscle_variability_hz: float = 0.5    # their bandwidth
    # Effort per unit of engagement, by joint name prefix (others: 1): holding the wrist away from rest takes more
    # effort than the same fraction of a finger's range.
    joint_effort: dict[str, float] = field(default_factory=dict)
    posture_deadzone: float = 0.1         # distance from the relaxed posture that is still relaxed
    posture_engaged: float = 0.4          # distance beyond the dead zone at which a joint is fully engaged
    cocontraction_peak: float = 0.3       # co-activation level reached in an episode
    cocontraction_episode_prob: float = 0.25
    closed_hand: float = 0.5              # mean finger closure at which the hand counts as closed (can squeeze)
    closed_hand_width: float = 0.05       # sharpness of that transition
    episode_s: tuple[float, float] = (0.5, 3.0)   # duration range of effort episodes
    episode_smoothing_s: float = 0.3


@dataclass(frozen=True)
class EMGParams:
    """From muscle activity to the signal at the electrodes (simulation/emg.py)."""
    n_sources: int = 6                    # point sources per muscle belly
    decay_mm: float = 12.0                # distance over which a source's gain at a pole falls by 1/e
    conduction_velocity: float = 4.0      # m/s along the fibres (sets the delay between the two poles)
    near_field_mm: float = float("inf")   # distance over which a source's signal stops travelling under both poles
                                          # (beyond it, it is common to them: bipolar rejection); inf: never
    fs: float = 2000.0                    # Hz, output sampling rate
    geometry_hz: float = 25.0             # Hz, rate of the posture-dependent source-electrode geometry (interpolated)
    filter_order: int = 2                 # band-pass of the source noise (band: Subject.spectrum_hz)
    superficial_depth_m: float = 0.004    # muscle depth below the fat used to normalise amplitudes
    elbow_margin_m: float = 0.02          # sources up to this far proximal of the elbow still count
    noise_to_mv: float = 3.0              # unit-variance source noise -> mV (before Subject.gain_mv)
    # Calibration against the point-electrode model this replaces (same spatial selectivity at decay 12 mm):
    # the bipolar derivation is 1.65x stronger, and it lowers the spectrum, compensated by Subject.spectrum_hz.
    bipolar_scale: float = 0.6


@dataclass(frozen=True)
class AcquisitionParams:
    """Shape of the recording disturbances (simulation/acquisition.py); their levels are device settings
    (Electrodes.noise_uv, mains_uv, motion_artifact)."""
    mains_modulation: float = 0.3         # relative slow modulation of the mains amplitude
    mains_modulation_hz: tuple[float, float] = (0.05, 0.3)
    mains_channel_spread: tuple[float, float] = (0.5, 1.5)   # per-channel mains amplitude factor
    mains_third_harmonic: float = 0.3     # relative to the fundamental
    artifact_lowpass_hz: float = 8.0      # movement artifacts are low-frequency
    artifact_mv: float = 0.03             # artifact level per rad/s of forearm / wrist speed
    amplifier_order: int = 4              # amplifier band-pass (band: Frontend.amplifier_hz)


@dataclass(frozen=True)
class QCParams:
    """Quality control: the simulated hand did what the motion says (simulator.py)."""
    max_rmse_deg: float = 5.0
    max_wrist_reserve_nm: float = 0.02
    max_saturation: float = 0.05          # fraction of saturated excitation samples
    saturated_above: float = 0.99         # excitation counted as saturated


@dataclass(frozen=True)
class SimParams:
    """All model parameters. `replace(stage={field: value})` returns a copy with some of them changed."""
    musculoskeletal: MusculoskeletalParams = field(default_factory=MusculoskeletalParams)
    control: ControlParams = field(default_factory=ControlParams)
    effort: EffortParams = field(default_factory=EffortParams)
    emg: EMGParams = field(default_factory=EMGParams)
    acquisition: AcquisitionParams = field(default_factory=AcquisitionParams)
    qc: QCParams = field(default_factory=QCParams)

    def replace(self, **stages: dict[str, Any]) -> SimParams:
        return replace(self, **{name: replace(getattr(self, name), **values) for name, values in stages.items()})
