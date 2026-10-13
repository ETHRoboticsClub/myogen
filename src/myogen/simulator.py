"""Simulator: hand motion in, sEMG recording out.

    motion (joint angles)
      -> ForearmModel       joint trajectory, gravity, armrest, passive springs at the relaxed posture
      -> MuscleController   muscle excitations that make the model follow the trajectory
      -> engine             clean bipolar sEMG at the electrodes (analytic | biomime)
      -> Electrodes.record  noise, amplifier, device
      -> Recording
run() = record(simulate_clean()).
"""
import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .body import HAND_JOINTS, ForearmModel
from .electrodes import Electrodes
from .engines import CleanEMG, make_engine
from .motion import REST_POSTURE, HandMotion
from .muscles import MuscleActivity, MuscleController
from .recording import Recording
from .subject import Subject

FS_CTRL = 100.0             # Hz: control rate and label rate
WARMUP = 0.5                # s: the first posture is held so the controller settles before the recording starts
LIMIT_TOLERANCE = 0.05      # rad beyond a joint limit before a warning (angles are clipped to the limits)

# Quality control: the simulated hand did what the motion says.
QC_MAX_RMSE_DEG = 5.0
QC_MAX_WRIST_RESERVE_NM = 0.02
QC_MAX_SATURATION = 0.05

log = logging.getLogger(__name__)


@dataclass
class CleanRun:
    """A simulated motion before the recording device: the muscle activity, its clean EMG and the labels."""
    activity: MuscleActivity
    clean: CleanEMG
    labels: dict
    filled_joints: list


class Simulator:
    def __init__(self, electrodes="myo", engine="analytic", subject=None, **engine_options):
        """electrodes: an Electrodes or a preset name (ring | myo | otto_bock); engine: "analytic" | "biomime" or
        an EMGEngine (engine_options go to its constructor, e.g. muap_bank=..., gain=...); subject: a Subject
        (default: the average subject)."""
        self.electrodes = electrodes if isinstance(electrodes, Electrodes) else Electrodes.preset(electrodes)
        self.subject = subject or Subject()
        self.model = ForearmModel()
        self.engine = make_engine(engine, **engine_options)
        self.engine.prepare(self.model, self.subject, self.electrodes)

    def with_electrodes(self, electrodes):
        """A simulator with the same model, subject and engine settings, and other electrodes."""
        other = copy.copy(self)
        other.electrodes, other.engine = electrodes, copy.copy(self.engine)
        other.engine.prepare(self.model, self.subject, electrodes)
        return other

    def run(self, motion, on_tick=None):
        """Simulate a HandMotion (or a motion file). Returns a Recording."""
        return self.record(self.simulate_clean(motion, on_tick))

    def simulate_clean(self, motion, on_tick=None):
        """The expensive part: motion -> muscle activity -> clean EMG at the electrodes. Returns a CleanRun, which
        record() turns into a Recording (calibrate() records one CleanRun with many device settings)."""
        motion = motion if isinstance(motion, HandMotion) else HandMotion.load(motion)
        sub = self.subject
        effort_rng = np.random.default_rng(np.random.SeedSequence([sub.seed, 0]))
        t0 = time.time()
        log.info(f"simulating {motion.duration:.1f} s: {self.engine.name} engine, {len(self.electrodes)} electrodes "
                 f"({self.electrodes.device})")

        q_ref, q_rest, hand = self.joint_trajectory(motion.resampled(FS_CTRL))
        co_activation, grip = effort(self.model, q_ref, q_rest, sub, effort_rng)
        fibre_rest = self.model.prepare(sub, q_ref[0], q_rest)
        controller = MuscleController(self.model, sub.recruitment_cost, FS_CTRL)
        activity = controller.track(q_ref, self.model.grip_torque(grip, finger_closure(hand)), self.model.support(sub),
                                    fibre_rest, effort=co_activation, on_tick=on_tick)

        start = int(WARMUP * FS_CTRL)
        activity = activity[start:]
        q_ref_next = np.vstack([q_ref[1:], q_ref[-1:]])[start:]     # labels are the state after each tick
        labels = {"q": activity.q, "q_ref": q_ref_next, "qd": activity.qd, "excitation": activity.excitation,
                  "activation": activity.activation, "reserve": activity.reserve,
                  "fibre_length": activity.fibre_length, "effort": activity.effort, "grip_load": grip[start:]}
        run = self.regenerate(CleanRun(activity, None, labels, list(motion.filled)))
        log.info(f"done in {time.time() - t0:.0f} s")
        return run

    def regenerate(self, run):
        """The CleanRun of the same muscle activity at this simulator's electrodes (re-runs only the engine)."""
        emg_rng = np.random.default_rng(np.random.SeedSequence([self.subject.seed, 1]))
        return CleanRun(run.activity, self.engine.generate(run.activity, emg_rng), run.labels, run.filled_joints)

    def record(self, run, electrodes=None):
        """The Recording of a CleanRun through the electrodes' device (default: the simulator's). Other
        electrodes must have the same positions: only the device settings (gain, noise, frontend) may differ."""
        electrodes = electrodes or self.electrodes
        if [(e.angle_deg, e.position_cm, e.ied_mm, e.orientation_deg) for e in electrodes.electrodes] != \
                [(e.angle_deg, e.position_cm, e.ied_mm, e.orientation_deg) for e in self.electrodes.electrodes]:
            raise ValueError("record() needs electrodes at the same positions as the simulator's")
        rng = np.random.default_rng(np.random.SeedSequence([self.subject.seed, 2]))
        joint_speed = np.abs(run.labels["qd"][:, :3]).sum(1)
        emg = electrodes.record(run.clean.emg, run.clean.fs, joint_speed, FS_CTRL, rng)
        qc = self.quality(emg, run.labels, run.activity.residual)
        if not qc["pass"]:
            log.warning(f"the simulated hand did not follow the motion well (tracking RMSE {qc['rmse_deg']:.1f} deg, "
                        f"saturation {qc['saturation']:.0%}): see recording.qc")
        return Recording(emg=emg, fs=electrodes.fs_out, labels=run.labels, fs_labels=FS_CTRL,
                         joint_names=self.model.joint_names, muscle_names=self.model.muscle_names,
                         electrodes=electrodes, subject=self.subject, engine=self.engine.name, qc=qc,
                         filled_joints=run.filled_joints, spikes=run.clean.spikes)

    def joint_trajectory(self, motion):
        """(q_ref (T, nq) with the warm-up hold, relaxed posture q_rest (nq,), hand angles (T, 23))."""
        hand = self._within_limits(motion.angles)
        elbow = motion.elbow if motion.elbow is not None else np.full(len(hand), np.radians(self.subject.elbow_angle_deg))
        n_warm = int(WARMUP * motion.fs)
        hand = np.vstack([np.repeat(hand[:1], n_warm, 0), hand])
        elbow = np.concatenate([np.repeat(elbow[:1], n_warm), elbow])
        q_ref = self.model.full_q(hand, elbow)
        rest = motion.rest
        if rest is not None and rest.sum() > motion.fs:            # at least 1 s of rest in the recording
            hand_rest = np.median(motion.angles[rest], axis=0)
        else:
            hand_rest = np.array([REST_POSTURE[j] for j in HAND_JOINTS])
        q_rest = self.model.full_q(hand_rest[None], np.radians(self.subject.elbow_angle_deg))[0]
        return q_ref, q_rest, hand

    def _within_limits(self, hand):
        lo, hi = self.model.qmin[self.model.hand_qpos], self.model.qmax[self.model.hand_qpos]
        beyond = [HAND_JOINTS[j] for j in np.flatnonzero(((hand < lo - LIMIT_TOLERANCE) |
                                                          (hand > hi + LIMIT_TOLERANCE)).any(0))]
        if beyond:
            log.warning(f"angles beyond the model's joint limits (clipped): {', '.join(beyond)}")
        return np.clip(hand, lo, hi)

    def quality(self, emg, labels, residual):
        m = self.model.m
        err = np.degrees(labels["q"] - labels["q_ref"])
        wrist = [m.jnt_dofadr[m.joint(n).id] for n in ("deviation_r", "flexion_r")]
        qc = {
            "rmse_deg": float(np.sqrt((err ** 2).mean())),
            "rmse_deg_per_joint": np.sqrt((err ** 2).mean(0)).round(2).tolist(),
            "max_err_deg": float(np.abs(err).max()),
            "wrist_reserve_rms_nm": float(np.sqrt((labels["reserve"][:, wrist] ** 2).mean())),
            "reserve_rms_nm": float(np.sqrt((labels["reserve"] ** 2).mean())),
            "saturation": float((labels["excitation"] > 0.99).mean()),
            "residual": float(residual.mean()),
            "finite": bool(np.isfinite(emg).all() and np.isfinite(labels["q"]).all()),
        }
        qc["pass"] = bool(qc["finite"] and qc["rmse_deg"] < QC_MAX_RMSE_DEG
                          and qc["wrist_reserve_rms_nm"] < QC_MAX_WRIST_RESERVE_NM
                          and qc["saturation"] < QC_MAX_SATURATION)
        return qc


def finger_closure(hand):
    """(T, 4) closure of fingers 2-5 in [0, 1], from the PIP angles (fully flexed PIP ~ 1.5 rad)."""
    return np.clip(hand[:, [HAND_JOINTS.index(f"pm{f}_flexion_r") for f in "2345"]] / 1.5, 0, 1)


def effort(model, q_ref, q_rest, subject, rng, fs=FS_CTRL):
    """(co-activation (T, nu), grip load (T,)): the effort beyond the mechanical minimum.

    A joint is engaged while it moves or is held away from the relaxed posture. Each muscle gets the effort of the
    engaged joints it acts on, in proportion to its moment arm relative to the joint's strongest muscle. A relaxed
    hand is nearly electrically silent, as in real recordings.
    """
    span = np.maximum(model.qmax - model.qmin, 1e-6)
    u, u_rest = q_ref / span, q_rest / span                     # joint angles as fractions of their range
    lp = signal.butter(2, 2.0, fs=fs, output="sos")
    speed = signal.sosfiltfilt(lp, np.abs(np.gradient(u, axis=0)) * fs, axis=0)
    moving = np.clip(speed - 0.15, 0, 1.5)                    # dead zone: sensor jitter is not movement
    away = np.clip(np.abs(u - u_rest) - 0.1, 0, None) / 0.4
    engaged = np.clip(signal.sosfiltfilt(lp, np.clip(away + moving / 0.5, 0, 1), axis=0), 0, 1)
    R = np.abs(model.moment_arms(q_rest))                       # (nu, nv); one dof per joint
    involvement = R / np.maximum(R.max(0), 1e-9)
    T = len(q_ref)
    per_muscle_engaged, per_muscle_moving = np.zeros((T, model.nu)), np.zeros((T, model.nu))
    for j in range(R.shape[1]):
        per_muscle_engaged = np.maximum(per_muscle_engaged, engaged[:, j:j + 1] * involvement[:, j])
        per_muscle_moving = np.maximum(per_muscle_moving, moving[:, j:j + 1] * involvement[:, j])
    level = _episodes(rng, T, fs, subject.cocontraction, 0.3, 0.25) + subject.hold_tone
    co_activation = level[:, None] * per_muscle_engaged + subject.move_cocontraction * per_muscle_moving
    hand = q_ref[:, model.hand_qpos]
    closed = 1 / (1 + np.exp(-(finger_closure(hand).mean(1) - 0.5) / 0.05))   # only a closed hand squeezes
    grip = _episodes(rng, T, fs, 0.0, 1.0, subject.grip_load) * closed
    return co_activation, grip


def _episodes(rng, T, fs, base, high, p_episode, tau=0.3):
    """(T,) slowly varying level at `base` with random episodes up to `high`, zero-phase smoothed with tau."""
    x = np.full(T, float(base))
    i = 0
    while i < T:
        n = int(rng.uniform(0.5, 3.0) * fs)
        if rng.random() < p_episode:
            x[i:i + n] = rng.uniform(base, high)
        i += n
    a = np.exp(-1 / (tau * fs))
    for _ in range(2):                         # first-order low-pass forward, then backward
        y, acc = np.empty_like(x), x[0]
        for k in range(T):
            acc = a * acc + (1 - a) * x[k]
            y[k] = acc
        x = y[::-1]
    return x
