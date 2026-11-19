from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..inputs.subject import Subject
from ..params import EffortParams
from .controller import MuscleActivity
from .kinematics import JointTrajectory
from .musculoskeletal import HAND_JOINTS, ForearmModel


@dataclass
class EffortPlan:
    """Effort beyond the mechanical minimum over a motion."""
    co_activation: np.ndarray   # (T, nu) torque-neutral drive added to the muscles (reaches the EMG, not the motion)
    grip: np.ndarray            # (T,) squeezing load in [0, 1] (reaches the motion, as torques on the fingers)


class Effort:
    """People drive their muscles harder than the mechanical minimum the controller finds. Hand-tuned heuristics
    (no outside reference): co-activation of the muscles of the engaged joints, grip episodes on a closed hand and a
    rest tone in every muscle."""

    def __init__(self, model: ForearmModel, subject: Subject, params: EffortParams | None = None,
                 fs: float = 100.0) -> None:
        self.model, self.sub, self.p, self.fs = model, subject, params or EffortParams(), fs

    def plan(self, q_ref: np.ndarray, q_rest: np.ndarray, rng: np.random.Generator) -> EffortPlan:
        """The effort over the reference trajectory q_ref (T, nq), relaxed at q_rest."""
        return EffortPlan(*effort(self.model, q_ref, q_rest, self.sub, rng, self.p, self.fs))

    def grip_torque(self, plan: EffortPlan, trajectory: JointTrajectory) -> np.ndarray:
        """(T, nv) torques opposing finger flexion while squeezing."""
        return self.model.grip_torque(plan.grip, finger_closure(trajectory.hand, self.p.pip_closed_rad))

    def apply(self, activity: MuscleActivity, plan: EffortPlan) -> MuscleActivity:
        """The activity with the co-activation and Subject.rest_tone added to the drive. Real recordings at rest are
        not noise: the rest EMG is non-Gaussian, varies from muscle to muscle and follows each electrode's movement
        pattern. (A tone carried over from recent use was tried: no better against real recordings.)"""
        drive = np.clip(activity.drive + plan.co_activation, 0, 1)
        return dataclasses.replace(activity, effort=np.asarray(plan.co_activation, float),
                                   drive=np.clip(drive + self.sub.rest_tone, 0, 1))


def effort(model: ForearmModel, q_ref: np.ndarray, q_rest: np.ndarray, subject: Subject, rng: np.random.Generator,
           params: EffortParams | None = None, fs: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """(co-activation (T, nu), grip load (T,)). People drive their muscles harder than the mechanical minimum.

    A joint is engaged while it moves or is held away from the relaxed posture, in the direction it moves or is held.
    Each muscle gets the effort of the engaged joints it acts on, in proportion to its moment arm relative to the
    joint's strongest muscle: in full if it pulls the joint that way (agonist), a fraction
    (Subject.antagonist_ratio) if it pulls the other way. A relaxed hand is nearly electrically silent, as in real
    recordings.
    """
    p = params or EffortParams()
    engaged, moving = joint_engagement(model, q_ref, q_rest, p, fs)
    involvement = muscle_involvement(model, q_rest)
    # Co-activation level: the subject's baseline with random episodes, plus tone while engaged and moving.
    level = _episodes(rng, len(q_ref), fs, base=subject.cocontraction, high=p.cocontraction_peak,
                      p_episode=p.cocontraction_episode_prob, duration_s=p.episode_s,
                      tau=p.episode_smoothing_s) + subject.hold_tone
    r = subject.antagonist_ratio
    co_activation = (level[:, None] * per_muscle(engaged, involvement, r)
                     + subject.move_cocontraction * per_muscle(moving, involvement, r))
    return co_activation, grip_load(model, q_ref, subject, rng, p, fs)


def joint_engagement(model: ForearmModel, q_ref: np.ndarray, q_rest: np.ndarray, p: EffortParams, fs: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(engaged (T, nv) in [-1, 1], moving (T, nv)): how far each joint is held from rest and how fast it moves,
    signed by direction (the sign of the joint angle change it calls for)."""
    span = np.maximum(model.qmax - model.qmin, 1e-6)
    u, u_rest = q_ref / span, q_rest / span                     # joint angles as fractions of their range
    lp = signal.butter(2, p.speed_lowpass_hz, fs=fs, output="sos")
    velocity = np.gradient(u, axis=0) * fs
    speed = signal.sosfiltfilt(lp, np.abs(velocity), axis=0)
    direction = np.sign(signal.sosfiltfilt(lp, velocity, axis=0))
    moving = direction * np.clip(speed - p.speed_deadzone, 0, p.speed_max)
    if p.passive_return:     # back towards the relaxed posture the hand is let go, not driven
        moving *= direction * np.sign(u - u_rest) > 0
    away = np.sign(u - u_rest) * np.clip(np.abs(u - u_rest) - p.posture_deadzone, 0, None) / p.posture_engaged
    engaged = np.clip(signal.sosfiltfilt(lp, np.clip(away + moving / p.speed_engaged, -1, 1), axis=0), -1, 1)
    return engaged, moving


def muscle_involvement(model: ForearmModel, q_rest: np.ndarray) -> np.ndarray:
    """(nu, nv) each muscle's moment arm relative to the strongest muscle of each joint (one dof per joint), signed
    by the direction the muscle turns the joint (positive: towards larger angles)."""
    R = -model.moment_arms(q_rest)                  # muscles pull: their force is negative in MuJoCo
    return R / np.maximum(np.abs(R).max(0), 1e-9)


def per_muscle(joint_level: np.ndarray, involvement: np.ndarray, antagonist_ratio: float = 1.0) -> np.ndarray:
    """(T, nu) the largest level among the joints each muscle acts on, weighted by its involvement: in full for the
    muscles that turn the joint in the level's direction, antagonist_ratio for the others (1: direction ignored)."""
    out = np.zeros((len(joint_level), involvement.shape[0]))
    for j in range(involvement.shape[1]):
        agonist = np.sign(joint_level[:, j:j + 1]) * np.sign(involvement[:, j]) > 0
        weight = np.abs(involvement[:, j]) * np.where(agonist, 1.0, antagonist_ratio)
        out = np.maximum(out, np.abs(joint_level[:, j:j + 1]) * weight)
    return out


def grip_load(model: ForearmModel, q_ref: np.ndarray, subject: Subject, rng: np.random.Generator,
              p: EffortParams, fs: float) -> np.ndarray:
    """(T,) random squeezing episodes in [0, 1], only while the hand is closed."""
    closure = finger_closure(q_ref[:, model.hand_qpos], p.pip_closed_rad).mean(1)
    closed = 1 / (1 + np.exp(-(closure - p.closed_hand) / p.closed_hand_width))
    return _episodes(rng, len(q_ref), fs, base=0.0, high=1.0, p_episode=subject.grip_load, duration_s=p.episode_s,
                     tau=p.episode_smoothing_s) * closed


def finger_closure(hand: np.ndarray, pip_closed_rad: float = EffortParams.pip_closed_rad) -> np.ndarray:
    """(T, 4) closure of fingers 2-5 in [0, 1], from the PIP angles."""
    return np.clip(hand[:, [HAND_JOINTS.index(f"pm{f}_flexion_r") for f in "2345"]] / pip_closed_rad, 0, 1)


def _episodes(rng: np.random.Generator, T: int, fs: float, base: float, high: float, p_episode: float,
              duration_s: tuple[float, float], tau: float) -> np.ndarray:
    """(T,) slowly varying level at `base` with random episodes up to `high`, zero-phase smoothed with tau."""
    x = np.full(T, float(base))
    i = 0
    while i < T:
        n = int(rng.uniform(*duration_s) * fs)
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
