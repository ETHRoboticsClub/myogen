from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..inputs.motion import REST_POSTURE, HandMotion
from ..inputs.subject import Subject
from ..params import ControlParams
from .musculoskeletal import HAND_JOINTS, ForearmModel

log = logging.getLogger(__name__)


@dataclass
class JointTrajectory:
    """A motion as the model's joint trajectory at the control rate, after a warm-up hold of its first posture."""
    q_ref: np.ndarray       # (T, nq) reference joint state, warm-up included
    q_rest: np.ndarray      # (nq,) relaxed posture
    hand: np.ndarray        # (T, 23) hand angles (HAND_JOINTS), within the model's limits
    warmup: int             # ticks of warm-up at the start (left out of the outputs)


class Kinematics:
    """HandMotion -> JointTrajectory: the motion within the model's joint limits, the elbow (the motion's or the
    subject's), and the relaxed posture (the median of the motion's rest frames, else REST_POSTURE)."""

    def __init__(self, model: ForearmModel, subject: Subject, params: ControlParams | None = None) -> None:
        self.model, self.sub, self.p = model, subject, params or ControlParams()

    def __call__(self, motion: HandMotion) -> JointTrajectory:
        """motion: at the control rate (HandMotion.resampled)."""
        p = self.p
        hand = self._within_limits(motion.angles)
        elbow_rest = np.radians(self.sub.elbow_angle_deg)
        elbow = motion.elbow if motion.elbow is not None else np.full(len(hand), elbow_rest)
        n_warm = int(p.warmup_s * motion.fs)
        hand = np.vstack([np.repeat(hand[:1], n_warm, 0), hand])
        elbow = np.concatenate([np.repeat(elbow[:1], n_warm), elbow])
        q_ref = self.model.full_q(hand, elbow)
        rest = motion.rest
        if rest is not None and rest.sum() > p.min_rest_s * motion.fs:
            hand_rest = np.median(motion.angles[rest], axis=0)
        else:
            hand_rest = np.array([REST_POSTURE[j] for j in HAND_JOINTS])
        q_rest = self.model.full_q(hand_rest[None], elbow_rest)[0]
        return JointTrajectory(q_ref, q_rest, hand, int(p.warmup_s * p.fs))

    def _within_limits(self, hand: np.ndarray) -> np.ndarray:
        lo, hi = self.model.qmin[self.model.hand_qpos], self.model.qmax[self.model.hand_qpos]
        tol = self.p.joint_limit_tolerance
        beyond = [HAND_JOINTS[j] for j in np.flatnonzero(((hand < lo - tol) | (hand > hi + tol)).any(0))]
        if beyond:
            log.warning(f"angles beyond the model's joint limits (clipped): {', '.join(beyond)}")
        return np.clip(hand, lo, hi)
