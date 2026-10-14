from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .inputs.electrodes import Electrodes
from .inputs.motion import REST_POSTURE, HandMotion
from .inputs.subject import Subject
from .params import SimParams
from .recording import Recording
from .simulation import acquisition
from .simulation.emg import EMGModel
from .simulation.muscle_activity import MuscleActivity, MuscleController, effort, finger_closure
from .simulation.musculoskeletal import HAND_JOINTS, ForearmModel

log = logging.getLogger(__name__)


@dataclass
class CleanRun:
    """A simulated motion before the recording device: the muscle activity, its clean EMG and the labels."""
    activity: MuscleActivity
    emg: np.ndarray                   # (N, C) clean bipolar sEMG in mV at SimParams.emg.fs
    labels: dict[str, np.ndarray]
    filled_joints: list[str]


class Simulator:
    """Movement in, sEMG recording out: run() = record(simulate_clean())."""

    def __init__(self, electrodes: Electrodes | str = "myo", subject: Subject | None = None,
                 params: SimParams | None = None) -> None:
        """electrodes: an Electrodes or a preset name (ring | myo | otto_bock); subject: a Subject (default: the
        average subject); params: the model parameters (default: SimParams())."""
        self.electrodes = electrodes if isinstance(electrodes, Electrodes) else Electrodes.preset(electrodes)
        self.subject = subject or Subject()
        self.params = params or SimParams()
        self.model = ForearmModel(params=self.params.musculoskeletal)
        self.emg = EMGModel(self.model, self.subject, self.electrodes, self.params.emg)

    def with_electrodes(self, electrodes: Electrodes) -> Simulator:
        """A simulator with the same model, subject and parameters, and other electrodes."""
        other = copy.copy(self)
        other.electrodes = electrodes
        other.emg = EMGModel(self.model, self.subject, electrodes, self.params.emg)
        return other

    def run(self, motion: HandMotion | str | Path, on_tick: Callable[[int, int], None] | None = None) -> Recording:
        """Simulate a HandMotion (or a motion file). Returns a Recording."""
        return self.record(self.simulate_clean(motion, on_tick))

    def simulate_clean(self, motion: HandMotion | str | Path,
                       on_tick: Callable[[int, int], None] | None = None) -> CleanRun:
        """The expensive part: movement -> muscle activity -> clean EMG at the electrodes. Returns a CleanRun, which
        record() turns into a Recording (calibrate() records one CleanRun with many device settings)."""
        motion = motion if isinstance(motion, HandMotion) else HandMotion.load(motion)
        sub, ctrl = self.subject, self.params.control
        effort_rng = np.random.default_rng(np.random.SeedSequence([sub.seed, 0]))
        t0 = time.time()
        log.info(f"simulating {motion.duration:.1f} s: {len(self.electrodes)} electrodes ({self.electrodes.device})")

        # The movement as the model's joint trajectory, and the loads on the skeleton.
        q_ref, q_rest, hand = self.joint_trajectory(motion.resampled(ctrl.fs))
        co_activation, grip = effort(self.model, q_ref, q_rest, sub, effort_rng, self.params.effort, ctrl.fs)
        fibre_rest = self.model.prepare(sub, q_ref[0], q_rest)
        grip_torque = self.model.grip_torque(grip, finger_closure(hand, self.params.effort.pip_closed_rad))
        # Muscle activity that reproduces it.
        controller = MuscleController(self.model, sub.recruitment_cost, ctrl)
        activity = controller.track(q_ref, grip_torque, self.model.support(sub), fibre_rest, effort=co_activation,
                                    on_tick=on_tick)
        start = int(ctrl.warmup_s * ctrl.fs)
        activity = activity[start:]
        # The EMG it produces at the electrodes.
        run = CleanRun(activity, self.emg.generate(activity, self._emg_rng()),
                       self._labels(activity, q_ref, grip, start), list(motion.filled))
        log.info(f"done in {time.time() - t0:.0f} s")
        return run

    def regenerate(self, run: CleanRun) -> CleanRun:
        """The CleanRun of the same muscle activity at this simulator's electrodes (re-runs only the EMG step)."""
        return CleanRun(run.activity, self.emg.generate(run.activity, self._emg_rng()), run.labels, run.filled_joints)

    def record(self, run: CleanRun, electrodes: Electrodes | None = None) -> Recording:
        """The Recording of a CleanRun through the electrodes' device (default: the simulator's). Other
        electrodes must have the same positions: only the device settings (gain, noise, frontend) may differ."""
        electrodes = electrodes or self.electrodes
        if [(e.angle_deg, e.position_cm, e.ied_mm, e.orientation_deg) for e in electrodes.electrodes] != \
                [(e.angle_deg, e.position_cm, e.ied_mm, e.orientation_deg) for e in self.electrodes.electrodes]:
            raise ValueError("record() needs electrodes at the same positions as the simulator's")
        rng = np.random.default_rng(np.random.SeedSequence([self.subject.seed, 2]))
        joint_speed = np.abs(run.labels["qd"][:, :3]).sum(1)
        fs_labels = self.params.control.fs
        emg = acquisition.record(run.emg, self.params.emg.fs, electrodes, joint_speed, fs_labels, rng,
                                 self.params.acquisition)
        qc = self.quality(emg, run.labels, run.activity)
        if not qc["pass"]:
            log.warning(f"the simulated hand did not follow the motion well (tracking RMSE {qc['rmse_deg']:.1f} deg, "
                        f"saturation {qc['saturation']:.0%}): see recording.qc")
        return Recording(emg=emg, fs=electrodes.fs_out, labels=run.labels, fs_labels=fs_labels,
                         joint_names=self.model.joint_names, muscle_names=self.model.muscle_names,
                         electrodes=electrodes, subject=self.subject, qc=qc, filled_joints=run.filled_joints)

    def joint_trajectory(self, motion: HandMotion) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q_ref (T, nq) with the warm-up hold, relaxed posture q_rest (nq,), hand angles (T, 23))."""
        hand = self._within_limits(motion.angles)
        elbow_rest = np.radians(self.subject.elbow_angle_deg)
        elbow = motion.elbow if motion.elbow is not None else np.full(len(hand), elbow_rest)
        ctrl = self.params.control
        n_warm = int(ctrl.warmup_s * motion.fs)
        hand = np.vstack([np.repeat(hand[:1], n_warm, 0), hand])
        elbow = np.concatenate([np.repeat(elbow[:1], n_warm), elbow])
        q_ref = self.model.full_q(hand, elbow)
        rest = motion.rest
        if rest is not None and rest.sum() > ctrl.min_rest_s * motion.fs:
            hand_rest = np.median(motion.angles[rest], axis=0)
        else:
            hand_rest = np.array([REST_POSTURE[j] for j in HAND_JOINTS])
        q_rest = self.model.full_q(hand_rest[None], np.radians(self.subject.elbow_angle_deg))[0]
        return q_ref, q_rest, hand

    def quality(self, emg: np.ndarray, labels: dict[str, np.ndarray], activity: MuscleActivity) -> dict[str, Any]:
        """Did the simulated hand do what the motion says? (thresholds: SimParams.qc)"""
        m, p = self.model.m, self.params.qc
        err = np.degrees(labels["q"] - labels["q_ref"])
        wrist = [m.jnt_dofadr[m.joint(n).id] for n in ("deviation_r", "flexion_r")]
        qc = {
            "rmse_deg": float(np.sqrt((err ** 2).mean())),
            "rmse_deg_per_joint": np.sqrt((err ** 2).mean(0)).round(2).tolist(),
            "max_err_deg": float(np.abs(err).max()),
            "wrist_reserve_rms_nm": float(np.sqrt((labels["reserve"][:, wrist] ** 2).mean())),
            "reserve_rms_nm": float(np.sqrt((labels["reserve"] ** 2).mean())),
            "saturation": float((labels["excitation"] > p.saturated_above).mean()),
            "residual": float(activity.residual.mean()),
            "recovered_ticks": [] if activity.recovered is None else np.flatnonzero(activity.recovered).tolist(),
            "finite": bool(np.isfinite(emg).all() and np.isfinite(labels["q"]).all()),
        }
        qc["pass"] = bool(qc["finite"] and not qc["recovered_ticks"] and qc["rmse_deg"] < p.max_rmse_deg
                          and qc["wrist_reserve_rms_nm"] < p.max_wrist_reserve_nm
                          and qc["saturation"] < p.max_saturation)
        return qc

    def _labels(self, activity: MuscleActivity, q_ref: np.ndarray, grip: np.ndarray, start: int
                ) -> dict[str, np.ndarray]:
        q_ref_next = np.vstack([q_ref[1:], q_ref[-1:]])[start:]     # labels are the state after each tick
        return {"q": activity.q, "q_ref": q_ref_next, "qd": activity.qd, "excitation": activity.excitation,
                "activation": activity.activation, "reserve": activity.reserve,
                "fibre_length": activity.fibre_length, "effort": activity.effort, "grip_load": grip[start:]}

    def _emg_rng(self) -> np.random.Generator:
        """The EMG step's randomness: the same for every electrode set, so re-runs differ only by geometry."""
        return np.random.default_rng(np.random.SeedSequence([self.subject.seed, 1]))

    def _within_limits(self, hand: np.ndarray) -> np.ndarray:
        lo, hi = self.model.qmin[self.model.hand_qpos], self.model.qmax[self.model.hand_qpos]
        tol = self.params.control.joint_limit_tolerance
        beyond = [HAND_JOINTS[j] for j in np.flatnonzero(((hand < lo - tol) | (hand > hi + tol)).any(0))]
        if beyond:
            log.warning(f"angles beyond the model's joint limits (clipped): {', '.join(beyond)}")
        return np.clip(hand, lo, hi)
