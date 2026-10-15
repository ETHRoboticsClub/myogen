from __future__ import annotations

import copy
import logging
import queue
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, overload

import numpy as np

from .inputs.electrodes import Electrodes
from .inputs.motion import REST_POSTURE, HandMotion, read_motion_file
from .inputs.subject import Subject
from .parallel import mp_context, n_workers, process_pool
from .params import SimParams
from .progress import ProgressFn, RunProgress
from .recording import Recording
from .simulation import acquisition
from .simulation.emg import EMGModel
from .simulation.muscle_activity import MuscleActivity, MuscleController, effort, finger_closure
from .simulation.musculoskeletal import HAND_JOINTS, ForearmModel

log = logging.getLogger(__name__)

MotionLike = HandMotion | str | Path
R = TypeVar("R")


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

    @overload
    def run(self, motion: MotionLike, *, workers: int | None = ..., progress: bool = ...  # type: ignore[overload-overlap]
            ) -> Recording: ...
    @overload
    def run(self, motion: Sequence[MotionLike], *, workers: int | None = ..., progress: bool = ...
            ) -> list[Recording]: ...

    def run(self, motion: MotionLike | Sequence[MotionLike], *, workers: int | None = None,
            progress: bool = True) -> Recording | list[Recording]:
        """Simulate a HandMotion (or a myogen motion file: read_motion_file). Returns a Recording.

        A list of motions returns a list of Recordings in the same order, simulated in `workers` parallel processes
        (default: one per motion, up to the CPU count; 1 = in this process). Each result is the same as running its
        motion alone. progress: show progress bars."""
        return self._map(Simulator._run_one, motion, workers, progress)

    @overload
    def simulate_clean(self, motion: MotionLike, *, workers: int | None = ...,  # type: ignore[overload-overlap]
                       progress: bool = ...) -> CleanRun: ...
    @overload
    def simulate_clean(self, motion: Sequence[MotionLike], *, workers: int | None = ..., progress: bool = ...
                       ) -> list[CleanRun]: ...

    def simulate_clean(self, motion: MotionLike | Sequence[MotionLike], *, workers: int | None = None,
                       progress: bool = False) -> CleanRun | list[CleanRun]:
        """The expensive part: movement -> muscle activity -> clean EMG at the electrodes. Returns a CleanRun, which
        record() turns into a Recording (calibrate() records one CleanRun with many device settings). A list of
        motions runs as in run()."""
        return self._map(Simulator._clean_one, motion, workers, progress)

    def _run_one(self, motion: MotionLike, on_progress: ProgressFn | None = None) -> Recording:
        return self.record(self._clean_one(motion, on_progress))

    def _clean_one(self, motion: MotionLike, on_progress: ProgressFn | None = None) -> CleanRun:
        """simulate_clean() of one motion; on_progress(stage, done, total) is called as it goes."""
        report = (lambda stage: None) if on_progress is None else \
            (lambda stage: lambda k, n: on_progress(stage, k + 1 if stage == "control" else k, n))
        motion = motion if isinstance(motion, HandMotion) else HandMotion.load(motion, read_motion_file)
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
                                    on_tick=report("control"))
        start = int(ctrl.warmup_s * ctrl.fs)
        activity = activity[start:]
        # The EMG it produces at the electrodes.
        run = CleanRun(activity, self.emg.generate(activity, self._emg_rng(), on_tick=report("emg")),
                       self._labels(activity, q_ref, grip, start), list(motion.filled))
        log.info(f"done in {time.time() - t0:.0f} s")
        return run

    def _map(self, simulate: Callable[[Simulator, MotionLike, ProgressFn | None], R],
             motions: MotionLike | Sequence[MotionLike], workers: int | None, progress: bool) -> R | list[R]:
        """simulate(self, motion, on_progress) for one motion, or for each of a list in worker processes."""
        single = isinstance(motions, HandMotion | str | Path)
        batch: list[MotionLike] = [motions] if single else list(motions)  # type: ignore[list-item, arg-type]
        names = [Path(m).stem if isinstance(m, str | Path) else "motion" if single else f"motion {i}"
                 for i, m in enumerate(batch)]
        workers = n_workers(len(batch), workers)
        with RunProgress(names, enabled=progress) as bars:
            if workers <= 1:
                results = []
                for i, m in enumerate(batch):
                    results.append(simulate(self, m, bars.reporter(i) if progress else None))
                    bars.finished(i)
            else:
                results = self._map_parallel(simulate, batch, workers, bars if progress else None)
        return results[0] if single else results

    def _map_parallel(self, simulate: Callable[[Simulator, MotionLike, ProgressFn | None], R],
                      motions: list[MotionLike], workers: int, bars: RunProgress | None) -> list[R]:
        """Each worker process builds its own Simulator (a pickled one is ~80 MB) and reports its progress through a
        queue the bars are drawn from here."""
        ticks = mp_context().Queue() if bars is not None else None
        results: list[Any] = [None] * len(motions)

        def drain() -> None:
            while ticks is not None and bars is not None:
                try:
                    bars.update(*ticks.get_nowait())
                except queue.Empty:
                    return

        with process_pool(workers, _worker_init, (self.electrodes, self.subject, self.params, ticks)) as pool:
            futures: dict[Future[R], int] = {pool.submit(_worker_simulate, simulate, i, m): i
                                             for i, m in enumerate(motions)}
            pending: set[Future[R]] = set(futures)
            try:
                while pending:
                    done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    drain()
                    for f in done:
                        results[futures[f]] = f.result()
                        if bars is not None:
                            bars.finished(futures[f])
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        return results

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


# -- worker processes for batches of motions ---------------------------------------------------------------------

_worker: tuple[Simulator, Any] | None = None      # this worker's Simulator and progress queue


def _worker_init(electrodes: Electrodes, subject: Subject, params: SimParams, ticks: Any) -> None:
    global _worker
    _worker = (Simulator(electrodes, subject, params), ticks)


def _worker_simulate(simulate: Callable[[Simulator, MotionLike, ProgressFn | None], R], i: int,
                     motion: MotionLike) -> R:
    assert _worker is not None
    sim, ticks = _worker
    if ticks is None:
        return simulate(sim, motion, None)
    last: dict[str, int] = {}

    def report(stage: str, done: int, total: int) -> None:
        step = max(1, total // 100)                 # ~100 updates per stage through the queue
        if done == total or done - last.get(stage, -step) >= step:
            last[stage] = done
            ticks.put((i, stage, done, total))
    return simulate(sim, motion, report)
