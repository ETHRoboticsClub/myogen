from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal

from ..parallel import n_workers, process_pool
from ..progress import Counter, progress_bars

if TYPE_CHECKING:
    from rich.progress import Progress

    from ..inputs.electrodes import Electrodes
    from ..inputs.motion import HandMotion
    from ..inputs.subject import Subject
    from ..params import SimParams
    from ..simulator import CleanRun, Simulator

Stats = dict[str, Any]       # statistic -> value (None where a recording has none)

ENV_FS = 100.0               # Hz, envelope rate of every statistic
RMS_WINDOW = 0.15            # s, moving RMS for devices that output raw EMG
BISECT_STEPS = 24            # evaluations per fitted setting and round

log = logging.getLogger(__name__)


def envelope(emg: np.ndarray, fs: float, device: str) -> np.ndarray:
    """(T, C) amplitude envelope at ENV_FS: the Otto Bock output as is, otherwise a moving RMS."""
    if device == "otto_bock":
        return np.asarray(emg, float)
    x = np.asarray(emg, float)
    w = max(1, int(RMS_WINDOW * fs))
    env = np.sqrt(np.maximum(signal.fftconvolve(x ** 2, np.ones((w, 1)) / w, mode="same", axes=0), 0))
    return env[::max(1, int(round(fs / ENV_FS)))]


def statistics(emg: np.ndarray, fs: float, device: str, rest: np.ndarray | None = None) -> Stats:
    """Device-level statistics of one recording. rest: (N,) bool at the EMG rate (None: all movement)."""
    env = envelope(emg, fs, device)
    rest = np.zeros(len(env), bool) if rest is None else _resize(rest, len(env))
    move = ~rest if (~rest).any() else np.ones(len(env), bool)
    zs = (env - env.mean(0)) / (env.std(0) + 1e-12)
    lag = lambda n: float(np.mean([np.corrcoef(zs[:-n, c], zs[n:, c])[0, 1] for c in range(env.shape[1])]))
    stats = {
        "amplitude": float(np.median(np.percentile(env[move], 95, axis=0))),
        "rest_floor": float(np.median(np.median(env[rest], axis=0))) if rest.any() else None,
        "autocorr_50ms": lag(int(0.05 * ENV_FS)),
        "autocorr_200ms": lag(int(0.2 * ENV_FS)),
        "neighbour_correlation": float(np.mean([np.corrcoef(env[:, c], env[:, c + 1])[0, 1]
                                                for c in range(env.shape[1] - 1)])),
    }
    stats["rest_move_ratio"] = stats["rest_floor"] / (np.median(env[move]) + 1e-12) if rest.any() else None
    if device != "otto_bock":
        f, P = signal.welch(np.asarray(emg, float), fs=fs, nperseg=min(256, len(emg)), axis=0)
        cdf = np.cumsum(P.mean(1)) / P.mean(1).sum()
        stats["median_freq_hz"] = float(f[np.searchsorted(cdf, 0.5)])
    return stats


def compare(real_emg: np.ndarray, sim_emg: np.ndarray, fs: float, device: str, rest: np.ndarray | None = None
            ) -> dict[str, tuple[float | None, float | None]]:
    """{statistic: (real, simulated)} for a real and a simulated recording of the same motion, plus
    channel_correlation: (None, mean correlation of each channel's real and simulated envelopes over time)."""
    real, sim = statistics(real_emg, fs, device, rest), statistics(sim_emg, fs, device, rest)
    out = {k: (real[k], sim[k]) for k in real}
    out["channel_correlation"] = (None, channel_correlation(real_emg, sim_emg, fs, device))
    return out


def channel_correlation(real_emg: np.ndarray, sim_emg: np.ndarray, fs: float, device: str) -> float:
    """Mean over channels of the correlation between the real and the simulated envelope over time."""
    a, b = envelope(real_emg, fs, device), envelope(sim_emg, fs, device)
    n = min(len(a), len(b))
    return float(np.mean([np.corrcoef(a[:n, c], b[:n, c])[0, 1] for c in range(a.shape[1])
                          if a[:n, c].std() > 0 and b[:n, c].std() > 0]))


@dataclass
class Calibration:
    electrodes: Electrodes                              # the fitted device profile
    fitted: dict[str, tuple[float, float]]              # parameter -> (before, after)
    report: dict[str, dict[str, float | None]]          # statistic -> {"real", "before", "after"}

    def __str__(self) -> str:
        lines = [f"{'statistic':24s}{'real':>10s}{'before':>10s}{'after':>10s}"]
        for k, v in self.report.items():
            lines.append(f"{k:24s}" + "".join(f"{_fmt(v[c]):>10s}" for c in ("real", "before", "after")))
        lines.append("fitted: " + ", ".join(f"{k} {b:.4g} -> {a:.4g}" for k, (b, a) in self.fitted.items()))
        return "\n".join(lines)


def calibrate(simulator: Simulator, motions: Sequence[HandMotion], real_emgs: Sequence[np.ndarray],
              rests: Sequence[np.ndarray | None] | None = None, rounds: int = 3, fit_rotation: bool = False,
              rotation_steps: int = 16, workers: int | None = None, progress: bool = True) -> Calibration:
    """Fit the device settings of simulator.electrodes to real recordings of the same motions.

    motions: HandMotions (what the hand did during each recording); real_emgs: (N, C) device outputs at the
    device's rate, aligned with the motions; rests: optional (N,) bool masks of rest (else motion.rest);
    fit_rotation: also fit the armband rotation, trying rotation_steps rotations around the forearm;
    workers: processes simulating the motions and trying the rotations (default: up to the CPU count; 1 = in this
    process); progress: show progress bars.
    """
    with progress_bars(progress) as bars:
        return _calibrate(simulator, motions, real_emgs, rests, rounds, fit_rotation, rotation_steps, workers,
                          progress, bars)


def _calibrate(simulator: Simulator, motions: Sequence[HandMotion], real_emgs: Sequence[np.ndarray],
               rests: Sequence[np.ndarray | None] | None, rounds: int, fit_rotation: bool, rotation_steps: int,
               workers: int | None, progress: bool, bars: Progress) -> Calibration:
    device = simulator.electrodes.device
    fs = simulator.electrodes.fs_out
    runs = simulator.simulate_clean(list(motions), workers=workers, progress=progress)
    rests = rests or [m.rest if m.rest is not None else None for m in motions]
    real = _pooled([statistics(x, fs, device, r) for x, r in zip(real_emgs, rests)])
    rotation, match_before = 0.0, _match(simulator, runs, real_emgs)
    if fit_rotation:
        search = Counter(bars, "armband rotation", rotation_steps, "rotations")
        scores = _rotation_scores(simulator, runs, real_emgs, [360.0 * k / rotation_steps
                                                                for k in range(rotation_steps)], workers, search)
        rotation = max(scores, key=lambda delta: scores[delta])
        simulator = simulator.with_electrodes(rotated(simulator.electrodes, rotation))
        runs = [simulator.regenerate(run) for run in runs]
        search.close()

    # Every device setting tried: before and after, and each step of each bisection.
    n_fits = (real["rest_floor"] is not None) + 1 + (device == "otto_bock")
    fit = Counter(bars, "device settings", 2 + rounds * n_fits * BISECT_STEPS, "settings")

    def simulated(electrodes: Electrodes) -> Stats:
        fit.advance()
        return _pooled([
            statistics(simulator.record(run, electrodes).emg, fs, device, r)
                    for run, r in zip(runs, rests)
        ])

    start = rotated(simulator.electrodes, -rotation)
    fitted = copy.deepcopy(simulator.electrodes)
    before = simulated(fitted)
    for _ in range(rounds):
        if real["rest_floor"] is not None:
            fitted.noise_uv = _bisect(lambda v: simulated(_with(fitted, noise_uv=v))["rest_floor"],
                                      real["rest_floor"], 1e-3, 1e3)
        fitted.gain = _bisect(lambda v: simulated(_with(fitted, gain=v))["amplitude"], real["amplitude"], 1e-3, 1e3)
        if device == "otto_bock":
            # Smoother envelope (lower cut-off) -> higher autocorrelation.
            fitted.frontend.smooth_hz = _bisect(
                lambda v: -simulated(_with(fitted, smooth_hz=v))["autocorr_50ms"], -real["autocorr_50ms"], 0.5, 20.0)
    after = simulated(fitted)
    fit.close()
    names = ["gain", "noise_uv"] + (["smooth_hz"] if device == "otto_bock" else [])
    fitted_params = {n: (_get(start, n), _get(fitted, n)) for n in names}
    if fit_rotation:
        fitted_params["rotation_deg"] = (0.0, rotation)
    report = {k: {"real": real[k], "before": before[k], "after": after[k]} for k in real}
    report["channel_correlation"] = {"real": None, "before": match_before, "after": _match(simulator, runs, real_emgs)}
    return Calibration(fitted, fitted_params, report)


def rotated(electrodes: Electrodes, delta_deg: float) -> Electrodes:
    """A copy with the ring electrodes (names ring*; all if there are none) rotated by delta_deg."""
    out = copy.deepcopy(electrodes)
    ring = [e for e in out.electrodes if e.name.startswith("ring")] or out.electrodes
    for e in ring:
        e.angle_deg = (e.angle_deg + delta_deg + 180.0) % 360.0 - 180.0
    return out


def _rotation_scores(simulator: Simulator, runs: Sequence[CleanRun], real_emgs: Sequence[np.ndarray],
                     rotations: list[float], workers: int | None, search: Counter) -> dict[float, float]:
    """rotation -> channel correlation of the runs re-generated with the electrodes rotated by it, in `workers`
    processes (each re-runs only the EMG step: the runs' muscle activity is shared)."""
    def done(delta: float, score: float) -> None:
        log.info(f"rotation {delta:6.1f} deg: channel correlation {score:.3f}")
        search.advance()

    workers = n_workers(len(rotations), workers)
    scores = {}
    if workers <= 1:
        for delta in rotations:
            scores[delta] = _rotation_score(simulator, runs, real_emgs, delta)
            done(delta, scores[delta])
        return scores
    light = [dataclasses.replace(run, emg=run.emg[:0]) for run in runs]    # the workers only need the activity
    with process_pool(workers, _worker_init, (simulator.electrodes, simulator.subject, simulator.params, light,
                                              list(real_emgs))) as pool:
        futures = {pool.submit(_worker_score, delta): delta for delta in rotations}
        try:
            for f in as_completed(futures):
                scores[futures[f]] = f.result()
                done(futures[f], scores[futures[f]])
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return {delta: scores[delta] for delta in rotations}


def _rotation_score(simulator: Simulator, runs: Sequence[CleanRun], real_emgs: Sequence[np.ndarray],
                    delta: float) -> float:
    candidate = simulator.with_electrodes(rotated(simulator.electrodes, delta))
    return _match(candidate, [candidate.regenerate(run) for run in runs], real_emgs)


_worker: tuple[Simulator, list[CleanRun], list[np.ndarray]] | None = None    # a rotation worker's inputs


def _worker_init(electrodes: Electrodes, subject: Subject, params: SimParams, runs: list[CleanRun],
                 real_emgs: list[np.ndarray]) -> None:
    from ..simulator import Simulator
    global _worker
    _worker = (Simulator(electrodes, subject, params), runs, real_emgs)


def _worker_score(delta: float) -> float:
    assert _worker is not None
    return _rotation_score(*_worker, delta)


def _match(simulator: Simulator, runs: Sequence[CleanRun], real_emgs: Sequence[np.ndarray]) -> float:
    """Mean per-channel envelope correlation between the simulated runs and the real recordings."""
    fs, device = simulator.electrodes.fs_out, simulator.electrodes.device
    return float(np.mean([channel_correlation(x, simulator.record(run).emg, fs, device)
                          for run, x in zip(runs, real_emgs)]))


def _with(electrodes: Electrodes, **values: float) -> Electrodes:
    e = copy.deepcopy(electrodes)
    for k, v in values.items():
        setattr(e.frontend if k == "smooth_hz" else e, k, v)
    return e


def _get(electrodes: Electrodes, name: str) -> float:
    return electrodes.frontend.smooth_hz if name == "smooth_hz" else getattr(electrodes, name)


def _bisect(f: Callable[[float], float], target: float, lo: float, hi: float, iterations: int = BISECT_STEPS
            ) -> float:
    """x in [lo, hi] (log scale) with f(x) ~ target, for an increasing f."""
    lo, hi = np.log(lo), np.log(hi)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if f(np.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def _pooled(stats: Sequence[Stats]) -> Stats:
    """Mean of each statistic over recordings (None where a recording has none)."""
    out = {}
    for k in stats[0]:
        v = [s[k] for s in stats if s[k] is not None]
        out[k] = float(np.mean(v)) if v else None
    return out


def _resize(mask: np.ndarray, n: int) -> np.ndarray:
    """A boolean mask resampled to n samples (nearest)."""
    mask = np.asarray(mask, bool)
    return mask[np.minimum((np.arange(n) * len(mask) / n).astype(int), len(mask) - 1)]


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.4g}"
