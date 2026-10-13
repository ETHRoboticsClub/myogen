"""`myogen generate`: a dataset of many subjects, sessions and trials, simulated in parallel.

Output: <out>/shard_XXX.h5 (one per worker, one group per trial) + manifest.jsonl (one line per trial, with QC)
+ config.json. Every trial is reproducible from (seed, subject, session, trial).
"""
import os

# One BLAS thread per worker process: parallelism comes from the process pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools
import json
import logging
import time
from multiprocessing import Manager, Pool
from pathlib import Path
from queue import Empty

import h5py
import numpy as np
from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table

from ..body.model import shared_model
from ..config import as_path
from ..emg.acquisition import DEVICE_FS, DEVICES
from ..log import console
from ..simulation.trial import sample_motion, simulate_trial, write_recording
from ..subject import channel_count, sample_session, sample_subject

TICK_EVERY = 50      # control ticks between progress messages (0.5 s of simulated time)
LIVE_EVERY = 3       # control ticks between live-view states (33 Hz)

log = logging.getLogger(__name__)


def run(config):
    """Generate the dataset described by config (generate, electrodes.<device>). Returns the manifest records."""
    g = config["generate"]
    if g["device"] not in DEVICES:
        raise ValueError(f"generate.device must be one of {DEVICES}")
    _check_inputs(config)
    out = as_path(g["out"])
    out.mkdir(parents=True, exist_ok=True)
    cfg = _worker_config(config)
    (out / "config.json").write_text(json.dumps({**cfg, "config_file": config["_file"]}, indent=2))

    groups = _plan(g)
    n_trials = sum(len(grp) for grp in groups)
    n = min(g["workers"] or max(1, (os.cpu_count() or 2) - 2), len(groups))
    log.info(f"generating [bold]{n_trials}[/] trials ({n_trials * g['duration'] / 3600:.2f} h, {g['device']}, "
             f"{g['emg_model']} EMG, {g['motion']} motion) on {n} workers -> {out}")

    live = None
    if g["live_view"]:
        from ..viewer.live import LiveView
        live = LiveView(config["view"]["port"], fs_label=cfg["fs_label"])
    manager = Manager()
    queue = manager.Queue()
    jobs = [(i, str(out / f"shard_{i:03d}.h5"), [it for grp in groups[i::n] for it in grp], cfg, queue,
             live is not None and i == 0) for i in range(n)]
    board = ProgressBoard(n, n_trials, g["duration"])
    pool = Pool(n)
    t0 = time.time()
    try:
        with board.progress, open(out / "manifest.jsonl", "a") as manifest:
            pending = pool.map_async(_worker, jobs)
            while True:
                finished = pending.ready()
                for ev in _drain(queue):
                    record = board.handle(ev)
                    if record is not None:
                        manifest.write(json.dumps(record) + "\n")
                        manifest.flush()
                    elif live is not None and ev[0] in ("setup", "state"):
                        live.handle(ev, len(board.results), n_trials)
                if live is not None:
                    live.render()
                if finished:
                    break
                time.sleep(0.02)
            pending.get()                            # re-raise worker exceptions
    except KeyboardInterrupt:
        pool.terminate()
        log.warning(f"interrupted: {len(board.results)} finished trials are in {out} (listed in manifest.jsonl)")
        raise
    finally:
        pool.close()
        manager.shutdown()
    if log.isEnabledFor(logging.INFO):
        _summary(board.results, g["duration"], time.time() - t0, out)
    if live is not None:
        live.finished(len(board.results))
    return board.results


def _worker_config(config):
    """The plain dict each worker gets (also written to config.json)."""
    g = config["generate"]
    cfg = {k: v for k, v in g.items() if k not in ("out", "workers", "live_view")}
    cfg["electrodes"] = config["electrodes"][g["device"]]
    cfg["channels"] = channel_count(cfg["electrodes"])
    cfg["fs_emg"] = DEVICE_FS.get(g["device"], g["fs_emg"])
    cfg["bank"] = str(as_path(config["paths"]["latent_bank"]))
    cfg["muap_bank"] = str(as_path(config["paths"]["muap_bank"]))
    return cfg


def _plan(g):
    """Trials grouped by (subject, session): a group stays on one worker so per-session setup is built once."""
    amputee = np.random.default_rng(g["seed"]).random(g["subjects"]) < g["amputee_frac"]
    groups = {}
    for s in range(g["subjects"]):
        for x in range(g["sessions"]):
            for t in range(g["trials"]):
                groups.setdefault((s, x), []).append({"subject": s, "session": x, "trial": t,
                                                      "amputee": bool(amputee[s])})
    return list(groups.values())


def _check_inputs(config):
    """Fail before starting the workers if a configured input is missing."""
    g, paths = config["generate"], config["paths"]
    if g["motion"] in ("ninapro", "mix") and not Path(paths["latent_bank"]).exists():
        raise FileNotFoundError(f"generate.motion: {g['motion']} needs the latent bank ({paths['latent_bank']}): "
                                f"`myogen download latent-bank`, or use generate.motion=synthetic")
    if g["emg_model"] == "biomime" and not Path(paths["muap_bank"]).exists():
        raise FileNotFoundError(f"generate.emg_model: biomime needs the MUAP bank ({paths['muap_bank']}): "
                                f"`myogen download muap-bank`")


# -- worker process ------------------------------------------------------------------------------------------

def _worker(args):
    """Simulate a list of trials into one shard; progress (and live state) go to `queue`."""
    wid, shard_path, items, cfg, queue, live = args
    records = []
    with h5py.File(shard_path, "a") as f:
        for item in items:
            t0 = time.time()
            name = f"s{item['subject']:04d}_x{item['session']:02d}_t{item['trial']:02d}"
            subject, session = _subject_session(cfg, item)
            motion_kind, rng = _trial_randomness(cfg, item)
            bank = _latent_bank(cfg["bank"], tuple(cfg["bank_datasets"])) if motion_kind == "ninapro" else None
            motion = sample_motion(rng, motion_kind, cfg["duration"], cfg["fs_label"], bank)
            synth = None
            if cfg["emg_model"] == "biomime":
                queue.put(("phase", wid, name, "BioMime setup"))
                synth = _biomime_synth(subject, session, (item["subject"], item["session"]), cfg)

            def on_tick(k, T, d, u):
                if k % TICK_EVERY == 0 or k == T - 1:
                    queue.put(("tick", wid, name, k + 1, T))
                if live and k % LIVE_EVERY == 0:
                    queue.put(("state", d.qpos.copy(), d.act.copy(), np.asarray(u, np.float32), k / cfg["fs_label"]))

            trial = simulate_trial(subject, session, motion, rng, fs_ctrl=cfg["fs_label"], fs_sim=cfg["fs_sim"],
                                   fs_out=cfg["fs_emg"], biomime=synth, save_spikes=cfg["save_spikes"],
                                   on_setup=(lambda setup: queue.put(("setup", name, setup))) if live else None,
                                   on_tick=on_tick)
            queue.put(("phase", wid, name, "EMG + write"))
            _write_trial(f, name, item, cfg, subject, session, trial, synth)
            records.append({"id": name, "shard": Path(shard_path).name, **item, "qc": trial.qc,
                            "seconds": cfg["duration"], "wall_s": round(time.time() - t0, 1)})
            queue.put(("done", wid, records[-1]))
    queue.put(("idle", wid))
    return records


def _subject_session(cfg, item):
    s, x = item["subject"], item["session"]
    _, _, anat = shared_model()
    subject = sample_subject(np.random.default_rng(np.random.SeedSequence([cfg["seed"], s])),
                             len(anat.muscle_names), anat.extrinsic, cfg["electrodes"],
                             amputee=item["amputee"], device=cfg["device"])
    session = sample_session(np.random.default_rng(np.random.SeedSequence([cfg["seed"], s, x])), subject,
                             support=cfg["support"])
    return subject, session


def _trial_randomness(cfg, item):
    """(motion kind, the trial's random generator). motion "mix" picks ninapro with probability real_frac."""
    s, x, t = item["subject"], item["session"], item["trial"]
    kind = cfg["motion"]
    if kind == "mix":
        pick = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 7, s, x, t]))
        kind = "ninapro" if pick.random() < cfg["real_frac"] else "synthetic"
    return kind, np.random.default_rng(np.random.SeedSequence([cfg["seed"], s, x, t]))


@functools.cache
def _latent_bank(path, datasets):
    from ..motion.ninapro import LatentBank
    return LatentBank(path, datasets=datasets)


@functools.cache
def _muap_bank(path):
    from ..emg.biomime.bank import MuapBank
    return MuapBank(path)


_synth_cache = {}


def _biomime_synth(subject, session, key, cfg):
    """BioMimeEMG for this (subject, session), built once and reused across its trials."""
    from ..emg.biomime.synth import BioMimeEMG
    if key not in _synth_cache:
        _synth_cache.clear()                     # trials come grouped by (subject, session)
        _synth_cache[key] = BioMimeEMG(_muap_bank(cfg["muap_bank"]), subject, session, shared_model()[2],
                                       pool=cfg["pool"], gain=cfg["biomime_gain"])
    return _synth_cache[key]


def _write_trial(f, name, item, cfg, subject, session, trial, synth):
    g = f.create_group(name)
    write_recording(g, trial.emg, trial.labels, trial.spikes, synth.muscles if synth is not None else None)
    g.attrs.update({"subject": item["subject"], "session": item["session"], "trial": item["trial"],
                    "amputee": item["amputee"], "fs_emg": cfg["fs_emg"], "fs_label": cfg["fs_label"],
                    "subject_params": json.dumps(subject.to_dict()),
                    "session_params": json.dumps(session.to_dict()),
                    "meta": json.dumps(trial.meta), "qc": json.dumps(trial.qc)})
    f.flush()


# -- progress display ----------------------------------------------------------------------------------------

class ProgressBoard:
    """One bar for all trials and one row per worker; handle() consumes the workers' events."""

    def __init__(self, n_workers, n_trials, duration):
        self.duration, self.results = duration, []
        self.progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(bar_width=30),
                                 TaskProgressColumn(), TextColumn("{task.fields[info]}"), TimeElapsedColumn(),
                                 TextColumn("eta"), TimeRemainingColumn(), console=console, transient=False)
        self.total = self.progress.add_task("[bold]trials", total=n_trials, info="")
        self.rows = [self.progress.add_task(f"  w{i:02d}", total=1, info="[dim]starting") for i in range(n_workers)]
        self.current = [None] * n_workers         # trial each worker is on (its row restarts per trial)

    def handle(self, ev):
        """Update the display; returns the manifest record when a trial finished."""
        kind = ev[0]
        if kind in ("tick", "phase") and self.current[ev[1]] != ev[2]:
            self.current[ev[1]] = ev[2]
            self.progress.reset(self.rows[ev[1]], description=f"  w{ev[1]:02d} {ev[2]}")
        if kind == "tick":
            _, wid, _, k, T = ev
            self.progress.update(self.rows[wid], completed=k, total=T, info="[dim]simulating")
        elif kind == "phase":
            self.progress.update(self.rows[ev[1]], info=f"[dim]{ev[3]}")
        elif kind == "idle":
            self.progress.update(self.rows[ev[1]], visible=False)
        elif kind == "done":
            record = ev[2]
            self.results.append(record)
            _log_trial(record)
            fails = sum(not r["qc"]["pass"] for r in self.results)
            info = f"{len(self.results) * self.duration / 3600:.2f} h" + (f" · [red]{fails} QC fail" if fails else "")
            self.progress.update(self.total, advance=1, info=info)
            return record
        return None


def _drain(queue):
    while True:
        try:
            yield queue.get_nowait()
        except Empty:
            return


def _log_trial(r):
    msg = f"{r['id']}  rmse {r['qc']['rmse_deg']:.2f}°  ({r['wall_s']:.0f} s)"
    if r["qc"]["pass"]:
        log.debug(msg)
    else:
        log.warning(f"{msg}  [red]failed QC[/] (kept, flagged in the manifest)")


def _summary(results, duration, wall, out):
    passed = sum(r["qc"]["pass"] for r in results)
    hours = len(results) * duration / 3600
    rmse = np.median([r["qc"]["rmse_deg"] for r in results]) if results else float("nan")
    t = Table(title="generation summary", show_header=False, box=None, title_justify="left")
    t.add_row("trials", f"{len(results)}  ({passed} passed QC)")
    t.add_row("data", f"{hours:.2f} h")
    t.add_row("wall time", f"{wall:.0f} s  ({hours * 3600 / max(wall, 1e-9):.1f}x real time)")
    t.add_row("median tracking RMSE", f"{rmse:.2f}°")
    t.add_row("output", str(out))
    console.print(t)
