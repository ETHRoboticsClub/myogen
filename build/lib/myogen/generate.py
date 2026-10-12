"""Synthetic pretraining data generator: `myogen generate`, parameters in config.yml (generate and
electrodes.<device>).

Layout: <out>/shard_XXX.h5 (one per worker) + <out>/manifest.jsonl (one line per trial, with QC) +
<out>/config.json. Every trial is reproducible from (master seed, subject, session, trial) indices.
"""
import json
import os
import time

# One BLAS thread per worker process: parallelism comes from the process pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import logging
from multiprocessing import Manager, Pool
from pathlib import Path
from queue import Empty

import h5py
import mujoco
import numpy as np
from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from scipy import signal

from .config import resolve
from .emg import EMGGeometry, acquire, synthesize
from .log import console
from .model import Anatomy, fibre_lengths, load
from .motion import N_LATENT, REST, JointMap, intent_trajectory, smooth_profile
from .subject import DEVICE_FS, DEVICES, channel_count, sample_session, sample_subject
from .tracker import MuscleTracker, TrackerConfig

# QC thresholds: a trial is rejected (kept on disk but flagged) if the hand did not do what the labels say.
QC_MAX_RMSE_DEG = 5.0
QC_MAX_WRIST_RESERVE_NM = 0.02
QC_MAX_SATURATION = 0.05

log = logging.getLogger(__name__)
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        m, d = load()
        _MODEL = (m, d, Anatomy.from_model(m, d))
    return _MODEL


# Passive joint stiffness (Nm/rad) around the relaxed posture: joint capsules, ligaments and resting muscle
# tone. A relaxed hand sits at this equilibrium with (almost) no EMG; without it the muscles would have to hold
# any glove-reported rest posture actively. Order-of-magnitude values from passive-stiffness studies.
PASSIVE_STIFFNESS = {"pro_sup_r": 0.1, "deviation_r": 0.25, "flexion_r": 0.2, "cmc_flexion_r": 0.03,
                     "cmc_abduction_r": 0.03, "mp_flexion_r": 0.02, "ip_flexion_r": 0.01, "mcp": 0.04,
                     "mcp_abd": 0.02, "pm": 0.02, "md": 0.01}


def set_passive_springs(m, d, anat, q_rest, scale):
    """Joint springs on all hand/wrist joints, calibrated so q_rest (full qpos) is the passive equilibrium.

    The spring reference is offset so that, at q_rest, spring torque balances gravity and the passive muscle
    forces: a relaxed hand rests where all passive forces cancel, with no muscle drive.
    """
    k_all = np.zeros(m.njnt)
    for name in anat.joint_names:
        if name.startswith("mcp") and "abduction" in name:
            k = PASSIVE_STIFFNESS["mcp_abd"]
        elif name[:3] in ("mcp",) or name[:2] in ("pm", "md"):
            k = PASSIVE_STIFFNESS[name[:3] if name.startswith("mcp") else name[:2]]
        else:
            k = PASSIVE_STIFFNESS.get(name, 0.0)
        k_all[m.joint(name).id] = k * scale
    m.jnt_stiffness[:] = k_all
    m.qpos_spring[:] = q_rest
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q_rest
    mujoco.mj_forward(m, d)
    static = d.qfrc_bias - d.qfrc_actuator           # torque the springs must supply (actuator = passive muscle)
    for jid in np.flatnonzero(k_all):
        a, v = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        m.qpos_spring[a] = q_rest[a] + np.clip(static[v] / k_all[jid], -0.6, 0.6)


def grip_load_torque(m, anat, latent, load_level):
    """Joint torques opposing finger flexion, as if squeezing an object (T, nv). load_level in [0, 1]."""
    T = len(latent)
    tau = np.zeros((T, m.nv))
    per_joint = {"mcp": 0.35, "pm": 0.18, "md": 0.08}  # Nm at load_level 1 (~5-10 N fingertip force)
    for k, f in enumerate("2345"):
        closure = latent[:, 1 + k]
        for j, s in per_joint.items():
            dof = m.jnt_dofadr[m.joint(f"{j}{f}_flexion_r").id]
            tau[:, dof] = -s * load_level * np.clip(closure / 0.6, 0, 1)
    return tau


_BANK = None


def _bank(path, datasets=("db1", "db5")):
    global _BANK
    if _BANK is None:
        from .ninapro import LatentBank
        _BANK = LatentBank(path, datasets=tuple(datasets))
    return _BANK


_SYNTH = {}


def biomime_synth(subject, session, key, muap_bank_path, pool="fuglevand", gain=1.0):
    """BioMimeEMG for this (subject, session), built once per worker and reused across its trials."""
    from .biomime.bank import MuapBank
    from .biomime.synth import BioMimeEMG
    if key not in _SYNTH:
        _SYNTH.clear()                         # trials are grouped by (subject, session): keep only the current one
        if "bank" not in _SYNTH_BANK:
            _SYNTH_BANK["bank"] = MuapBank(muap_bank_path)
        _SYNTH[key] = BioMimeEMG(_SYNTH_BANK["bank"], subject, session, _model()[2], pool=pool, gain=gain)
    return _SYNTH[key]


_SYNTH_BANK = {}


def simulate_trial(subject, session, duration, seed, fs_ctrl=100, warmup=0.5, n_src=6,
                   fs_emg=2000, fs_out=2000, tracker_kwargs=None, motion="synthetic", bank=None,
                   biomime=None, save_spikes=False, on_setup=None, on_tick=None):
    """motion: "synthetic" (generator prior) or "ninapro" (real glove-derived intent from `bank`).
    biomime: a BioMimeEMG for this subject/session -> motor-unit-level EMG; None -> analytic source model.
    on_setup(setup): called once the recording setup is fixed (subject / session params, electrode meta, q0);
    on_tick(k, T, d, u): called after every control tick (progress, live view)."""
    m, d, anat = _model()
    rng = np.random.default_rng(seed)
    every = int(round(1 / (fs_ctrl * m.opt.timestep)))
    jmap = JointMap.sample(np.random.default_rng(subject.seed), *anat.hand_limits())
    n_warm = int(warmup * fs_ctrl)
    n_total = int(round((duration + warmup) * fs_ctrl))
    source, restim = "synthetic", np.full(n_total, -1, np.int16)
    if motion == "ninapro":
        z, restim, key, i0 = bank.sample(rng, n_total)
        source = f"{key}@{i0}"
    else:
        z = intent_trajectory(rng, duration + warmup, fs_ctrl, speed=rng.uniform(0.7, 1.4))[:n_total]
    z[:n_warm] = z[n_warm]
    T = len(z)
    # Relaxed posture: the subject's rest in the real recording, or the generator's REST.
    z_rest = np.median(z[restim == 0], axis=0) if (restim == 0).sum() > fs_ctrl else REST
    elbow = session.elbow_angle + 0.15 * (smooth_profile(rng, T, fs_ctrl, 0.0, 1.0, 0.3, tau=1.0) - 0.5)
    q_ref = anat.full_q(jmap(z), elbow)
    q_rest = anat.full_q(jmap(z_rest[None]), session.elbow_angle)[0]
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q_ref[0]
    mujoco.mj_forward(m, d)
    if session.arm_support > 0.5:
        # Forearm resting on a horizontal support: gravity perpendicular to the forearm axis (at this elbow angle).
        ax = d.xpos[m.body("lunate_r").id] - d.xpos[m.body("ulna_r").id]
        ax /= np.linalg.norm(ax)
        g = session.gravity_dir - (session.gravity_dir @ ax) * ax
        session.gravity_dir = g / np.linalg.norm(g)
    m.opt.gravity[:] = 9.81 * session.gravity_dir
    set_passive_springs(m, d, anat, q_rest, subject.passive_stiffness)
    fibre_rest = np.maximum(fibre_lengths(m, d, anat), 1e-3)       # d is at q_rest: reference fibre lengths
    qd_ref = np.gradient(q_ref, 1 / fs_ctrl, axis=0)
    qdd_ref = np.gradient(qd_ref, 1 / fs_ctrl, axis=0)
    # Engagement: how far the hand is from its relaxed posture or how fast it moves. Co-contraction only happens
    # when engaged; a relaxed hand is (nearly) electrically silent, as in real recordings.
    lp = signal.butter(2, 2.0, fs=fs_ctrl, output="sos")
    speed = signal.sosfiltfilt(lp, np.linalg.norm(np.gradient(z[:, :7], axis=0), axis=1) * fs_ctrl)
    moving = np.clip(speed - 0.15, 0, 1.5)                    # dead-zone: glove jitter during rest is not movement
    away = np.clip(np.linalg.norm(z[:, :7] - z_rest[:7], axis=1) - 0.1, 0, None) / 0.4
    engaged = np.clip(signal.sosfiltfilt(lp, np.clip(away + moving / 0.5, 0, 1)), 0, 1)
    cocon = (smooth_profile(rng, T, fs_ctrl, session.cocontraction_mean, 0.3, 0.25) + session.hold_tone) * engaged
    cocon = cocon + session.cocontraction_move_gain * moving   # stiffening while moving
    closed = 1 / (1 + np.exp(-(z[:, 1:5].mean(1) - 0.5) / 0.05))   # only squeeze when the hand is closed
    grip = smooth_profile(rng, T, fs_ctrl, 0.0, 1.0, session.load_prob) * closed
    load_tau = grip_load_torque(m, anat, z, grip)

    mujoco.mj_resetData(m, d)
    d.qpos[:] = q_ref[0]
    mujoco.mj_forward(m, d)
    tracker = MuscleTracker(m, d, TrackerConfig(ctrl_every=every, muscle_cost=subject.recruitment_cost,
                                                **(tracker_kwargs or {})))
    geom = EMGGeometry(m, d, anat, subject, session, n_src=n_src) if biomime is None else None
    setup_meta = {"gravity": session.gravity_dir.tolist(), "motion": source,
                  "emg_model": "analytic" if biomime is None else "biomime"}
    if geom is not None:
        setup_meta["electrodes_m"] = geom.electrodes.tolist()
    else:
        setup_meta["electrodes"] = [{"name": ch.name, "poles": ch.poles} for ch in biomime.channels]
        setup_meta["biomime"] = biomime.summary()
    if on_setup is not None:
        on_setup({"subject_params": subject.to_dict(), "session_params": session.to_dict(), "meta": setup_meta,
                  "q0": d.qpos.copy()})

    rec = {k: np.zeros((T, n)) for k, n in [("q", m.nq), ("qd", m.nv), ("act", m.nu), ("exc", m.nu),
                                            ("reserve", m.nv), ("fibre", m.nu)]}
    G = np.zeros((T, m.nu * n_src, subject.layout.n_channels), np.float32) if geom is not None else None
    residual = np.zeros(T)
    support = np.zeros(m.nv)
    for name, frac in [("elbow_flexion_r", session.arm_support), ("pro_sup_r", 0.7 * session.arm_support),
                       ("deviation_r", session.wrist_support), ("flexion_r", session.wrist_support)]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            support[m.jnt_dofadr[jid]] = frac
    for k in range(T):
        u, _, r = tracker.control(q_ref[k], qd_ref[k], qdd_ref[k], cocon[k], load_tau[k], support)
        if geom is not None:
            G[k] = geom.gains(d)      # state is current after mj_inverse
        d.ctrl[:] = u
        rec["exc"][k], rec["reserve"][k], residual[k] = u, r, tracker.last_residual
        for _ in range(every):
            mujoco.mj_step(m, d)
        rec["q"][k], rec["qd"][k], rec["act"][k] = d.qpos, d.qvel, d.act
        rec["fibre"][k] = fibre_lengths(m, d, anat)
        if on_tick is not None:
            on_tick(k, T, d, u)

    s = slice(n_warm, None)
    fibre_rel = rec["fibre"] / fibre_rest                       # normalised fibre length (1 = relaxed posture)
    spikes = None
    if biomime is None:
        emg = synthesize(rec["exc"][s], G[s], rec["qd"][s], subject, session, fs_ctrl,
                         fs=fs_emg, fs_out=fs_out, rng=rng)
    else:
        # Same smooth neural drive as the analytic model (the controller's fast corrections are not neural).
        u = signal.sosfiltfilt(signal.butter(2, 4.0, fs=fs_ctrl, output="sos"), rec["exc"][s], axis=0)
        x, spikes = biomime.synthesize(np.clip(u, 0, 1), fibre_rel[s], fs_ctrl, rng, return_spikes=True)
        g = np.gcd(int(fs_emg), 2048)
        x = signal.resample_poly(x, int(fs_emg) // g, 2048 // g, axis=0) if fs_emg != 2048 else x
        emg = acquire(x, rec["qd"][s], subject, session, fs_ctrl, fs=fs_emg, fs_out=fs_out, rng=rng)
    # Labels describe the state after each tick; q_ref[k+1] is the matching reference.
    q_ref_next = np.vstack([q_ref[1:], q_ref[-1:]])
    labels = {
        "q": rec["q"][s], "qd": rec["qd"][s], "q_ref": q_ref_next[s], "latent": z[s],
        "act": rec["act"][s], "exc": rec["exc"][s], "reserve": rec["reserve"][s],
        "cocontraction": cocon[s], "grip_load": grip[s], "restimulus": restim[s], "fibre_len": fibre_rel[s],
    }
    err = np.degrees(labels["q"] - labels["q_ref"])
    wrist = [m.jnt_dofadr[m.joint(n).id] for n in ("deviation_r", "flexion_r")]
    qc = {
        "rmse_deg": float(np.sqrt((err ** 2).mean())),
        "rmse_deg_per_joint": np.sqrt((err ** 2).mean(0)).round(2).tolist(),
        "max_err_deg": float(np.abs(err).max()),
        "wrist_reserve_rms_nm": float(np.sqrt((labels["reserve"][:, wrist] ** 2).mean())),
        "reserve_rms_nm": float(np.sqrt((labels["reserve"] ** 2).mean())),
        "saturation": float((labels["exc"] > 0.99).mean()),
        "residual": float(residual[s].mean()),
        "emg_rms_mv": np.sqrt((emg.astype(np.float64) ** 2).mean(0)).round(4).tolist(),
        "finite": bool(np.isfinite(emg).all() and np.isfinite(labels["q"]).all()),
    }
    qc["pass"] = bool(qc["finite"] and qc["rmse_deg"] < QC_MAX_RMSE_DEG
                      and qc["wrist_reserve_rms_nm"] < QC_MAX_WRIST_RESERVE_NM
                      and qc["saturation"] < QC_MAX_SATURATION)
    meta = dict(setup_meta)
    if save_spikes and spikes is not None:
        meta["_spikes"] = spikes
    return emg, labels, qc, meta


TICK_EVERY = 50      # control ticks between progress messages (0.5 s of simulated time)
LIVE_EVERY = 3       # control ticks between live-view states (33 Hz)


def _job(args):
    """One worker: simulates its trials into its own shard; progress (and live state) go to `queue`."""
    wid, shard_path, items, cfg, queue, live = args
    out = []
    with h5py.File(shard_path, "a") as f:
        for it in items:
            t0 = time.time()
            name = f"s{it['subject']:04d}_x{it['session']:02d}_t{it['trial']:02d}"
            ss = np.random.SeedSequence([cfg["seed"], it["subject"]])
            sub_rng = np.random.default_rng(ss)
            _, _, anat = _model()
            subject = sample_subject(sub_rng, len(anat.muscle_names), anat.extrinsic, cfg["electrodes"],
                                     amputee=it["amputee"], device=cfg["device"])
            ses_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], it["subject"], it["session"]]))
            session = sample_session(ses_rng, subject, support=cfg["support"])
            seed = np.random.SeedSequence([cfg["seed"], it["subject"], it["session"], it["trial"]])
            mot_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 7, it["subject"], it["session"],
                                                                    it["trial"]]))
            motion = cfg["motion"]
            if motion == "mix":
                motion = "ninapro" if mot_rng.random() < cfg["real_frac"] else "synthetic"
            bank = _bank(cfg["bank"], cfg["bank_datasets"]) if motion == "ninapro" else None
            synth = None
            if cfg["emg_model"] == "biomime":
                queue.put(("phase", wid, name, "BioMime setup"))
                synth = biomime_synth(subject, session, (it["subject"], it["session"]), cfg["muap_bank"], cfg["pool"],
                                      cfg["biomime_gain"])

            def on_tick(k, T, d, u):
                if k % TICK_EVERY == 0 or k == T - 1:
                    queue.put(("tick", wid, name, k + 1, T))
                if live and k % LIVE_EVERY == 0:
                    queue.put(("state", d.qpos.copy(), d.act.copy(), np.asarray(u, np.float32), k / cfg["fs_label"]))

            on_setup = (lambda setup: queue.put(("setup", name, setup))) if live else None
            emg, labels, qc, meta = simulate_trial(subject, session, cfg["duration"], seed,
                                                   fs_ctrl=cfg["fs_label"], fs_emg=cfg["fs_sim"],
                                                   fs_out=cfg["fs_emg"], motion=motion, bank=bank,
                                                   biomime=synth, save_spikes=cfg["save_spikes"],
                                                   on_setup=on_setup, on_tick=on_tick)
            queue.put(("phase", wid, name, "EMG + write"))
            spikes = meta.pop("_spikes", None)
            g = f.create_group(name)
            g.create_dataset("emg", data=emg, compression="gzip", compression_opts=4, chunks=True)
            for k, v in labels.items():
                v = v if v.dtype.kind in "iu" else v.astype(np.float32)
                g.create_dataset(f"labels/{k}", data=v, compression="gzip")
            if spikes is not None:
                # Motor-unit ground truth: spike (unit, sample @ 2048 Hz) pairs + the unit table.
                g.create_dataset("spikes/unit", data=spikes[0], compression="gzip")
                g.create_dataset("spikes/sample", data=spikes[1], compression="gzip")
                g["spikes"].attrs["fs"] = 2048
                ut = synth.units
                for k in ("muscle", "rank", "threshold", "size", "cond", "weight"):
                    g.create_dataset(f"spikes/units/{k}", data=getattr(ut, k), compression="gzip")
                g["spikes/units"].attrs["muscles"] = synth.muscles
            g.attrs.update({"subject": it["subject"], "session": it["session"], "trial": it["trial"],
                            "amputee": it["amputee"], "fs_emg": cfg["fs_emg"], "fs_label": cfg["fs_label"],
                            "subject_params": json.dumps(subject.to_dict()),
                            "session_params": json.dumps(session.to_dict()),
                            "meta": json.dumps(meta), "qc": json.dumps(qc)})
            f.flush()
            out.append({"id": name, "shard": Path(shard_path).name, **it, "qc": qc,
                        "seconds": cfg["duration"], "wall_s": round(time.time() - t0, 1)})
            queue.put(("done", wid, out[-1]))
    queue.put(("idle", wid))
    return out


def run(config):
    """Generate the dataset described by config (see config.yml: generate, electrodes.<device>). Returns the
    manifest records of the generated trials."""
    g = config["generate"]
    if g["device"] not in DEVICES:
        raise ValueError(f"generate.device must be one of {DEVICES}")
    _check_inputs(config)
    out = resolve(config, g["out"])
    out.mkdir(parents=True, exist_ok=True)
    workers = g["workers"] or max(1, (os.cpu_count() or 2) - 2)
    cfg = {k: v for k, v in g.items() if k not in ("out", "workers", "live_view")}
    cfg["electrodes"] = config["electrodes"][g["device"]]
    cfg["channels"] = channel_count(cfg["electrodes"])
    cfg["fs_emg"] = DEVICE_FS.get(g["device"], g["fs_emg"])
    cfg["bank"] = str(resolve(config, config["paths"]["latent_bank"]))
    cfg["muap_bank"] = str(resolve(config, config["paths"]["muap_bank"]))
    (out / "config.json").write_text(json.dumps({**cfg, "config_file": config["_file"]}, indent=2))

    amp_rng = np.random.default_rng(g["seed"])
    amputee = amp_rng.random(g["subjects"]) < g["amputee_frac"]
    items = [{"subject": s, "session": x, "trial": t, "amputee": bool(amputee[s])}
             for s in range(g["subjects"]) for x in range(g["sessions"]) for t in range(g["trials"])]
    # Keep each (subject, session) on one worker so per-session setup (e.g. BioMime projection) is built once.
    groups = {}
    for it in items:
        groups.setdefault((it["subject"], it["session"]), []).append(it)
    groups = list(groups.values())
    n = min(workers, len(groups))
    log.info(f"generating [bold]{len(items)}[/] trials ({len(items) * g['duration'] / 3600:.2f} h, "
             f"{g['device']}, {g['emg_model']} EMG, {g['motion']} motion) on {n} workers -> {out}")

    manager = Manager()
    queue = manager.Queue()
    jobs = [(i, str(out / f"shard_{i:03d}.h5"), [it for grp in groups[i::n] for it in grp], cfg, queue,
             bool(g["live_view"]) and i == 0) for i in range(n)]
    live = None
    if g["live_view"]:
        from .live import LiveView
        live = LiveView(config["view"]["port"], fs_label=cfg["fs_label"])
    results, t0 = [], time.time()
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(bar_width=30),
                        TaskProgressColumn(), TextColumn("{task.fields[info]}"), TimeElapsedColumn(),
                        TextColumn("eta"), TimeRemainingColumn(), console=console, transient=False)
    pool = Pool(n)
    try:
        with progress, open(out / "manifest.jsonl", "a") as manifest:
            total = progress.add_task("[bold]trials", total=len(items), info="")
            rows = [progress.add_task(f"  w{i:02d}", total=1, info="[dim]starting", start=True) for i in range(n)]
            current = [None] * n                     # trial each worker is on (its row restarts per trial)
            async_result = pool.map_async(_job, jobs)
            while True:
                finished = async_result.ready()
                for ev in _drain(queue):
                    kind = ev[0]
                    if kind in ("tick", "phase") and current[ev[1]] != ev[2]:
                        current[ev[1]] = ev[2]
                        progress.reset(rows[ev[1]], description=f"  w{ev[1]:02d} {ev[2]}")
                    if kind == "tick":
                        _, wid, name, k, T = ev
                        progress.update(rows[wid], completed=k, total=T, info="[dim]simulating")
                    elif kind == "phase":
                        progress.update(rows[ev[1]], info=f"[dim]{ev[3]}")
                    elif kind == "done":
                        r = ev[2]
                        results.append(r)
                        manifest.write(json.dumps(r) + "\n")
                        manifest.flush()
                        _log_trial(r)
                        fails = sum(not x["qc"]["pass"] for x in results)
                        hours = len(results) * g["duration"] / 3600
                        info = f"{hours:.2f} h" + (f" · [red]{fails} QC fail" if fails else "")
                        progress.update(total, advance=1, info=info)
                    elif kind == "idle":
                        progress.update(rows[ev[1]], visible=False)
                    elif live is not None:
                        live.handle(ev, len(results), len(items))
                if live is not None:
                    live.render()
                if finished:
                    break
                time.sleep(0.02)
            async_result.get()                       # re-raise worker exceptions
    except KeyboardInterrupt:
        pool.terminate()
        log.warning(f"interrupted: {len(results)} finished trials are in {out} (manifest.jsonl lists them)")
        raise
    finally:
        pool.close()
        manager.shutdown()
    if log.isEnabledFor(logging.INFO):
        _summary(results, g["duration"], time.time() - t0, out)
    if live is not None:
        live.finished(len(results))
    return results


def _check_inputs(config):
    """Fail before starting workers if a configured input is missing."""
    g, paths = config["generate"], config["paths"]
    if g["motion"] in ("ninapro", "mix") and not Path(paths["latent_bank"]).exists():
        raise FileNotFoundError(f"generate.motion: {g['motion']} needs the latent bank ({paths['latent_bank']}): "
                                f"`myogen download latent-bank`, or use generate.motion=synthetic")
    if g["emg_model"] == "biomime" and not Path(paths["muap_bank"]).exists():
        raise FileNotFoundError(f"generate.emg_model: biomime needs the MUAP bank ({paths['muap_bank']}): "
                                f"`myogen download muap-bank`")


def _drain(queue):
    while True:
        try:
            yield queue.get_nowait()
        except Empty:
            return


def _log_trial(r):
    qc = r["qc"]
    msg = f"{r['id']}  rmse {qc['rmse_deg']:.2f}°  ({r['wall_s']:.0f} s)"
    if qc["pass"]:
        log.debug(msg)
    else:
        log.warning(f"{msg}  [red]failed QC[/] (kept, flagged in the manifest)")


def _summary(results, duration, wall, out):
    t = Table(title="generation summary", show_header=False, box=None, title_justify="left")
    passed = sum(r["qc"]["pass"] for r in results)
    hours = len(results) * duration / 3600
    rmse = np.median([r["qc"]["rmse_deg"] for r in results]) if results else float("nan")
    t.add_row("trials", f"{len(results)}  ({passed} passed QC)")
    t.add_row("data", f"{hours:.2f} h")
    t.add_row("wall time", f"{wall:.0f} s  ({hours * 3600 / max(wall, 1e-9):.1f}x real time)")
    t.add_row("median tracking RMSE", f"{rmse:.2f}°")
    t.add_row("output", str(out))
    console.print(t)
