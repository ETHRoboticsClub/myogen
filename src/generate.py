"""Synthetic pretraining data generator.

    python -m src.generate --out data/pilot --subjects 8 --sessions 2 --trials 2 --duration 30

Layout: <out>/shard_XXX.h5 (one per worker) + <out>/manifest.jsonl (one line per trial, with QC) +
<out>/config.json. Every trial is reproducible from (master seed, subject, session, trial) indices.
"""
import argparse
import json
import os
import time

# One BLAS thread per worker process: parallelism comes from the process pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy import signal

from .emg import EMGGeometry, acquire, synthesize
from .model import Anatomy, fibre_lengths, load
from .motion import N_LATENT, REST, JointMap, intent_trajectory, smooth_profile
from .subject import DEVICE_FS, DEVICES, sample_session, sample_subject
from .tracker import MuscleTracker, TrackerConfig

# QC thresholds: a trial is rejected (kept on disk but flagged) if the hand did not do what the labels say.
QC_MAX_RMSE_DEG = 5.0
QC_MAX_WRIST_RESERVE_NM = 0.02
QC_MAX_SATURATION = 0.05

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


def biomime_synth(subject, session, key, muap_bank_path, pool="fuglevand"):
    """BioMimeEMG for this (subject, session), built once per worker and reused across its trials."""
    from .biomime.bank import MuapBank
    from .biomime.synth import BioMimeEMG
    if key not in _SYNTH:
        _SYNTH.clear()                         # trials are grouped by (subject, session): keep only the current one
        if "bank" not in _SYNTH_BANK:
            _SYNTH_BANK["bank"] = MuapBank(muap_bank_path)
        _SYNTH[key] = BioMimeEMG(_SYNTH_BANK["bank"], subject, session, _model()[2], pool=pool)
    return _SYNTH[key]


_SYNTH_BANK = {}


def simulate_trial(subject, session, duration, seed, fs_ctrl=100, warmup=0.5, n_src=6,
                   fs_emg=2000, fs_out=2000, tracker_kwargs=None, motion="synthetic", bank=None,
                   biomime=None, save_spikes=False):
    """motion: "synthetic" (generator prior) or "ninapro" (real glove-derived intent from `bank`).
    biomime: a BioMimeEMG for this subject/session -> motor-unit-level EMG; None -> analytic source model."""
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
    meta = {"gravity": session.gravity_dir.tolist(), "motion": source,
            "emg_model": "analytic" if biomime is None else "biomime"}
    if geom is not None:
        meta["electrodes_m"] = geom.electrodes.tolist()
    else:
        meta["electrodes"] = [{"name": ch.name, "poles": ch.poles} for ch in biomime.channels]
        meta["biomime"] = biomime.summary()
    if save_spikes and spikes is not None:
        meta["_spikes"] = spikes
    return emg, labels, qc, meta


def _job(args):
    (shard_path, items, cfg) = args
    out = []
    with h5py.File(shard_path, "a") as f:
        for it in items:
            t0 = time.time()
            ss = np.random.SeedSequence([cfg["seed"], it["subject"]])
            sub_rng = np.random.default_rng(ss)
            _, _, anat = _model()
            subject = sample_subject(sub_rng, len(anat.muscle_names), anat.extrinsic, n_channels=cfg["channels"],
                                     rings=cfg["rings"], amputee=it["amputee"], device=cfg["device"])
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
                synth = biomime_synth(subject, session, (it["subject"], it["session"]), cfg["muap_bank"], cfg["pool"])
            emg, labels, qc, meta = simulate_trial(subject, session, cfg["duration"], seed,
                                                   fs_ctrl=cfg["fs_label"], fs_emg=cfg["fs_sim"],
                                                   fs_out=cfg["fs_emg"], motion=motion, bank=bank,
                                                   biomime=synth, save_spikes=cfg["save_spikes"])
            spikes = meta.pop("_spikes", None)
            name = f"s{it['subject']:04d}_x{it['session']:02d}_t{it['trial']:02d}"
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
            print(f"[{Path(shard_path).stem}] {name} rmse={qc['rmse_deg']:.2f}deg pass={qc['pass']} "
                  f"({out[-1]['wall_s']}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=8)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--duration", type=float, default=30.0, help="seconds per trial")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--rings", type=int, default=1)
    ap.add_argument("--amputee-frac", type=float, default=0.3)
    ap.add_argument("--device", choices=DEVICES, default="raw",
                    help="raw sEMG, or emulate NinaPro hardware: otto_bock (DB1, 10 ch envelope @100 Hz), "
                         "myo (DB5, 16 ch int8 @200 Hz); sets channels and output rate")
    ap.add_argument("--motion", choices=["synthetic", "ninapro", "mix"], default="mix",
                    help="intent source: generator prior, real NinaPro glove trajectories, or a mix")
    ap.add_argument("--real-frac", type=float, default=0.5, help="fraction of ninapro motion when --motion mix")
    ap.add_argument("--bank", default=str(Path(__file__).resolve().parent.parent / "data" / "real" / "latent_bank.h5"))
    ap.add_argument("--support", choices=["random", "supported", "free"], default="random",
                    help="forearm on an armrest (NinaPro protocol), held up, or a 50/50 mix")
    ap.add_argument("--bank-datasets", nargs="+", default=["db1", "db5"], help="restrict real motion to these")
    ap.add_argument("--emg-model", choices=["analytic", "biomime"], default="analytic",
                    help="analytic interference model (fast) or BioMime motor-unit-level synthesis")
    ap.add_argument("--muap-bank", default=str(Path(__file__).resolve().parent.parent / "data" / "biomime" / "muap_bank.h5"))
    ap.add_argument("--pool", choices=["fuglevand", "lif"], default="fuglevand", help="motor-unit pool (biomime)")
    ap.add_argument("--save-spikes", action="store_true", help="store motor-unit spike trains (biomime)")
    ap.add_argument("--fs-emg", type=int, default=2000, help="output EMG rate for --device raw")
    ap.add_argument("--fs-sim", type=int, default=2000, help="internal EMG synthesis rate")
    ap.add_argument("--fs-label", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.device in DEVICE_FS:
        args.fs_emg = DEVICE_FS[args.device]
        args.channels = {"otto_bock": 10, "myo": 16}[args.device]
    cfg = {k: v for k, v in vars(args).items() if k not in ("out", "workers")}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    amp_rng = np.random.default_rng(args.seed)
    amputee = amp_rng.random(args.subjects) < args.amputee_frac
    items = [{"subject": s, "session": x, "trial": t, "amputee": bool(amputee[s])}
             for s in range(args.subjects) for x in range(args.sessions) for t in range(args.trials)]
    # Keep each (subject, session) on one worker so per-session setup (e.g. BioMime projection) is built once.
    groups = {}
    for it in items:
        groups.setdefault((it["subject"], it["session"]), []).append(it)
    groups = list(groups.values())
    n = min(args.workers, len(groups))
    jobs = [(str(out / f"shard_{i:03d}.h5"), [it for grp in groups[i::n] for it in grp], cfg) for i in range(n)]
    t0 = time.time()
    with Pool(n) as pool:
        results = [r for rs in pool.map(_job, jobs) for r in rs]
    with open(out / "manifest.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    passed = sum(r["qc"]["pass"] for r in results)
    hours = len(results) * args.duration / 3600
    print(f"done: {len(results)} trials ({hours:.2f} h), {passed} passed QC, {time.time() - t0:.0f}s wall")


if __name__ == "__main__":
    main()
