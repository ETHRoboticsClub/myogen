"""One trial: a latent trajectory drives the forearm model, the muscle tracker produces excitations, and an EMG
engine (analytic or BioMime) turns them into the recording. Shared by `simulate` and `myogen generate`."""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import signal

from ..body.loads import grip_load_torque, set_gravity, set_passive_springs, support_fractions
from ..body.model import fibre_lengths, set_pose, shared_model
from ..body.tracker import MuscleTracker, TrackerConfig
from ..emg.acquisition import acquire, resample
from ..emg.analytic import EMGGeometry, synthesize
from ..motion.latent import REST, JointMap
from ..motion.synthetic import intent_trajectory, smooth_profile

FS_CTRL = 100           # Hz: control rate and label rate
FS_SIM = 2000           # Hz: internal EMG synthesis rate
FS_BIOMIME = 2048       # Hz: BioMime MUAP / spike rate
WARMUP = 0.5            # s: the first posture is held so the tracker settles before the recording starts
N_SOURCES = 6           # analytic engine: point sources per muscle

# Quality control: a failed trial is kept but flagged (the hand did not do what the labels say).
QC_MAX_RMSE_DEG = 5.0
QC_MAX_WRIST_RESERVE_NM = 0.02
QC_MAX_SATURATION = 0.05


@dataclass
class TrialResult:
    emg: np.ndarray             # (N, C) device output
    labels: dict                # name -> (T, ...) at fs_ctrl
    qc: dict
    meta: dict                  # recording setup: gravity, motion source, engine, electrodes
    spikes: dict | None = None  # BioMime spike table (BioMimeEMG.spike_table) when requested


def sample_motion(rng, kind, duration, fs=FS_CTRL, bank=None):
    """(latent incl. warm-up, restimulus, source) for kind "synthetic" or "ninapro" (a window of `bank`)."""
    n = int(round((duration + WARMUP) * fs))
    if kind == "ninapro":
        z, restim, key, i0 = bank.sample(rng, n)
        return z, restim, f"{key}@{i0}"
    z = intent_trajectory(rng, duration + WARMUP, fs, speed=rng.uniform(0.7, 1.4))[:n]
    return z, np.full(n, -1, np.int16), "synthetic"


def given_motion(latent, restimulus=None, fs=FS_CTRL):
    """(latent incl. warm-up, restimulus, source) for a given (T, 10) latent at fs."""
    latent = np.asarray(latent, float)
    n_warm = int(WARMUP * fs)
    z = np.vstack([np.repeat(latent[:1], n_warm, 0), latent])
    restim = np.full(len(z), -1, np.int16)
    if restimulus is not None:
        restim[n_warm:] = restimulus
    return z, restim, "given"


def simulate_trial(subject, session, motion, rng, fs_ctrl=FS_CTRL, fs_sim=FS_SIM, fs_out=FS_SIM, biomime=None,
                   save_spikes=False, on_setup=None, on_tick=None):
    """motion: (latent, restimulus, source) from sample_motion / given_motion. biomime: a BioMimeEMG for this
    subject / session, None for the analytic engine. on_setup(setup) is called once the recording setup is fixed,
    on_tick(k, T, d, u) after every control tick (progress, live view)."""
    m, d, anat = shared_model()
    z, restim, source = motion
    n_warm = int(WARMUP * fs_ctrl)
    z[:n_warm] = z[n_warm]
    T = len(z)

    # Reference joint trajectory and relaxed posture (the recording's rest if it has one).
    jmap = JointMap.sample(np.random.default_rng(subject.seed), *anat.hand_limits())
    z_rest = np.median(z[restim == 0], axis=0) if (restim == 0).sum() > fs_ctrl else REST
    elbow = session.elbow_angle + 0.15 * (smooth_profile(rng, T, fs_ctrl, 0.0, 1.0, 0.3, tau=1.0) - 0.5)
    q_ref = anat.full_q(jmap(z), elbow)
    q_rest = anat.full_q(jmap(z_rest[None]), session.elbow_angle)[0]

    # Environment: gravity, passive springs at the relaxed posture, effort profiles.
    set_pose(m, d, q_ref[0])
    set_gravity(m, d, session)
    set_passive_springs(m, d, anat, q_rest, subject.passive_stiffness)
    fibre_rest = np.maximum(fibre_lengths(m, d, anat), 1e-3)       # d is at q_rest
    cocontraction, grip = effort_profiles(rng, z, z_rest, session, fs_ctrl)

    set_pose(m, d, q_ref[0])
    geom = EMGGeometry(m, d, anat, subject, session, n_src=N_SOURCES) if biomime is None else None
    meta = setup_meta(session, source, geom, biomime)
    if on_setup is not None:
        on_setup({"subject_params": subject.to_dict(), "session_params": session.to_dict(), "meta": meta,
                  "q0": d.qpos.copy()})

    rec = track(m, d, anat, subject, session, q_ref, cocontraction, grip_load_torque(m, z, grip), fs_ctrl,
                geom, on_tick)

    s = slice(n_warm, None)
    rec["fibre"] /= fibre_rest                                      # normalised fibre length (1 = relaxed)
    if biomime is None:
        emg = synthesize(rec["exc"][s], rec["gains"][s], rec["qd"][s], subject, session, fs_ctrl,
                         fs=fs_sim, fs_out=fs_out, rng=rng)
        spikes = None
    else:
        emg, spikes = biomime_emg(biomime, rec["exc"][s], rec["fibre"][s], rec["qd"][s], subject, session, rng,
                                  fs_ctrl, fs_sim, fs_out)
    q_ref_next = np.vstack([q_ref[1:], q_ref[-1:]])                  # labels are the state after each tick
    labels = {
        "q": rec["q"][s], "qd": rec["qd"][s], "q_ref": q_ref_next[s], "latent": z[s],
        "act": rec["act"][s], "exc": rec["exc"][s], "reserve": rec["reserve"][s],
        "cocontraction": cocontraction[s], "grip_load": grip[s], "restimulus": restim[s], "fibre_len": rec["fibre"][s],
    }
    return TrialResult(emg=emg, labels=labels, qc=quality(m, emg, labels, rec["residual"][s]), meta=meta,
                       spikes=biomime.spike_table(spikes) if save_spikes and spikes is not None else None)


def effort_profiles(rng, z, z_rest, session, fs):
    """(co-contraction, grip load), each (T,). Both only when the hand is engaged (away from rest or moving):
    a relaxed hand is nearly electrically silent, as in real recordings."""
    lp = signal.butter(2, 2.0, fs=fs, output="sos")
    speed = signal.sosfiltfilt(lp, np.linalg.norm(np.gradient(z[:, :7], axis=0), axis=1) * fs)
    moving = np.clip(speed - 0.15, 0, 1.5)                    # dead zone: glove jitter at rest is not movement
    away = np.clip(np.linalg.norm(z[:, :7] - z_rest[:7], axis=1) - 0.1, 0, None) / 0.4
    engaged = np.clip(signal.sosfiltfilt(lp, np.clip(away + moving / 0.5, 0, 1)), 0, 1)
    T = len(z)
    cocontraction = (smooth_profile(rng, T, fs, session.cocontraction_mean, 0.3, 0.25) + session.hold_tone) * engaged
    cocontraction = cocontraction + session.cocontraction_move_gain * moving     # stiffening while moving
    closed = 1 / (1 + np.exp(-(z[:, 1:5].mean(1) - 0.5) / 0.05))                 # only a closed hand squeezes
    grip = smooth_profile(rng, T, fs, 0.0, 1.0, session.load_prob) * closed
    return cocontraction, grip


def track(m, d, anat, subject, session, q_ref, cocontraction, load_tau, fs, geom=None, on_tick=None):
    """Run the muscle tracker along q_ref. Returns the per-tick recordings (and the analytic gain matrices)."""
    every = int(round(1 / (fs * m.opt.timestep)))
    qd_ref = np.gradient(q_ref, 1 / fs, axis=0)
    qdd_ref = np.gradient(qd_ref, 1 / fs, axis=0)
    tracker = MuscleTracker(m, d, TrackerConfig(ctrl_every=every, muscle_cost=subject.recruitment_cost))
    support = support_fractions(m, session)
    T = len(q_ref)
    rec = {k: np.zeros((T, n)) for k, n in [("q", m.nq), ("qd", m.nv), ("act", m.nu), ("exc", m.nu),
                                            ("reserve", m.nv), ("fibre", m.nu)]}
    rec["residual"] = np.zeros(T)
    if geom is not None:
        rec["gains"] = np.zeros((T, m.nu * geom.K, subject.layout.n_channels), np.float32)
    for k in range(T):
        u, _, reserve = tracker.control(q_ref[k], qd_ref[k], qdd_ref[k], cocontraction[k], load_tau[k], support)
        if geom is not None:
            rec["gains"][k] = geom.gains(d)          # the state is current after mj_inverse
        d.ctrl[:] = u
        rec["exc"][k], rec["reserve"][k], rec["residual"][k] = u, reserve, tracker.last_residual
        for _ in range(every):
            mujoco.mj_step(m, d)
        rec["q"][k], rec["qd"][k], rec["act"][k] = d.qpos, d.qvel, d.act
        rec["fibre"][k] = fibre_lengths(m, d, anat)
        if on_tick is not None:
            on_tick(k, T, d, u)
    return rec


def biomime_emg(biomime, exc, fibre_rel, qd, subject, session, rng, fs_ctrl, fs_sim, fs_out):
    """(device output, spikes) from the BioMime engine, through the same recording chain as the analytic one."""
    # Same smooth neural drive as the analytic engine (the controller's fast corrections are not neural).
    u = signal.sosfiltfilt(signal.butter(2, 4.0, fs=fs_ctrl, output="sos"), exc, axis=0)
    x, spikes = biomime.synthesize(np.clip(u, 0, 1), fibre_rel, fs_ctrl, rng, return_spikes=True)
    x = resample(x, FS_BIOMIME, fs_sim)
    return acquire(x, qd, subject, session, fs_ctrl, fs=fs_sim, fs_out=fs_out, rng=rng), spikes


def setup_meta(session, source, geom, biomime):
    meta = {"gravity": session.gravity_dir.tolist(), "motion": source,
            "emg_model": "analytic" if biomime is None else "biomime"}
    if biomime is None:
        meta["electrodes_m"] = geom.electrodes.tolist()
    else:
        meta["electrodes"] = [{"name": ch.name, "poles": ch.poles} for ch in biomime.channels]
        meta["biomime"] = biomime.summary()
    return meta


def quality(m, emg, labels, residual):
    err = np.degrees(labels["q"] - labels["q_ref"])
    wrist = [m.jnt_dofadr[m.joint(n).id] for n in ("deviation_r", "flexion_r")]
    qc = {
        "rmse_deg": float(np.sqrt((err ** 2).mean())),
        "rmse_deg_per_joint": np.sqrt((err ** 2).mean(0)).round(2).tolist(),
        "max_err_deg": float(np.abs(err).max()),
        "wrist_reserve_rms_nm": float(np.sqrt((labels["reserve"][:, wrist] ** 2).mean())),
        "reserve_rms_nm": float(np.sqrt((labels["reserve"] ** 2).mean())),
        "saturation": float((labels["exc"] > 0.99).mean()),
        "residual": float(residual.mean()),
        "emg_rms_mv": np.sqrt((emg.astype(np.float64) ** 2).mean(0)).round(4).tolist(),
        "finite": bool(np.isfinite(emg).all() and np.isfinite(labels["q"]).all()),
    }
    qc["pass"] = bool(qc["finite"] and qc["rmse_deg"] < QC_MAX_RMSE_DEG
                      and qc["wrist_reserve_rms_nm"] < QC_MAX_WRIST_RESERVE_NM
                      and qc["saturation"] < QC_MAX_SATURATION)
    return qc


def write_recording(group, emg, labels, spikes=None, spike_muscles=None):
    """EMG, labels and spikes as datasets of an HDF5 group (one trial)."""
    group.create_dataset("emg", data=emg, compression="gzip", compression_opts=4, chunks=True)
    for k, v in labels.items():
        group.create_dataset(f"labels/{k}", data=v if v.dtype.kind in "iu" else v.astype(np.float32),
                             compression="gzip")
    if spikes is not None:
        for k, v in spikes.items():
            group.create_dataset(f"spikes/{k}", data=v, compression="gzip")
        group["spikes"].attrs["fs"] = FS_BIOMIME
        group["spikes/units"].attrs["muscles"] = spike_muscles
