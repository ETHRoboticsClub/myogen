"""Simulate the EMG of a NinaPro recording from its glove motion, and compare it with the real EMG.

    python scripts/ninapro_compare.py data/real/ninapro_db5/s1/S1_E2_A1.mat --engine biomime --start 20 --duration 30
    python scripts/ninapro_compare.py data/real/ninapro_db1/s1/S1_A1_E2.mat --engine analytic

Needs myogen (pip install -e .), and matplotlib for the figure. Writes into --out:
    simulated.h5      the myogen Recording (myogen replay simulated.h5)
    comparison.json   real vs simulated statistics (myogen.compare)
    comparison.png    real and simulated EMG, envelopes and the joint motion
With --calibrate PROFILE.yml it also fits the device settings on this window (with --fit-rotation also the
armband rotation, unknown in NinaPro) and writes the profile. The comparison is then no longer independent:
calibrate on other subjects than the ones you report.

Glove -> joint angles: the CyberGlove sensors are normalised per subject (1st-99th percentile over all the
subject's files in the folder, rest-centred for the wrist), turned into a hand posture (digit closures, thumb
abduction, spread, wrist) and mapped to the model's joints with fixed finger coupling. Pronation / supination
comes from the accelerometer inside the Myo armbands (DB5: gravity's roll around the forearm, relative to rest)
or the movement labels (DB1, exercise 2); during labelled rest it returns to the rest reference (the subject is
relaxed; this also removes accelerometer transients). DB1's wrist-pitch sensor duplicates sensor 12, so its wrist
flexion also comes from the labels. This is a heuristic retargeting: the glove is uncalibrated.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat

import myogen as mg

# CyberGlove 22 sensor indices (checked against the single-finger movements of exercise 1).
THUMB_ROT, THUMB_MCP, THUMB_IP, THUMB_ABD = 0, 1, 2, 3
FINGERS = [(4, 5, 6), (7, 8, 9), (11, 12, 13), (15, 16, 17)]           # index, middle, ring, little: MCP PIP DIP
FINGER_ABD = (10, 14, 18)
WRIST_PITCH, WRIST_YAW = 20, 21
# Exercise-2 movement id -> (posture channel, target value), for channels the glove does not measure.
EX2_TARGETS = {9: ("pro_sup", 0.7), 10: ("pro_sup", -0.7), 11: ("pro_sup", 0.7), 12: ("pro_sup", -0.7),
               13: ("wrist_flex", 0.7), 14: ("wrist_flex", -0.7), 17: ("wrist_flex", -0.6)}


def load_mat(path):
    d = loadmat(path)
    db5 = "acc" in d
    rec = {"path": Path(path), "dataset": "db5" if db5 else "db1", "fs": 200.0 if db5 else 100.0,
           "emg": d["emg"].astype(np.float32), "glove": d["glove"].astype(float),
           "restimulus": d["restimulus"].ravel().astype(int),
           "exercise": int(Path(path).stem.split("_E")[1][0])}            # the file name is authoritative
    if db5:
        rec["acc"] = d["acc"].astype(float)
    return rec


def glove_to_angles(rec, subject_recs):
    """{model joint: (T,) rad} for one recording; subject_recs (all files of the subject) set the normalisation."""
    glove_all = np.concatenate([r["glove"] for r in subject_recs])
    lo, hi = np.percentile(glove_all, [1, 99], axis=0)
    span = np.maximum(hi - lo, 1e-6)
    rest_med = np.median(np.concatenate([r["glove"][r["restimulus"] == 0] for r in subject_recs]), axis=0)
    fs = rec["fs"]
    g = _lowpass(rec["glove"], 3.0, fs)
    n = np.clip((g - lo) / span, 0, 1)                                 # 0..1 per sensor
    c = np.clip((g - rest_med) / (span / 2), -1, 1)                    # rest-centred, -1..1

    thumb = np.clip(1 - n[:, [THUMB_MCP, THUMB_IP]].mean(1), 0, 1)    # thumb flexion lowers sensors 1, 2
    fingers = [np.clip(n[:, list(idx)].mean(1), 0, 1) for idx in FINGERS]
    thumb_abd = np.clip(0.5 * (n[:, THUMB_ROT] + 1 - n[:, THUMB_ABD]), 0, 1)
    spread = np.clip(1 - n[:, list(FINGER_ABD)].mean(1), 0, 1)
    pitch_broken = rec["dataset"] == "db1" and np.corrcoef(glove_all[:, WRIST_PITCH], glove_all[:, 12])[0, 1] > 0.95
    wrist_flex = _from_labels(rec, "wrist_flex") if pitch_broken else c[:, WRIST_PITCH]
    wrist_dev = c[:, WRIST_YAW]
    pro_sup = _pro_sup(rec, subject_recs)

    # Posture -> joints with fixed coupling (MCP takes 87.5 % of a finger's closure, DIP = 0.725 PIP).
    lerp = lambda a, b, t: a + (b - a) * t
    q = {"pro_sup_r": 1.3 * pro_sup, "deviation_r": np.where(wrist_dev > 0, 0.4, 0.17) * wrist_dev,
         "flexion_r": 0.75 * wrist_flex * np.where(wrist_flex > 0, 1 - 0.55 * np.mean(fingers, 0), 1.0),
         "cmc_flexion_r": lerp(0.3, -0.6, thumb), "cmc_abduction_r": lerp(-0.4, 0.7, thumb_abd),
         "mp_flexion_r": lerp(0.2, -0.6, thumb), "ip_flexion_r": lerp(0.2, -1.0, thumb)}
    for k, (f, closure) in enumerate(zip("2345", fingers)):
        q[f"mcp{f}_flexion_r"] = 1.45 * np.clip(closure * 0.875 / 0.9, 0, 1)
        q[f"mcp{f}_abduction_r"] = 0.22 * [-1.0, -0.2, 0.6, 1.0][k] * (spread - 0.3) * (1 - closure)
        q[f"pm{f}_flexion_r"] = 1.5 * closure
        q[f"md{f}_flexion_r"] = 0.725 * 1.5 * closure
    return q


def _pro_sup(rec, subject_recs):
    """Pronation / supination in [-1, 1]: Myo accelerometer roll (DB5) or the movement labels (DB1)."""
    if "acc" not in rec:
        return _from_labels(rec, "pro_sup")
    acc_rest = np.concatenate([r["acc"][r["restimulus"] == 0] for r in subject_recs])
    roll_ref = np.median(np.arctan2(acc_rest[:, 1], acc_rest[:, 2]))
    gain = 1.0
    ex2 = [r for r in subject_recs if r["exercise"] == 2]
    if ex2:
        # The armband follows only part of the radius's rotation: scale the labelled pro/supinations to +-0.7.
        roll2 = np.arctan2(ex2[0]["acc"][:, 1], ex2[0]["acc"][:, 2]) - roll_ref
        peaks = [np.percentile(np.abs(roll2[ex2[0]["restimulus"] == mv]), 80) for mv in (9, 10, 11, 12)
                 if (ex2[0]["restimulus"] == mv).any()]
        if peaks and np.mean(peaks) > 1e-3:
            gain = np.clip(0.7 * 1.3 / np.mean(peaks), 1.0, 8.0)
    acc = _lowpass(rec["acc"], 3.0, rec["fs"])
    pro_sup = np.clip((np.unwrap(np.arctan2(acc[:, 1], acc[:, 2])) - roll_ref) * gain / 1.3, -1, 1)
    at_rest = np.clip(_lowpass((rec["restimulus"] == 0).astype(float), 1.5, rec["fs"]), 0, 1)
    return pro_sup * (1 - at_rest)                                     # relaxed forearm at rest


def _from_labels(rec, channel):
    """A smooth trajectory for a channel the glove does not measure, from the exercise-2 movement labels."""
    target = np.zeros(len(rec["restimulus"]))
    if rec["exercise"] == 2:
        for mv, (ch, value) in EX2_TARGETS.items():
            if ch == channel:
                target[rec["restimulus"] == mv] = value
    return _lowpass(target, 1.5, rec["fs"])


def _lowpass(x, fc, fs):
    return signal.sosfiltfilt(signal.butter(2, fc, fs=fs, output="sos"), x, axis=0)


def plot(path, real, sim, motion, fs, title, channels=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(real)) / fs
    C = min(channels, real.shape[1])
    fig, ax = plt.subplots(4, 1, figsize=(14, 13), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.5, 1.5]})
    for a, x, name in ((ax[0], real, "real"), (ax[1], sim, "simulated")):
        off = 3 * np.percentile(np.abs(x[:, :C]), 99) + 1e-9
        for c in range(C):
            a.plot(t, x[:, c] - c * off, lw=0.4)
        a.set_title(f"{name} EMG (first {C} channels)")
        a.set_yticks([])
    env_real = mg.calibration.envelope(real, fs, motion["device"]).mean(1)
    env_sim = mg.calibration.envelope(sim, fs, motion["device"]).mean(1)
    te = np.arange(len(env_real)) / mg.calibration.ENV_FS
    ax[2].plot(te, env_real / env_real.max(), label="real")
    ax[2].plot(te, env_sim / env_sim.max(), label="simulated")
    ax[2].set_title("mean envelope (each normalised to its max)")
    ax[2].legend()
    tm = np.arange(len(motion["angles"])) / motion["fs"]
    for name in ("pm3_flexion_r", "flexion_r", "pro_sup_r"):
        ax[3].plot(tm, np.degrees(motion["angles"][:, mg.body.HAND_JOINTS.index(name)]), label=name)
    ax[3].fill_between(tm, *ax[3].get_ylim(), where=motion["rest"], color="0.9", label="rest")
    ax[3].set_title("joint angles (deg)")
    ax[3].legend(ncol=4, fontsize=8)
    ax[3].set_xlabel("s")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", help="a NinaPro DB1 / DB5 .mat file")
    ap.add_argument("--engine", choices=("analytic", "biomime"), default="analytic")
    ap.add_argument("--muap-bank", help="biomime: MUAP bank (default: where `myogen biomime-setup` puts it)")
    ap.add_argument("--gain", type=float, default=1.0, help="biomime: global amplitude")
    ap.add_argument("--electrodes", help="electrode profile YAML (default: the dataset's preset, myo / otto_bock)")
    ap.add_argument("--subject-seed", type=int, help="a random subject (Subject.random) instead of the average one")
    ap.add_argument("--start", type=float, default=0.0, help="s into the recording")
    ap.add_argument("--duration", type=float, default=30.0, help="s to simulate (simulation runs ~2x real time)")
    ap.add_argument("--calibrate", metavar="PROFILE", help="fit the device settings on this window, write PROFILE")
    ap.add_argument("--fit-rotation", action="store_true", help="with --calibrate: also fit the armband rotation")
    ap.add_argument("--out", default="ninapro_compare", help="output directory")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rec = load_mat(a.mat)
    siblings = [load_mat(p) for p in sorted(Path(a.mat).parent.glob("*.mat"))]
    fs = rec["fs"]
    i0, i1 = int(a.start * fs), int((a.start + a.duration) * fs)
    if i1 > len(rec["emg"]):
        raise SystemExit(f"{a.mat} is {len(rec['emg']) / fs:.0f} s long")
    angles = {k: v[i0:i1] for k, v in glove_to_angles(rec, siblings).items()}
    rest = rec["restimulus"][i0:i1] == 0
    motion = mg.JointAdapter()(angles, fs, rest=rest)
    real = rec["emg"][i0:i1]

    device = "myo" if rec["dataset"] == "db5" else "otto_bock"
    electrodes = mg.Electrodes.load(a.electrodes) if a.electrodes else mg.Electrodes.preset(device)
    subject = mg.Subject.random(a.subject_seed) if a.subject_seed is not None else mg.Subject()
    options = {"muap_bank": a.muap_bank, "gain": a.gain} if a.engine == "biomime" else {}
    sim = mg.Simulator(electrodes, engine=a.engine, subject=subject, **options)
    run = sim.simulate_clean(motion)
    recording = sim.record(run)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"file": str(a.mat), "engine": a.engine, "window_s": [a.start, a.start + a.duration],
              "tracking_rmse_deg": recording.qc["rmse_deg"]}
    if a.calibrate:
        cal = mg.calibrate(sim, [motion], [real], [rest], fit_rotation=a.fit_rotation)
        print(cal)
        print(f"wrote {cal.electrodes.save(a.calibrate)}")
        sim = sim.with_electrodes(cal.electrodes) if a.fit_rotation else sim
        recording = sim.record(sim.regenerate(run), cal.electrodes)
        report["calibration"] = {"fitted": cal.fitted, "statistics": cal.report}
    stats = mg.compare(real, recording.emg, fs, electrodes.device, rest)
    report["statistics"] = {k: {"real": r, "simulated": s} for k, (r, s) in stats.items()}

    recording.save(out / "simulated.h5")
    (out / "comparison.json").write_text(json.dumps(report, indent=1))
    print(f"\n{rec['path'].name}  {a.engine} engine  {a.start:.0f}-{a.start + a.duration:.0f} s  "
          f"tracking RMSE {recording.qc['rmse_deg']:.2f} deg")
    print(f"{'statistic':24s}{'real':>10s}{'simulated':>11s}")
    for k, (r, s) in stats.items():
        print(f"{k:24s}" + "".join("        -" if v is None else f"{v:10.4g}" for v in (r, s)))
    try:
        plot(out / "comparison.png", real, recording.emg, {"angles": motion.angles, "fs": fs, "rest": rest,
                                                            "device": electrodes.device}, fs,
             f"{rec['path'].name} ({a.start:.0f}-{a.start + a.duration:.0f} s): real vs {a.engine}")
    except ImportError:
        print("matplotlib not installed: no figure")
    print(f"wrote {out}/simulated.h5, comparison.json, comparison.png  (myogen replay {out}/simulated.h5)")


if __name__ == "__main__":
    main()
