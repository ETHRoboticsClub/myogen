"""Simulate the EMG of a NinaPro recording from its glove motion, and compare it with the real EMG.

    python scripts/ninapro_compare.py data/real/ninapro_db5/s1/S1_E2_A1.mat --start 20 --duration 30
    python scripts/ninapro_compare.py data/real/ninapro_db1/s1/S1_A1_E2.mat

Needs myogen (pip install -e .), and matplotlib for the figure. Writes into --out:
    simulated.h5      the myogen Recording (myogen replay simulated.h5)
    comparison.json   real vs simulated statistics (myogen.compare)
    comparison.png    real and simulated EMG, envelopes and the joint motion
With --calibrate PROFILE.yml it also fits the device settings on this window (with --fit-rotation also the
armband rotation, unknown in NinaPro) and writes the profile. The comparison is then no longer independent:
calibrate on other subjects than the ones you report.

Glove -> joint angles: see ninapro.py (a heuristic retargeting: the glove is uncalibrated).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

import myogen as mg

try:
    from ninapro import glove_to_angles, load_mat
except ImportError:                                   # run as python -m scripts.ninapro_compare
    from scripts.ninapro import glove_to_angles, load_mat


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
    sim = mg.Simulator(electrodes, subject=subject)
    run = sim.simulate_clean(motion)
    recording = sim.record(run)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"file": str(a.mat), "window_s": [a.start, a.start + a.duration],
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
    print(f"\n{rec['path'].name}  {a.start:.0f}-{a.start + a.duration:.0f} s  "
          f"tracking RMSE {recording.qc['rmse_deg']:.2f} deg")
    print(f"{'statistic':24s}{'real':>10s}{'simulated':>11s}")
    for k, (r, s) in stats.items():
        print(f"{k:24s}" + "".join("        -" if v is None else f"{v:10.4g}" for v in (r, s)))
    try:
        plot(out / "comparison.png", real, recording.emg, {"angles": motion.angles, "fs": fs, "rest": rest,
                                                            "device": electrodes.device}, fs,
             f"{rec['path'].name} ({a.start:.0f}-{a.start + a.duration:.0f} s): real vs simulated")
    except ImportError:
        print("matplotlib not installed: no figure")
    print(f"wrote {out}/simulated.h5, comparison.json, comparison.png  (myogen replay {out}/simulated.h5)")


if __name__ == "__main__":
    main()
